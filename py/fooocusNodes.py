import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np
import folder_paths
from comfy.samplers import *
import fooocus_modules.config as config
import fooocus_modules.default_pipeline as pipeline
import fooocus_modules.core as core
from extras.expansion import FooocusExpansion
from extras.expansion import safe_str
import extras.preprocessors as preprocessors
from fooocus import get_local_filepath
from nodes import SaveImage, PreviewImage
from fooocus_modules.util import (
    remove_empty_str,
    HWC3,
    resize_image,
    get_image_shape_ceil,
    set_image_shape_ceil,
    get_shape_ceil,
    resample_image,
    erode_or_dilate,
)
from fooocus_modules.upscaler import perform_upscale
import fooocus_modules.inpaint_worker as inpaint_worker
import fooocus_modules.patch
from typing import   Tuple

import comfy.samplers

MIN_SEED = 0
MAX_SEED = 2**63 - 1

# lora
class FooocusLoraStack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 10
        inputs = {
            "required": {
                "toggle": ([True, False],),
                "num_loras": ("INT", {"default": 1, "min": 0, "max": max_lora_num}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
            },
        }

        for i in range(1, max_lora_num + 1):
            inputs["optional"][f"lora_{i}_name"] = (
                ["None"] + folder_paths.get_filename_list("loras"),
                {"default": "None"},
            )
            inputs["optional"][f"lora_{i}_strength"] = (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
            )
        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"

    CATEGORY = "Fooocus"

    def stack(self, toggle, num_loras, lora_stack=None, **kwargs):
        loras = []
        if (toggle in [False, None, "False"]) or not kwargs:
            return (loras,)

        # Import Stack values
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        # Import Lora values
        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_{i}_name")

            if not lora_name or lora_name == "None":
                continue

            lora_strength = float(kwargs.get(f"lora_{i}_strength"))
            loras.append([lora_name, lora_strength])

        return (loras,)


class FooocusLoader:
    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [
            f"{width} x {height}" for width, height in config.BASE_RESOLUTIONS
        ]
        return {
            "required": {
                "base_model_name": (folder_paths.get_filename_list("checkpoints"), {"default": "juggernautXL_v8Rundiffusion.safetensors"},),
                "refiner_model_name": (["None"] + folder_paths.get_filename_list("checkpoints"), {"default": "None"},),
                "refiner_switch": ("FLOAT", {"default": 0.618, "min": 0.1, "max": 1, "step": 0.1},),
                "refiner_swap_method": (["joint", "separate", "vae"],),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "prompt_expansion": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "resolution": (resolution_strings, {"default": "1024 x 1024"}),
                "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8},),
                "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8},),
                "image_number": ("INT", {"default": 1, "min": 1, "max": 100}), },
            "optional": {"optional_lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    FUNCTION = "fooocus_loader"
    CATEGORY = "Fooocus"

    @classmethod
    def get_resolution_strings(cls):
        return [f"{width} x {height}" for width, height in config.BASE_RESOLUTIONS]

    def process_resolution(self, resolution: str) -> Tuple[int, int]:
        if resolution == "自定义 x 自定义":
            return None
        try:
            width, height = map(int, resolution.split(" x "))
            return width, height
        except ValueError:
            raise ValueError("Invalid base_resolution format.")

    def fooocus_loader(self, optional_lora_stack=[], **kwargs):
        resolution = kwargs.pop("resolution")
        empty_latent_width, empty_latent_height = self.process_resolution(
            resolution
        )
        pipe = {
            # 将关键字参数赋值给pipe字典
            key: value
            for key, value in kwargs.items()
            if key not in ("empty_latent_width", "empty_latent_height")
        }
        positive_prompt = kwargs["positive_prompt"]
        if positive_prompt != "" and pipe["prompt_expansion"]:
            pipeline.final_expansion = FooocusExpansion()
            positive_prompt = pipeline.final_expansion(
                positive_prompt,
                pipe["seed"],
            )
            print("PromptExpansion is:  " + positive_prompt)
        else:
            print("PromptExpansion is off or positive prompt is none!!  ")

        pipe.update(
            {
                "positive_prompt": positive_prompt,
                "negative_prompt": kwargs["negative_prompt"],
                "latent_width": empty_latent_width,
                "latent_height": empty_latent_height,
                "optional_lora_stack": optional_lora_stack,
                "use_cn": False,
            }
        )

        return {
            "ui": {
                "positive": pipe["positive_prompt"],
                "negative": pipe["negative_prompt"],
            },
            "result": (
                pipe,

            ),
        }


class FooocusPreKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "generation_mode": (["text_or_images_to_images", "inpaint", "outpaint"], {"default": "text_or_images_to_images"}, ),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.5},),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m_sde_gpu", },),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", },),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01},),
                "settings": (["Simple", "Advanced"], {"default": "Simple"}),
                "sharpness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0}),
                "adaptive_cfg": ("FLOAT", {"default": 7, "min": 0.0, "max": 100.0}),
                "adm_scaler_positive": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.1},),
                "adm_scaler_negative": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.1},),
                "adm_scaler_end": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1},),
                "controlnet_softness": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},),
                "inpaint_respective_field": ("FLOAT", {"default": 0.618, "min": 0.1, "max": 1.0, "step": 0.1},),
                "inpaint_engine":(list(config.FOOOCUS_INPAINT_PATCH.keys()),),
                "top": ("BOOLEAN", {"default": False}),
                "bottom": ("BOOLEAN", {"default": False}),
                "left": ("BOOLEAN", {"default": False}),
                "right": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "inpaint_image": ("IMAGE",),
                "inpaint_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("pipe", "model", "clip", "vae")

    FUNCTION = "fooocus_preKSampler"
    CATEGORY = "Fooocus"

    def fooocus_preKSampler(self, pipe: dict, inpaint_image=None, inpaint_mask=None,inpaint_engine=None, **kwargs):
        # 检查pipe非空
        assert pipe is not None, "请先调用 FooocusLoader 进行初始化！"
        pipe.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in ("switch", "refiner_switch")
            }
        )
        if pipe["sampler_name"] == "lcm":
            pipe["scheduler"] = "lcm"
            pipe["refiner_model_name"] = "None"
            pipe["refiner_switch"] = 1.0
            fooocus_modules.patch.sharpness = 0.0
            fooocus_modules.patch.adaptive_cfg = 1.0
            fooocus_modules.patch.positive_adm_scale = 1.0
            fooocus_modules.patch.negative_adm_scale = 1, 0
            fooocus_modules.patch.adm_scaler_end = 0.0

        config.controlnet_softness = kwargs.pop("controlnet_softness")

        fooocus_modules.patch.adaptive_cfg = kwargs.pop("adaptive_cfg")
        fooocus_modules.patch.sharpness = kwargs.pop("sharpness")
        fooocus_modules.patch.positive_adm_scale = kwargs.pop("adm_scaler_positive")
        fooocus_modules.patch.negative_adm_scale = kwargs.pop("adm_scaler_negative")
        fooocus_modules.patch.adm_scaler_end = kwargs.pop("adm_scaler_end")
        denoising_strength = kwargs.pop("denoise")
        inpaint_respective_field = pipe["inpaint_respective_field"]
        # 更新pipe参数
        steps = kwargs.get("steps")
        refiner_switch = pipe["refiner_switch"]
        base_model_additional_loras = []
        inpaint_worker.current_task = None
        use_synthetic_refiner = False
        if inpaint_image is None:
            pipe["generation_mode"] = "text_or_images_to_images"
        if pipe["generation_mode"] == "inpaint":
            pipe["top"] = False
            pipe["bottom"] = False
            pipe["left"] = False
            pipe["right"] = False

        if (pipe["generation_mode"] == "inpaint" or pipe["generation_mode"] == "outpaint"):
            head_file = get_local_filepath(config.FOOOCUS_INPAINT_HEAD["fooocus_inpaint_head"]["model_url"], config.INPAINT_DIR)
            patch_file = get_local_filepath(config.FOOOCUS_INPAINT_PATCH[inpaint_engine]["model_url"], config.INPAINT_DIR)
            base_model_additional_loras += [(patch_file, 1.0)]
            if pipe["refiner_model_name"] == "None":
                use_synthetic_refiner = True
                refiner_switch = 0.5
        switch = int(round(steps * refiner_switch))
        # 加载模型
        pipeline.refresh_everything(
            refiner_model_name=pipe["refiner_model_name"],
            base_model_name=pipe["base_model_name"],
            loras=pipe["optional_lora_stack"],
            base_model_additional_loras=base_model_additional_loras,
            use_synthetic_refiner=use_synthetic_refiner,
        )
        # 处理提示词
        prompts = remove_empty_str(
            [safe_str(p) for p in pipe["positive_prompt"].splitlines()], default=""
        )
        negative_prompts = remove_empty_str(
            [safe_str(p) for p in pipe["negative_prompt"].splitlines()],
            default="",
        )
        positive = pipeline.clip_encode(prompts, len(prompts))
        negative = pipeline.clip_encode(
            negative_prompts, len(negative_prompts))
        # 预处理latent
        if pipe["generation_mode"] == "text_or_images_to_images":
            if inpaint_image == None:
                initial_latent = core.generate_empty_latent(
                    pipe["latent_width"], pipe["latent_height"])
                denoising_strength = 1.0
            else:
                candidate_vae, _ = pipeline.get_candidate_vae(
                    steps=steps,
                    switch=switch,
                    denoise=denoising_strength,
                    refiner_swap_method=pipe["refiner_swap_method"],
                )
                if isinstance(inpaint_image, list):
                    inpaint_image = inpaint_image[0]
                    inpaint_image = inpaint_image.unsqueeze(0)
                initial_latent = core.encode_vae(candidate_vae, inpaint_image)
        else:
            inpaint_image = inpaint_image[0].numpy()
            inpaint_image = (inpaint_image * 255).astype(np.uint8)
            if pipe["top"] or pipe["bottom"] or pipe["left"] or pipe["right"]:
                print("启用扩图！")
                inpaint_mask = np.zeros(inpaint_image.shape, dtype=np.uint8)
                inpaint_mask = inpaint_mask[:, :, 0]
                H, W, C = inpaint_image.shape
                if pipe["top"]:
                    inpaint_image = np.pad(
                        inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode="edge"
                    )
                    inpaint_mask = np.pad(
                        inpaint_mask,
                        [[int(H * 0.3), 0], [0, 0]],
                        mode="constant",
                        constant_values=255,
                    )
                if pipe["bottom"]:
                    inpaint_image = np.pad(
                        inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode="edge"
                    )
                    inpaint_mask = np.pad(
                        inpaint_mask,
                        [[0, int(H * 0.3)], [0, 0]],
                        mode="constant",
                        constant_values=255,
                    )

                H, W, C = inpaint_image.shape
                if pipe["left"]:
                    inpaint_image = np.pad(
                        inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode="edge"
                    )
                    inpaint_mask = np.pad(
                        inpaint_mask,
                        [[0, 0], [int(H * 0.3), 0]],
                        mode="constant",
                        constant_values=255,
                    )
                if pipe["right"]:
                    inpaint_image = np.pad(
                        inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode="edge"
                    )
                    inpaint_mask = np.pad(
                        inpaint_mask,
                        [[0, 0], [0, int(H * 0.3)]],
                        mode="constant",
                        constant_values=255,
                    )
                inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
                denoising_strength = 1.0
                inpaint_respective_field = 1.0

            elif inpaint_mask == None:
                raise Exception("inpaint_mask is None!!")
            else:
                inpaint_mask = inpaint_mask[0].numpy()
                inpaint_mask = (inpaint_mask * 255).astype(np.uint8)
            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=inpaint_respective_field,
            )
            inpaint_pixel_fill = core.numpy_to_pytorch(
                inpaint_worker.current_task.interested_fill
            )
            inpaint_pixel_image = core.numpy_to_pytorch(
                inpaint_worker.current_task.interested_image
            )
            inpaint_pixel_mask = core.numpy_to_pytorch(
                inpaint_worker.current_task.interested_mask
            )
            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=pipe["steps"],
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=pipe["refiner_swap_method"],
            )
            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask, vae=candidate_vae, pixels=inpaint_pixel_image
            )
            latent_swap = None
            if candidate_vae_swap is not None:
                print("正在编码SD1.5局部重绘VAE……")
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap, pixels=inpaint_pixel_fill
                )["samples"]

            print("正在编码VAE……")
            latent_fill = core.encode_vae(vae=candidate_vae, pixels=inpaint_pixel_fill)[
                "samples"
            ]

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill,
                latent_mask=latent_mask,
                latent_swap=latent_swap,
            )
            pipeline.final_unet = inpaint_worker.current_task.patch(
                inpaint_head_model_path=head_file,
                inpaint_latent=latent_inpaint,
                inpaint_latent_mask=latent_mask,
                model=pipeline.final_unet,
            )
            initial_latent = {"samples": latent_fill}

            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f"最终分辨率是 {str((final_height, final_width))}.")
        B, C, H, W = initial_latent["samples"].shape
        height, width = H * 8, W * 8
        print(f"初始分辨率是 {str((height, width))}.")
        pipe.update(
            {
                "positive": positive,
                "negative": negative,
                "denoise": denoising_strength,
                "latent": initial_latent,
                "model": pipeline.final_unet,
                "inpaint_respective_field": inpaint_respective_field,
                "height": height,
                "width": width,
                "switch": switch,
            }
        )
        new_pipe = pipe.copy()
        del pipe
        return {"ui": {"value": [new_pipe["seed"]]}, "result": (new_pipe, pipeline.final_unet, pipeline.final_clip, pipeline.final_vae)}


class FooocusKsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save",], {"default": "Preview"},),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "optional": {"model": ("MODEL",), },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE")
    RETURN_NAMES = ("pipe", "image")
    OUTPUT_NODE = True
    FUNCTION = "ksampler"
    CATEGORY = "Fooocus"

    def ksampler(self, pipe, image_output, save_prefix, model=None, prompt=None, extra_pnginfo=None):
        if model is not None:
            pipeline.final_unet = model
        if pipe["use_cn"]:
            positive = pipe["cn_positive"]
            negative = pipe["cn_negative"]
        else:
            positive = pipe["positive"]
            negative = pipe["negative"]
        all_imgs = []
        for i in range(1, pipe["image_number"] + 1):
            print(f"正在生成第 {i} 张图像……")
            imgs = pipeline.process_diffusion(
                positive_cond=positive,
                negative_cond=negative,
                steps=pipe["steps"],
                switch=pipe["switch"],
                width=pipe["latent_width"],
                height=pipe["latent_height"],
                image_seed=pipe["seed"]+i,
                callback=None,
                sampler_name=pipe["sampler_name"],
                scheduler_name=pipe["scheduler"],
                latent=pipe["latent"],
                denoise=pipe["denoise"],
                tiled=False,
                cfg_scale=pipe["cfg"],
                refiner_swap_method=pipe["refiner_swap_method"],
            )
            if inpaint_worker.current_task is not None:
                imgs = [inpaint_worker.current_task.post_process(
                    x) for x in imgs]
            imgs = [np.array(img).astype(np.float32) / 255.0 for img in imgs]
            imgs = [torch.from_numpy(img) for img in imgs]
            all_imgs.extend(imgs)

        if image_output in ("Save", "Hide/Save"):
            saveimage = SaveImage()
            results = saveimage.save_images(
                all_imgs, save_prefix, prompt, extra_pnginfo)

        if image_output == "Preview":
            previewimage = PreviewImage()
            results = previewimage.save_images(
                all_imgs, save_prefix, prompt, extra_pnginfo)

        if image_output == "Hide":
            return {"ui": {"value": list()}, "result": (pipe, all_imgs)}

        results["result"] = pipe, all_imgs
        return results


class FooocusHirefix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "upscale": ([1.5, 2.0], {"default": 1.5, }),
                "steps": ("INT", {"default": 18, "min": 10, "max": 100}),
                "denoise": ("FLOAT", {"default": 0.382, "min": 0.00, "max": 1.00, "step": 0.001},),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save",], {"default": "Preview"},),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    RETURN_TYPES = ("PIPE_LINE", "IMAGE")
    RETURN_NAMES = ("pipe", "image")
    OUTPUT_NODE = True
    FUNCTION = "fooocusHirefix"
    CATEGORY = "Fooocus"

    def fooocusHirefix(self, pipe, image, upscale, denoise, steps, image_output, save_prefix, prompt=None, extra_pnginfo=None):
        if isinstance(image, list):
            image = image[0]
            image = image.unsqueeze(0)

        image = image[0].numpy()
        image = (image * 255).astype(np.uint8)
        image = HWC3(image)
        H, W, C = image.shape
        print(f'放大中的图片来自于 {str((H, W))} ...')
        if isinstance(image, list):
            image = np.array(image)
        uov_input_image = perform_upscale(image)
        print(f'图片已放大。')
        f = upscale
        shape_ceil = get_shape_ceil(H * f, W * f)
        if shape_ceil < 1024:
            print(f'[放大] 图像因尺寸过小已被重新调整。')
            uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
            shape_ceil = 1024
        else:
            uov_input_image = resample_image(
                uov_input_image, width=W * f, height=H * f)
        initial_pixels = core.numpy_to_pytorch(uov_input_image)
        candidate_vae, _ = pipeline.get_candidate_vae(
            steps=pipe["steps"],
            switch=pipe["switch"],
            denoise=denoise,
            refiner_swap_method=pipe["refiner_swap_method"]
        )

        initial_latent = core.encode_vae(
            vae=candidate_vae,
            pixels=initial_pixels, tiled=True)
        B, C, H, W = initial_latent['samples'].shape
        width = W * 8
        height = H * 8
        print(f'最终解决方案是 {str((height, width))}.')

        imgs = pipeline.process_diffusion(
            positive_cond=pipe["positive"],
            negative_cond=pipe["negative"],
            steps=steps,
            switch=pipe["switch"],
            width=pipe["latent_width"],
            height=pipe["latent_height"],
            image_seed=pipe["seed"],
            callback=None,
            sampler_name=pipe["sampler_name"],
            scheduler_name=pipe["scheduler"],
            latent=initial_latent,
            denoise=denoise,
            tiled=True,
            cfg_scale=pipe["cfg"],
            refiner_swap_method=pipe["refiner_swap_method"],
        )

        if inpaint_worker.current_task is not None:
            imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

        imgs = [np.array(img).astype(np.float32) / 255.0 for img in imgs]
        imgs = [torch.from_numpy(img) for img in imgs]
        if image_output in ("Save", "Hide/Save"):
            saveimage = SaveImage()
            results = saveimage.save_images(
                imgs, save_prefix, prompt, extra_pnginfo)

        if image_output == "Preview":
            previewimage = PreviewImage()
            results = previewimage.save_images(
                imgs, save_prefix, prompt, extra_pnginfo)

        if image_output == "Hide":
            return {"ui": {"value": list()}, "result": (pipe, imgs)}

        results["result"] = pipe, imgs
        return results


class FooocusControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "controlnet": (["None"] + folder_paths.get_filename_list("controlnet"), {"default": "None"},),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},),
                "preprocess": ("BOOLEAN", {"default": True},),

            },
        }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE")
    RETURN_NAMES = ("pipe", "image")
    OUTPUT_NODE = True
    FUNCTION = "apply_controlnet"
    CATEGORY = "Fooocus"

    def apply_controlnet(
        self, pipe, image, controlnet, strength, start_percent, end_percent, preprocess
    ):
        if controlnet == "None":
            return {"ui": {"value": list()}, "result": (pipe, image)}

        cn_path = folder_paths.get_full_path("controlnet", controlnet)
        # pipeline.refresh_controlnets([cn_path])

        image = image[0].numpy()
        image = (image * 255).astype(np.uint8)
        image = resize_image(HWC3(image), pipe["width"], pipe["height"])
        if preprocess:
            image = preprocessors.canny_pyramid(image)
            image = HWC3(image)
        image = core.numpy_to_pytorch(image)

        positive_cond, negative_cond = core.apply_controlnet(
            pipe["positive"],
            pipe["negative"],
            core.load_controlnet(cn_path),
            image,
            strength,
            start_percent,
            end_percent,
        )
        new_pipe = pipe.copy()
        new_pipe["use_cn"] = True
        new_pipe["cn_positive"] = positive_cond
        new_pipe["cn_negative"] = negative_cond
        return (new_pipe, image,)


NODE_CLASS_MAPPINGS = {

    "Fooocus Loader": FooocusLoader,
    "Fooocus PreKSampler": FooocusPreKSampler,
    "Fooocus KSampler": FooocusKsampler,
    "Fooocus Hirefix": FooocusHirefix,
    "Fooocus LoraStack": FooocusLoraStack,
    "Fooocus Controlnet": FooocusControlnet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fooocus positive": "Positive",
    "Fooocus negative": "Negative",
    "Fooocus stylesSelector": "stylesPromptSelector",

}
