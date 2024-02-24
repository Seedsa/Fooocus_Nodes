import os
import sys

sys.path.append(os.path.dirname(__file__))
modules_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import folder_paths
from comfy.samplers import *
import config as config
import modules.default_pipeline as pipeline
import modules.core as core
from modules.sdxl_styles import apply_style,apply_wildcards,fooocus_expansion
from extras.expansion import FooocusExpansion
from extras.expansion import safe_str
import extras.face_crop as face_crop
import modules.flags as flags

import extras.preprocessors as preprocessors
import extras.ip_adapter as ip_adapter
from fooocus import get_local_filepath
from nodes import SaveImage, PreviewImage
from modules.util import (
    remove_empty_str,
    HWC3,
    resize_image,
    get_image_shape_ceil,
    set_image_shape_ceil,
    get_shape_ceil,
    resample_image,
    erode_or_dilate,
)
import ldm_patched.modules.model_management as model_management

from modules.upscaler import perform_upscale
import modules.inpaint_worker as inpaint_worker
import modules.patch
from typing import   Tuple
from log import log_node_info,log_node_error,log_node_success
import random
import time
import copy

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
                "resolution": (resolution_strings, {"default": "1024 x 1024"}),
                "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8},),
                "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8},),
                "image_number": ("INT", {"default": 1, "min": 1, "max": 100}), },
            "optional": {"optional_lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("PIPE_LINE",)
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
        if resolution != "自定义 x 自定义":
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")
        else:
            empty_latent_width = kwargs.pop("empty_latent_width")
            empty_latent_height = kwargs.pop("empty_latent_height")
        pipe = {
            # 将关键字参数赋值给pipe字典
            key: value
            for key, value in kwargs.items()
            if key not in ("empty_latent_width", "empty_latent_height")
        }
        positive_prompt = kwargs["positive_prompt"]
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
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.5},),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m_sde_gpu", },),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", },),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01},),
                "settings": (["Simple", "Advanced"], {"default": "Simple"}),
                "sharpness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0}),
                "adaptive_cfg": ("FLOAT", {"default": 7, "min": 0.0, "max": 100.0}),
                "adm_scaler_positive": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.1},),
                "adm_scaler_negative": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.1},),
                "adm_scaler_end": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1},),
                "controlnet_softness": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},),
            },
            "optional": {
                "image_to_latent": ("IMAGE",),
                "latent":("LATENT",),
                "fooocus_inpaint":("FOOOCUS_INPAINT",),
                "fooocus_styles":("FOOOCUS_STYLES",),
            },
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CLIP", "VAE","CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("pipe", "model", "clip", "vae","CONDITIONING+","CONDITIONING-")

    FUNCTION = "fooocus_preKSampler"
    CATEGORY = "Fooocus"

    def fooocus_preKSampler(self, pipe: dict,image_to_latent=None, latent=None,fooocus_inpaint=None,fooocus_styles=None,**kwargs):
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
            modules.patch.sharpness = 0.0
            modules.patch.adaptive_cfg = 1.0
            modules.patch.positive_adm_scale = 1.0
            modules.patch.negative_adm_scale = 1, 0
            modules.patch.adm_scaler_end = 0.0

        config.controlnet_softness = kwargs.pop("controlnet_softness")

        modules.patch.adaptive_cfg = kwargs.pop("adaptive_cfg")
        print(f'[Parameters] Adaptive CFG = {modules.patch.adaptive_cfg}')

        modules.patch.sharpness = kwargs.pop("sharpness")
        print(f'[Parameters] Sharpness = {modules.patch.sharpness}')
        modules.patch.positive_adm_scale = kwargs.pop("adm_scaler_positive")
        modules.patch.negative_adm_scale = kwargs.pop("adm_scaler_negative")
        modules.patch.adm_scaler_end = kwargs.pop("adm_scaler_end")
        print(f'[Parameters] ADM Scale = '
              f'{modules.patch.positive_adm_scale} : '
              f'{modules.patch.negative_adm_scale} : '
              f'{modules.patch.adm_scaler_end}')
        print(f'[Parameters] CFG = {kwargs.get("cfg")}')
        print(f'[Parameters] Seed = {kwargs.get("seed")}')

        denoising_strength = kwargs.pop("denoise")
        height = pipe["latent_height"]
        width = pipe["latent_width"]
        if fooocus_styles is not None:
            style_selections = fooocus_styles
        else:
            style_selections= []
        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        # 更新pipe参数
        steps = kwargs.get("steps")
        refiner_switch = pipe["refiner_switch"]
        base_model_additional_loras = []
        inpaint_worker.current_task = None
        use_synthetic_refiner = False

        if fooocus_inpaint is not None:
            inpaint_engine = fooocus_inpaint.get("inpaint_engine")
            if inpaint_engine != 'None':
              head_file = get_local_filepath(config.FOOOCUS_INPAINT_HEAD["fooocus_inpaint_head"]["model_url"], config.INPAINT_DIR)
              patch_file = get_local_filepath(config.FOOOCUS_INPAINT_PATCH[inpaint_engine]["model_url"], config.INPAINT_DIR)
              base_model_additional_loras += [(patch_file, 1.0)]
              if pipe["refiner_model_name"] == "None":
                  use_synthetic_refiner = True
                  refiner_switch = 0.5
        switch = int(round(steps * refiner_switch))
        print(f'[Parameters] Sampler = {pipe["sampler_name"]} - {pipe["scheduler"]}')
        print(f'[Parameters] Steps = {steps} - {switch}')
        # 加载模型
        print('Loading models ...')
        pipeline.refresh_everything(
            refiner_model_name=pipe["refiner_model_name"],
            base_model_name=pipe["base_model_name"],
            loras=pipe["optional_lora_stack"],
            base_model_additional_loras=base_model_additional_loras,
            use_synthetic_refiner=use_synthetic_refiner,
        )
        tasks = []
        print('Processing prompts ...')
        prompts = remove_empty_str([safe_str(p) for p in pipe["positive_prompt"].splitlines()], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in pipe["negative_prompt"].splitlines()], default='')
        prompt = prompts[0]
        negative_prompt = negative_prompts[0]
        positive = pipeline.clip_encode(prompts, len(prompts))
        negative = pipeline.clip_encode(
            negative_prompts, len(negative_prompts))
        if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                use_expansion = False
        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []
        for i in range(pipe["image_number"]):
            task_seed = (kwargs.get("seed") + i) % (MAX_SEED + 1)  # randint is inclusive, % is not
            task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future
            task_prompt = apply_wildcards(prompt, task_rng)
            task_negative_prompt = apply_wildcards(negative_prompt, task_rng)
            task_extra_positive_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
            task_extra_negative_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_negative_prompts]
            positive_basic_workloads = []
            negative_basic_workloads = []
            if use_style:
                for s in style_selections:
                    p, n = apply_style(s, positive=task_prompt)
                    positive_basic_workloads = positive_basic_workloads + p
                    negative_basic_workloads = negative_basic_workloads + n
            else:
                positive_basic_workloads.append(task_prompt)

            negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

            positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
            negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

            positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
            negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)
            tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
            ))
        if use_expansion:
            for i, t in enumerate(tasks):
                    print(f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    print(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.
        for i, t in enumerate(tasks):
                print(f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])
        for i, t in enumerate(tasks):
                if abs(float(kwargs.get("cfg")) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    print(f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])
        if image_to_latent is not None:
            candidate_vae, _ = pipeline.get_candidate_vae(
                    steps=steps,
                    switch=switch,
                    denoise=denoising_strength,
                    refiner_swap_method=pipe["refiner_swap_method"],
            )
            if isinstance(image_to_latent, list):
                    image_to_latent = image_to_latent[0]
                    image_to_latent = image_to_latent.unsqueeze(0)
            initial_latent = core.encode_vae(candidate_vae, image_to_latent)
        elif latent is not None:
            initial_latent = latent
        else:
            initial_latent = None
        if fooocus_inpaint is not None:
            inpaint_image = fooocus_inpaint.get("image")
            inpaint_mask =  fooocus_inpaint.get("mask")
            inpaint_engine = fooocus_inpaint.get("inpaint_engine")
            inpaint_disable_initial_latent = fooocus_inpaint.get("inpaint_disable_initial_latent")
            inpaint_respective_field = fooocus_inpaint.get("inpaint_respective_field")
            top=fooocus_inpaint.get("top")
            bottom=fooocus_inpaint.get("bottom")
            left=fooocus_inpaint.get("left")
            right=fooocus_inpaint.get("right")
            inpaint_image = inpaint_image[0].numpy()
            inpaint_image = (inpaint_image * 255).astype(np.uint8)
            if top or bottom or left or right:
                print("启用扩图！")
                inpaint_mask = np.zeros(inpaint_image.shape, dtype=np.uint8)
                inpaint_mask = inpaint_mask[:, :, 0]
                H, W, C = inpaint_image.shape
                if top:
                    inpaint_image = np.pad(
                        inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode="edge"
                    )
                    inpaint_mask = np.pad(
                        inpaint_mask,
                        [[int(H * 0.3), 0], [0, 0]],
                        mode="constant",
                        constant_values=255,
                    )
                if bottom:
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
                if left:
                    inpaint_image = np.pad(
                        inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode="edge"
                    )
                    inpaint_mask = np.pad(
                        inpaint_mask,
                        [[0, 0], [int(H * 0.3), 0]],
                        mode="constant",
                        constant_values=255,
                    )
                if right:
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
            if inpaint_engine != 'None':
              pipeline.final_unet = inpaint_worker.current_task.patch(
                  inpaint_head_model_path=head_file,
                  inpaint_latent=latent_inpaint,
                  inpaint_latent_mask=latent_mask,
                  model=pipeline.final_unet,
              )
            if not inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}
            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        print(f'[Parameters] Denoising Strength = {denoising_strength}')
        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        pipe.update(
            {
                "tasks":tasks,
                "positive": prompt,
                "seed":kwargs.get("seed"),
                "negative": negative_prompt,
                "denoise": denoising_strength,
                "latent": initial_latent,
                "model": pipeline.final_unet,
                "height": height,
                "width": width,
                "switch": switch,
            }
        )
        new_pipe = pipe.copy()
        del pipe
        return {"ui": {"value": [new_pipe["seed"]]}, "result": (new_pipe, pipeline.final_unet, pipeline.final_clip, pipeline.final_vae,positive,negative)}


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
        all_imgs = []
        for current_task_id, task in enumerate(pipe["tasks"]):
            try:
                print(f"正在生成第 {current_task_id + 1} 张图像……")
                positive_cond, negative_cond = task['c'], task['uc']
                if "cn_tasks" in pipe and len(pipe["cn_tasks"]) > 0:
                    for cn_path,cn_img, cn_stop, cn_weight in pipe["cn_tasks"]:
                        positive_cond, negative_cond = core.apply_controlnet(
                            positive_cond, negative_cond,
                            core.load_controlnet(cn_path), cn_img, cn_weight, 0, cn_stop)
                imgs = pipeline.process_diffusion(
                  positive_cond=positive_cond,
                  negative_cond=negative_cond,
                  steps=pipe["steps"],
                  switch=pipe["switch"],
                  width=pipe["latent_width"],
                  height=pipe["latent_height"],
                  image_seed=task['task_seed'],
                  callback=None,
                  sampler_name=pipe["sampler_name"],
                  scheduler_name=pipe["scheduler"],
                  latent=pipe["latent"],
                  denoise=pipe["denoise"],
                  tiled=False,
                  cfg_scale=pipe["cfg"],
                  refiner_swap_method=pipe["refiner_swap_method"],
                )
                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]
                imgs = [np.array(img).astype(np.float32) / 255.0 for img in imgs]
                imgs = [torch.from_numpy(img) for img in imgs]
                all_imgs.extend(imgs)
            except model_management.InterruptProcessingException as e:
                print('task stopped')

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
                "cn_type":(config.cn_list, {"default": config.default_cn}, ),
                "cn_stop": ("FLOAT", {"default": config.default_parameters[config.default_cn][0], "min": 0.0, "max": 1.0, "step": 0.01},),
                "cn_weight": ("FLOAT", {"default": config.default_parameters[config.default_cn][1], "min": 0.0, "max": 2.0, "step": 0.01},),
                "skip_cn_preprocess": ("BOOLEAN", {"default": False},),
            },
        }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE")
    RETURN_NAMES = ("pipe", "image")
    OUTPUT_NODE = True
    FUNCTION = "apply_controlnet"
    CATEGORY = "Fooocus"

    def apply_controlnet(
        self, pipe, image, cn_type, cn_stop, cn_weight, skip_cn_preprocess
    ):
        print("process controlnet...")
        if cn_type == config.cn_canny:
          cn_path = get_local_filepath(config.FOOOCUS_IMAGE_PROMPT[config.cn_canny]["model_url"],config.CONTROLNET_DIR)
          image = image[0].numpy()
          image = (image * 255).astype(np.uint8)
          image = resize_image(HWC3(image), pipe["width"], pipe["height"])
          if not skip_cn_preprocess:
            image = preprocessors.canny_pyramid(image)
        if cn_type == config.cn_cpds:
          cn_path = get_local_filepath(config.FOOOCUS_IMAGE_PROMPT[config.cn_cpds]["model_url"],config.CONTROLNET_DIR)
          image = image[0].numpy()
          image = (image * 255).astype(np.uint8)
          image = resize_image(HWC3(image), pipe["width"], pipe["height"])
          if not skip_cn_preprocess:
            image = preprocessors.cpds(image)
        image = HWC3(image)
        image = core.numpy_to_pytorch(image)
        new_pipe = pipe.copy()
        if "cn_tasks" not in new_pipe:
            new_pipe["cn_tasks"] = []
        new_pipe["cn_tasks"].append([cn_path,image,cn_stop,cn_weight])
        return (new_pipe, image,)

class FooocusImagePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ip_type":(config.ip_list, {"default": config.default_ip}, ),
                "ip_stop": ("FLOAT", {"default": config.default_parameters[config.default_ip][0], "min": 0.0, "max": 1.0, "step": 0.01},),
                "ip_weight": ("FLOAT", {"default": config.default_parameters[config.default_ip][1], "min": 0.0, "max": 2.0, "step": 0.01},),
                "skip_cn_preprocess": ("BOOLEAN", {"default": False},),
            },
        }

    RETURN_TYPES = ("IMAGE_PROMPT",)
    RETURN_NAMES = ("image_prompt",)
    OUTPUT_NODE = True
    FUNCTION = "image_prompt"
    CATEGORY = "Fooocus"

    def image_prompt(
        self, image, ip_type, ip_stop, ip_weight, skip_cn_preprocess
    ):
        if ip_type == config.cn_ip:
          clip_vision_path, ip_negative_path, ip_adapter_path = config.downloading_ip_adapters('ip')
          ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
          image = image[0].numpy()
          image = (image * 255).astype(np.uint8)
          image = resize_image(HWC3(image), width=224, height=224, resize_mode=0)
          task = [image,ip_stop,ip_weight]
          task[0] = ip_adapter.preprocess(image, ip_adapter_path=ip_adapter_path)
        if ip_type == config.cn_ip_face:
          clip_vision_path, ip_negative_path,  ip_adapter_face_path = config.downloading_ip_adapters('face')
          ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)
          image = image[0].numpy()
          image = (image * 255).astype(np.uint8)
          image = HWC3(image)
          if not skip_cn_preprocess:
            image = face_crop.crop_image(image)
          image = resize_image(image, width=224, height=224, resize_mode=0)
          task = [image,ip_stop,ip_weight]
          task[0] = ip_adapter.preprocess(image, ip_adapter_path=ip_adapter_face_path)
        return (task, )

class FooocusApplyImagePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "image_prompt_a": ("IMAGE_PROMPT",),
                "image_prompt_b": ("IMAGE_PROMPT",),
                "image_prompt_c": ("IMAGE_PROMPT",),
                "image_prompt_d": ("IMAGE_PROMPT",),
            },
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_NODE = True
    FUNCTION = "apply_image_prompt"
    CATEGORY = "Fooocus"

    def apply_image_prompt(
        self, model, image_prompt_a=None, image_prompt_b=None, image_prompt_c=None, image_prompt_d=None
    ):
        image_prompt_tasks = []
        if image_prompt_a:
            image_prompt_tasks.append(image_prompt_a)
        if image_prompt_b:
            image_prompt_tasks.append(image_prompt_b)
        work_model = model.clone()
        new_model = ip_adapter.patch_model(work_model, image_prompt_tasks)
        return (new_model, )

class FooocusInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "inpaint_disable_initial_latent":("BOOLEAN", {"default": False}),
                "inpaint_respective_field": ("FLOAT", {"default": 0.618, "min": 0, "max": 1.0, "step": 0.1},),
                "inpaint_engine":(config.inpaint_engine_versions,{"default":"v2.6"},),
                "top": ("BOOLEAN", {"default": False}),
                "bottom": ("BOOLEAN", {"default": False}),
                "left": ("BOOLEAN", {"default": False}),
                "right": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }
    RETURN_TYPES = ("FOOOCUS_INPAINT",)
    RETURN_NAMES = ("fooocus_inpaint",)
    OUTPUT_NODE = True
    FUNCTION = "fooocus_inpaint"
    CATEGORY = "Fooocus"

    def fooocus_inpaint(
        self, image,inpaint_disable_initial_latent,inpaint_respective_field,inpaint_engine,top,bottom,left,right,mask=None
    ):
        fooocus_inpaint = {
            "image":image,
            "mask":mask,
            "inpaint_disable_initial_latent":inpaint_disable_initial_latent,
            "inpaint_respective_field":inpaint_respective_field,
            "inpaint_engine":inpaint_engine,
            "top":top,
            "bottom":bottom,
            "left":left,
            "right":right
        }
        return (fooocus_inpaint, )

NODE_CLASS_MAPPINGS = {

    "Fooocus Loader": FooocusLoader,
    "Fooocus PreKSampler": FooocusPreKSampler,
    "Fooocus KSampler": FooocusKsampler,
    "Fooocus Hirefix": FooocusHirefix,
    "Fooocus LoraStack": FooocusLoraStack,
    "Fooocus Controlnet": FooocusControlnet,
    "Fooocus ImagePrompt": FooocusImagePrompt,
    "Fooocus ApplyImagePrompt": FooocusApplyImagePrompt,
    "Fooocus Inpaint":FooocusInpaint
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fooocus positive": "Positive",
    "Fooocus negative": "Negative",
    "Fooocus stylesSelector": "stylesPromptSelector",

}
