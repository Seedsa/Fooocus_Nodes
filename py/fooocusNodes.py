import os
import sys

modules_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(modules_path)
from modules.patch import PatchSettings, patch_settings, patch_all
patch_all()
import numpy as np
import folder_paths
from comfy.samplers import *
import modules.config
import modules.flags
import modules.default_pipeline as pipeline
import modules.core as core
from modules.sdxl_styles import apply_style, fooocus_expansion, apply_arrays, random_style_name, get_random_style
from modules.util import apply_wildcards
from extras.expansion import safe_str
import extras.face_crop as face_crop
import modules.advanced_parameters as advanced_parameters
from extras.expansion import FooocusExpansion as Expansion
import extras.preprocessors as preprocessors
import extras.ip_adapter as ip_adapter
from nodes import SaveImage, PreviewImage, MAX_RESOLUTION, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from modules.util import (
    remove_empty_str,
    HWC3,
    resize_image,
    set_image_shape_ceil,
    get_shape_ceil,
    resample_image,
)
from libs.utils import easySave
from modules.upscaler import perform_upscale
import modules.inpaint_worker as inpaint_worker
import modules.patch
from typing import Tuple
from log import log_node_info
import random
import time
import copy

import comfy.samplers

MIN_SEED = 0
MAX_SEED = 2**63 - 1

# lora

pid = os.getpid()
print(f'Started worker with PID {pid}')


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
            f"{width} x {height}" for width, height in modules.config.BASE_RESOLUTIONS
        ]
        return {
            "required": {
                "base_model_name": (folder_paths.get_filename_list("checkpoints"), {"default": "juggernautXL_v8Rundiffusion.safetensors"},),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "refiner_model_name": (["None"] + folder_paths.get_filename_list("checkpoints"), {"default": "None"},),
                "refiner_switch": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1, "step": 0.1},),
                "refiner_swap_method": (["joint", "separate", "vae"],),
                "clip_skip": ("INT", {"default": 2, "min": -24, "max": 12, "step": 1}),
                "positive": ("STRING", {"default":"", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default":"", "placeholder": "Negative", "multiline": True}),
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
        return [f"{width} x {height}" for width, height in modules.config.BASE_RESOLUTIONS]

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
        positive_prompt = kwargs["positive"]
        negative_prompt = kwargs["negative"]
        vae_name = kwargs["vae_name"]
        pipe.update(
            {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "vae_name": vae_name,
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
                "freeu_enabled": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "image_to_latent": ("IMAGE",),
                "latent": ("LATENT",),
                "fooocus_inpaint": ("FOOOCUS_INPAINT",),
                "fooocus_styles": ("FOOOCUS_STYLES",),
            },
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CLIP", "VAE",
                    "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "model", "clip", "vae",
                    "CONDITIONING+", "CONDITIONING-")

    FUNCTION = "fooocus_preKSampler"
    CATEGORY = "Fooocus"

    def fooocus_preKSampler(self, pipe: dict, image_to_latent=None, latent=None, fooocus_inpaint=None, fooocus_styles=None, **kwargs):
        # 检查pipe非空
        assert pipe is not None, "请先调用 FooocusLoader 进行初始化！"
        execution_start_time = time.perf_counter()
        pipe.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in ("switch", "refiner_switch")
            }
        )
        prompt = pipe["positive_prompt"]
        negative_prompt = pipe["negative_prompt"]
        if fooocus_styles is not None:
            style_selections = fooocus_styles
        else:
            style_selections = []

        image_number = pipe["image_number"]
        image_seed = kwargs.get("seed")
        read_wildcards_in_order = False
        sharpness = kwargs.get('sharpness')
        guidance_scale = kwargs.get("cfg")
        base_model_name = pipe["base_model_name"]
        vae_name = pipe["vae_name"]
        refiner_model_name = pipe["refiner_model_name"]
        refiner_switch = pipe["refiner_switch"]
        clip_skip = pipe["clip_skip"]
        loras = pipe["optional_lora_stack"]
        outpaint_selections = []
        disable_seed_increment = False
        adm_scaler_positive = kwargs.pop("adm_scaler_positive")
        adm_scaler_negative = kwargs.pop("adm_scaler_negative")
        adm_scaler_end = kwargs.pop("adm_scaler_end")
        adaptive_cfg = kwargs.pop("adaptive_cfg")
        sampler_name = pipe["sampler_name"]
        scheduler_name = pipe["scheduler"]
        refiner_swap_method = pipe["refiner_swap_method"]
        controlnet_softness = kwargs.pop("controlnet_softness")
        freeu_enabled = kwargs.pop("freeu_enabled")

        base_model_additional_loras = []

        if fooocus_expansion in style_selections:
            use_expansion = True
        else:
            use_expansion = False

        if use_expansion:
            use_style = len(style_selections) > 1
        else:
            use_style = len(style_selections) > 0

        if base_model_name == refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            refiner_model_name = 'None'

        steps = kwargs.pop("steps")

        if pipe["sampler_name"] == "lcm":
            print('Enter LCM mode.')
            if refiner_model_name != 'None':
                print(f'Refiner disabled in LCM mode.')

            refiner_model_name = "None"
            pipe["sampler_name"] = "lcm"
            pipe["scheduler"] = "lcm"
            modules.patch.sharpness = 0.0
            cfg_scale = guidance_scale = 1.0
            refiner_switch = 1.0
            modules.patch.adaptive_cfg = 1.0
            modules.patch.positive_adm_scale = 1.0
            modules.patch.negative_adm_scale = 1, 0
            modules.patch.adm_scaler_end = 0.0
        seed = int(image_seed)
        print(f'[Parameters] Adaptive CFG = {adaptive_cfg}')
        print(f'[Parameters] CLIP Skip = {clip_skip}')
        print(f'[Parameters] Sharpness = {sharpness}')
        print(f'[Parameters] ControlNet Softness = {controlnet_softness}')
        print(f'[Parameters] ADM Scale = '
              f'{adm_scaler_positive} : '
              f'{adm_scaler_negative} : '
              f'{adm_scaler_end}')
        print(f'[Parameters] Seed = {seed}')

        patch_settings[pid] = PatchSettings(
            sharpness,
            adm_scaler_end,
            adm_scaler_positive,
            adm_scaler_negative,
            controlnet_softness,
            adaptive_cfg
        )

        cfg_scale = float(guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = kwargs.pop("denoise")

        height = pipe["latent_height"]
        width = pipe["latent_width"]
        width, height = int(width), int(height)

        skip_prompt_processing = False

        inpaint_worker.current_task = None
        inpaint_parameterized = False
        if fooocus_inpaint is not None:
            inpaint_engine = fooocus_inpaint.get("inpaint_engine")
            top = fooocus_inpaint.get("top")
            bottom = fooocus_inpaint.get("bottom")
            left = fooocus_inpaint.get("left")
            right = fooocus_inpaint.get("right")
            if top is not None and top is True:
                outpaint_selections.append('top')
            if bottom is not None and bottom is True:
                outpaint_selections.append('bottom')
            if left is not None and left is True:
                outpaint_selections.append('left')
            if right is not None and right is True:
                outpaint_selections.append('right')
            if inpaint_engine != 'None':
                inpaint_parameterized = True

        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None
        use_synthetic_refiner = False
        sampler_name = pipe["sampler_name"]
        scheduler_name = pipe["scheduler"]

        goals = []
        tasks = []

        if fooocus_inpaint is not None:
            inpaint_image = fooocus_inpaint.get("image")
            inpaint_image = core.pytorch_to_numpy(inpaint_image)[0]

            inpaint_mask = fooocus_inpaint.get("mask")
            if inpaint_mask is not None:
                inpaint_mask = core.pytorch_to_numpy(inpaint_mask)[0]
                inpaint_mask = HWC3(inpaint_mask)
                if isinstance(inpaint_mask, np.ndarray):
                    if inpaint_mask.ndim == 3:
                        H, W, C = inpaint_image.shape
                        inpaint_mask = resample_image(
                            inpaint_mask, width=W, height=H)
                        inpaint_mask = np.mean(inpaint_mask, axis=2)
                        inpaint_mask = (inpaint_mask > 127).astype(
                            np.uint8) * 255
                        inpaint_mask = np.maximum(inpaint_mask, inpaint_mask)

            inpaint_engine = fooocus_inpaint.get("inpaint_engine")
            inpaint_disable_initial_latent = fooocus_inpaint.get(
                "inpaint_disable_initial_latent")
            inpaint_respective_field = fooocus_inpaint.get(
                "inpaint_respective_field")

            inpaint_image = HWC3(inpaint_image)
            log_node_info('Downloading upscale models ...')
            modules.config.downloading_upscale_model()
            if inpaint_parameterized:
                print('Downloading inpainter ...')
                inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                    inpaint_engine)
                base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                if refiner_model_name == "None":
                    use_synthetic_refiner = True
                    refiner_switch = 0.8
            else:
                inpaint_head_model_path, inpaint_patch_model_path = None, None
                print(f'[Inpaint] Parameterized inpaint is disabled.')
            goals.append('inpaint')

        switch = int(round(steps * refiner_switch))
        print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        print(f'[Parameters] Steps = {steps} - {switch}')

        log_node_info('Initializing ...')

        if not skip_prompt_processing:

            prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                use_expansion = False

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            log_node_info('Loading models ...')
            pipeline.refresh_everything(
                refiner_model_name=refiner_model_name,
                base_model_name=base_model_name,
                loras=loras,
                base_model_additional_loras=base_model_additional_loras,
                use_synthetic_refiner=use_synthetic_refiner,
                vae_name=vae_name
            )
            pipeline.set_clip_skip(clip_skip)
            log_node_info('Processing prompts ...')

            # for node output
            positive = pipeline.clip_encode(prompts, len(prompts))
            negative = pipeline.clip_encode(
                negative_prompts, len(negative_prompts))

            tasks = []
            for i in range(image_number):
                if disable_seed_increment:
                    task_seed = seed % (MAX_SEED + 1)
                else:
                    task_seed = (seed + i) % (MAX_SEED + 1)  # randint is inclusive, % is not

                # may bind to inpaint noise in the future
                task_rng = random.Random(task_seed)

                task_prompt = apply_wildcards(prompt, task_rng, i, read_wildcards_in_order)
                task_prompt = apply_arrays(task_prompt, i)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, read_wildcards_in_order)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                if use_style:
                    placeholder_replaced = False

                    for s in style_selections:
                        if s == fooocus_expansion:
                            continue
                        if s == random_style_name:
                            s = get_random_style(task_rng)
                            print(f'Using Fooocus Random Style:{s}')
                        p, n, style_has_placeholder = apply_style(s, positive=task_prompt)
                        if style_has_placeholder:
                            placeholder_replaced = True
                        positive_basic_workloads = positive_basic_workloads + p
                        negative_basic_workloads = negative_basic_workloads + n

                    if not placeholder_replaced:
                        positive_basic_workloads = [task_prompt] + positive_basic_workloads
                else:
                    positive_basic_workloads.append(task_prompt)

                # Always use independent workload for negative.
                negative_basic_workloads.append(task_negative_prompt)

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
                    log_node_info(f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    print(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(tasks):
                log_node_info(f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    log_node_info(f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if len(goals) > 0:
            log_node_info('Image processing ...')

        if image_to_latent is not None:
            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )
            if isinstance(image_to_latent, list):
                image_to_latent = image_to_latent[0]
                image_to_latent = image_to_latent.unsqueeze(0)
            initial_latent = core.encode_vae(candidate_vae, image_to_latent)
        elif latent is not None:
            initial_latent = latent

        if 'inpaint' in goals:
            if len(outpaint_selections) > 0:
                inpaint_mask = np.zeros(inpaint_image.shape, dtype=np.uint8)
                inpaint_mask = inpaint_mask[:, :, 0]
                H, W, C = inpaint_image.shape
                if 'top' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                          constant_values=255)
                if 'bottom' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                          constant_values=255)

                H, W, C = inpaint_image.shape
                if 'left' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [int(W * 0.3), 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(W * 0.3), 0]], mode='constant',
                                          constant_values=255)
                if 'right' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(W * 0.3)], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(W * 0.3)]], mode='constant',
                                          constant_values=255)

                inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
                denoising_strength = 1.0
                inpaint_respective_field = 1.0

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=inpaint_respective_field
            )

            log_node_info('VAE Inpaint encoding ...')

            inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                log_node_info("VAE SD15 encoding ...")
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap, pixels=inpaint_pixel_fill
                )["samples"]

            log_node_info("VAE encoding ..")
            latent_fill = core.encode_vae(vae=candidate_vae, pixels=inpaint_pixel_fill)[
                "samples"
            ]

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if inpaint_parameterized:
                pipeline.final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=pipeline.final_unet
                )

            if not inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if freeu_enabled:
            print(f'FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                1.01,
                1.02,
                0.99,
                0.95
            )

        print(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name == 'lcm':
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            print('Using lcm scheduler.')

        log_node_info('Moving model to GPU ...')
        new_pipe = {
            "image_number": 1,  # should be reseted to one
            "base_model_name": pipe['base_model_name'],
            "refiner_model_name": pipe['refiner_model_name'],
            "optional_lora_stack": pipe['optional_lora_stack'],
            "use_cn": pipe['use_cn'],
            "refiner_switch": switch,
            "positive_prompt": pipe["positive_prompt"],
            "negative_prompt": pipe["negative_prompt"],
            "tasks": tasks,
            "positive": positive,
            "negative": negative,
            "seed": seed,
            "steps": steps,
            "cfg": cfg_scale,
            "refiner_swap_method": refiner_swap_method,
            "sampler_name": final_sampler_name,
            "scheduler": final_scheduler_name,
            "denoise": denoising_strength,
            "latent": initial_latent,
            "model": pipeline.final_unet,
            "latent_height": height,
            "latent_width": width,
            "switch": switch,
            "clip": pipeline.final_clip,
            "vae": pipeline.final_vae,
        }
        del pipe
        return {"ui": {"value": [new_pipe["seed"]]}, "result": (new_pipe, pipeline.final_unet, pipeline.final_clip, pipeline.final_vae, positive, negative)}


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
        all_steps = pipe["steps"] * len(pipe["tasks"])
        pbar = comfy.utils.ProgressBar(all_steps)

        def callback(step, x0, x, total_steps, y):
            preview_bytes = None
            # done_steps = current_task_id * pipe["steps"] + step
            pbar.update_absolute(step + 1, total_steps, preview_bytes)

        for current_task_id, task in enumerate(pipe["tasks"]):
            try:
                print(f"Current Task {current_task_id + 1} ……")
                positive_cond, negative_cond = task['c'], task['uc']

                if "cn_tasks" in pipe and len(pipe["cn_tasks"]) > 0:
                    for cn_path, cn_img, cn_stop, cn_weight in pipe["cn_tasks"]:
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
                    callback=callback,
                    sampler_name=pipe["sampler_name"],
                    scheduler_name=pipe["scheduler"],
                    latent=pipe["latent"],
                    denoise=pipe["denoise"],
                    tiled=False,
                    cfg_scale=pipe["cfg"],
                    refiner_swap_method=pipe["refiner_swap_method"],
                    disable_preview=True
                )

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(
                        x) for x in imgs]

                imgs = [np.array(img).astype(np.float32) /
                        255.0 for img in imgs]
                imgs = [torch.from_numpy(img) for img in imgs]
                all_imgs.extend(imgs)
            except Exception as e:
                print('task stopped')
                raise e

        new_pipe = {
            **pipe,
            "images": all_imgs,
        }
        del pipe

        # Combine the processed images back into a single tensor
        base_image = torch.stack([tensor.squeeze() for tensor in all_imgs])

        if image_output in ("Save", "Hide/Save"):
            saveimage = SaveImage()
            results = saveimage.save_images(
                all_imgs, save_prefix, prompt, extra_pnginfo)

        if image_output == "Preview":
            previewimage = PreviewImage()
            results = previewimage.save_images(
                all_imgs, save_prefix, prompt, extra_pnginfo)

        if image_output == "Hide":
            return {"ui": {"value": list()}, "result": (new_pipe, base_image)}

        results["result"] = new_pipe, base_image
        return results


class FooocusUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "upscale": ([1.5, 2.0], {"default": 1.5, }),
                "steps": ("INT", {"default": 18, "min": 1, "max": 100}),
                "denoise": ("FLOAT", {"default": 0.382, "min": 0.00, "max": 1.00, "step": 0.001},),
                "fast": ("BOOLEAN", {"default": False}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save",], {"default": "Preview"},),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("PIPE_LINE", "IMAGE")
    RETURN_NAMES = ("pipe", "image")
    OUTPUT_NODE = True
    FUNCTION = "FooocusUpscale"
    CATEGORY = "Fooocus"

    def FooocusUpscale(self, pipe, upscale, denoise, fast, steps, image_output, save_prefix, image=None, prompt=None, extra_pnginfo=None):
        all_imgs = []
        if "images" in pipe:
            images = pipe["images"]
        else:
            images = image
        all_steps = steps * len(images)
        pbar = comfy.utils.ProgressBar(all_steps)
        log_node_info('Downloading upscale models ...')
        modules.config.downloading_upscale_model()

        def callback(step, x0, x, total_steps, y):
            preview_bytes = None
            pbar.update_absolute(step + 1, total_steps, preview_bytes)

        for image_id, img in enumerate(images):
            print(f'upscale image #{image_id+1} ...')
            img = img.unsqueeze(0)
            img = img[0].numpy()
            img = (img * 255).astype(np.uint8)
            img = HWC3(img)
            H, W, C = img.shape
            log_node_info(f'Upscaling image from {str((H, W))} ...')
            if isinstance(img, list):
                img = np.array(img)

            uov_input_image = perform_upscale(img)
            print(f'Image upscaled.')
            f = upscale
            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                print(f'[Upscale] Image is resized because it is too small.')
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

            if fast:
                direct_return = True
            elif image_is_super_large:
                print('Image is too large. Directly returned the SR image. '
                      'Usually directly return SR image at 4K resolution '
                      'yields better results than SDXL diffusion.')
                direct_return = True
            elif pipe is None:
                direct_return = True
            else:
                direct_return = False

            if direct_return:
                print('upscaled image direct_return')
                uov_input_image = core.numpy_to_pytorch(uov_input_image)
                all_imgs.extend(uov_input_image)
                continue

            tiled = True

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            log_node_info('VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
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
            print(f'Final resolution is {str((height, width))}.')

            if pipe['tasks'] is not None and len(pipe['tasks']) == len(images):
                task = pipe["tasks"][image_id]
                positive_cond, negative_cond = task['c'], task['uc']
            else:
                positive_cond = pipe["positive"]
                negative_cond = pipe["negative"]

            imgs = pipeline.process_diffusion(
                positive_cond=positive_cond,
                negative_cond=negative_cond,
                steps=steps,
                switch=pipe["switch"],
                width=pipe["latent_width"],
                height=pipe["latent_height"],
                image_seed=pipe["seed"],
                callback=callback,
                sampler_name=pipe["sampler_name"],
                scheduler_name=pipe["scheduler"],
                latent=initial_latent,
                denoise=denoise,
                tiled=tiled,
                cfg_scale=pipe["cfg"],
                refiner_swap_method=pipe["refiner_swap_method"],
                disable_preview=True
            )

            if inpaint_worker.current_task is not None:
                imgs = [inpaint_worker.current_task.post_process(
                    x) for x in imgs]

            imgs = [np.array(img).astype(np.float32) / 255.0 for img in imgs]
            imgs = [torch.from_numpy(img) for img in imgs]
            all_imgs.extend(imgs)

        new_pipe = {
            **pipe,
            "images": all_imgs,
        }
        del pipe
        if image_output in ("Save", "Hide/Save"):
            saveimage = SaveImage()
            results = saveimage.save_images(
                all_imgs, save_prefix, prompt, extra_pnginfo)

        if image_output == "Preview":
            previewimage = PreviewImage()
            results = previewimage.save_images(
                all_imgs, save_prefix, prompt, extra_pnginfo)

        if image_output == "Hide":
            return {"ui": {"value": list()}, "result": (new_pipe, all_imgs)}

        results["result"] = new_pipe, all_imgs
        return results


class FooocusControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "cn_type": (modules.flags.cn_list, {"default": modules.flags.default_cn}, ),
                "cn_stop": ("FLOAT", {"default": modules.flags.default_parameters[modules.flags.default_cn][0], "min": 0.0, "max": 1.0, "step": 0.01},),
                "cn_weight": ("FLOAT", {"default": modules.flags.default_parameters[modules.flags.default_cn][1], "min": 0.0, "max": 2.0, "step": 0.01},),
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
        if cn_type == modules.flags.cn_canny:
            cn_path = modules.config.downloading_controlnet_canny()
            image = image[0].numpy()
            image = (image * 255).astype(np.uint8)
            image = resize_image(
                HWC3(image), pipe["latent_width"], pipe["latent_height"])
            if not skip_cn_preprocess:
                image = preprocessors.canny_pyramid(image)
        if cn_type == modules.flags.cn_cpds:
            cn_path = modules.config.downloading_controlnet_cpds()
            image = image[0].numpy()
            image = (image * 255).astype(np.uint8)
            image = resize_image(
                HWC3(image), pipe["latent_width"], pipe["latent_height"])
            if not skip_cn_preprocess:
                image = preprocessors.cpds(image)
        image = HWC3(image)
        image = core.numpy_to_pytorch(image)
        new_pipe = pipe.copy()
        if "cn_tasks" not in new_pipe:
            new_pipe["cn_tasks"] = []
        new_pipe["cn_tasks"].append([cn_path, image, cn_stop, cn_weight])
        return (new_pipe, image,)


class FooocusImagePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ip_type": (modules.flags.ip_list, {"default": modules.flags.default_ip}, ),
                "ip_stop": ("FLOAT", {"default": modules.flags.default_parameters[modules.flags.default_ip][0], "min": 0.0, "max": 1.0, "step": 0.01},),
                "ip_weight": ("FLOAT", {"default": modules.flags.default_parameters[modules.flags.default_ip][1], "min": 0.0, "max": 2.0, "step": 0.01},),
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
        if ip_type == modules.flags.cn_ip:
            clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters(
                'ip')
            ip_adapter.load_ip_adapter(
                clip_vision_path, ip_negative_path, ip_adapter_path)
            image = image[0].numpy()
            image = (image * 255).astype(np.uint8)
            image = resize_image(HWC3(image), width=224,
                                 height=224, resize_mode=0)
            task = [image, ip_stop, ip_weight]
            task[0] = ip_adapter.preprocess(
                image, ip_adapter_path=ip_adapter_path)
        if ip_type == modules.flags.cn_ip_face:
            clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                'face')
            ip_adapter.load_ip_adapter(
                clip_vision_path, ip_negative_path, ip_adapter_face_path)
            image = image[0].numpy()
            image = (image * 255).astype(np.uint8)
            image = HWC3(image)
            if not skip_cn_preprocess:
                image = face_crop.crop_image(image)
            image = resize_image(image, width=224, height=224, resize_mode=0)
            task = [image, ip_stop, ip_weight]
            task[0] = ip_adapter.preprocess(
                image, ip_adapter_path=ip_adapter_face_path)
        return (task, )


class FooocusApplyImagePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "image_prompt_1": ("IMAGE_PROMPT",),
                "image_prompt_2": ("IMAGE_PROMPT",),
                "image_prompt_3": ("IMAGE_PROMPT",),
                "image_prompt_4": ("IMAGE_PROMPT",),
            },
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_NODE = True
    FUNCTION = "apply_image_prompt"
    CATEGORY = "Fooocus"

    def apply_image_prompt(
        self, model, image_prompt_1=None, image_prompt_2=None, image_prompt_3=None, image_prompt_4=None
    ):
        image_prompt_tasks = []
        if image_prompt_1:
            image_prompt_tasks.append(image_prompt_1)
        if image_prompt_2:
            image_prompt_tasks.append(image_prompt_2)
        if image_prompt_3:
            image_prompt_tasks.append(image_prompt_3)
        if image_prompt_4:
            image_prompt_tasks.append(image_prompt_4)
        work_model = model.clone()
        new_model = ip_adapter.patch_model(work_model, image_prompt_tasks)
        return (new_model, )


class FooocusInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "inpaint_disable_initial_latent": ("BOOLEAN", {"default": False}),
                "inpaint_respective_field": ("FLOAT", {"default": 0.618, "min": 0, "max": 1.0, "step": 0.1},),
                "inpaint_engine": (modules.flags.inpaint_engine_versions, {"default": "v2.6"},),
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
        self, image, inpaint_disable_initial_latent, inpaint_respective_field, inpaint_engine, top, bottom, left, right, mask=None
    ):
        fooocus_inpaint = {
            "image": image,
            "mask": mask,
            "inpaint_disable_initial_latent": inpaint_disable_initial_latent,
            "inpaint_respective_field": inpaint_respective_field,
            "inpaint_engine": inpaint_engine,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right
        }
        return (fooocus_inpaint, )


class FooocusPipeOut:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING",
                    "CONDITIONING", "LATENT", "VAE", "FLOAT",)
    RETURN_NAMES = ("pipe", "model", "pos", "neg", "latent",
                    "vae", "switch",)
    FUNCTION = "flush"

    CATEGORY = "Fooocus"

    def flush(self, pipe, my_unique_id=None):
        model = pipe.get("model")
        pos = pipe.get("positive")
        neg = pipe.get("negative")
        latent = pipe.get("latent")
        vae = pipe.get("vae")
        switch = pipe.get("switch")

        return pipe, model, pos, neg, latent, vae, switch


class preDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipe": ("PIPE_LINE",),
            "guide_size": ("FLOAT", {"default": 384, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
            "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
            "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
            "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
            "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
        },
            "optional": {
                "bbox_segm_pipe": ("PIPE_LINE",),
                "sam_pipe": ("PIPE_LINE",),
                "optional_image": ("IMAGE",),
        },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "doit"

    CATEGORY = "Fooocus/Fix"

    def doit(self, pipe, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler, denoise, feather, noise_mask, force_inpaint, drop_size, wildcard, cycle, bbox_segm_pipe=None, sam_pipe=None, optional_image=None):
        tasks = pipe["tasks"] if "tasks" in pipe else None
        if tasks is None:
            raise Exception(f"[ERROR] pipe['tasks'] is missing")
        model = pipe["model"] if "model" in pipe else None
        if model is None:
            raise Exception(f"[ERROR] pipe['model'] is missing")
        clip = pipe["clip"] if "clip" in pipe else None
        if clip is None:
            raise Exception(f"[ERROR] pipe['clip'] is missing")
        vae = pipe["vae"] if "vae" in pipe else None
        if vae is None:
            raise Exception(f"[ERROR] pipe['vae'] is missing")
        if optional_image is not None:
            images = optional_image
        else:
            images = pipe["images"] if "images" in pipe else None
            if images is None:
                raise Exception(f"[ERROR] pipe['image'] is missing")
        bbox_segm_pipe = bbox_segm_pipe or (pipe["bbox_segm_pipe"] if pipe and "bbox_segm_pipe" in pipe else None)
        if bbox_segm_pipe is None:
            raise Exception(f"[ERROR] bbox_segm_pipe or pipe['bbox_segm_pipe'] is missing")
        sam_pipe = sam_pipe or (pipe["sam_pipe"] if pipe and "sam_pipe" in pipe else None)
        if sam_pipe is None:
            raise Exception(f"[ERROR] sam_pipe or pipe['sam_pipe'] is missing")

        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}
        new_pipe = {
            **pipe,
            "tasks": tasks,
            "images": images,
            "model": model,
            "clip": clip,
            "vae": vae,
            "seed": seed,
            "bbox_segm_pipe": bbox_segm_pipe,
            "sam_pipe": sam_pipe,
            "loader_settings": loader_settings,
            "detail_fix_settings": {
                "guide_size": guide_size,
                "guide_size_for": guide_size_for,
                "max_size": max_size,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "feather": feather,
                "noise_mask": noise_mask,
                "force_inpaint": force_inpaint,
                "drop_size": drop_size,
                "wildcard": wildcard,
                "cycle": cycle
            }
        }

        del bbox_segm_pipe
        del sam_pipe

        return (new_pipe,)


class detailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipe": ("PIPE_LINE",),
            "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Sender", "Sender/Save"], {"default": "Preview"}),
            "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
            "save_prefix": ("STRING", {"default": "ComfyUI"}),
        },
            "optional": {
                "model": ("MODEL",),
        },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID", }
        }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "doit"

    CATEGORY = "Fooocus/Fix"

    def doit(self, pipe, image_output, link_id, save_prefix, model=None, prompt=None, extra_pnginfo=None, my_unique_id=None):

        my_unique_id = int(my_unique_id)

        model = model or (pipe["model"] if "model" in pipe else None)
        if model is None:
            raise Exception(f"[ERROR] model or pipe['model'] is missing")

        bbox_segm_pipe = pipe["bbox_segm_pipe"] if pipe and "bbox_segm_pipe" in pipe else None
        if bbox_segm_pipe is None:
            raise Exception(f"[ERROR] bbox_segm_pipe or pipe['bbox_segm_pipe'] is missing")
        sam_pipe = pipe["sam_pipe"] if "sam_pipe" in pipe else None
        if sam_pipe is None:
            raise Exception(f"[ERROR] sam_pipe or pipe['sam_pipe'] is missing")
        bbox_detector_opt, bbox_threshold, bbox_dilation, bbox_crop_factor, segm_detector_opt = bbox_segm_pipe
        sam_model_opt, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative = sam_pipe

        detail_fix_settings = pipe["detail_fix_settings"] if "detail_fix_settings" in pipe else None
        if detail_fix_settings is None:
            raise Exception(f"[ERROR] detail_fix_settings or pipe['detail_fix_settings'] is missing")
        tasks = pipe["tasks"]
        image = pipe["images"]
        clip = pipe["clip"]
        vae = pipe["vae"]
        seed = pipe["seed"]
        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}
        guide_size = pipe["detail_fix_settings"]["guide_size"]
        guide_size_for = pipe["detail_fix_settings"]["guide_size_for"]
        max_size = pipe["detail_fix_settings"]["max_size"]
        steps = pipe["detail_fix_settings"]["steps"]
        cfg = pipe["detail_fix_settings"]["cfg"]
        sampler_name = pipe["detail_fix_settings"]["sampler_name"]
        scheduler = pipe["detail_fix_settings"]["scheduler"]
        denoise = pipe["detail_fix_settings"]["denoise"]
        feather = pipe["detail_fix_settings"]["feather"]
        noise_mask = pipe["detail_fix_settings"]["noise_mask"]
        force_inpaint = pipe["detail_fix_settings"]["force_inpaint"]
        drop_size = pipe["detail_fix_settings"]["drop_size"]
        wildcard = pipe["detail_fix_settings"]["wildcard"]
        cycle = pipe["detail_fix_settings"]["cycle"]

        cls = ALL_NODE_CLASS_MAPPINGS["FaceDetailer"]
        result_imgs = []
        # 细节修复初始时间
        start_time = int(time.time() * 1000)
        for current_task_id, task in enumerate(tasks):
            img = image[current_task_id]
            result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, d_pipe, result_cnet_images = cls().doit(
                [img], model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                scheduler,
                task['c'], task['uc'], denoise, feather, noise_mask, force_inpaint,
                bbox_threshold, bbox_dilation, bbox_crop_factor,
                sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                sam_mask_hint_use_negative, drop_size, bbox_detector_opt, wildcard, cycle, sam_model_opt, segm_detector_opt,
                detailer_hook=None)
            result_imgs.extend(result_img)

        # 细节修复结束时间
        end_time = int(time.time() * 1000)

        spent_time = '细节修复:' + str((end_time - start_time) / 1000) + '秒'

        results = easySave(result_imgs, save_prefix, image_output, prompt, extra_pnginfo)
        new_pipe = {
            **pipe,
            "tasks": tasks,
            "images": result_imgs,
            "model": model,
            "clip": clip,
            "vae": vae,
            "seed": seed,
            "wildcard": wildcard,
            "bbox_segm_pipe": bbox_segm_pipe,
            "sam_pipe": sam_pipe,

            "loader_settings": {
                **loader_settings,
                "spent_time": spent_time
            },
            "detail_fix_settings": detail_fix_settings
        }

        del bbox_segm_pipe
        del sam_pipe
        del pipe

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": (new_pipe, result_imgs)}

        if image_output in ("Sender", "Sender/Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        return {"ui": {"images": results}, "result": (new_pipe, result_imgs)}


class ultralyticsDetectorForDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {"required":
                {"model_name": (bboxs + segms,),
                 "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                 "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                 }
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("bbox_segm_pipe",)
    FUNCTION = "doit"

    CATEGORY = "Fooocus/Fix"

    def doit(self, model_name, bbox_threshold, bbox_dilation, bbox_crop_factor):
        if 'UltralyticsDetectorProvider' not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use UltralyticsDetectorProvider, you need to install 'Impact Pack'")
        cls = ALL_NODE_CLASS_MAPPINGS['UltralyticsDetectorProvider']
        bbox_detector, segm_detector = cls().doit(model_name)
        pipe = (bbox_detector, bbox_threshold, bbox_dilation, bbox_crop_factor, segm_detector)
        return (pipe,)


class samLoaderForDetailerFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("sams"),),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"], {"default": "AUTO"}),
                "sam_detection_hint": (
                    ["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points",
                     "mask-point-bbox", "none"],),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),
            }
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("sam_pipe",)
    FUNCTION = "doit"

    CATEGORY = "Fooocus/Fix"

    def doit(self, model_name, device_mode, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative):
        if 'SAMLoader' not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use SAMLoader, you need to install 'Impact Pack'")
        cls = ALL_NODE_CLASS_MAPPINGS['SAMLoader']
        (sam_model,) = cls().load_model(model_name, device_mode)
        pipe = (sam_model, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative)
        return (pipe,)

class FooocusDescribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_type": (["Photo", "Anime"], {"default": "Photo"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe"
    CATEGORY = "Fooocus"
    OUTPUT_NODE = True

    def describe(self, image, image_type):
        img = image[0].numpy()
        img = (img * 255).astype(np.uint8)
        img = HWC3(img)
        if image_type == 'Photo':
            from extras.interrogate import default_interrogator as default_interrogator_photo
            interrogator = default_interrogator_photo
        else:
            from extras.wd14tagger import default_interrogator as default_interrogator_anime
            interrogator = default_interrogator_anime
        tags = interrogator(img)
        print(tags)
        return {"ui": {"tags": tags}, "result": (tags,)}


class FooocusExpansion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "log_prompt": ("BOOLEAN", {"default": False},),
            },
        }
    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("final_prompt", "seed",)
    FUNCTION = "expansion"

    CATEGORY = "Fooocus"

    def expansion(self, prompt, seed, log_prompt):
        expansion = Expansion()
        prompt = remove_empty_str([safe_str(prompt)], default='')[0]
        max_seed = int(1024 * 1024 * 1024)
        if not isinstance(seed, int):
            seed = random.randint(1, max_seed)
        if seed < 0:
            seed = - seed
        seed = seed % max_seed
        expansion = expansion(prompt, seed)
        if log_prompt:
            print(f'[Prompt Expansion] {expansion}')
        return {"ui": {"expansion": expansion}, "result": (expansion,)}

NODE_CLASS_MAPPINGS = {

    "Fooocus Loader": FooocusLoader,
    "Fooocus PreKSampler": FooocusPreKSampler,
    "Fooocus KSampler": FooocusKsampler,
    "Fooocus Upscale": FooocusUpscale,
    "Fooocus LoraStack": FooocusLoraStack,
    "Fooocus Controlnet": FooocusControlnet,
    "Fooocus ImagePrompt": FooocusImagePrompt,
    "Fooocus ApplyImagePrompt": FooocusApplyImagePrompt,
    "Fooocus Inpaint": FooocusInpaint,
    "Fooocus PipeOut": FooocusPipeOut,
    "Fooocus Describe": FooocusDescribe,
    "Fooocus Expansion": FooocusExpansion,
    # fix
    "Fooocus preDetailerFix": preDetailerFix,
    "Fooocus ultralyticsDetectorPipe": ultralyticsDetectorForDetailerFix,
    "Fooocus samLoaderPipe": samLoaderForDetailerFix,
    "Fooocus detailerFix": detailerFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fooocus positive": "Positive",
    "Fooocus negative": "Negative",
    "Fooocus stylesSelector": "stylesPromptSelector",

}
