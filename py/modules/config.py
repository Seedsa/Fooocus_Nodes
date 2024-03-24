import os
import folder_paths
from pathlib import Path
from .model_loader import load_file_from_url
from .flags import inpaint_engine_versions


BASE_RESOLUTIONS = [
    (512, 512),
    (512, 768),
    (768, 512),
    (576, 1024),
    (768, 1024),
    (768, 1280),
    (768, 1344),
    (768, 1536),
    (832, 1152),
    (896, 1152),
    (896, 1088),
    (1024, 1024),
    (1024, 576),
    (1024, 768),
    (1088, 896),
    (1152, 832),
    (1152, 896),
    (1280, 768),
    (1344, 768),
    (1536, 640),
    (1536, 768),
    ("自定义", "自定义")
]
wildcard_filenames = []
paths_checkpoints = folder_paths.get_folder_paths('checkpoints')[0]
path_checkpoints = folder_paths.get_folder_paths('checkpoints')[0]
path_loras = folder_paths.get_folder_paths('loras')[0]
path_embeddings = folder_paths.get_folder_paths('embeddings')[0]
path_vae_approx = folder_paths.get_folder_paths('vae_approx')[0]
path_controlnet = folder_paths.get_folder_paths('controlnet')[0]
path_clip_vision = folder_paths.get_folder_paths('clip_vision')[0]
path_upscale_models = folder_paths.get_folder_paths('upscale_models')[0]
path_fooocus_expansion = folder_paths.models_dir + "/prompt_expansion/fooocus_expansion"
path_inpaint = folder_paths.models_dir + "/inpaint"
path_ipadapter = folder_paths.models_dir + "/ipadapter"

path_styles = os.path.join(
    Path(__file__).parent.parent.parent, "sdxl_styles")
path_wildcards = os.path.join(
    Path(__file__).parent.parent.parent, "wildcards")


use_model_cache = False
default_refiner_model_name = 'None'
default_base_model_name = "juggernautXL_v8Rundiffusion.safetensors"
default_loras = [
    [
        "None",
        1.0
    ],
    [
        "None",
        1.0
    ],
    [
        "None",
        1.0
    ],
    [
        "None",
        1.0
    ],
    [
        "None",
        1.0
    ]
]
default_loras = [(y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in default_loras]


def downloading_inpaint_models(v):
    assert v in inpaint_engine_versions

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v25.fooocus.patch')

    if v == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v26.fooocus.patch')

    return head_file, patch_file


def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=path_controlnet,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(path_controlnet, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(path_controlnet, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(path_clip_vision,
                             'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=path_ipadapter,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(path_ipadapter,
                             'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=path_ipadapter,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_ipadapter,
                                 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=path_ipadapter,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_ipadapter,
                                 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=path_upscale_models,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')
