import os
import folder_paths
from pathlib import Path


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

INPAINT_DIR = os.path.join(folder_paths.models_dir, "inpaint")
CONTROLNET_DIR = os.path.join(folder_paths.models_dir, "controlnet")
FOOOCUS_STYLES_DIR = os.path.join(
    Path(__file__).parent.parent.parent, "sdxl_styles")


FOOOCUS_INPAINT_HEAD = {
    "fooocus_inpaint_head": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"
    }
}
FOOOCUS_INPAINT_PATCH = {
    "v2.6": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch"
    },
    "v2.5": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch"
    },
    "v1": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch"
    },
}
FOOOCUS_IMAGE_PROMPT = {
    "PyraCanny": {
        "model_url": "https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors"
    },
    "CPDS": {
        "model_url": "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors"
    }
}

path_fooocus_expansion = folder_paths.models_dir + \
    "/prompt_expansion/fooocus_expansion"
path_loras = folder_paths.get_folder_paths('loras')[0]
path_embeddings = folder_paths.get_folder_paths('embeddings')[0]
path_vae_approx = folder_paths.get_folder_paths('vae_approx')[0]
path_controlnet = folder_paths.get_folder_paths('controlnet')[0]

use_model_cache = False
path_checkpoints = folder_paths.get_folder_paths('checkpoints')[0]
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
        model_dir=path_controlnet,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(path_controlnet,
                             'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet,
                                 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet,
                                 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=path_upscale_models,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')
