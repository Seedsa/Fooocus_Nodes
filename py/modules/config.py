import os
import folder_paths
from pathlib import Path
from .model_loader import load_file_from_url
from .flags import inpaint_engine_versions
import yaml

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


def get_base_path_from_yaml():
    """
    Returns:
        str or None: The value of base_path from the YAML file for confmyui or None if not found.
    """
    yaml_file = folder_paths.base_path + "/extra_model_paths.yaml"
    try:
        # Open the YAML file and load its content
        with open(yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)

        # Access the 'base_path' key from the YAML data
        base_path = yaml_data.get('comfyui', {}).get('base_path') + "/"
        return base_path

    except FileNotFoundError:
        # print(f"YAML file '{yaml_file}' not found.")
        return None
    except yaml.YAMLError as e:
        # print(f"Error parsing YAML file: {e}")
        return None


def get_external_paths(name):
    base_path = get_base_path_from_yaml()
    if "prompt_expansion/fooocus_expansion" in name:
        if base_path is not None:
            return base_path + "/prompt_expansion/fooocus_expansion"
        else:
            return folder_paths.models_dir + "/prompt_expansion/fooocus_expansion"
    if "ipadapter" in name:
        if base_path is not None:
            return base_path + "/ipadapter"
        else:
            return folder_paths.models_dir + "ipadapter"
    if "inpaint" in name:
        if base_path is not None:
            return base_path + "/inpaint"
        else:
            return folder_paths.models_dir + "/inpaint"
    try:
        return folder_paths.get_folder_paths(name)[-1]
    except Exception as e:
        return folder_paths.get_folder_paths(name)[0]


wildcard_filenames = []
paths_checkpoints = get_external_paths('checkpoints')
paths_loras = get_external_paths('loras')
path_embeddings = get_external_paths('embeddings')
path_vae_approx = get_external_paths('vae_approx')
path_controlnet = get_external_paths('controlnet')
path_clip_vision = get_external_paths('clip_vision')
path_upscale_models = get_external_paths('upscale_models')
path_fooocus_expansion = get_external_paths("prompt_expansion/fooocus_expansion")
path_inpaint = get_external_paths("inpaint")
path_ipadapter = get_external_paths("ipadapter")

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
