import os
import folder_paths
from pathlib import Path
import os
from urllib.parse import urlparse
from torch.hub import download_url_to_file
from log import log_node_warn

BASE_RESOLUTIONS = [
    (128, 128),
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
UPSCALE_DIR = os.path.join(folder_paths.models_dir, "upscale_models")
CONTROLNET_DIR = os.path.join(folder_paths.models_dir, "controlnet")
CLIP_VISION_DIR = os.path.join(folder_paths.models_dir, "clip_vision")
FOOOCUS_STYLES_DIR = os.path.join(
    Path(__file__).parent.parent.parent, "styles")

inpaint_engine_versions = ['None', 'v1', 'v2.5', 'v2.6']
FOOOCUS_INPAINT_HEAD = {
    "fooocus_inpaint_head": {
        "model_url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"
    }
}
FOOOCUS_UPSCALE_MODEL = {
    "model_url": "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin"
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

cn_ip = "ImagePrompt"
cn_ip_face = "FaceSwap"
cn_canny = "PyraCanny"
cn_cpds = "CPDS"

ip_list = [cn_ip, cn_ip_face]
cn_list = [cn_canny, cn_cpds]

default_cn = cn_canny
default_ip = cn_ip


default_parameters = {
    cn_ip: (0.5, 0.6), cn_ip_face: (0.9, 0.75), cn_canny: (0.5, 1.0), cn_cpds: (0.5, 1.0)
}  # stop, weight

path_fooocus_expansion = folder_paths.models_dir + \
    "/prompt_expansion/fooocus_expansion"


def get_local_filepath(url, dirname, local_file_name=None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)
    destination = os.path.join(dirname, local_file_name)
    if not os.path.exists(destination):
        log_node_warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []
    get_local_filepath('https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
                       CLIP_VISION_DIR, "clip_vision_vit_h.safetensors")
    results += [os.path.join(CLIP_VISION_DIR, 'clip_vision_vit_h.safetensors')]

    get_local_filepath('https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
                       CONTROLNET_DIR, "fooocus_ip_negative.safetensors")
    results += [os.path.join(CONTROLNET_DIR,
                             'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        get_local_filepath('https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
                           CONTROLNET_DIR, "ip-adapter-plus_sdxl_vit-h.bin")
        results += [os.path.join(CONTROLNET_DIR,
                                 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        get_local_filepath('https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
                           CONTROLNET_DIR, "ip-adapter-plus-face_sdxl_vit-h.bin")
        results += [os.path.join(CONTROLNET_DIR,
                                 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results
