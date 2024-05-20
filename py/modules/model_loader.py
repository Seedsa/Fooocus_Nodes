from urllib.parse import urlparse
from typing import Optional
from .config import *
from .flags import inpaint_engine_versions

allow_download = user_config.get('allow_download_models', True)

def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file) and allow_download:
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    if not os.path.exists(cached_file):
        print(f'Plz Manually Download: "{url}" to {cached_file}\n')
        return None
    return cached_file


def downloading_inpaint_models(v):
    assert v in inpaint_engine_versions

    load_file_from_url(
        url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
        model_dir=path_inpaint,
        file_name="fooocus_inpaint_head.pth",
    )
    head_file = os.path.join(path_inpaint, "fooocus_inpaint_head.pth")
    patch_file = None

    if v == "v1":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch",
            model_dir=path_inpaint,
            file_name="inpaint.fooocus.patch",
        )
        patch_file = os.path.join(path_inpaint, "inpaint.fooocus.patch")

    if v == "v2.5":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch",
            model_dir=path_inpaint,
            file_name="inpaint_v25.fooocus.patch",
        )
        patch_file = os.path.join(path_inpaint, "inpaint_v25.fooocus.patch")

    if v == "v2.6":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch",
            model_dir=path_inpaint,
            file_name="inpaint_v26.fooocus.patch",
        )
        patch_file = os.path.join(path_inpaint, "inpaint_v26.fooocus.patch")

    return head_file, patch_file


def downloading_controlnet_canny():
    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors",
        model_dir=path_controlnet,
        file_name="control-lora-canny-rank128.safetensors",
    )
    return os.path.join(path_controlnet, "control-lora-canny-rank128.safetensors")


def downloading_controlnet_cpds():
    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors",
        model_dir=path_controlnet,
        file_name="fooocus_xl_cpds_128.safetensors",
    )
    return os.path.join(path_controlnet, "fooocus_xl_cpds_128.safetensors")


def downloading_ip_adapters(v):
    assert v in ["ip", "face"]

    results = []

    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
        model_dir=path_clip_vision,
        file_name="clip_vision_vit_h.safetensors",
    )
    results += [os.path.join(path_clip_vision, "clip_vision_vit_h.safetensors")]

    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
        model_dir=path_ipadapter,
        file_name="fooocus_ip_negative.safetensors",
    )
    results += [os.path.join(path_ipadapter, "fooocus_ip_negative.safetensors")]

    if v == "ip":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
            model_dir=path_ipadapter,
            file_name="ip-adapter-plus_sdxl_vit-h.bin",
        )
        results += [os.path.join(path_ipadapter, "ip-adapter-plus_sdxl_vit-h.bin")]

    if v == "face":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
            model_dir=path_ipadapter,
            file_name="ip-adapter-plus-face_sdxl_vit-h.bin",
        )
        results += [os.path.join(path_ipadapter, "ip-adapter-plus-face_sdxl_vit-h.bin")]
    return results


def downloading_upscale_model():
    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
        model_dir=path_upscale_models,
        file_name="fooocus_upscaler_s409985e5.bin",
    )
    return os.path.join(path_upscale_models, "fooocus_upscaler_s409985e5.bin")
