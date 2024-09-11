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
    ("自定义", "自定义"),
]



wildcard_filenames = []
path_controlnet = folder_paths.get_folder_paths("controlnet")[0]
path_fooocus_expansion = folder_paths.get_folder_paths("fooocus_expansion")[0]


path_wildcards = os.path.join(Path(__file__).parent.parent.parent, "wildcards")

default_refiner_model_name = "None"
default_base_model_name = "juggernautXL_v8Rundiffusion.safetensors"
default_loras = [
    ["None", 1.0],
    ["None", 1.0],
    ["None", 1.0],
    ["None", 1.0],
    ["None", 1.0],
]
default_loras = [
    (y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in default_loras
]
wildcards_max_bfs_depth = 64
wildcard_filenames = []


def downloading_inpaint_models(v):
    assert v in inpaint_engine_versions

    load_file_from_url(
        url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
        model_dir="inpaint",
        file_name="fooocus_inpaint_head.pth",
    )
    head_file = folder_paths.get_full_path("inpaint", "fooocus_inpaint_head.pth")
    patch_file = None

    if v == "v1":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch",
            model_dir="inpaint",
            file_name="inpaint.fooocus.patch",
        )

        patch_file = folder_paths.get_full_path("inpaint", "inpaint.fooocus.patch")

    if v == "v2.5":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch",
            model_dir="inpaint",
            file_name="inpaint_v25.fooocus.patch",
        )
        patch_file = folder_paths.get_full_path("inpaint", "inpaint_v25.fooocus.patch")

    if v == "v2.6":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch",
            model_dir="inpaint",
            file_name="inpaint_v26.fooocus.patch",
        )
        patch_file = folder_paths.get_full_path("inpaint", "inpaint_v26.fooocus.patch")

    return head_file, patch_file


def downloading_controlnet_canny():
    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors",
        model_dir="controlnet",
        file_name="control-lora-canny-rank128.safetensors",
    )
    return folder_paths.get_full_path("controlnet","control-lora-canny-rank128.safetensors")


def downloading_controlnet_cpds():
    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors",
        model_dir="controlnet",
        file_name="fooocus_xl_cpds_128.safetensors",
    )
    return folder_paths.get_full_path("controlnet","fooocus_xl_cpds_128.safetensors")


def downloading_ip_adapters(v):
    assert v in ["ip", "face"]

    results = []

    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors",
        model_dir="clip_vision",
        file_name="clip_vision_vit_h.safetensors",
    )
    results += [folder_paths.get_full_path("clip_vision", "clip_vision_vit_h.safetensors")]

    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors",
        model_dir="ipadapter",
        file_name="fooocus_ip_negative.safetensors",
    )
    results += [folder_paths.get_full_path("ipadapter", "fooocus_ip_negative.safetensors")]

    if v == "ip":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin",
            model_dir="ipadapter",
            file_name="ip-adapter-plus_sdxl_vit-h.bin",
        )
        results += [folder_paths.get_full_path("ipadapter", "ip-adapter-plus_sdxl_vit-h.bin")]

    if v == "face":
        load_file_from_url(
            url="https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin",
            model_dir="ipadapter",
            file_name="ip-adapter-plus-face_sdxl_vit-h.bin",
        )
        results += [folder_paths.get_full_path("ipadapter", "ip-adapter-plus-face_sdxl_vit-h.bin")]
    return results


def downloading_upscale_model():
    load_file_from_url(
        url="https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin",
        model_dir="upscale_models",
        file_name="fooocus_upscaler_s409985e5.bin",
    )
    return folder_paths.get_full_path("upscale_models", "fooocus_upscaler_s409985e5.bin")
