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
FOOOCUS_STYLES_DIR = os.path.join(Path(__file__).parent.parent.parent, "sdxl_styles")


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
  "PyraCanny":{
    "model_url":"https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors"
  },
  "CPDS":{
    "model_url":"https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors"
  }
}

controlnet_softness=0.25
path_fooocus_expansion = folder_paths.models_dir+"/prompt_expansion/fooocus_expansion"
path_loras=folder_paths.get_folder_paths('loras')[0]
path_embeddings=folder_paths.get_folder_paths('embeddings')[0]
path_vae_approx=folder_paths.get_folder_paths('vae_approx')[0]
path_controlnet=folder_paths.get_folder_paths('controlnet')[0]

use_model_cache = True
path_checkpoints=folder_paths.models_dir+"/checkpoints"
default_refiner_model_name='None'
default_base_model_name="juggernautXL_v8Rundiffusion.safetensors"
default_loras=[
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
