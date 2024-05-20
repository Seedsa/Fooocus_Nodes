import os
import folder_paths
from pathlib import Path
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
    ("自定义", "自定义"),
]


def read_user_config(file_path):
    """
    Reads a user YAML configuration file and returns its contents as a dictionary.

    Parameters:
    file_path (str): The path to the configuration file.

    Returns:
    dict: The configuration as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error: The file {file_path} contains invalid YAML. Error details: {exc}")
        return None


user_config_path = os.path.join(Path(__file__).parent.parent.parent, "user-config.yaml")
user_config = read_user_config(user_config_path)


def get_path(sub_path):
    base_path = user_config.get("models_directory", None)
    default_paths = {
        "path_styles": "sdxl_styles",
        "path_wildcards": "wildcards",
    }
    for key, value in default_paths.items():
        if key in sub_path:
            return os.path.join(Path(__file__).parent.parent.parent, value)

    if base_path is not None and os.path.isdir(base_path):
        return os.path.join(base_path, sub_path)
    else:
        return os.path.join(folder_paths.models_dir, sub_path)


allow_download = user_config.get("allow_download_models", True)
wildcard_filenames = []
paths_checkpoints = get_path("checkpoints")
paths_loras = get_path("loras")
path_embeddings = get_path("embeddings")
path_vae_approx = get_path("vae_approx")
path_controlnet = get_path("controlnet")
path_clip_vision = get_path("clip_vision")
path_upscale_models = get_path("upscale_models")
path_fooocus_expansion = get_path("prompt_expansion/fooocus_expansion")
path_inpaint = get_path("inpaint")
path_ipadapter = get_path("ipadapter")
path_styles = get_path("path_styles")
path_wildcards = get_path("path_wildcards")

use_model_cache = False
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

# def debug_paths():
#     print(f"{paths_checkpoints=}")
#     print(f"{paths_loras=}")
#     print(f"{path_styles=}")
# debug_paths()
