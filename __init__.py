import os
import importlib
import shutil
import folder_paths


def create_folder_and_update_paths(folder_name):
    folder_path = os.path.join(folder_paths.models_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    folder_paths.folder_names_and_paths[folder_name] = (
        [
            folder_path,
            *folder_paths.folder_names_and_paths.get(folder_name, [[], set()])[0]
        ],
        folder_paths.supported_pt_extensions
    )
create_folder_and_update_paths("fooocus_expansion")
create_folder_and_update_paths("inpaint")
create_folder_and_update_paths("ipadapter")

from .py.modules.model_loader import load_file_from_url
from .py.modules.config import (
    path_fooocus_expansion as fooocus_expansion_path,
)
from .py import log


node_list = [
    "api",
    "fooocusNodes",
    "prompt"
]


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for module_name in node_list:
    imported_module = importlib.import_module(
        ".py.{}".format(module_name), __name__)
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS,
                           **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}


WEB_DIRECTORY = "./web"


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)
    dir = os.path.abspath(dir)
    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def install_expansion():
    src_dir = get_ext_dir("fooocus_expansion")
    if not os.path.exists(src_dir):
        log.log_node_error(
            "prompt_expansion is not exists. Please reinstall the extension.")
        return
    if not os.path.exists(fooocus_expansion_path):
        log.log_node_info('Copying Prompt Expansion files')
        shutil.copytree(src_dir, fooocus_expansion_path, dirs_exist_ok=True)


def download_models():
    vae_approx_filenames = [
        ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
        ('vaeapp_sd15.pth',
         'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
        ('xl-to-v1_interposer-v3.1.safetensors',
         'https://huggingface.co/lllyasviel/misc/resolve/main/xl-to-v1_interposer-v3.1.safetensors')
    ]

    for file_name, url in vae_approx_filenames:
        load_file_from_url(
            url=url, model_dir="vae_approx", file_name=file_name)

    install_expansion()
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir="fooocus_expansion",
        file_name='pytorch_model.bin'
    )


download_models()

__all__ = ['NODE_CLASS_MAPPINGS',
           'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
print("\033[0m\033[95m ComfyUI  Fooocus Nodes  :\033[0m \033[32mloaded\033[0m")
