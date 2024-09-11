"""
This module initializes and sets up the Fooocus extension for ComfyUI.
It handles folder creation, file downloads, and node mapping for the extension.
"""

import os
import importlib
import shutil
import folder_paths
import filecmp
def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    for full_folder_path in full_folder_paths:
        folder_paths.add_model_folder_path(folder_name, full_folder_path)
    if folder_name in folder_paths.folder_names_and_paths:
        current_paths, current_extensions = folder_paths.folder_names_and_paths[folder_name]
        updated_extensions = current_extensions | extensions
        folder_paths.folder_names_and_paths[folder_name] = (current_paths, updated_extensions)
    else:
        folder_paths.folder_names_and_paths[folder_name] = (full_folder_paths, extensions)

model_path = folder_paths.models_dir
add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(model_path, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("ultralytics_segm", [os.path.join(model_path, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("ultralytics", [os.path.join(model_path, "ultralytics")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("sams", [os.path.join(model_path, "sams")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("ipadapter", [os.path.join(model_path, "ipadapter")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("inpaint", [os.path.join(model_path, "inpaint")], folder_paths.supported_pt_extensions)
add_folder_path_and_extensions("fooocus_expansion", [os.path.join(model_path, "fooocus_expansion")], folder_paths.supported_pt_extensions)

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


def recursive_overwrite(src, dest, ignore=None):
    if os.path.isdir(src):
        if not os.path.isdir(dest):
            os.makedirs(dest)
        files = os.listdir(src)
        if ignore is not None:
            ignored = ignore(src, files)
        else:
            ignored = set()
        for f in files:
            if f not in ignored:
                recursive_overwrite(os.path.join(src, f),
                                    os.path.join(dest, f),
                                    ignore)
    else:
        if not os.path.exists(dest) or not filecmp.cmp(src, dest):
            shutil.copyfile(src, dest)
            log.log_node_info(f'Copying file from {src} to {dest}')

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
        os.makedirs(fooocus_expansion_path)
    recursive_overwrite(src_dir, fooocus_expansion_path)


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
