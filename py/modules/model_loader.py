import os
from urllib.parse import urlparse
from typing import Optional
import folder_paths

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
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    # 从所有文件夹中寻找
    cached_file = folder_paths.get_full_path(model_dir, file_name)
    if cached_file is None:
        os.makedirs(folder_paths.get_folder_paths(model_dir)[0], exist_ok=True)
        cached_file = os.path.join(folder_paths.get_folder_paths(model_dir)[0],file_name)

    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file
