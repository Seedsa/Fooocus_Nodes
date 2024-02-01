import os
from urllib.parse import urlparse
from typing import Optional
from urllib.error import URLError


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
    if not os.path.exists(cached_file):
        try:
            # 将 URL 和 cached_file 包含在引号内，确保PowerShell命令可以正确解析包含空格的路径
            os.system(f'PowerShell -Command "Write-Host \'正在下载模型（可能需要科学网络）: \"{url}\" 到模型文件夹 \"{model_dir}\"\' -Foreground Blue"')
            from torch.hub import download_url_to_file
            download_url_to_file(url, cached_file, progress=progress)
        except URLError as e:
            # 同样确保model_dir在PowerShell中被正确解析
            os.system(f"PowerShell -Command \"Write-Host '下载模型失败。请使用科学网络或者到UP主提供的网盘中下载官方默认模型{file_name}，放入模型文件夹：\"{model_dir}\"' -Foreground Red\"")
            raise
    return cached_file