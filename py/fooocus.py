import os
from urllib.parse import urlparse
from torch.hub import download_url_to_file
from log import log_node_warn

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
