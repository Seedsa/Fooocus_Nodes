import re
import os
import sys


modules_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(modules_path)
from server import PromptServer
from modules.sdxl_styles import legal_style_names

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Module 'aiohttp' not installed. Please install it via:")
    print("pip install aiohttp")
    sys.exit()


@PromptServer.instance.routes.get("/fooocus/prompt/styles")
async def getStylesList(request):
    if "name" in request.rel_url.query:
        name = request.rel_url.query["name"]
    return web.json_response(legal_style_names)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
