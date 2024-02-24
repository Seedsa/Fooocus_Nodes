import re
import os
import sys

sys.path.append(os.path.dirname(__file__))
modules_path = os.path.dirname(os.path.realpath(__file__))
import json
from server import PromptServer
from modules.sdxl_styles import legal_style_names

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Module 'aiohttp' not installed. Please install it via:")
    print("pip install aiohttp")
    sys.exit()

# parse csv
@PromptServer.instance.routes.post("/easyuse/upload/csv")
async def parse_csv(request):
    post = await request.post()
    csv = post.get("csv")
    if csv and csv.file:
        file = csv.file
        text = ''
        for line in file.readlines():
            line = str(line.strip())
            line = line.replace("'", "").replace("b",'')
            text += line + '; \n'
        return web.json_response(text)


#get style list
styles_dir = os.path.abspath(os.path.join(__file__, "../../styles"))
samples_dir= os.path.abspath(os.path.join(__file__, "../../styles/samples"))
resource_dir = os.path.abspath(os.path.join(__file__, "../../sdxl_styles"))
@PromptServer.instance.routes.get("/fooocus/prompt/styles")
async def getStylesList(request):
    if "name" in request.rel_url.query:
        name = request.rel_url.query["name"]
    return web.json_response(legal_style_names)

# get style preview image
fooocus_images_path = samples_dir
@PromptServer.instance.routes.get("/fooocus/prompt/styles/image")
async def getStylesImage(request):
    styles_name = request.rel_url.query["styles_name"] if "styles_name" in request.rel_url.query else None
    if "name" in request.rel_url.query:
        name = request.rel_url.query["name"]
        if os.path.exists(os.path.join(styles_dir, 'samples')):
            file = os.path.join(styles_dir, 'samples', name + '.jpg')
            if os.path.isfile(file):
                return web.FileResponse(file)
            elif styles_name == 'fooocus_styles':
                return web.Response(text=fooocus_images_path + name + '.jpg')
        elif styles_name == 'fooocus_styles':
            return web.Response(text=fooocus_images_path + name + '.jpg')
    return web.Response(status=400)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
