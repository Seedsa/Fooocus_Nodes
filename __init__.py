import os
import importlib


node_list = [
    "api",
    "fooocusNodes",
    "prompt"

]


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for module_name in node_list:
    imported_module = importlib.import_module(".py.{}".format(module_name), __name__)
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}





#Styles
styles_path = os.path.join(os.path.dirname(__file__), "styles")
samples_path = os.path.join(os.path.dirname(__file__), "styles", "samples")
if os.path.exists(styles_path):
    if not os.path.exists(samples_path):
        os.mkdir(samples_path)
else:
    os.mkdir(styles_path)
    os.mkdir(samples_path)


WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
print("\033[0m\033[95m ComfyUI  Fooocus  :\033[0m \033[32mloaded\033[0m")
