import os
import folder_paths
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
inpaint_head_model_path=folder_paths.models_dir+"/inpaint/fooocus_inpaint_head.pth"
inpaint_patch_model_path=folder_paths.models_dir+"/inpaint/inpaint.fooocus.patch"
path_fooocus_expansion = os.path.normpath(os.path.join( os.path.abspath(os.path.dirname(__file__)), '..'))+"/prompt_expansion/fooocus_expansion"