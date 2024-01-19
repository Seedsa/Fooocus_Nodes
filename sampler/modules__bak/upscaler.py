import os
import numpy as np
import torch
from comfy_extras.chainner_models.architecture.RRDB import RRDBNet as ESRGAN
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from collections import OrderedDict
import folder_paths

model_filename = os.path.join(folder_paths.get_folder_paths("upscale_models")[0], 'fooocus_upscaler_s409985e5.bin')
opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None
@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y


def perform_upscale(img):
    global model

    print(f'Upscaling image with shape {str(img.shape)} ...')

    if model is None:
        sd = torch.load(model_filename)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()

    img = numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = pytorch_to_numpy(img)[0]

    return img
