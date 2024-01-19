import os
import sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
import folder_paths
from comfy.samplers import *
import config as config
from nodes import SaveImage, PreviewImage,CLIPSetLastLayer,VAELoader
import modules.default_pipeline as pipeline
import modules.core as core 
from extras.expansion import FooocusExpansion
from extras.expansion import safe_str
from modules.util import remove_empty_str, HWC3, resize_image, \
    get_image_shape_ceil, set_image_shape_ceil, get_shape_ceil, resample_image, erode_or_dilate
import  modules.inpaint_worker as inpaint_worker
import modules.patch

# 全局变量用于存储 pipe 字典
global_pipe = None

class FooocusStyle:
    @classmethod
    def INPUT_TYPES(cls):
        
        return {"required":{ 
                    "positive": ("STRING",{"multiline": True}), 
                    "negative": ("STRING",{"multiline": True}),
                    "PromptExpansion":("BOOLEAN", {"default": True, }),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
                     },
                
                } 
    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("positive", "negative", )
    FUNCTION = "fooocus_style"
    CATEGORY = "Fooocus"
    def fooocus_style(self, positive, negative,seed, PromptExpansion):
        if positive !='' and PromptExpansion:
            fooocus_expansion = FooocusExpansion()
            positive=fooocus_expansion(positive,seed,)
        else:
            print("PromptExpansion is off of positive prompt is none!!  ")   
    
        return positive,negative,
# lora
class FooocusLoraStack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 10
        inputs = {
            "required": {
                "toggle": ([True, False],),
                # "mode": (["simple", "advanced"],),
                "num_loras": ("INT", {"default": 1, "min": 0, "max": max_lora_num}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
            },
        }

        for i in range(1, max_lora_num+1):
            inputs["optional"][f"lora_{i}_name"] = (
            ["None"] + folder_paths.get_filename_list("loras"), {"default": "None"})
            inputs["optional"][f"lora_{i}_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})


        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"

    CATEGORY = "Fooocus"

    def stack(self, toggle,  num_loras, lora_stack=None, **kwargs):
        loras = []
        if (toggle in [False, None, "False"]) or not kwargs:
            return (loras,)
        

        # Import Stack values
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        # Import Lora values
        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_{i}_name")

            if not lora_name or lora_name == "None":
                continue

            lora_strength = float(kwargs.get(f"lora_{i}_strength"))
            loras.append([lora_name, lora_strength])

        return (loras,)



class FooocusLoader:
    @classmethod
    def INPUT_TYPES(cls):
        resolution_strings = [f"{width} x {height}" for width, height in config.BASE_RESOLUTIONS]
        return {"required":{ 
                    "base_model_name": (folder_paths.get_filename_list("checkpoints"),{"default":"juggernautXL_v8Rundiffusion.safetensors"} ), 
                    "refiner_model_name": (["None"]+folder_paths.get_filename_list("checkpoints"),{"default":"None"} ), 
                    "refiner_switch": ("FLOAT", {"default": 0.618, "min": 0.1, "max": 1, "step": 0.1}), 
                    "refiner_swap_method": (['joint', 'separate', 'vae'], ),                    
                    "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),), 
                    # "stop_at_clip_layer": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),                
                    "positive_prompt": ("STRING",{"forceInput": True}), 
                    "negative_prompt":("STRING",{"forceInput": True}),      
                    "resolution": (resolution_strings,{"default":"1024 x 1024"}),
                    "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                    "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                     },
                "optional": {"optional_lora_stack": ("LORA_STACK",)},                 
                }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE","CLIP",)
    RETURN_NAMES = ("pipe", "model", "vae","clip",)
    FUNCTION = "fooocus_loader"
    CATEGORY = "Fooocus"
    
    def fooocus_loader(self, base_model_name, refiner_model_name,refiner_switch,refiner_swap_method,vae_name,  positive_prompt, negative_prompt, 
                       resolution,  empty_latent_width, empty_latent_height,batch_size,optional_lora_stack):  
       
        if resolution != "自定义 x 自定义":
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.") 

        pipeline.refresh_everything(refiner_model_name,base_model_name,optional_lora_stack)
        if vae_name not in ["Baked VAE", "Baked-VAE"]:
            vae=VAELoader()
            vae=vae.load_vae(vae_name)
            pipeline.model_base.vae=vae 
        prompts = remove_empty_str([safe_str(p) for p in positive_prompt.splitlines()], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')
        positive=pipeline.clip_encode(prompts,len(prompts))
        negative=pipeline.clip_encode(negative_prompts,len(negative_prompts))        
        global global_pipe
        global_pipe = {
            # ... 初始化 pipe 字典内容 ...
            "base_model_name":base_model_name,
            "refiner_model_name":refiner_model_name,
            "resolution": resolution,
            "refiner_switch":refiner_switch,
            "vae_name":vae_name,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "batch_size":batch_size,
            "optional_lora_stack":optional_lora_stack,
            "refiner_swap_method":refiner_swap_method,
            "batch_size": batch_size,           
            "model": pipeline.model_base.unet_with_lora,
            "vae":pipeline.model_base.vae,
            "clip": pipeline.model_base.clip_with_lora,
            "positive": positive,
            "negative": negative,
            "latent_width": empty_latent_width,
            "latent_height": empty_latent_height,
        }        

        
        return (global_pipe,pipeline.model_base.unet_with_lora,pipeline.model_base.vae,pipeline.model_base.clip_with_lora  )   

class FooocusPreKSampler:
    @classmethod
    def INPUT_TYPES(cls):  
        return {"required":
                    {"pipe":("PIPE_LINE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),                   
                    "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0,"step":0.5}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m_sde_gpu", }),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", }),                 
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.1}),
                    "sharpness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0}),  
                    "adaptive_cfg":("FLOAT", {"default": 7, "min": 0.0, "max": 100.0}),  
                    "adm_scaler_positive":  ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0,"step":0.1}),                
                    "adm_scaler_negative":  ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0,"step":0.1}),                
                    "adm_scaler_end":  ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0,"step":0.1}),                
                    },
                    "optional":
                    {                        
                    "image":("IMAGE",),
                    "latent":("LATENT",),   
                     }
                }  

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)

    FUNCTION = "fooocus_preKSampler"
    CATEGORY = "Fooocus"
    def fooocus_preKSampler(self, pipe, seed, steps,  cfg, sampler_name, scheduler,  
                            sharpness,denoise, adaptive_cfg,adm_scaler_positive,adm_scaler_negative,adm_scaler_end,
                            image=None,latent=None,):
        global global_pipe
        assert global_pipe is not None, "请先调用 FooocusLoader 进行初始化！"

        modules.patch.adaptive_cfg=adaptive_cfg
        modules.patch.sharpness = sharpness
        modules.patch.positive_adm_scale = adm_scaler_positive
        modules.patch.negative_adm_scale = adm_scaler_negative
        modules.patch.adm_scaler_end = adm_scaler_end 
        if image is not None:
            latent= core.encode_vae(global_pipe["vae"],image)
            print("Image is not None, use image as latent.")
        elif latent is not None:            
            print("Latent is not None, use latent as latent.")
        else:
            latent = core.generate_empty_latent( global_pipe["latent_width"], global_pipe["latent_height"],global_pipe["batch_size"])

        new_pipe={
            "latent":latent,
            "seed":seed,
            "steps":steps,
            "cfg":cfg,
            "sampler_name":sampler_name,
            "scheduler":scheduler,
            "sharpness":sharpness,
            "denoise":denoise,
        }
        global_pipe.update(new_pipe)
        del new_pipe
        return {"ui": {"value": [seed]}, "result": (global_pipe,)} 

class FooocusKSampler:
    @classmethod
    def INPUT_TYPES(cls):
       
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save", ],{"default": "Preview"}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),                 
                 },
                 "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", },
                }

    RETURN_TYPES = ("PIPE_LINE","IMAGE")
    RETURN_NAMES=("pipe","images")
    FUNCTION = "fooocus_kSampler"
    OUTPUT_NODE = True
    CATEGORY = "Fooocus"    
    def fooocus_kSampler(self,pipe,image_output,save_prefix, prompt=None, extra_pnginfo=None,):    
        switch=int(round(global_pipe["refiner_switch"]*global_pipe["steps"]))

        imgs = pipeline.process_diffusion(
                positive_cond=global_pipe["positive"],
                negative_cond=global_pipe["negative"],
                steps=global_pipe["steps"],
                switch=switch,
                width=global_pipe["latent_width"],
                height=global_pipe["latent_height"],
                image_seed=global_pipe['seed'],
                callback=None,
                sampler_name=global_pipe["sampler_name"],
                scheduler_name=global_pipe["scheduler"],
                latent=global_pipe["latent"],
                denoise=global_pipe["denoise"],
                tiled=False,
                cfg_scale=global_pipe["cfg"],
                refiner_swap_method=global_pipe["refiner_swap_method"]
            )
        if image_output in ("Save", "Hide/Save"):
            saveimage=SaveImage()
            results=saveimage.save_images(imgs,save_prefix,prompt,extra_pnginfo)

        if image_output == "Preview":
            previewimage=PreviewImage()
            results=previewimage.save_images(imgs,save_prefix,prompt,extra_pnginfo)
        if image_output == "Hide":
            return  {"ui": {"value": list()}, "result": (pipe,imgs)}
        
        results["result"]=pipe,imgs
        return  results


#扩图
class FooocusInpainting:
   
    @classmethod
    def INPUT_TYPES(s):
       
        return {"required":                    
                    {"pipe":("PIPE_LINE",), 
                     "inpaint_image":("IMAGE",),
                     
                     "inpaint_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.1}),
                     "inpaint_respective_field": ("FLOAT", {"default": 0.618, "min": 0, "max": 1.0, "step": 0.1}),
                     "top":("BOOLEAN",{'default':False }),                         
                     "bottom":("BOOLEAN",{'default':False}),   
                     "left":("BOOLEAN",{'default':False }),   
                     "right":("BOOLEAN",{'default':False }),   
                     "image_output": (["Hide", "Preview", "Save", "Hide/Save", ],{"default": "Preview"}),
                     "save_prefix": ("STRING", {"default": "ComfyUI"}),
                     },
                "opsional":{
                   "inpaint_mask":("MASK",),
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",                  
                     },

                }

    RETURN_TYPES = ("PIPE_LINE","IMAGE")
    RETURN_NAMES=("pipe","images")
    FUNCTION = "fooocus_inpainting"
    CATEGORY = "Fooocus"
    def fooocus_inpainting(self, pipe,inpaint_strength, inpaint_respective_field,inpaint_image,inpaint_mask, top,bottom,left,right,image_output,save_prefix,prompt=None,extra_pnginfo=None):
        refiner_model_name = 'None'
        if inpaint_mask==None:
            if not top and not bottom and not left and not right:
                return pipe, inpaint_image
            inpaint_image=inpaint_image[0].numpy()
            inpaint_image=(inpaint_image * 255).astype(np.uint8)
            inpaint_mask=np.zeros(inpaint_image.shape, dtype=np.uint8)
            inpaint_mask=inpaint_mask[:, :, 0] 

            H,W,C=inpaint_image.shape
            if top:
                inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                        constant_values=255)
            if bottom :
                inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                        constant_values=255)

            H, W, C = inpaint_image.shape
            if left:
                inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant',
                                        constant_values=255)
            if right:
                inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
                inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant',
                                        constant_values=255)
            inpaint_strength=1.0
            inpaint_respective_field=1.0
        denoising_strength=inpaint_strength
        inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=inpaint_respective_field
            )
        inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
        inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
        inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

        decode_image=None

        if image_output in ("Save", "Hide/Save"):
            saveimage=SaveImage()
            results=saveimage.save_images(decode_image,save_prefix,prompt,extra_pnginfo)

        if image_output == "Preview":
            previewimage=PreviewImage()
            results=previewimage.save_images(decode_image,save_prefix,prompt,extra_pnginfo)
        if image_output == "Hide":
            return  {"ui": {"value": list()}, "result": (pipe,decode_image)}
        
        results["result"]=pipe,decode_image
        return  results





NODE_CLASS_MAPPINGS = {
    "FooocusStyle":FooocusStyle,
    "FooocusLoader": FooocusLoader,
    "FooocusPreKSampler": FooocusPreKSampler,
    "FooocusKSampler": FooocusKSampler,
    "FooocusInpainting":FooocusInpainting,
    "FooocusLoraStack":FooocusLoraStack,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FooocusStyle":"FooocusStyle",
    "FooocusLoader" : "FooocusLoader",
    "FooocusPreKSampler" : "FooocusPreKSampler",
    "FooocusKSampler" : "FooocusKSampler",
    "FooocusInpainting":"FooocusInpainting",
    "FooocusLoraStack":"FooocusLoraStack",


    
}