# ComfyUI Fooocus Nodes

# Installation

1. Clone the repository:
   `git clone https://github.com/Seedsa/Fooocus_Nodes.git`  
   to your ComfyUI `custom_nodes` directory

# Update

1. Navigate to the cloned repo e.g. `custom_nodes/Fooocus_Nodes`
2. `git pull`

# How to Use

[Fooocus Expansion](https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin)download to /models/prompt_expansion/fooocus_expansion

[XL APPROX](https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth) download to /models/vae_approx

[SD1.5 APPROX](https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt) download to/models/vae_approx

# Model Cache

/py/modules/config.py/use_model_cache = True/False

# Features

- [x] Fooocus Txt2image&Img2img
- [x] Fooocus Inpaint&Outpaint
- [x] Fooocus Upscale
- [x] Fooocus ImagePrompt&FaceSwap
- [x] Fooocus Canny&CPDS
- [x] Fooocus Styles&PromptExpansion

# Workflows

## Basic

![basic](/workflow/basic.png)

## FooocusStyles

![basic](/workflow/basic+fooocus_styles.png)

## Canny&CPDS

![basic](/workflow/canny&cpds.png)

## imagePrompt&FaceSwap

![basic](/workflow/imagePrompt&faceswap.png)

## Inpaint&Outpaint

![basic](/workflow/inpaint&outpaint.png)
