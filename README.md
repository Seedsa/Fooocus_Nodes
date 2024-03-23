# ComfyUI Fooocus Nodes

# Installation

1. Clone the repository:
   `git clone https://github.com/Seedsa/Fooocus_Nodes.git`  
   to your ComfyUI `custom_nodes` directory

# Update

1. Navigate to the cloned repo e.g. `custom_nodes/Fooocus_Nodes`
2. `git pull`

# Model Cache

For improved model switch performance with RAM > 40GB, you can enable model caching:

Open /py/modules/config.py
Set use_model_cache to True:

```
use_model_cache = True
```

# Features

- [x] Fooocus Txt2image&Img2img
- [x] Fooocus Inpaint&Outpaint
- [x] Fooocus Upscale
- [x] Fooocus ImagePrompt&FaceSwap
- [x] Fooocus Canny&CPDS
- [x] Fooocus Styles&PromptExpansion

# Workflows

Here are some examples of basic and advanced workflows supported by Fooocus Nodes:

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

# Acknowledgments

This project builds upon and extends the original work found at [ComfyUI_Fooocus](https://github.com/17Retoucher/ComfyUI_Fooocus).
