import os

path_styles = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))

# 风格提示词选择器
class FooocusStyles:
    @classmethod
    def INPUT_TYPES(s):
        styles = ["fooocus_styles"]
        styles_dir = path_styles
        for file_name in os.listdir(styles_dir):
            file = os.path.join(styles_dir, file_name)
            if (
                os.path.isfile(file)
                and file_name.endswith(".json")
                and "styles" in file_name.split(".")[0]
            ):
                styles.append(file_name.split(".")[0])
        return {
            "required": {
                "styles": (styles, {"default": "fooocus_styles"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    #
    RETURN_TYPES = (
        "FOOOCUS_STYLES",
    )
    RETURN_NAMES = (
        "fooocus_styles",
    )

    CATEGORY = "Fooocus/Prompt"
    FUNCTION = "run"
    OUTPUT_MODE = True

    def run(
        self,
        styles,
        prompt=None,
        extra_pnginfo=None,
        my_unique_id=None,
    ):
        values = []
        if my_unique_id in prompt:
            if prompt[my_unique_id]["inputs"]["select_styles"]:
                values = prompt[my_unique_id]["inputs"]["select_styles"].split(
                    ",")

        return (values,)


# 正面提示词
class positivePrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"positive": ("STRING", {"default": "", "multiline": True, "placeholder": "Positive"},),
                         }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "main"

    CATEGORY = "Fooocus/Prompt"

    @staticmethod
    def main(positive):
        return (positive,)


# 负面提示词
class negativePrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "negative": (
                    "STRING",
                    {"default": "", "multiline": True, "placeholder": "Negative"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "main"

    CATEGORY = "Fooocus/Prompt"

    @staticmethod
    def main(negative):
        return (negative,)


NODE_CLASS_MAPPINGS = {
    "Fooocus positive": positivePrompt,
    "Fooocus negative": negativePrompt,
    "Fooocus Styles": FooocusStyles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fooocus positive": "Positive",
    "Fooocus negative": "Negative",
    "Fooocus stylesSelector": "stylesPromptSelector",
    "Fooocus Styles": "Fooocus Styles"
}
