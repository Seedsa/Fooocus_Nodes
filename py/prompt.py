import os
import json
from pathlib import Path
from fooocus_modules.config import FOOOCUS_STYLES_DIR,RESOURCES_DIR



# 风格提示词选择器
class stylesPromptSelector:
    @classmethod
    def INPUT_TYPES(s):
        styles = ["fooocus_styles"]
        styles_dir = FOOOCUS_STYLES_DIR
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
            "optional": {
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    #
    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "positive",
        "negative",
    )

    CATEGORY = "Fooocus/Prompt"
    FUNCTION = "run"
    OUTPUT_MODE = True

    def replace_repeat(self, prompt):
        arr = prompt.replace("，", ",").split(",")
        if len(arr) != len(set(arr)):
            for i in range(len(arr)):
                arr[i] = arr[i].strip()
            arr = list(set(arr))
            return ", ".join(arr)
        else:
            return prompt

    def run(
        self,
        styles,
        positive="",
        negative="",
        prompt=None,
        extra_pnginfo=None,
        my_unique_id=None,
    ):
        values = []
        all_styles = {}
        positive_prompt, negative_prompt = "", negative
        if styles == "fooocus_styles":
            file = os.path.join(RESOURCES_DIR,  styles + '.json')
        else:
            file = os.path.join(RESOURCES_DIR, styles + '.json')
        f = open(file, "r", encoding="utf-8")
        data = json.load(f)
        f.close()
        for d in data:
            all_styles[d["name"]] = d
        if my_unique_id in prompt:
            if prompt[my_unique_id]["inputs"]["select_styles"]:
                values = prompt[my_unique_id]["inputs"]["select_styles"].split(",")

        has_prompt = False
        for index, val in enumerate(values):
            if "prompt" in all_styles[val]:
                if "{prompt}" in all_styles[val]["prompt"] and has_prompt == False:
                    positive_prompt = all_styles[val]["prompt"].format(prompt=positive)
                    has_prompt = True
                else:
                    positive_prompt += ", " + all_styles[val]["prompt"].replace(
                        ", {prompt}", ""
                    ).replace("{prompt}", "")
            if "negative_prompt" in all_styles[val]:
                negative_prompt += (
                    ", " + all_styles[val]["negative_prompt"]
                    if negative_prompt
                    else all_styles[val]["negative_prompt"]
                )

        if has_prompt == False and positive:
            positive_prompt = positive + ", "

        # 去重
        positive_prompt = (
            self.replace_repeat(positive_prompt) if positive_prompt else ""
        )
        negative_prompt = (
            self.replace_repeat(negative_prompt) if negative_prompt else ""
        )

        return (positive_prompt, negative_prompt)


# 正面提示词
class positivePrompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"positive": ( "STRING", {"default": "", "multiline": True, "placeholder": "Positive"},),
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
    "Fooocus stylesSelector": stylesPromptSelector,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fooocus positive": "Positive",
    "Fooocus negative": "Negative",
    "Fooocus stylesSelector": "stylesPromptSelector",

}
