import { app } from "/scripts/app.js";

let origProps = {};

const findWidgetByName = (node, name) =>
  node.widgets.find((w) => w.name === name);

const doesInputWithNameExist = (node, name) =>
  node.inputs ? node.inputs.some((input) => input.name === name) : false;

function updateNodeHeight(node) {
  node.setSize([node.size[0], node.computeSize()[1]]);
}

function toggleWidget(node, widget, show = false, suffix = "") {
  if (!widget || doesInputWithNameExist(node, widget.name)) return;
  if (!origProps[widget.name]) {
    origProps[widget.name] = {
      origType: widget.type,
      origComputeSize: widget.computeSize,
    };
  }
  const origSize = node.size;

  widget.type = show ? origProps[widget.name].origType : "esayHidden" + suffix;
  widget.computeSize = show
    ? origProps[widget.name].origComputeSize
    : () => [0, -4];

  widget.linkedWidgets?.forEach((w) =>
    toggleWidget(node, w, ":" + widget.name, show)
  );

  const height = show
    ? Math.max(node.computeSize()[1], origSize[1])
    : node.size[1];
  node.setSize([node.size[0], height]);
}

function widgetLogic(node, widget) {
  if (widget.name === "lora_name") {
    if (widget.value === "None") {
      toggleWidget(node, findWidgetByName(node, "lora_model_strength"));
      toggleWidget(node, findWidgetByName(node, "lora_clip_strength"));
    } else {
      toggleWidget(node, findWidgetByName(node, "lora_model_strength"), true);
      toggleWidget(node, findWidgetByName(node, "lora_clip_strength"), true);
    }
  }

  if (widget.name === "image_output") {
    if (widget.value === "Sender" || widget.value === "Sender/Save") {
      toggleWidget(node, findWidgetByName(node, "link_id"), true);
    } else {
      toggleWidget(node, findWidgetByName(node, "link_id"));
    }
    if (
      widget.value === "Hide" ||
      widget.value === "Preview" ||
      widget.value === "Sender"
    ) {
      toggleWidget(node, findWidgetByName(node, "save_prefix"));
      toggleWidget(node, findWidgetByName(node, "output_path"));
      toggleWidget(node, findWidgetByName(node, "embed_workflow"));
      toggleWidget(node, findWidgetByName(node, "number_padding"));
      toggleWidget(node, findWidgetByName(node, "overwrite_existing"));
    } else if (
      widget.value === "Save" ||
      widget.value === "Hide/Save" ||
      widget.value === "Sender/Save"
    ) {
      toggleWidget(node, findWidgetByName(node, "save_prefix"), true);
      toggleWidget(node, findWidgetByName(node, "output_path"), true);
      toggleWidget(node, findWidgetByName(node, "embed_workflow"), true);
      toggleWidget(node, findWidgetByName(node, "number_padding"), true);
      toggleWidget(node, findWidgetByName(node, "overwrite_existing"), true);
    }
  }

  if (widget.name === "num_loras") {
    let number_to_show = widget.value + 1;
    for (let i = 0; i < number_to_show; i++) {
      toggleWidget(node, findWidgetByName(node, "lora_" + i + "_name"), true);
      toggleWidget(
        node,
        findWidgetByName(node, "lora_" + i + "_strength"),
        true
      );
    }
    for (let i = number_to_show; i < 21; i++) {
      toggleWidget(node, findWidgetByName(node, "lora_" + i + "_name"));
      toggleWidget(node, findWidgetByName(node, "lora_" + i + "_strength"));
    }
    updateNodeHeight(node);
  }

  if (widget.name === "generation_mode") {
    if (widget.value === "inpaint") {
      toggleWidget(
        node,
        findWidgetByName(node, "inpaint_respective_field"),
        true
      );
      toggleWidget(node, findWidgetByName(node, "inpaint_engine"), true);
      toggleWidget(node, findWidgetByName(node, "top"));
      toggleWidget(node, findWidgetByName(node, "bottom"));
      toggleWidget(node, findWidgetByName(node, "left"));
      toggleWidget(node, findWidgetByName(node, "right"));
      toggleWidget(node, findWidgetByName(node, "inpaint_mask"));
    }
    if (widget.value === "outpaint") {
      toggleWidget(
        node,
        findWidgetByName(node, "inpaint_respective_field"),
        true
      );
      toggleWidget(node, findWidgetByName(node, "inpaint_engine"), true);
      toggleWidget(node, findWidgetByName(node, "top"), true);
      toggleWidget(node, findWidgetByName(node, "bottom"), true);
      toggleWidget(node, findWidgetByName(node, "left"), true);
      toggleWidget(node, findWidgetByName(node, "right"), true);
    }
    if (widget.value === "text_or_images_to_images") {
      toggleWidget(node, findWidgetByName(node, "inpaint_respective_field"));
      toggleWidget(node, findWidgetByName(node, "inpaint_engine"));
      toggleWidget(node, findWidgetByName(node, "top"));
      toggleWidget(node, findWidgetByName(node, "bottom"));
      toggleWidget(node, findWidgetByName(node, "left"));
      toggleWidget(node, findWidgetByName(node, "right"));
      toggleWidget(node, findWidgetByName(node, "inpaint_mask"));
    }
    updateNodeHeight(node);
  }
  if (widget.name === "settings") {
    if (widget.value === "Simple") {
      toggleWidget(node, findWidgetByName(node, "sharpness"), false);
      toggleWidget(node, findWidgetByName(node, "adaptive_cfg"), false);
      toggleWidget(node, findWidgetByName(node, "adm_scaler_positive"), false);
      toggleWidget(node, findWidgetByName(node, "adm_scaler_negative"), false);
      toggleWidget(node, findWidgetByName(node, "adm_scaler_end"), false);
    }
    if (widget.value === "Advanced") {
      toggleWidget(node, findWidgetByName(node, "sharpness"), true);
      toggleWidget(node, findWidgetByName(node, "adaptive_cfg"), true);
      toggleWidget(node, findWidgetByName(node, "adm_scaler_positive"), true);
      toggleWidget(node, findWidgetByName(node, "adm_scaler_negative"), true);
      toggleWidget(node, findWidgetByName(node, "adm_scaler_end"), true);
    }
    updateNodeHeight(node);
  }
  if (widget.name === "resolution") {
    if (widget.value === "自定义 x 自定义") {
      toggleWidget(node, findWidgetByName(node, "empty_latent_width"), true);
      toggleWidget(node, findWidgetByName(node, "empty_latent_height"), true);
    } else {
      toggleWidget(node, findWidgetByName(node, "empty_latent_width"), false);
      toggleWidget(node, findWidgetByName(node, "empty_latent_height"), false);
    }
    updateNodeHeight(node);
  }
  if (widget.name === "toggle") {
    widget.type = "toggle";
    widget.options = { on: "Enabled", off: "Disabled" };
  }
}

app.registerExtension({
  name: "comfy.fooocus.dynamicWidgets",

  nodeCreated(node) {
    switch (node.comfyClass) {
      case "Fooocus LoraStack":
      case "Fooocus Loader":
      case "Fooocus KSampler":
      case "Fooocus PreKSampler":
      case "Fooocus Upscale":
        getSetters(node);
        break;
    }
  },
});

const getSetWidgets = [
  "image_output",
  "num_loras",
  "toggle",
  "resolution",
  "generation_mode",
  "settings",
];

function getSetters(node) {
  if (node.widgets)
    for (const w of node.widgets) {
      if (getSetWidgets.includes(w.name)) {
        widgetLogic(node, w);
        let widgetValue = w.value;

        // Define getters and setters for widget values
        Object.defineProperty(w, "value", {
          get() {
            return widgetValue;
          },
          set(newVal) {
            if (newVal !== widgetValue) {
              widgetValue = newVal;
              widgetLogic(node, w);
            }
          },
        });
      }
    }
}
