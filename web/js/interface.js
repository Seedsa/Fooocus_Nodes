import { app } from "/scripts/app.js";

// 增加Slot颜色
const customPipeLineLink = "#7737AA";
const customPipeLineSDXLLink = "#7737AA";
const customIntLink = "#29699C";
const customXYPlotLink = "#74DA5D";
const customXYLink = "#38291f";
const STRINGLink = "#00aa8c";

var customLinkColors =
  JSON.parse(localStorage.getItem("Comfy.Settings.ttN.customLinkColors")) || {};
if (
  !customLinkColors["PIPE_LINE"] ||
  !LGraphCanvas.link_type_colors["PIPE_LINE"]
) {
  customLinkColors["PIPE_LINE"] = customPipeLineLink;
}
if (
  !customLinkColors["PIPE_LINE_SDXL"] ||
  !LGraphCanvas.link_type_colors["PIPE_LINE_SDXL"]
) {
  customLinkColors["PIPE_LINE_SDXL"] = customPipeLineSDXLLink;
}
if (!customLinkColors["INT"] || !LGraphCanvas.link_type_colors["INT"]) {
  customLinkColors["INT"] = customIntLink;
}
if (!customLinkColors["XYPLOT"] || !LGraphCanvas.link_type_colors["XYPLOT"]) {
  customLinkColors["XYPLOT"] = customXYPlotLink;
}
if (!customLinkColors["X_Y"] || !LGraphCanvas.link_type_colors["X_Y"]) {
  customLinkColors["X_Y"] = customXYLink;
}
if (!customLinkColors["STRING"] || !LGraphCanvas.link_type_colors["STRING"]) {
  customLinkColors["STRING"] = STRINGLink;
}

localStorage.setItem(
  "Comfy.Settings.fooocus.customLinkColors",
  JSON.stringify(customLinkColors)
);

// 节点颜色
const COLOR_THEMES = LGraphCanvas.node_colors;
const NODE_COLORS = {
  "Fooocus positive": "green",
  "Fooocus negative": "red",
};

function setNodeColors(node, theme) {
  if (!theme) {
    return;
  }
  if (theme.color) node.color = theme.color;
  if (theme.bgcolor) node.bgcolor = theme.bgcolor;
}

app.registerExtension({
  name: "comfy.fooocus.interface",
  setup() {
    Object.assign(app.canvas.default_connection_color_byType, customLinkColors);
    Object.assign(LGraphCanvas.link_type_colors, customLinkColors);
  },

  nodeCreated(node) {
    if (NODE_COLORS.hasOwnProperty(node.comfyClass)) {
      const colorKey = NODE_COLORS[node.comfyClass];
      const theme = COLOR_THEMES[colorKey];
      setNodeColors(node, theme);
    }
  },
});
