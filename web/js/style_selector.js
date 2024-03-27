// 1.0.3
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { $el } from "/scripts/ui.js";

// 添加样式
const styleElement = document.createElement("style");
const cssCode = `
    .fooocus-prompt-styles .tools{
        display:flex;
        justify-content:flex-between;
        height:30px;
        padding-bottom:10px;
        border-bottom:2px solid var(--border-color);
    }
    .fooocus-prompt-styles .tools button.delete{
        height:30px;
        border-radius: 8px;
        border: 2px solid var(--border-color);
        font-size:11px;
        background:var(--comfy-input-bg);
        color:var(--error-text);
        box-shadow:none;
        cursor:pointer;
    }
    .fooocus-prompt-styles .tools button.delete:hover{
        filter: brightness(1.2);
    }
    .fooocus-prompt-styles .tools textarea.search{
        flex:1;
        margin-left:10px;
        height:20px;
        line-height:20px;
        border-radius: 8px;
        border: 2px solid var(--border-color);
        font-size:11px;
        background:var(--comfy-input-bg);
        color:var(--input-text);
        box-shadow:none;
        padding:4px 10px;
        outline: none;
        resize: none;
        appearance:none;
    }
    .fooocus-prompt-styles-list{
        list-style: none;
        padding: 0;
        margin: 0;
        min-height: 150px;
        height: calc(100% - 40px);
        overflow: auto;
        // display: flex;
        // flex-wrap: wrap;
    }
    .fooocus-prompt-styles-tag{
        display: inline-block;
        vertical-align: middle;
        margin-top: 8px;
        margin-right: 8px;
        padding:4px;
        color: var(--input-text);
        background-color: var(--comfy-input-bg);
        border-radius: 8px;
        border: 2px solid var(--border-color);
        font-size:11px;
        cursor:pointer;
    }
    .fooocus-prompt-styles-tag.hide{
        display:none;
    }
    .fooocus-prompt-styles-tag:hover{
       filter: brightness(1.2);
    }
    .fooocus-prompt-styles-tag input{
        --ring-color: transparent;
        position: relative;
        box-shadow: none;
        border: 2px solid var(--border-color);
        border-radius: 2px;
        background: linear-gradient(135deg, var(--comfy-menu-bg) 0%, var(--comfy-input-bg) 60%);
    }
    .fooocus-prompt-styles-tag img{
        --ring-color: transparent;
        position: relative;
        border: 2px solid var(--border-color);
        border-radius: 2px;

    }


    .fooocus-prompt-styles-tag input[type=checkbox]:checked{
        border: 1px solid #006691;
        background-image: url("data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3e%3c/svg%3e");
        background-color: #006691;
    }
    .fooocus-prompt-styles-tag input[type=checkbox]{
        color-adjust: exact;
        display: inline-block;
        flex-shrink: 0;
        vertical-align: middle;
        appearance: none;
        border: 2px solid var(--border-color);
        background-origin: border-box;
        padding: 0;
        width: 1rem;
        height: 1rem;
        border-radius:4px;
        color:#006691;
        user-select: none;
    }
    .fooocus-prompt-styles-tag span{
        margin:0 4px;
        vertical-align: middle;
    }
    #show_image_id{
        width:128px;
        height:128px;
    }
`;
styleElement.innerHTML = cssCode;
document.head.appendChild(styleElement);
// 获取风格列表
let styles_list_cache = {};
async function getStylesList(name) {
  if (styles_list_cache[name]) return styles_list_cache[name];
  else {
    const resp = await api.fetchApi(`/fooocus/prompt/styles?name=${name}`);
    if (resp.status === 200) {
      let data = await resp.json();
      styles_list_cache[name] = data;
      return data;
    }
    return undefined;
  }
}

function getTagList(tags, styleName, language = "en-US") {
  let rlist = [];
  tags.forEach((k, i) => {
    rlist.push(
      $el(
        "label.fooocus-prompt-styles-tag",
        {
          dataset: {
            tag: k,
            name: k,
            index: i,
          },
          $: (el) => {
            el.children[0].onclick = () => {
              el.classList.toggle("fooocus-prompt-styles-tag-selected");
            };
          },
        },
        [
          $el("input", {
            type: "checkbox",
            name: k,
          }),
          $el("span", {
            textContent: k,
          }),
        ]
      )
    );
  });
  return rlist;
}

app.registerExtension({
  name: "comfy.fooocus.styleSelector",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name == "Fooocus Styles") {
      // 创建时
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated ? onNodeCreated?.apply(this, arguments) : undefined;
        const styles_id = this.widgets.findIndex((w) => w.name == "styles");
        const language =
          localStorage["AGL.Locale"] ||
          localStorage["Comfy.Settings.AGL.Locale"] ||
          "en-US";

        const list = $el("ul.fooocus-prompt-styles-list", []);
        let styles_values = "";
        this.setProperty("values", []);

        let selector = this.addDOMWidget(
          "select_styles",
          "btn",
          $el("div.fooocus-prompt-styles", [
            $el("div.tools", [
              $el("button.delete", {
                textContent: language == "zh-CN" ? "清空所有" : "Empty All",
                style: {},
                onclick: () => {
                  selector.element.children[1]
                    .querySelectorAll(".fooocus-prompt-styles-tag-selected")
                    .forEach((el) => {
                      el.classList.remove("fooocus-prompt-styles-tag-selected");
                      el.children[0].checked = false;
                    });
                  this.setProperty("values", []);
                },
              }),
              $el("textarea.search", {
                dir: "ltr",
                style: { "overflow-y": "scroll" },
                rows: 1,
                placeholder:
                  language == "zh-CN"
                    ? "🔎 在此处输入以搜索样式 ..."
                    : "🔎 Type here to search styles ...",
                oninput: (e) => {
                  let value = e.target.value.toLowerCase(); // 将输入值转换为小写
                  selector.element.children[1]
                    .querySelectorAll(".fooocus-prompt-styles-tag")
                    .forEach((el) => {
                      // 将每个标签的相关属性也转换为小写进行比较
                      let name = el.dataset.name.toLowerCase();
                      let tag = el.dataset.tag.toLowerCase();
                      let classValue = el.classList.value.toLowerCase();
                      // 检查是否包含输入值，以实现模犊搜索
                      if (
                        name.includes(value) ||
                        tag.includes(value) ||
                        classValue.includes(
                          "fooocus-prompt-styles-tag-selected"
                        )
                      ) {
                        el.classList.remove("hide");
                      } else {
                        el.classList.add("hide");
                      }
                    });
                },
              }),
            ]),
            list,
          ])
        );

        Object.defineProperty(this.widgets[styles_id], "value", {
          set: (value) => {
            styles_values = value;
            if (styles_values) {
              getStylesList(styles_values).then((_) => {
                selector.element.children[1].innerHTML = "";
                if (styles_list_cache[styles_values]) {
                  let tags = styles_list_cache[styles_values];
                  // 重新排序
                  if (selector.value)
                    tags = tags.sort(
                      (a, b) =>
                        selector.value.includes(b.name) -
                        selector.value.includes(a.name)
                    );
                  let list = getTagList(tags, value, language);
                  selector.element.children[1].append(...list);
                  selector.element.children[1]
                    .querySelectorAll(".fooocus-prompt-styles-tag")
                    .forEach((el) => {
                      if (this.properties["values"].includes(el.dataset.tag)) {
                        el.classList.add("fooocus-prompt-styles-tag-selected");
                      }
                      this.setSize([425, 500]);
                    });
                }
              });
            }
          },
          get: () => {
            return styles_values;
          },
        });

        let style_select_values = "";
        Object.defineProperty(selector, "value", {
          set: (value) => {
            setTimeout((_) => {
              selector.element.children[1]
                .querySelectorAll(".fooocus-prompt-styles-tag")
                .forEach((el) => {
                  let arr = value.split(",");
                  if (arr.includes(el.dataset.tag)) {
                    el.classList.add("fooocus-prompt-styles-tag-selected");
                    el.children[0].checked = true;
                  }
                });
            }, 300);
          },
          get: () => {
            selector.element.children[1]
              .querySelectorAll(".fooocus-prompt-styles-tag")
              .forEach((el) => {
                if (
                  el.classList.value.indexOf(
                    "fooocus-prompt-styles-tag-selected"
                  ) >= 0
                ) {
                  if (!this.properties["values"].includes(el.dataset.tag)) {
                    this.properties["values"].push(el.dataset.tag);
                  }
                } else {
                  if (this.properties["values"].includes(el.dataset.tag)) {
                    this.properties["values"] = this.properties[
                      "values"
                    ].filter((v) => v != el.dataset.tag);
                  }
                }
              });
            style_select_values = this.properties["values"].join(",");
            return style_select_values;
          },
        });

        let old_values = "";
        let style_lists_dom = selector.element.children[1];
        style_lists_dom.addEventListener("mouseenter", function (e) {
          let value = "";
          style_lists_dom
            .querySelectorAll(".fooocus-prompt-styles-tag-selected")
            .forEach((el) => (value += el.dataset.tag));
          old_values = value;
        });
        style_lists_dom.addEventListener("mouseleave", function (e) {
          let value = "";
          style_lists_dom
            .querySelectorAll(".fooocus-prompt-styles-tag-selected")
            .forEach((el) => (value += el.dataset.tag));
          let new_values = value;
          if (old_values != new_values) {
            // console.log("选项发生了变化")
            // 获取搜索值
            const search_value =
              document.getElementsByClassName("search")[0]["value"];
            // 重新排序
            const tags = styles_list_cache[styles_values].sort(
              (a, b) =>
                new_values.includes(b.name) - new_values.includes(a.name)
            );
            style_lists_dom.innerHTML = "";
            let list = getTagList(tags, styles_values, language);
            style_lists_dom.append(...list);
            style_lists_dom
              .querySelectorAll(".fooocus-prompt-styles-tag")
              .forEach((el) => {
                if (new_values.includes(el.dataset.tag)) {
                  el.classList.add("fooocus-prompt-styles-tag-selected");
                  el.children[0].checked = true;
                }
                if (search_value) {
                  if (
                    el.dataset.name.indexOf(search_value) != -1 ||
                    el.dataset.tag.indexOf(search_value) != -1 ||
                    el.classList.value.indexOf(
                      "fooocus-prompt-styles-tag-selected"
                    ) != -1
                  ) {
                    el.classList.remove("hide");
                  } else {
                    el.classList.add("hide");
                  }
                }
              });
          }
        });

        // 初始化
        setTimeout((_) => {
          if (!styles_values) {
            styles_values = "fooocus_styles";
            getStylesList(styles_values).then((_) => {
              selector.element.children[1].innerHTML = "";
              if (styles_list_cache[styles_values]) {
                let tags = styles_list_cache[styles_values];

                // 重新排序
                if (selector.value)
                  tags = tags.sort(
                    (a, b) =>
                      selector.value.includes(b.name) -
                      selector.value.includes(a.name)
                  );
                let list = getTagList(tags, styles_values, language);
                selector.element.children[1].append(...list);
              }
            });
          }
          this.setSize([425, 500]);
        }, 100);

        return onNodeCreated;
      };
    }
  },
});
