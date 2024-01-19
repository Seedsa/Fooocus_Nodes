import json
import gradio as gr
import modules.config


def load_parameter_button_click(raw_prompt_txt, is_generating):
    loaded_parameter_dict = json.loads(raw_prompt_txt)
    assert isinstance(loaded_parameter_dict, dict)

    results = [True, 1]

    try:
        h = loaded_parameter_dict.get('正向提示词', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('反向提示词', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('风格', None)
        h = eval(h)
        assert isinstance(h, list)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('性能', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('分辨率', None)
        width, height = eval(h)
        formatted = modules.config.add_ratio(f'{width}*{height}')
        if formatted in modules.config.available_aspect_ratios:
            results.append(formatted)
            results.append(-1)
            results.append(-1)
        else:
            results.append(gr.update())
            results.append(width)
            results.append(height)
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('采样锐度 Sharpness', None)
        assert h is not None
        h = float(h)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('导向缩放倍数', None)
        assert h is not None
        h = float(h)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('ADM导向', None)
        p, n, e = eval(h)
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('基础模型', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('精炼模型', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('精炼开关', None)
        assert h is not None
        h = float(h)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('采样器 Sampler', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('调度器 Scheduler', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('种子 Seed', None)
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except:
        results.append(gr.update())
        results.append(gr.update())

    if is_generating:
        results.append(gr.update())
    else:
        results.append(gr.update(visible=True))
    
    results.append(gr.update(visible=False))

    for i in range(1, 6):
        try:
            n, w = loaded_parameter_dict.get(f'LoRA {i}').split(' : ')
            w = float(w)
            results.append(n)
            results.append(w)
        except:
            results.append(gr.update())
            results.append(gr.update())

    return results
