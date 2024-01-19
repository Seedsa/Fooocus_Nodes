# Consistent with Kohya/A1111 to reduce differences between model training and inference.

import os
import torch
import comfy.cldm.cldm
import comfy.k_diffusion.sampling
import comfy.ldm.modules.attention
import comfy.ldm.modules.diffusionmodules.model
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.ldm.modules.diffusionmodules.openaimodel
# import comfy.args_parser
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.samplers
import comfy.sd
import comfy.sd1_clip
import comfy.clip_vision
import comfy.ops as ops

from modules.ops import use_patched_ops
from transformers import CLIPTextModel, CLIPTextConfig, modeling_utils, CLIPVisionConfig, CLIPVisionModelWithProjection


def patched_encode_token_weights(self, token_weight_pairs):
    to_encode = list()
    max_token_len = 0
    has_weights = False
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        max_token_len = max(len(tokens), max_token_len)
        has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
        to_encode.append(tokens)

    sections = len(to_encode)
    if has_weights or sections == 0:
        to_encode.append(comfy.sd1_clip.gen_empty_tokens(self.special_tokens, max_token_len))

    out, pooled = self.encode(to_encode)
    if pooled is not None:
        first_pooled = pooled[0:1].to(comfy.model_management.intermediate_device())
    else:
        first_pooled = pooled

    output = []
    for k in range(0, sections):
        z = out[k:k + 1]
        if has_weights:
            original_mean = z.mean()
            z_empty = out[-1]
            for i in range(len(z)):
                for j in range(len(z[i])):
                    weight = token_weight_pairs[k][j][1]
                    if weight != 1.0:
                        z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            new_mean = z.mean()
            z = z * (original_mean / new_mean)
        output.append(z)

    if len(output) == 0:
        return out[-1:].to(comfy.model_management.intermediate_device()), first_pooled
    return torch.cat(output, dim=-2).to(comfy.model_management.intermediate_device()), first_pooled


def patched_SDClipModel__init__(self, max_length=77, freeze=True, layer="last", layer_idx=None,
                                textmodel_json_config=None, dtype=None, special_tokens=None,
                                layer_norm_hidden_state=True, **kwargs):
    torch.nn.Module.__init__(self)
    assert layer in self.LAYERS

    if special_tokens is None:
        special_tokens = {"start": 49406, "end": 49407, "pad": 49407}

    if textmodel_json_config is None:
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(comfy.sd1_clip.__file__)),
                                             "sd1_clip_config.json")

    config = CLIPTextConfig.from_json_file(textmodel_json_config)
    self.num_layers = config.num_hidden_layers

    with use_patched_ops(ops.manual_cast):
        with modeling_utils.no_init_weights():
            self.transformer = CLIPTextModel(config)

    if dtype is not None:
        self.transformer.to(dtype)

    self.transformer.text_model.embeddings.to(torch.float32)

    if freeze:
        self.freeze()

    self.max_length = max_length
    self.layer = layer
    self.layer_idx = None
    self.special_tokens = special_tokens
    self.text_projection = torch.nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
    self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
    self.enable_attention_masks = False

    self.layer_norm_hidden_state = layer_norm_hidden_state
    if layer == "hidden":
        assert layer_idx is not None
        assert abs(layer_idx) < self.num_layers
        self.clip_layer(layer_idx)
    self.layer_default = (self.layer, self.layer_idx)


def patched_SDClipModel_forward(self, tokens):
    backup_embeds = self.transformer.get_input_embeddings()
    device = backup_embeds.weight.device
    tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
    tokens = torch.LongTensor(tokens).to(device)

    attention_mask = None
    if self.enable_attention_masks:
        attention_mask = torch.zeros_like(tokens)
        max_token = self.transformer.get_input_embeddings().weight.shape[0] - 1
        for x in range(attention_mask.shape[0]):
            for y in range(attention_mask.shape[1]):
                attention_mask[x, y] = 1
                if tokens[x, y] == max_token:
                    break

    outputs = self.transformer(input_ids=tokens, attention_mask=attention_mask,
                               output_hidden_states=self.layer == "hidden")
    self.transformer.set_input_embeddings(backup_embeds)

    if self.layer == "last":
        z = outputs.last_hidden_state
    elif self.layer == "pooled":
        z = outputs.pooler_output[:, None, :]
    else:
        z = outputs.hidden_states[self.layer_idx]
        if self.layer_norm_hidden_state:
            z = self.transformer.text_model.final_layer_norm(z)

    if hasattr(outputs, "pooler_output"):
        pooled_output = outputs.pooler_output.float()
    else:
        pooled_output = None

    if self.text_projection is not None and pooled_output is not None:
        pooled_output = pooled_output.float().to(self.text_projection.device) @ self.text_projection.float()

    return z.float(), pooled_output


def patched_ClipVisionModel__init__(self, json_config):
    config = CLIPVisionConfig.from_json_file(json_config)

    self.load_device = comfy.model_management.text_encoder_device()
    self.offload_device = comfy.model_management.text_encoder_offload_device()

    if comfy.model_management.should_use_fp16(self.load_device, prioritize_performance=False):
        self.dtype = torch.float16
    else:
        self.dtype = torch.float32

    with use_patched_ops(ops.manual_cast):
        with modeling_utils.no_init_weights():
            self.model = CLIPVisionModelWithProjection(config)

    self.model.to(self.dtype)
    self.patcher = comfy.model_patcher.ModelPatcher(
        self.model,
        load_device=self.load_device,
        offload_device=self.offload_device
    )


def patched_ClipVisionModel_encode_image(self, image):
    comfy.model_management.load_model_gpu(self.patcher)
    pixel_values = comfy.clip_vision.clip_preprocess(image.to(self.load_device))
    outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

    for k in outputs:
        t = outputs[k]
        if t is not None:
            if k == 'hidden_states':
                outputs["penultimate_hidden_states"] = t[-2].to(comfy.model_management.intermediate_device())
                outputs["hidden_states"] = None
            else:
                outputs[k] = t.to(comfy.model_management.intermediate_device())

    return outputs


def patch_all_clip():
    comfy.sd1_clip.ClipTokenWeightEncoder.encode_token_weights = patched_encode_token_weights
    comfy.sd1_clip.SDClipModel.__init__ = patched_SDClipModel__init__
    comfy.sd1_clip.SDClipModel.forward = patched_SDClipModel_forward
    comfy.clip_vision.ClipVisionModel.__init__ = patched_ClipVisionModel__init__
    comfy.clip_vision.ClipVisionModel.encode_image = patched_ClipVisionModel_encode_image
    return
