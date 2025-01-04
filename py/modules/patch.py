import os
import torch
import time
import math
import ldm_patched.modules.model_base
import ldm_patched.ldm.modules.diffusionmodules.openaimodel
import comfy.model_management
import modules.anisotropic as anisotropic
import ldm_patched.ldm.modules.attention
import ldm_patched.k_diffusion.sampling
import ldm_patched.modules.sd1_clip
import modules.inpaint_worker as inpaint_worker
import ldm_patched.ldm.modules.diffusionmodules.openaimodel
import ldm_patched.ldm.modules.diffusionmodules.model
import ldm_patched.modules.sd
import ldm_patched.controlnet.cldm
import ldm_patched.modules.model_patcher
import ldm_patched.modules.samplers
import ldm_patched.modules.args_parser
import warnings
import safetensors.torch
import modules.constants as constants

from ldm_patched.modules.samplers import calc_cond_uncond_batch
from ldm_patched.k_diffusion.sampling import BatchedBrownianTree
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, apply_control
from modules.patch_precision import patch_all_precision
from modules.patch_clip import patch_all_clip


class PatchSettings:
    def __init__(self,
                 sharpness=2.0,
                 adm_scaler_end=0.3,
                 positive_adm_scale=1.5,
                 negative_adm_scale=0.8,
                 controlnet_softness=0.25,
                 adaptive_cfg=7.0):
        self.sharpness = sharpness
        self.adm_scaler_end = adm_scaler_end
        self.positive_adm_scale = positive_adm_scale
        self.negative_adm_scale = negative_adm_scale
        self.controlnet_softness = controlnet_softness
        self.adaptive_cfg = adaptive_cfg
        self.global_diffusion_progress = 0
        self.eps_record = None


patch_settings = {}


def weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype):
    dora_scale = comfy.model_management.cast_to_device(dora_scale, weight.device, intermediate_dtype)
    lora_diff *= alpha
    weight_calc = weight + lora_diff.type(weight.dtype)
    weight_norm = (
        weight_calc.transpose(0, 1)
        .reshape(weight_calc.shape[1], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
        .transpose(0, 1)
    )

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * (weight_calc)
    else:
        weight[:] = weight_calc
    return weight

def pad_tensor_to_shape(tensor: torch.Tensor, new_shape: list[int]) -> torch.Tensor:
    """
    Pad a tensor to a new shape with zeros.

    Args:
        tensor (torch.Tensor): The original tensor to be padded.
        new_shape (List[int]): The desired shape of the padded tensor.

    Returns:
        torch.Tensor: A new tensor padded with zeros to the specified shape.

    Note:
        If the new shape is smaller than the original tensor in any dimension,
        the original tensor will be truncated in that dimension.
    """
    if any([new_shape[i] < tensor.shape[i] for i in range(len(new_shape))]):
        raise ValueError("The new shape must be larger than the original tensor in all dimensions")

    if len(new_shape) != len(tensor.shape):
        raise ValueError("The new shape must have the same number of dimensions as the original tensor")

    # Create a new tensor filled with zeros
    padded_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    # Create slicing tuples for both tensors
    orig_slices = tuple(slice(0, dim) for dim in tensor.shape)
    new_slices = tuple(slice(0, dim) for dim in tensor.shape)

    # Copy the original tensor into the new tensor
    padded_tensor[new_slices] = tensor[orig_slices]

    return padded_tensor

def calculate_weight_patched(patches, weight, key, intermediate_dtype=torch.float32):
    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]
        if function is None:
            function = lambda a: a

        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (calculate_weight_patched(v[1:], v[0].clone(), key, intermediate_dtype=intermediate_dtype), )

        if len(v) == 1:
            patch_type = "diff"
        elif len(v) == 2:
            patch_type = v[0]
            v = v[1]
        if patch_type == "diff":
            diff: torch.Tensor = v[0]
            # An extra flag to pad the weight if the diff's shape is larger than the weight
            do_pad_weight = len(v) > 1 and v[1]['pad_weight']
            if do_pad_weight and diff.shape != weight.shape:
                print("Pad weight {} from {} to shape: {}".format(key, weight.shape, diff.shape))
                weight = pad_tensor_to_shape(weight, diff.shape)

            if strength != 0.0:
                if diff.shape != weight.shape:
                    print("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, diff.shape, weight.shape))
                else:
                    weight += function(strength * comfy.model_management.cast_to_device(diff, weight.device, weight.dtype))
        elif patch_type == "lora": #lora/locon
            mat1 = comfy.model_management.cast_to_device(v[0], weight.device, intermediate_dtype)
            mat2 = comfy.model_management.cast_to_device(v[1], weight.device, intermediate_dtype)
            dora_scale = v[4]
            if v[2] is not None:
                alpha = v[2] / mat2.shape[0]
            else:
                alpha = 1.0

            if v[3] is not None:
                #locon mid weights, hopefully the math is fine because I didn't properly test it
                mat3 = comfy.model_management.cast_to_device(v[3], weight.device, intermediate_dtype)
                final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
            try:
                lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                if dora_scale is not None:
                    weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype))
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
            except Exception as e:
                print("ERROR {} {} {}".format(patch_type, key, e))
        elif patch_type == "fooocus":
            w1 = comfy.model_management.cast_to_device(v[0], weight.device, torch.float32)
            w_min = comfy.model_management.cast_to_device(v[1], weight.device, torch.float32)
            w_max = comfy.model_management.cast_to_device(v[2], weight.device, torch.float32)
            w1 = (w1 / 255.0) * (w_max - w_min) + w_min
            if strength != 0.0:
                if w1.shape != weight.shape:
                    print("WARNING SHAPE MISMATCH {} FOOOCUS WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                else:
                    weight += strength * comfy.model_management.cast_to_device(w1, weight.device, weight.dtype)
        elif patch_type == "lokr":
            w1 = v[0]
            w2 = v[1]
            w1_a = v[3]
            w1_b = v[4]
            w2_a = v[5]
            w2_b = v[6]
            t2 = v[7]
            dora_scale = v[8]
            dim = None

            if w1 is None:
                dim = w1_b.shape[0]
                w1 = torch.mm(comfy.model_management.cast_to_device(w1_a, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w1_b, weight.device, intermediate_dtype))
            else:
                w1 = comfy.model_management.cast_to_device(w1, weight.device, intermediate_dtype)

            if w2 is None:
                dim = w2_b.shape[0]
                if t2 is None:
                    w2 = torch.mm(comfy.model_management.cast_to_device(w2_a, weight.device, intermediate_dtype),
                                    comfy.model_management.cast_to_device(w2_b, weight.device, intermediate_dtype))
                else:
                    w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                        comfy.model_management.cast_to_device(t2, weight.device, intermediate_dtype),
                                        comfy.model_management.cast_to_device(w2_b, weight.device, intermediate_dtype),
                                        comfy.model_management.cast_to_device(w2_a, weight.device, intermediate_dtype))
            else:
                w2 = comfy.model_management.cast_to_device(w2, weight.device, intermediate_dtype)

            if len(w2.shape) == 4:
                w1 = w1.unsqueeze(2).unsqueeze(2)
            if v[2] is not None and dim is not None:
                alpha = v[2] / dim
            else:
                alpha = 1.0

            try:
                lora_diff = torch.kron(w1, w2).reshape(weight.shape)
                if dora_scale is not None:
                    weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype))
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
            except Exception as e:
                print("ERROR {} {} {}".format(patch_type, key, e))
        elif patch_type == "loha":
            w1a = v[0]
            w1b = v[1]
            if v[2] is not None:
                alpha = v[2] / w1b.shape[0]
            else:
                alpha = 1.0

            w2a = v[3]
            w2b = v[4]
            dora_scale = v[7]
            if v[5] is not None: #cp decomposition
                t1 = v[5]
                t2 = v[6]
                m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                    comfy.model_management.cast_to_device(t1, weight.device, intermediate_dtype),
                                    comfy.model_management.cast_to_device(w1b, weight.device, intermediate_dtype),
                                    comfy.model_management.cast_to_device(w1a, weight.device, intermediate_dtype))

                m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                    comfy.model_management.cast_to_device(t2, weight.device, intermediate_dtype),
                                    comfy.model_management.cast_to_device(w2b, weight.device, intermediate_dtype),
                                    comfy.model_management.cast_to_device(w2a, weight.device, intermediate_dtype))
            else:
                m1 = torch.mm(comfy.model_management.cast_to_device(w1a, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w1b, weight.device, intermediate_dtype))
                m2 = torch.mm(comfy.model_management.cast_to_device(w2a, weight.device, intermediate_dtype),
                                comfy.model_management.cast_to_device(w2b, weight.device, intermediate_dtype))

            try:
                lora_diff = (m1 * m2).reshape(weight.shape)
                if dora_scale is not None:
                    weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype))
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
            except Exception as e:
                print("ERROR {} {} {}".format(patch_type, key, e))
        elif patch_type == "glora":
            dora_scale = v[5]

            old_glora = False
            if v[3].shape[1] == v[2].shape[0] == v[0].shape[0] == v[1].shape[1]:
                rank = v[0].shape[0]
                old_glora = True

            if v[3].shape[0] == v[2].shape[1] == v[0].shape[1] == v[1].shape[0]:
                if old_glora and v[1].shape[0] == weight.shape[0] and weight.shape[0] == weight.shape[1]:
                    pass
                else:
                    old_glora = False
                    rank = v[1].shape[0]

            a1 = comfy.model_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, intermediate_dtype)
            a2 = comfy.model_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, intermediate_dtype)
            b1 = comfy.model_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, intermediate_dtype)
            b2 = comfy.model_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, intermediate_dtype)

            if v[4] is not None:
                alpha = v[4] / rank
            else:
                alpha = 1.0

            try:
                if old_glora:
                    lora_diff = (torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1).to(dtype=intermediate_dtype), a2), a1)).reshape(weight.shape) #old lycoris glora
                else:
                    if weight.dim() > 2:
                        lora_diff = torch.einsum("o i ..., i j -> o j ...", torch.einsum("o i ..., i j -> o j ...", weight.to(dtype=intermediate_dtype), a1), a2).reshape(weight.shape)
                    else:
                        lora_diff = torch.mm(torch.mm(weight.to(dtype=intermediate_dtype), a1), a2).reshape(weight.shape)
                    lora_diff += torch.mm(b1, b2).reshape(weight.shape)

                if dora_scale is not None:
                    weight = function(weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype))
                else:
                    weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
            except Exception as e:
                print("ERROR {} {} {}".format(patch_type, key, e))
        else:
            print("patch type not recognized {} {}".format(patch_type, key))

        if old_weight is not None:
            weight = old_weight

    return weight

class BrownianTreeNoiseSamplerPatched:
    transform = None
    tree = None

    @staticmethod
    def global_init(x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        if comfy.model_management.directml_enabled:
            cpu = True

        t0, t1 = transform(torch.as_tensor(sigma_min)), transform(torch.as_tensor(sigma_max))

        BrownianTreeNoiseSamplerPatched.transform = transform
        BrownianTreeNoiseSamplerPatched.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __call__(sigma, sigma_next):
        transform = BrownianTreeNoiseSamplerPatched.transform
        tree = BrownianTreeNoiseSamplerPatched.tree

        t0, t1 = transform(torch.as_tensor(sigma)), transform(torch.as_tensor(sigma_next))
        return tree(t0, t1) / (t1 - t0).abs().sqrt()


def compute_cfg(uncond, cond, cfg_scale, t):
    pid = os.getpid()
    mimic_cfg = float(patch_settings[pid].adaptive_cfg)
    real_cfg = float(cfg_scale)

    real_eps = uncond + real_cfg * (cond - uncond)

    if cfg_scale > patch_settings[pid].adaptive_cfg:
        mimicked_eps = uncond + mimic_cfg * (cond - uncond)
        return real_eps * t + mimicked_eps * (1 - t)
    else:
        return real_eps


def patched_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
    pid = os.getpid()

    if math.isclose(cond_scale, 1.0) and not model_options.get("disable_cfg1_optimization", False):
        final_x0 = calc_cond_uncond_batch(model, cond, None, x, timestep, model_options)[0]

        if patch_settings[pid].eps_record is not None:
            patch_settings[pid].eps_record = ((x - final_x0) / timestep).cpu()

        return final_x0

    positive_x0, negative_x0 = calc_cond_uncond_batch(model, cond, uncond, x, timestep, model_options)

    positive_eps = x - positive_x0
    negative_eps = x - negative_x0

    alpha = 0.001 * patch_settings[pid].sharpness * patch_settings[pid].global_diffusion_progress

    positive_eps_degraded = anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=positive_x0)
    positive_eps_degraded_weighted = positive_eps_degraded * alpha + positive_eps * (1.0 - alpha)

    final_eps = compute_cfg(uncond=negative_eps, cond=positive_eps_degraded_weighted,
                            cfg_scale=cond_scale, t=patch_settings[pid].global_diffusion_progress)

    if patch_settings[pid].eps_record is not None:
        patch_settings[pid].eps_record = (final_eps / timestep).cpu()

    return x - final_eps


def round_to_64(x):
    h = float(x)
    h = h / 64.0
    h = round(h)
    h = int(h)
    h = h * 64
    return h


def sdxl_encode_adm_patched(self, **kwargs):
    clip_pooled = ldm_patched.modules.model_base.sdxl_pooled(kwargs, self.noise_augmentor)
    width = kwargs.get("width", 1024)
    height = kwargs.get("height", 1024)
    target_width = width
    target_height = height
    pid = os.getpid()

    if kwargs.get("prompt_type", "") == "negative":
        width = float(width) * patch_settings[pid].negative_adm_scale
        height = float(height) * patch_settings[pid].negative_adm_scale
    elif kwargs.get("prompt_type", "") == "positive":
        width = float(width) * patch_settings[pid].positive_adm_scale
        height = float(height) * patch_settings[pid].positive_adm_scale

    def embedder(number_list):
        h = self.embedder(torch.tensor(number_list, dtype=torch.float32))
        h = torch.flatten(h).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return h

    width, height = int(width), int(height)
    target_width, target_height = round_to_64(target_width), round_to_64(target_height)

    adm_emphasized = embedder([height, width, 0, 0, target_height, target_width])
    adm_consistent = embedder([target_height, target_width, 0, 0, target_height, target_width])

    clip_pooled = clip_pooled.to(adm_emphasized)
    final_adm = torch.cat((clip_pooled, adm_emphasized, clip_pooled, adm_consistent), dim=1)

    return final_adm


def patched_KSamplerX0Inpaint_forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, model_options={}, seed=None):
    if inpaint_worker.current_task is not None:
        latent_processor = self.inner_model.inner_model.process_latent_in
        inpaint_latent = latent_processor(inpaint_worker.current_task.latent).to(x)
        inpaint_mask = inpaint_worker.current_task.latent_mask.to(x)

        if getattr(self, 'energy_generator', None) is None:
            # avoid bad results by using different seeds.
            self.energy_generator = torch.Generator(device='cpu').manual_seed((seed + 1) % constants.MAX_SEED)

        energy_sigma = sigma.reshape([sigma.shape[0]] + [1] * (len(x.shape) - 1))
        current_energy = torch.randn(
            x.size(), dtype=x.dtype, generator=self.energy_generator, device="cpu").to(x) * energy_sigma
        x = x * inpaint_mask + (inpaint_latent + current_energy) * (1.0 - inpaint_mask)

        out = self.inner_model(x, sigma,
                               cond=cond,
                               uncond=uncond,
                               cond_scale=cond_scale,
                               model_options=model_options,
                               seed=seed)

        out = out * inpaint_mask + inpaint_latent * (1.0 - inpaint_mask)
    else:
        out = self.inner_model(x, sigma,
                               cond=cond,
                               uncond=uncond,
                               cond_scale=cond_scale,
                               model_options=model_options,
                               seed=seed)
    return out


def timed_adm(y, timesteps):
    if isinstance(y, torch.Tensor) and int(y.dim()) == 2 and int(y.shape[1]) == 5632:
        y_mask = (timesteps > 999.0 * (1.0 - float(patch_settings[os.getpid()].adm_scaler_end))).to(y)[..., None]
        y_with_adm = y[..., :2816].clone()
        y_without_adm = y[..., 2816:].clone()
        return y_with_adm * y_mask + y_without_adm * (1.0 - y_mask)
    return y


def patched_cldm_forward(self, x, hint, timesteps, context, y=None, **kwargs):
    t_emb = ldm_patched.ldm.modules.diffusionmodules.openaimodel.timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
    emb = self.time_embed(t_emb)
    pid = os.getpid()

    guided_hint = self.input_hint_block(hint, emb, context)

    y = timed_adm(y, timesteps)

    outs = []

    hs = []
    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x
    for module, zero_conv in zip(self.input_blocks, self.zero_convs):
        if guided_hint is not None:
            h = module(h, emb, context)
            h += guided_hint
            guided_hint = None
        else:
            h = module(h, emb, context)
        outs.append(zero_conv(h, emb, context))

    h = self.middle_block(h, emb, context)
    outs.append(self.middle_block_out(h, emb, context))

    if patch_settings[pid].controlnet_softness > 0:
        for i in range(10):
            k = 1.0 - float(i) / 9.0
            outs[i] = outs[i] * (1.0 - patch_settings[pid].controlnet_softness * k)

    return outs


def patched_unet_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    self.current_step = 1.0 - timesteps.to(x) / 999.0
    patch_settings[os.getpid()].global_diffusion_progress = float(self.current_step.detach().cpu().numpy().tolist()[0])

    y = timed_adm(y, timesteps)

    transformer_options["original_shape"] = list(x.shape)
    transformer_options["transformer_index"] = 0
    transformer_patches = transformer_options.get("patches", {})

    num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
    image_only_indicator = kwargs.get("image_only_indicator", self.default_image_only_indicator)
    time_context = kwargs.get("time_context", None)

    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = ldm_patched.ldm.modules.diffusionmodules.openaimodel.timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x
    for id, module in enumerate(self.input_blocks):
        transformer_options["block"] = ("input", id)
        h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = apply_control(h, control, 'input')
        if "input_block_patch" in transformer_patches:
            patch = transformer_patches["input_block_patch"]
            for p in patch:
                h = p(h, transformer_options)

        hs.append(h)
        if "input_block_patch_after_skip" in transformer_patches:
            patch = transformer_patches["input_block_patch_after_skip"]
            for p in patch:
                h = p(h, transformer_options)

    transformer_options["block"] = ("middle", 0)
    h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
    h = apply_control(h, control, 'middle')

    for id, module in enumerate(self.output_blocks):
        transformer_options["block"] = ("output", id)
        hsp = hs.pop()
        hsp = apply_control(hsp, control, 'output')

        if "output_block_patch" in transformer_patches:
            patch = transformer_patches["output_block_patch"]
            for p in patch:
                h, hsp = p(h, hsp, transformer_options)

        h = torch.cat([h, hsp], dim=1)
        del hsp
        if len(hs) > 0:
            output_shape = hs[-1].shape
        else:
            output_shape = None
        h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
    h = h.type(x.dtype)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)


def build_loaded(module, loader_name):
    original_loader_name = loader_name + '_origin'

    if not hasattr(module, original_loader_name):
        setattr(module, original_loader_name, getattr(module, loader_name))

    original_loader = getattr(module, original_loader_name)

    def loader(*args, **kwargs):
        result = None
        try:
            result = original_loader(*args, **kwargs)
        except Exception as e:
            result = None
            exp = str(e) + '\n'
            for path in list(args) + list(kwargs.values()):
                if isinstance(path, str):
                    if os.path.exists(path):
                        exp += f'File corrupted: {path} \n'
                        corrupted_backup_file = path + '.corrupted'
                        if os.path.exists(corrupted_backup_file):
                            os.remove(corrupted_backup_file)
                        os.replace(path, corrupted_backup_file)
                        if os.path.exists(path):
                            os.remove(path)
                        exp += f'Fooocus has tried to move the corrupted file to {corrupted_backup_file} \n'
                        exp += f'You may try again now and Fooocus will download models again. \n'
            raise ValueError(exp)
        return result

    setattr(module, loader_name, loader)
    return


def patch_all():
    if comfy.model_management.directml_enabled:
        comfy.model_management.lowvram_available = True
        comfy.model_management.OOM_EXCEPTION = Exception

    patch_all_precision()
    patch_all_clip()
    comfy.lora.calculate_weight = calculate_weight_patched
    # ldm_patched.modules.model_patcher.FooocusModelPatcher.calculate_weight = calculate_weight_patched
    ldm_patched.controlnet.cldm.ControlNet.forward = patched_cldm_forward
    ldm_patched.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = patched_unet_forward
    ldm_patched.modules.model_base.SDXL.encode_adm = sdxl_encode_adm_patched
    ldm_patched.modules.samplers.KSamplerX0Inpaint.forward = patched_KSamplerX0Inpaint_forward
    ldm_patched.k_diffusion.sampling.BrownianTreeNoiseSampler = BrownianTreeNoiseSamplerPatched
    ldm_patched.modules.samplers.sampling_function = patched_sampling_function

    warnings.filterwarnings(action='ignore', module='torchsde')

    build_loaded(safetensors.torch, 'load_file')
    build_loaded(torch, 'load')

    return
