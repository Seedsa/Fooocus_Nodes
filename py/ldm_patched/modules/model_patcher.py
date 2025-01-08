"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations
from typing import Optional, Callable
import torch
import copy
import inspect
import logging
import uuid
import collections
import math

import comfy.utils
import comfy.float
import comfy.model_management
import comfy.lora
import comfy.hooks
from comfy.model_patcher import ModelPatcher, wipe_lowvram_weight
import comfy.patcher_extension
from comfy.patcher_extension import CallbacksMP, WrappersMP, PatcherInjection

try:
    from comfy.comfy_types import UnetWrapperFunction
except Exception:
    from comfy.types import UnetWrapperFunction

class LowVramPatch:
    def __init__(self, key, patches):
        self.key = key
        self.patches = patches
    def __call__(self, weight):
        return comfy.lora.calculate_weight(self.patches[self.key], weight, self.key, intermediate_dtype=weight.dtype)

class FooocusModelPatcher(ModelPatcher):
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            self.unpatch_hooks()
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            loading = self._load_list()

            load_completely = []
            loading.sort(reverse=True)
            for x in loading:
                n = x[1]
                m = x[2]
                params = x[3]
                module_mem = x[0]

                lowvram_weight = False

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if mem_counter + module_mem >= lowvram_model_memory:
                        lowvram_weight = True
                        lowvram_counter += 1
                        if hasattr(m, "prev_comfy_cast_weights"): #Already lowvramed
                            continue

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if lowvram_weight:
                    if weight_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(weight_key)
                        else:
                            m.weight_function = LowVramPatch(weight_key, self.patches)
                            patch_counter += 1
                    if bias_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(bias_key)
                        else:
                            m.bias_function = LowVramPatch(bias_key, self.patches)
                            patch_counter += 1

                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True
                else:
                    if hasattr(m, "comfy_cast_weights"):
                        if m.comfy_cast_weights:
                            wipe_lowvram_weight(m)

                    if full_load or mem_counter + module_mem < lowvram_model_memory:
                        mem_counter += module_mem
                        load_completely.append((module_mem, n, m, params))

            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]
                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights == True:
                        continue

                for param in params:
                    self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            if lowvram_counter > 0:
                logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
                self.model.model_lowvram = True
            else:
                logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
                self.model.model_lowvram = False
                if full_load:
                    self.model.to(device_to)
                    mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            # self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.current_weight_patches_uuid = self.patches_uuid

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            if self.model.model_lowvram:
                for m in self.model.modules():
                    wipe_lowvram_weight(m)

                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    comfy.utils.copy_to_param(self.model, k, bk.weight)
                else:
                    comfy.utils.set_attr_param(self.model, k, bk.weight)

            self.model.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                # self.model.device = device_to
            self.model.model_loaded_weight_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()