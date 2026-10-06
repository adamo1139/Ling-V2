#!/usr/bin/env python3
"""Check Megatron DCP checkpoints for zeroed q/k layernorm weights.

Run 5's saves came back with 35 of 128 q_layernorm entries set to exactly 0 in the
first layer of 7 of the 8 pipeline stages, while the in-memory model trained fine.
RMSNorm weights are never exactly 0 after training, so any exact zero is corruption.

  PYTHONPATH=Ling-V2/Megatron-LM-core_v0.13.0 \
    python3 Ling-V2/tools/check_dcp_qk_norm.py <iter_dir> [<iter_dir> ...]

Prints one line per checkpoint; exit status 1 if any checkpoint has zeros.
"""
import sys

import torch
import torch.distributed.checkpoint.default_planner as default_planner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys


# Same workaround as convert_dcp_to_safetensors_apt4_poziomka5.py: Megatron DCP
# metadata has no planner_data, which the stock key-filtered loader requires.
def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
    for key, value in metadata.state_dict_metadata.items():
        if key in self.keys:
            state_dict[key] = (torch.empty(value.size, dtype=value.properties.dtype)
                               if isinstance(value, TensorStorageMetadata) else value)
    super(default_planner._EmptyStateDictLoadPlanner, self).set_up_planner(
        state_dict, metadata, is_coordinator)


default_planner._EmptyStateDictLoadPlanner.set_up_planner = set_up_planner


def check(checkpoint, num_layers=16):
    keys = [f"decoder.layers.{layer}.self_attention.{norm}_layernorm.weight"
            for layer in range(num_layers) for norm in ("q", "k")]
    state = _load_state_dict_from_keys(set(keys), checkpoint_id=checkpoint)
    return {key: int((state[key] == 0).sum()) for key in keys}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bad = False
    for checkpoint in sys.argv[1:]:
        zeros = check(checkpoint)
        broken = {key.split(".")[2] + "." + key.split(".")[4][0]: count
                  for key, count in zeros.items() if count}
        if broken:
            bad = True
            print(f"CORRUPT {checkpoint}: zeros (layer.q/k: count) {broken}", flush=True)
        else:
            print(f"OK      {checkpoint}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
