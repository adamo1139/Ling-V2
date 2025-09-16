import torch

import sys
import os, json
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from safetensors.torch import save_file as safe_save_file

from pathlib import Path
import argparse

import json
import os
import re
import gc
from safetensors.torch import save_file as safe_save_file
from huggingface_hub import split_torch_state_dict_into_shards
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint import FileSystemReader
import logging
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def per_block_cast_to_fp8(x: torch.Tensor, scale_ue8m0):
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_scale = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4) / 448.0
    if scale_ue8m0:
        x_scale = x_scale.maximum(torch.tensor(1e-10, device=x.device)).log2().ceil().exp2()
    x_scaled = (x_view / x_scale).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), x_scale.view(x_view.size(0), x_view.size(2))


def set_up_planner(self, state_dict, metadata=None, is_coordinator=False) -> None:
    assert not state_dict
    assert metadata is not None

    # rebuild the state dict from the metadata
    for k, v in metadata.state_dict_metadata.items():
        if k not in self.keys:
            continue

        if isinstance(v, TensorStorageMetadata):
            v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
        state_dict[k] = v

    super(
        torch.distributed.checkpoint.default_planner._EmptyStateDictLoadPlanner, self
    ).set_up_planner(state_dict, metadata, is_coordinator)


torch.distributed.checkpoint.default_planner._EmptyStateDictLoadPlanner.set_up_planner = (
    set_up_planner
)


class KeyFilter:
    def __init__(self, fn):
        self.fn = fn

    def __contains__(self, item):
        return self.fn(item)


def key_to_keep(key):
    return "optimizer" not in key and "_extra_state" not in key and 'rng_state' not in key and 'chained_0' not in key


def transform_qkv_weight(args, weight):
    """
                    tp0                tp1
                               │
               ┌───────┬───┬───│───────┬───┬───┐
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
      mcore    │   Q   │ K │ V │   Q   │ K │ V │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               └───┬───┴─┬─┴─┬─│───┬───┴─┬─┴──┬┘
                   │     │   │ │   │     │    │
                   │     │   └─┴───┼─────┼┐   │
                   │     │┌────────┘     ││   │
                   │     ││              ││   │
                   │     └┼─────┐   ┌────┘│   │
                   │      │     │   │     │   │
               ┌───▼──────▼────┬▼───▼──┬──▼───▼┐
               │  Q0      Q1   │K0  K1 │  V0 V1│
               │               │       │       │
               │               │       │       │
    bailing    │       Q       │   K   │   V   │
               │               │       │       │
               │               │       │       │
               │               │       │       │
               └───────────────┴───────┴───────┘
    """
    hidden_size = args.hidden_size
    each_kv_size = hidden_size // args.num_attention_heads
    each_q_size = each_kv_size * args.num_attention_heads // args.num_query_groups

    qs = []
    ks = []
    vs = []
    for qkv in torch.chunk(weight, args.num_query_groups, dim=0):
        q, k, v = qkv.split([each_q_size, each_kv_size, each_kv_size], dim=0)
        qs.append(q)
        ks.append(k)
        vs.append(v)
    all_q = torch.cat(qs, dim=0)
    all_k = torch.cat(ks, dim=0)
    all_v = torch.cat(vs, dim=0)
    return torch.cat([all_q, all_k, all_v], dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="config for convert dcp checkpoint script")

    parser.add_argument(
        '--checkpoint-path', type=str, default=None, required=True, help='Path tp DCP checkpoint.'
    )
    parser.add_argument(
        '--target-path', type=str, default=None, required=True, help='Path to save checkpoint'
    )
    parser.add_argument(
        '--megatron-path',
        type=str,
        default=None,
        required=False,
        help='Base directory of Megatron repository',
    )
    parser.add_argument(
        '--force-bf16',
        action='store_true',
        default=False,
        help='Whether to force convert to bf16'
    )
    parser.add_argument(
        '--force-fp8',
        action='store_true',
        default=False,
        help='Whether to force convert to fp8'
    )
    parser.add_argument(
        '--mtp-path',
        type=str,
        default=None,
        required=False,
        # This option is suitable for debugging model.
        help='Where to export the mtp model as a separated model'
    )
    parser.add_argument(
        '--mtp-as-extra-layer',
        action='store_true',
        default=False,
        # This option is suitable for production model.
        help='Whether to export mtp model as an extra layer to --target-path.'
    )
    parser.add_argument(
        '--override-tokenizer-path',
        type=str,
        default=None,
        required=False,
        help='use this tokenizer instead of the one in the checkpoint'
    )
    return parser.parse_args()


fp8_quant_config = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [
    128,
    128
    ]
}

gen_config = lambda args, tokenizer: f"""{{
    "architectures": [
        "BailingMoeV2ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "auto_map": {{
        "AutoConfig": "configuration_bailing_moe_v2.BailingMoeV2Config",
        "AutoModel": "modeling_bailing_moe_v2.BailingMoeV2Model",
        "AutoModelForCausalLM": "modeling_bailing_moe_v2.BailingMoeV2ForCausalLM"
    }},
    "num_hidden_layers": {args.num_layers},
    "hidden_size": {args.hidden_size},
    "intermediate_size": {args.ffn_hidden_size},
    "eos_token_id": {tokenizer.encode(tokenizer.eos_token)[0]},
    "pad_token_id": null,
    "first_k_dense_replace": {args.first_k_dense if hasattr(args, 'first_k_dense') else args.first_k_dense_replace},
    "hidden_act": "silu",
    "max_position_embeddings": {args.seq_length},
    "model_type": "bailing_moe",
    "moe_intermediate_size": {args.moe_ffn_hidden_size},
    "norm_topk_prob": true,
    "num_experts_per_tok": {args.moe_router_topk},
    "num_attention_heads": {args.num_attention_heads},
    "num_experts": {args.num_experts},
    "num_shared_experts": null,
    "num_key_value_heads": {args.num_query_groups},
    "rope_theta": {args.rotary_base},
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.52.3",
    "use_bias": false,
    "use_rmsnorm": true,
    "rms_norm_eps": {args.norm_epsilon},
    "head_dim": {args.hidden_size // args.num_attention_heads},
    "use_cache": true,
    "use_qkv_bias": false,
    "embedding_dropout": 0.0,
    "output_dropout": 0.0,
    "vocab_size": {(args.vocab_size + args.make_vocab_size_divisible_by - 1) // args.make_vocab_size_divisible_by * args.make_vocab_size_divisible_by},
    "partial_rotary_factor": {args.rotary_percent},
    "router_dtype": "{args.moe_router_dtype}",
    "moe_router_enable_expert_bias": {str(args.moe_router_enable_expert_bias).lower()},
    "routed_scaling_factor": {1.0 if args.moe_router_topk_scaling_factor is None else args.moe_router_topk_scaling_factor},
    "n_group": {0 if args.moe_router_num_groups is None else args.moe_router_num_groups},
    "topk_group": {0 if args.moe_router_group_topk is None else args.moe_router_group_topk},
    "use_qk_norm": {str(args.qk_layernorm).lower()},
    "score_function": "{args.moe_router_score_function}"
}}
"""  # noqa: E731

def split_list(lst, n):
    m = len(lst)
    q, r = divmod(m, n)
    return [lst[i * q + min(i, r): (i + 1) * q + min(i + 1, r)] for i in range(n)]


class DCPKeyMapper:
    quant_device = 'cpu'
    layer_pattern = re.compile(r'decoder\.layers\.(\d+)\.(.+)')
    hf_layer_prefix = 'model.layers'
    key_map = {
        "embedding.word_embeddings.weight": "model.word_embeddings.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
        # map for layers, split layer idx
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.down_weight": "mlp.linear_down_proj.weight",
        "mlp.up_weight": "mlp.linear_up_proj.weight",
        "mlp.router.expert_bias": "mlp.gate.expert_bias",
        "mlp.router.weight": "mlp.gate.weight",
        "self_attention.q_layernorm.weight": "attention.query_layernorm.weight",
        "self_attention.k_layernorm.weight": "attention.key_layernorm.weight",
        "self_attention.linear_proj.weight": "attention.dense.weight",
        "mlp.linear_fc2.weight": "mlp.down_proj.weight",
        # for qkv split
        "self_attention.linear_q.weight": "attention.linear_q.weight",
        "self_attention.linear_k.weight": "attention.linear_k.weight",
        "self_attention.linear_v.weight": "attention.linear_v.weight",
        "self_attention.linear_q.layer_norm_weight": "input_layernorm.q.weight",
        "self_attention.linear_k.layer_norm_weight": "input_layernorm.k.weight",
        "self_attention.linear_v.layer_norm_weight": "input_layernorm.v.weight",
        # for mtp
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "eh_proj.weight": "eh_proj.weight",
        "final_layernorm.weight": "final_layernorm.weight",
    }
    split_key_map = {
        "mlp.linear_fc1.weight": "mlp",
    }
    qkv_key_map = {
        "self_attention.linear_qkv.weight": "attention.query_key_value.weight",
        "self_attention.linear_qkv.bias": "attention.query_key_value.bias",
    }
    expert_key_map = {
        "mlp.experts.experts.linear_fc1.weight": "mlp.experts.{expert_id}",
        "mlp.experts.experts.linear_fc2.weight": "mlp.experts.{expert_id}.down_proj.weight"
    }
    fp8_weight = {
        'mlp.up_proj.weight',
        'mlp.down_proj.weight',
        'mlp.gate_proj.weight',
        # mlp.experts.expert_{expert_id}.gate/up/down_proj.weight should be added by set_num_experts
        'attention.dense.weight',
        'attention.query_key_value.weight',
        # for split qkv
        'attention.linear_q.weight',
        'attention.linear_k.weight',
        'attention.linear_v.weight',
    }

    mtp_pattern = re.compile(r'mtp\.layers\.(\d+)\.(transformer_layer\.)?(.+)')

    @classmethod
    def set_num_experts(cls, num_experts):
        for i in range(num_experts):
            cls.fp8_weight.add(f'mlp.experts.{i}.gate_proj.weight')
            cls.fp8_weight.add(f'mlp.experts.{i}.up_proj.weight')
            cls.fp8_weight.add(f'mlp.experts.{i}.down_proj.weight')
    
    @classmethod
    def split_gate_up_proj(cls, key, value):
        if value is None:
            return [f'{key}.gate_proj.weight', f'{key}.up_proj.weight']
        else:
            gate_proj, up_proj = torch.chunk(value, 2, dim=0)
            return {
                f'{key}.gate_proj.weight': gate_proj,
                f'{key}.up_proj.weight': up_proj
            }

    @classmethod
    def parse_key_or_kv(cls, key, value):
        if value is None:
            return [key]
        else:
            return {key: value}

    @classmethod
    def transform_qkv(cls, key, value, args):
        if value is None:
            return [key]
        else:
            qkv = transform_qkv_weight(args, value)
            return {
                key: qkv
            }
    
    @classmethod
    def merge_list_or_dict(cls, arr):
        if isinstance(arr[0], list):
            return [x for a in arr for x in a]
        else:
            return {k: v for a in arr for k, v in a.items()}

    @classmethod
    def map_kv(self, key, value, args, mtp_as_extra_layer):
        m = re.match(self.layer_pattern, key)
        if m:
            # weight in layers
            layer_idx, post_key = m.groups()
            layer_prefix = f'{self.hf_layer_prefix}.{layer_idx}'
            if post_key in self.split_key_map:
                updated = self.split_gate_up_proj(self.split_key_map[post_key], value)
            elif post_key in self.qkv_key_map:
                updated = self.transform_qkv(f'{self.qkv_key_map[post_key]}', value, args)
            elif post_key in self.expert_key_map:
                def split_expert(ei):
                    expert_value = None if value is None else value[ei]
                    return self.split_gate_up_proj(f'{self.expert_key_map[post_key].format(expert_id=ei)}', expert_value)
                if 'down_proj' in self.expert_key_map[post_key]:
                    updated = self.merge_list_or_dict([self.parse_key_or_kv(f'{self.expert_key_map[post_key].format(expert_id=ei)}', None if value is None else value[ei]) for ei in range(args.num_experts)])
                else:
                    updated = self.merge_list_or_dict([split_expert(ei) for ei in range(args.num_experts)])
            else:
                updated = self.parse_key_or_kv(f'{self.key_map[post_key]}', value)
            if value is None:
                return [f'{layer_prefix}.{x}' for k in updated for x in self.quant_if_needed(k, None, args)]
            else:
                return {f'{layer_prefix}.{new_k}': new_v for k, v in updated.items() for new_k, new_v in self.quant_if_needed(k, v, args).items()}
        elif mtp_as_extra_layer and (m := re.match(self.mtp_pattern, key)):
            return self.map_kv_mtp(key, value, args, mtp_as_extra_layer=True)

        # weight out of layers
        return self.parse_key_or_kv(self.key_map[key], value)
   
    @classmethod
    def quant_if_needed(cls, key, value, args):
        if not args.fp8:
            if value is None:
                return [key]
            else:
                return {key: value}
        if value is None:
            if key in cls.fp8_weight:
                return [key, f'{key}_scale_inv']
            else:
                return [key]
        # try use gpu to quant for speed
        if key in cls.fp8_weight:
            scale_ue8m0 = False
            if key == 'attention.dense.weight':
                scale_ue8m0 = True
            if 'mlp.experts' in key and 'down_proj' not in key:
                scale_ue8m0 = True
            qw, scale = per_block_cast_to_fp8(value.to(cls.quant_device), scale_ue8m0)
            return {
                key: qw.to('cpu'),
                f'{key}_scale_inv': scale.to('cpu')
            }
        else:
            return {key: value}

    @classmethod
    def map_kv_mtp(cls, key, value, args, mtp_as_extra_layer=False):
        m = re.match(cls.mtp_pattern, key)
        if m:
            layer_idx, is_transformer, subkey = m.groups()
            if mtp_as_extra_layer:
                assert layer_idx == "0", f"For mtp_as_extra_layer, only 1-layer mtp is supported，current {key=} current {layer_idx=}"
                layer_idx = args.num_layers
            if subkey in cls.split_key_map:
                updated = cls.split_gate_up_proj(cls.split_key_map[subkey], value)
            elif subkey in cls.qkv_key_map:
                updated = cls.transform_qkv(cls.qkv_key_map[subkey], value, args)
            elif subkey in cls.expert_key_map:
                def split_expert(ei):
                    expert_value = None if value is None else value[ei]
                    return cls.split_gate_up_proj(cls.expert_key_map[subkey].format(expert_id=ei), expert_value)
                if 'down_proj' in cls.expert_key_map[subkey]:
                    updated = cls.merge_list_or_dict([cls.parse_key_or_kv(f'{cls.expert_key_map[subkey].format(expert_id=ei)}', None if value is None else value[ei]) for ei in range(args.num_experts)])
                else:
                    updated = cls.merge_list_or_dict([split_expert(ei) for ei in range(args.num_experts)])
            else:
                updated = cls.parse_key_or_kv(cls.key_map[subkey], value)
            if value is None:
                for k in updated:
                    if k in cls.fp8_weight:
                        print(f'{cls.hf_layer_prefix}.{k}')
                return [f'{cls.hf_layer_prefix}.{layer_idx}.{x}' for k in updated for x in cls.quant_if_needed(k, None, args)]
            else:
                return {f'{cls.hf_layer_prefix}.{layer_idx}.{new_k}': new_v for k, v in updated.items() for new_k, new_v in cls.quant_if_needed(k, v, args).items()}
        else:
            raise Exception(f'Unsupported key {key}')


class SafetensorPart:
    def __init__(self, idx, total, keys, nbytes, converter):
        self.part_filename = f'model-{idx + 1:05d}-of-{total:05d}.safetensors'
        self.ckpt = converter.ckpt
        self.keys = keys
        self.nbytes = nbytes
        self.dst = converter.dst
        self.args = converter.args
        self.mtp_as_extra_layer = converter.mtp_as_extra_layer
        self.idx = idx

    def rewrite(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            DCPKeyMapper.quant_device = f'cuda:{self.idx % torch.cuda.device_count()}'
        else:
            DCPKeyMapper.quant_device = 'cpu'
        DCPKeyMapper.set_num_experts(self.args.num_experts)
        logging.info(f'convert for {self.part_filename} with {self.nbytes} bytes')
        state_dict = _load_state_dict_from_keys(self.keys, checkpoint_id=self.ckpt)
        logging.info('load done')
        new_state_dict = {new_k: new_v for k, v in state_dict.items() for new_k, new_v in DCPKeyMapper.map_kv(k, v, self.args, mtp_as_extra_layer=self.mtp_as_extra_layer).items()}
        logging.info('convert done')
        safe_save_file(new_state_dict, os.path.join(self.dst, self.part_filename), metadata={"format": "pt"})
        logging.info('save done')
        index_map = {k: self.part_filename for k in new_state_dict}
        del state_dict
        del new_state_dict
        gc.collect()
        return index_map
    
    def gen_index_map(self, mtp_as_extra_layer):
        new_keys = [new_k for k in self.keys for new_k in DCPKeyMapper.map_kv(k, None, self.args, mtp_as_extra_layer=mtp_as_extra_layer)]
        return {k: self.part_filename for k in new_keys}


def rewrite_part(part):
    return part.rewrite()


class DCPConverter:
    def __init__(self, ckpt, dst, force_bf16=False, force_fp8=False, mtp_as_extra_layer=False, override_tokenizer_path=None):
        self.ckpt = ckpt
        self.dst = dst
        if not os.path.exists(dst):
            os.mkdir(dst)
        self.args = torch.load(os.path.join(ckpt, 'common.pt'), weights_only=False)['args']
        if force_bf16:
            self.args.fp8 = None
        if force_fp8:
            self.args.fp8 = 'e4m3'
        if override_tokenizer_path:
            self.args.spm_tokenizer_path = override_tokenizer_path
        if not hasattr(self.args, 'first_k_dense_replace'):
            self.args.first_k_dense_replace = self.args.moe_layer_freq.index(1)
        self.mtp_as_extra_layer = mtp_as_extra_layer
        self.reader = FileSystemReader(self.ckpt)

    def key_to_keep(self, key):
        if self.mtp_as_extra_layer:
            return "optimizer" not in key and "_extra_state" not in key and 'rng_state' not in key and 'chained_0' not in key and 'rerun_state_machine_state' not in key
        else:
            return "optimizer" not in key and "_extra_state" not in key and 'rng_state' not in key and 'chained_0' not in key and 'rerun_state_machine_state' not in key and not key.startswith('mtp.')

    def key_to_keep_mtp(self, key):
        return "optimizer" not in key and "_extra_state" not in key and 'rng_state' not in key and 'chained_0' not in key and 'rerun_state_machine_state' not in key and key.startswith('mtp.')

    def part_generator(self, threshold=10*2**30):
        meta = [(k, v.properties.dtype.itemsize * v.size.numel()) for k, v in self.reader.read_metadata().state_dict_metadata.items() if self.key_to_keep(k)]
        start = 0
        total_bytes = 0
        for i, (k, size_in_bytes) in enumerate(meta):
            if total_bytes + size_in_bytes > threshold:
                yield {x[0] for x in meta[start:i]}, total_bytes
                total_bytes = size_in_bytes
                start = i
            else:
                total_bytes += size_in_bytes
        if start < len(meta):
            yield {k for k, _ in meta[start:]}, sum([v for _, v in meta[start:]])
        
    def rewrite_to_safetensors(self, world_size, rank):
        logging.info(f'start rewrite safetensors {rank}/{world_size}...')
        parts = [(i, k, v) for i, (k, v) in enumerate(self.part_generator())]
        total_bytes = sum([nbytes for _, _, nbytes in parts])
        meta = {
            "metadata": {
                "total_size": total_bytes
            },
            "weight_map": {}
        }
        nparts = len(parts)
        parts = [SafetensorPart(i, nparts, k, v, self) for i, k, v in parts]

        segs = split_list(parts, world_size)
        if len(segs[rank]) > 0:
            multiprocessing.set_start_method('spawn')
            with multiprocessing.Pool(64) as pool:
                pool.map(rewrite_part, segs[rank])
        
        if rank == 0:
            meta['weight_map'] = {k: v for part in parts for k, v in part.gen_index_map(self.mtp_as_extra_layer).items()}
            with open(os.path.join(self.dst, 'model.safetensors.index.json'), 'w') as f:
                json.dump(meta, f, indent=2)

    def write_config(self):
        with open(os.path.join(self.dst, 'config.json'), 'w') as f:
            config = gen_config(self.args, self.tokenizer)
            config = json.loads(config)
            if self.args.fp8:
                config['quantization_config'] = fp8_quant_config
            if self.mtp_as_extra_layer:
                config['num_nextn_predict_layers'] = 1
            json.dump(config, f, indent=4)

    def write_mtp_config(self, dst):
        with open(os.path.join(dst, 'config.json'), 'w') as f:
            config = gen_config(self.args)
            config = json.loads(config)
            config["num_hidden_layers"] = 1
            config["first_k_dense_replace"] = 0
            config['num_nextn_predict_layers'] = 1
            if self.args.fp8:
                config['quantization_config'] = fp8_quant_config
            json.dump(config, f, indent=4)

    def copy_tokenizer_config(self, dst=None):
        dst = dst if dst is not None else self.dst
        if not os.path.exists(self.args.spm_tokenizer_path):
            logging.warning(f'tokenizer path {self.args.spm_tokenizer_path} not exists!')
            self.tokenizer = None
            return
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.spm_tokenizer_path, trust_remote_code=True)
        shutil.copy2(os.path.join(self.args.spm_tokenizer_path, 'special_tokens_map.json'), os.path.join(dst, 'special_tokens_map.json'))
        shutil.copy2(os.path.join(self.args.spm_tokenizer_path, 'tokenizer.json'), os.path.join(dst, 'tokenizer.json'))
        shutil.copy2(os.path.join(self.args.spm_tokenizer_path, 'generation_config.json'), os.path.join(dst, 'generation_config.json'))
        with open(os.path.join(self.args.spm_tokenizer_path, 'tokenizer_config.json'), 'r') as inf:
            with open(os.path.join(dst, 'tokenizer_config.json'), 'w') as outf:
                data = json.load(inf)
                data['tokenizer_class'] = 'PreTrainedTokenizerFast'
                json.dump(data, outf, indent=2)

    def rewrite_mtp(self, mtp_path):
        if not os.path.exists(mtp_path):
            os.mkdir(mtp_path)
        part_filename = 'mtp_layers_parameters.safetensors'
        keys = {k for k, _ in self.reader.read_metadata().state_dict_metadata.items() if self.key_to_keep_mtp(k)}
        state_dict = _load_state_dict_from_keys(keys, checkpoint_id=self.ckpt)
        new_state_dict = {new_k: new_v for k, v in state_dict.items() for new_k, new_v in DCPKeyMapper.map_kv_mtp(k, v, self.args).items()}
        safe_save_file(new_state_dict, os.path.join(mtp_path, part_filename), metadata={"format": "pt"})
        metadata = {
            "weight_map": {k: part_filename for k in new_state_dict}
        }
        with open(os.path.join(mtp_path, 'model.safetensors.index.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

    def copy_model_config_module(self, dst=None):
        dst = dst if dst is not None else self.dst
        
        # Get the script directory and construct path to bailing_arch_files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ling_v2_dir = os.path.dirname(script_dir)  # Go up from tools/ to Ling-V2/
        bailing_arch_dir = os.path.join(ling_v2_dir, 'bailing_arch_files')
        
        # Copy configuration file
        config_src = os.path.join(bailing_arch_dir, 'configuration_bailing_moe_v2.py')
        if os.path.exists(config_src):
            shutil.copy2(config_src, dst)
            print(f'Copied configuration_bailing_moe_v2.py from {config_src}')
        else:
            print(f'Warning: configuration_bailing_moe_v2.py not found at {config_src}')
            
        # Copy modeling file
        modeling_src = os.path.join(bailing_arch_dir, 'modeling_bailing_moe_v2.py')
        if os.path.exists(modeling_src):
            shutil.copy2(modeling_src, dst)
            print(f'Copied modeling_bailing_moe_v2.py from {modeling_src}')
        else:
            print(f'Warning: modeling_bailing_moe_v2.py not found at {modeling_src}')



if __name__ == '__main__':
    args = parse_args()
    # currently use force_bf16
    args.force_bf16 = True
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    if args.mtp_as_extra_layer:
        assert args.mtp_path is None, "--mtp-as-extra-layer and --mtp-path cannot be both set."

    # convert_ckpt(args.checkpoint_path, args.target_path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    torch.set_num_threads(1)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    converter = DCPConverter(args.checkpoint_path, args.target_path, 
        force_bf16=args.force_bf16, 
        force_fp8=args.force_fp8, 
        mtp_as_extra_layer=args.mtp_as_extra_layer, 
        override_tokenizer_path=args.override_tokenizer_path)
    DCPKeyMapper.set_num_experts(converter.args.num_experts)
    converter.rewrite_to_safetensors(world_size, rank)
    if rank == 0:
        converter.copy_tokenizer_config()
        converter.write_config()
        converter.copy_model_config_module()
        if args.mtp_path:
            converter.rewrite_mtp(args.mtp_path)
            converter.write_mtp_config(args.mtp_path)
            converter.copy_tokenizer_config(args.mtp_path)
            converter.copy_model_config_module(args.mtp_path)
