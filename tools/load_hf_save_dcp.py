"""
Load HF BailingMoeV2 checkpoint into a Megatron model and save as DCP.

Run with torchrun using the same parallelism as training:
    PYTHONPATH=Megatron-LM-core_v0.13.0:$PYTHONPATH \
    torchrun --nproc_per_node 8 tools/load_hf_save_dcp.py \
        --hf-path /path/to/hf/checkpoint \
        <all the same Megatron args as training>
"""

import argparse
import json
import os
import re
import sys
import torch
from safetensors import safe_open


def load_hf_state_dict(hf_path):
    """Load all tensors from HF safetensors checkpoint."""
    index_path = os.path.join(hf_path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index['weight_map'].values()))
    else:
        shard_files = ['model.safetensors']

    state_dict = {}
    for shard_file in shard_files:
        shard_path = os.path.join(hf_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def reverse_qkv_weight(weight, hidden_size, num_attention_heads, num_query_groups):
    """
    HF (bailing) format:  [Q_all, K_all, V_all]
    Megatron (mcore) format: [Q0, K0, V0, Q1, K1, V1, ...] interleaved per group
    """
    head_dim = hidden_size // num_attention_heads
    q_size = hidden_size
    kv_size = num_query_groups * head_dim

    all_q, all_k, all_v = weight.split([q_size, kv_size, kv_size], dim=0)

    heads_per_group = num_attention_heads // num_query_groups

    q_chunks = torch.chunk(all_q, num_query_groups, dim=0)
    k_chunks = torch.chunk(all_k, num_query_groups, dim=0)
    v_chunks = torch.chunk(all_v, num_query_groups, dim=0)

    groups = []
    for qi, ki, vi in zip(q_chunks, k_chunks, v_chunks):
        groups.append(torch.cat([qi, ki, vi], dim=0))
    return torch.cat(groups, dim=0)


def load_hf_into_model(model, hf_state_dict, args):
    """Copy HF weights into a Megatron model (single PP stage)."""
    from megatron.core import mpu

    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups if args.group_query_attention else args.num_attention_heads
    first_k_dense = args.moe_layer_freq.index(1) if 1 in args.moe_layer_freq else args.num_layers

    # Compute global layer offset for this PP stage
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = args.pipeline_model_parallel_size
    num_layers_per_stage = args.num_layers // pp_size
    layer_offset = pp_rank * num_layers_per_stage

    # Embeddings (only on first PP stage)
    if hasattr(model, 'embedding'):
        model.embedding.word_embeddings.weight.data.copy_(
            hf_state_dict['model.word_embeddings.weight'])

    # Output layer (only on last PP stage)
    if hasattr(model, 'output_layer'):
        model.output_layer.weight.data.copy_(
            hf_state_dict['lm_head.weight'])

    # Final layernorm (only on last PP stage)
    if hasattr(model, 'decoder') and getattr(model.decoder, 'final_layernorm', None) is not None:
        model.decoder.final_layernorm.weight.data.copy_(
            hf_state_dict['model.norm.weight'])

    # Transformer layers (local indices -> global HF layer indices)
    for local_idx, layer in enumerate(model.decoder.layers):
        layer_idx = layer_offset + local_idx
        prefix = f'model.layers.{layer_idx}'
        is_moe = layer_idx >= first_k_dense

        # Attention
        qkv_weight = hf_state_dict[f'{prefix}.attention.query_key_value.weight']
        layer.self_attention.linear_qkv.weight.data.copy_(
            reverse_qkv_weight(qkv_weight, hidden_size, num_attention_heads, num_query_groups))
        layer.self_attention.linear_proj.weight.data.copy_(
            hf_state_dict[f'{prefix}.attention.dense.weight'])

        # Input layernorm (fused into linear_qkv in TE)
        layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(
            hf_state_dict[f'{prefix}.input_layernorm.weight'])

        # QK layernorm
        if hasattr(layer.self_attention, 'q_layernorm'):
            layer.self_attention.q_layernorm.weight.data.copy_(
                hf_state_dict[f'{prefix}.attention.query_layernorm.weight'])
            layer.self_attention.k_layernorm.weight.data.copy_(
                hf_state_dict[f'{prefix}.attention.key_layernorm.weight'])

        # Post-attention layernorm
        # Dense layers with TE: fused into mlp.linear_fc1.layer_norm_weight
        # MoE layers: separate pre_mlp_layernorm module
        if is_moe:
            layer.pre_mlp_layernorm.weight.data.copy_(
                hf_state_dict[f'{prefix}.post_attention_layernorm.weight'])
        else:
            layer.mlp.linear_fc1.layer_norm_weight.data.copy_(
                hf_state_dict[f'{prefix}.post_attention_layernorm.weight'])

        if is_moe:
            # Router
            layer.mlp.router.weight.data.copy_(
                hf_state_dict[f'{prefix}.mlp.gate.weight'])
            if hasattr(layer.mlp.router, 'expert_bias'):
                layer.mlp.router.expert_bias.data.copy_(
                    hf_state_dict[f'{prefix}.mlp.gate.expert_bias'])

            # Shared experts
            if hasattr(layer.mlp, 'shared_experts'):
                shared = layer.mlp.shared_experts
                gate = hf_state_dict[f'{prefix}.mlp.shared_experts.gate_proj.weight']
                up = hf_state_dict[f'{prefix}.mlp.shared_experts.up_proj.weight']
                shared.linear_fc1.weight.data.copy_(torch.cat([gate, up], dim=0))
                shared.linear_fc2.weight.data.copy_(
                    hf_state_dict[f'{prefix}.mlp.shared_experts.down_proj.weight'])

            # Routed experts
            experts = layer.mlp.experts
            num_experts = args.num_experts
            # Handle SequentialMLP (local_experts list), GroupedMLP (weight1/weight2),
            # and TEGroupedMLP (linear_fc1/linear_fc2 with per-expert weight0..weightN)
            if hasattr(experts, 'local_experts'):
                for ei in range(len(experts.local_experts)):
                    gate = hf_state_dict[f'{prefix}.mlp.experts.{ei}.gate_proj.weight']
                    up = hf_state_dict[f'{prefix}.mlp.experts.{ei}.up_proj.weight']
                    experts.local_experts[ei].linear_fc1.weight.data.copy_(
                        torch.cat([gate, up], dim=0))
                    experts.local_experts[ei].linear_fc2.weight.data.copy_(
                        hf_state_dict[f'{prefix}.mlp.experts.{ei}.down_proj.weight'])
            elif hasattr(experts, 'weight1'):
                # GroupedMLP stores all expert weights as stacked tensors
                all_gate = []
                all_up = []
                all_down = []
                for ei in range(num_experts):
                    all_gate.append(hf_state_dict[f'{prefix}.mlp.experts.{ei}.gate_proj.weight'])
                    all_up.append(hf_state_dict[f'{prefix}.mlp.experts.{ei}.up_proj.weight'])
                    all_down.append(hf_state_dict[f'{prefix}.mlp.experts.{ei}.down_proj.weight'])
                experts.weight1.data.copy_(
                    torch.stack([torch.cat([g, u], dim=0) for g, u in zip(all_gate, all_up)], dim=0))
                experts.weight2.data.copy_(torch.stack(all_down, dim=0))
            elif hasattr(experts, 'linear_fc1'):
                # TEGroupedMLP: linear_fc1/linear_fc2 are TEGroupedLinear
                # with per-expert weights as weight0, weight1, ..., weightN
                fc1 = experts.linear_fc1
                fc2 = experts.linear_fc2
                for ei in range(num_experts):
                    gate = hf_state_dict[f'{prefix}.mlp.experts.{ei}.gate_proj.weight']
                    up = hf_state_dict[f'{prefix}.mlp.experts.{ei}.up_proj.weight']
                    getattr(fc1, f'weight{ei}').data.copy_(torch.cat([gate, up], dim=0))
                    getattr(fc2, f'weight{ei}').data.copy_(
                        hf_state_dict[f'{prefix}.mlp.experts.{ei}.down_proj.weight'])
        else:
            # Dense MLP
            gate = hf_state_dict[f'{prefix}.mlp.gate_proj.weight']
            up = hf_state_dict[f'{prefix}.mlp.up_proj.weight']
            layer.mlp.linear_fc1.weight.data.copy_(torch.cat([gate, up], dim=0))
            layer.mlp.linear_fc2.weight.data.copy_(
                hf_state_dict[f'{prefix}.mlp.down_proj.weight'])

        print(f'  Loaded layer {layer_idx} ({"MoE" if is_moe else "dense"})')


def main():
    # Insert our custom --hf-path arg before Megatron parses everything
    # We need to extract it before Megatron's parse_args eats sys.argv
    hf_path = None
    save_iteration = 8000
    filtered_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--hf-path':
            hf_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--save-iteration':
            save_iteration = int(sys.argv[i + 1])
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1
    sys.argv = filtered_argv

    assert hf_path is not None, "Must specify --hf-path"

    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import set_global_variables, get_args
    from megatron.training.initialize import initialize_megatron
    from megatron.training.checkpointing import save_checkpoint
    from megatron.core import mpu
    from pretrain_gpt import model_provider

    # Initialize Megatron (sets up distributed, builds tokenizer, etc.)
    # We pass --no-load-optim --no-load-rng --no-initialization to skip loading
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            'no_load_optim': True,
            'no_load_rng': True,
            'no_save_optim': True,
            'no_save_rng': True,
            'use_cpu_initialization': True,
        }
    )

    args = get_args()
    rank = torch.distributed.get_rank()

    # Build model for this rank's PP/TP/EP stage
    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()
    model = model_provider(pre_process, post_process).to(args.params_dtype)

    if rank == 0:
        print(f'Model built.')

    # Each rank loads HF weights sequentially to avoid OOM from all ranks
    # holding the full state dict simultaneously
    world_size = torch.distributed.get_world_size()
    for loading_rank in range(world_size):
        if rank == loading_rank:
            print(f'[rank {rank}] Loading HF weights from {hf_path}...')
            hf_state_dict = load_hf_state_dict(hf_path)
            load_hf_into_model(model, hf_state_dict, args)
            del hf_state_dict
            import gc; gc.collect()
            print(f'[rank {rank}] Done loading weights.')
        torch.distributed.barrier()

    if rank == 0:
        print(f'Saving DCP checkpoint at iteration {save_iteration}...')

    # Save using Megatron's checkpoint saver
    save_checkpoint(save_iteration, [model], None, None,
                    num_floating_point_operations_so_far=0)

    if rank == 0:
        print('Done! Checkpoint saved.')

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
