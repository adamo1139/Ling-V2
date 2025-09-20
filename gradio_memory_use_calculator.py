#!/usr/bin/env python3
"""
Theoretical Memory Usage Calculator (Megatron-LM Style)

This tool computes theoretical memory footprints for model training/inference:
- Weight + Optimizer Memory (parameters sharded across parallelism + optimizer states)
- Activation Memory (forward/backward, with pipeline adjustments)
- Total Memory per GPU (in GB, BF16/FP16 weights, FP32 grads/activations)
- Supports dense and MoE models; approximations for EP/ETP sharding
- Configurations for 1x GPU (no parallelism) and 8x GPU (default DP=8, configurable TP/PP/EP/ETP)

Notes:
- Based on Megatron-LM's theoretical_memory_usage.py formulas
- Assumes sequence parallelism and selective recompute (for activation savings)
- MoE: Counts full experts, shards by TP/ETP and EP (local_experts = num_experts // EP)
- Excludes overheads (KV cache, comms, ~10-20% extra in practice)
- Precision: 2 bytes/param (BF16), 4 bytes/activation (FP32), 18 bytes/param total for AdamW
- For 8x: Per-GPU memory; assumes world_size = DP * TP * PP * EP * ETP >=8
"""

import gradio as gr
import math

NUM_BYTES_IN_GIGABYTE = 1024 * 1024 * 1024
NUM_BYTES_IN_MEGABYTE = 1024 * 1024

def format_number(num):
    """Format large numbers with appropriate units"""
    if num >= 1e12:
        return f"{num / 1e12:.2f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    else:
        return str(int(num))

def format_memory_gb(num_bytes):
    """Format bytes to GB"""
    return f"{num_bytes / NUM_BYTES_IN_GIGABYTE:.2f} GB"

def safe_int(val, default=0):
    """Safely convert to int, handling empty/None/invalid and non-strings"""
    if val is None:
        return default
    if isinstance(val, str):
        if val == '' or val.lower() == 'none':
            return default
    try:
        return int(val)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid integer value '{val}' (expected number, got non-numeric)")

def safe_float(val, default=0.0):
    """Safely convert to float, handling empty/None/invalid and non-strings"""
    if val is None:
        return default
    if isinstance(val, str):
        if val == '' or val.lower() == 'none':
            return default
    try:
        return float(val)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid float value '{val}' (expected number, got non-numeric)")

def calculate_memory_usage(
    # Base model parameters
    num_layers,
    hidden_size,
    vocab_size,
    num_attention_heads,
    n_kv_heads,
    ffn_hidden_size,
    moe_ffn_hidden_size,
    num_experts,
    moe_router_topk,
    shared_experts,
    moe_layer_freq,
    shared_expert_intermediate_size,
    untie_embeddings_and_output_weights,
    mtp_num_layers,
    gated_linear_multiplier,

    # Parallelism
    data_parallel_size,
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    expert_model_parallel_size,
    expert_tensor_model_parallel_size,

    # Training/Activation
    micro_batch_size,
    seq_length,
    num_microbatches,
    virtual_pipeline_model_parallel_size,
    sequence_parallel,
    recompute_granularity,

    # Optimizer/Memory
    use_distributed_optimizer,
    precision_bytes_param,  # e.g., 2 for BF16
    precision_bytes_activation  # e.g., 4 for FP32
):
    try:
        # Convert and validate inputs
        num_layers = safe_int(num_layers, 1)
        hidden_size = safe_int(hidden_size, 768)
        vocab_size = safe_int(vocab_size, 32000)  # Padded vocab
        num_attention_heads = safe_int(num_attention_heads, 6)
        n_kv_heads = safe_int(n_kv_heads, 2)
        ffn_hidden_size = safe_int(ffn_hidden_size, 128)
        num_experts = safe_int(num_experts, 1)
        moe_router_topk = safe_int(moe_router_topk, 2)
        shared_experts = safe_int(shared_experts, 0)
        moe_ffn_hidden_size = safe_int(moe_ffn_hidden_size, 128) if num_experts > 1 else 0
        shared_expert_intermediate_size = safe_int(shared_expert_intermediate_size, 0)
        untie_embeddings_and_output_weights = bool(untie_embeddings_and_output_weights)
        mtp_num_layers = safe_int(mtp_num_layers, 0)
        if mtp_num_layers == 0:
            mtp_num_layers = None
        gated_linear_multiplier = safe_float(gated_linear_multiplier, 1.5)

        data_parallel_size = safe_int(data_parallel_size, 1)
        tensor_model_parallel_size = safe_int(tensor_model_parallel_size, 1)
        pipeline_model_parallel_size = safe_int(pipeline_model_parallel_size, 1)
        expert_model_parallel_size = safe_int(expert_model_parallel_size, 1)
        expert_tensor_model_parallel_size = safe_int(expert_tensor_model_parallel_size, 1)

        micro_batch_size = safe_int(micro_batch_size, 2)
        seq_length = safe_int(seq_length, 32768)
        num_microbatches = safe_int(num_microbatches, None)
        if num_microbatches == 0:
            num_microbatches = None
        virtual_pipeline_model_parallel_size = safe_int(virtual_pipeline_model_parallel_size, None)
        if virtual_pipeline_model_parallel_size == 0:
            virtual_pipeline_model_parallel_size = None
        sequence_parallel = bool(sequence_parallel)
        recompute_granularity = str(recompute_granularity) if recompute_granularity else 'selective'

        use_distributed_optimizer = bool(use_distributed_optimizer)
        precision_bytes_param = safe_float(precision_bytes_param, 2.0)
        precision_bytes_activation = safe_float(precision_bytes_activation, 4.0)

        # Validations
        if num_layers <= 0 or hidden_size <= 0 or vocab_size <= 0:
            raise ValueError("Base params must be positive")
        if num_attention_heads <= 0 or n_kv_heads <= 0 or n_kv_heads > num_attention_heads:
            raise ValueError("Invalid attention heads")
        if num_experts < 1:
            num_experts = 1  # Dense
        if num_experts > 1 and moe_ffn_hidden_size <= 0:
            raise ValueError("moe_ffn_hidden_size must be positive for MoE")
        if num_experts > 1 and moe_router_topk > num_experts:
            raise ValueError("moe_router_topk must be <= num_experts")
        if shared_experts < 0 or shared_experts > num_experts:
            raise ValueError("shared_experts must be >=0 and <= num_experts")
        if num_experts > 1 and expert_model_parallel_size > 1 and num_experts % expert_model_parallel_size != 0:
            raise ValueError("num_experts must be multiple of EP")
        if data_parallel_size < 1 or tensor_model_parallel_size < 1 or pipeline_model_parallel_size < 1 or expert_model_parallel_size < 1:
            raise ValueError("Parallel sizes must be >=1")
        if expert_tensor_model_parallel_size < 1:
            expert_tensor_model_parallel_size = tensor_model_parallel_size  # Default ETP=TP
        if micro_batch_size <= 0 or seq_length <= 0:
            raise ValueError("Batch/seq must be positive")
        if not (1 <= precision_bytes_param <= 4) or not (1 <= precision_bytes_activation <= 4):
            raise ValueError("Precision bytes in [1,4]")
        moe_layer_freq = str(moe_layer_freq) if moe_layer_freq else 'none'
        if num_experts > 1 and moe_layer_freq == 'every':
            moe_layer_pattern = [1] * num_layers
        else:
            moe_layer_pattern = [0] * num_layers  # Dense

        num_dense_layers = num_layers - sum(moe_layer_pattern)
        num_moe_layers = sum(moe_layer_pattern)
        num_local_experts = num_experts // expert_model_parallel_size if num_experts > 1 else 1

        # Attention projection size (GQA)
        kv_channels = (hidden_size // num_attention_heads)  # per-head dim
        query_projection_size = kv_channels * num_attention_heads
        query_projection_to_hidden_size_ratio = query_projection_size / hidden_size
        gqa_factor = n_kv_heads / num_attention_heads

        # Self-attention params per layer (standard, no MLA)
        # Match Megatron: 2*H*H * ((1 + (num_query_groups/num_heads)) * (query_proj_size / H))
        self_attn_term = 2 * hidden_size * hidden_size * (
            (1 + gqa_factor) * query_projection_to_hidden_size_ratio
        )

        # Dense FFN per layer
        dense_ffn_term = 2 * hidden_size * (ffn_hidden_size * gated_linear_multiplier + 2)  # +2 layernorms
        num_parameters_in_transformer_layer_dense = dense_ffn_term + self_attn_term

        # MoE FFN per layer (full experts, will shard later)
        moe_ffn_term = 2 * hidden_size * (
            moe_ffn_hidden_size * num_experts * gated_linear_multiplier +
            shared_expert_intermediate_size * gated_linear_multiplier + 2
        )
        num_parameters_in_transformer_layer_moe = moe_ffn_term + self_attn_term

        # Total transformer params
        num_parameters_in_transformer_block = (
            num_parameters_in_transformer_layer_dense * num_dense_layers +
            num_parameters_in_transformer_layer_moe * num_moe_layers
        )
        embedding_size = hidden_size * vocab_size
        final_layernorm = 2 * hidden_size
        num_parameters_in_embedding_layers = embedding_size * (2 if untie_embeddings_and_output_weights else 1)
        num_parameters_in_transformer_block += final_layernorm

        # MTP block if enabled
        num_parameters_in_mtp_block = 0
        if mtp_num_layers:
            mtp_layer_is_moe = moe_layer_pattern[-1] if moe_layer_pattern else 0
            mtp_num_moe_layers = mtp_layer_is_moe * mtp_num_layers
            mtp_num_dense_layers = (1 - mtp_layer_is_moe) * mtp_num_layers
            num_parameters_in_mtp_block = (
                num_parameters_in_transformer_layer_dense * mtp_num_dense_layers +
                num_parameters_in_transformer_layer_moe * mtp_num_moe_layers
            )

        total_params = (
            num_parameters_in_transformer_block +
            num_parameters_in_mtp_block +
            num_parameters_in_embedding_layers
        )

        # Activated parameters per forward pass (heuristic)
        activated_embed = num_parameters_in_embedding_layers
        # Attention: full per layer
        attn_params_per_layer = self_attn_term + 2 * hidden_size  # + layernorms
        activated_attn = attn_params_per_layer * (num_layers + (mtp_num_layers or 0))
        # Router: per MoE layer, hidden * num_experts (logits)
        activated_router = num_moe_layers * hidden_size * num_experts
        # MoE FFN: sparse, (top_k + shared) * 3 * hidden * moe_ffn (SwiGLU)
        activated_moe_ffn = num_moe_layers * (moe_router_topk + shared_experts) * 3 * hidden_size * moe_ffn_hidden_size
        # Dense FFN: full
        activated_dense_ffn = num_dense_layers * 2 * hidden_size * ffn_hidden_size * gated_linear_multiplier
        # MTP if enabled
        activated_mtp = 0
        if mtp_num_layers:
            activated_mtp = attn_params_per_layer * mtp_num_layers + activated_moe_ffn / num_moe_layers * mtp_num_moe_layers if num_moe_layers > 0 else activated_dense_ffn / num_dense_layers * mtp_num_dense_layers if num_dense_layers > 0 else 0
        total_activated = (
            activated_embed + activated_attn + activated_router + activated_moe_ffn +
            activated_dense_ffn + activated_mtp + final_layernorm
        )
        activation_ratio = total_activated / total_params if total_params > 0 else 1.0

        # Sharding for most loaded shard (first/last PP stage)
        # Adjust MoE for local experts and ETP
        moe_sharding_factor = (num_local_experts / num_experts) / expert_tensor_model_parallel_size if num_experts > 1 else 1.0
        transformer_params_sharded = num_parameters_in_transformer_block * moe_sharding_factor
        num_parameters_on_most_loaded_model_shard = (
            (transformer_params_sharded / pipeline_model_parallel_size) +
            num_parameters_in_mtp_block * moe_sharding_factor +
            embedding_size
        ) / tensor_model_parallel_size
        if untie_embeddings_and_output_weights and pipeline_model_parallel_size == 1:
            num_parameters_on_most_loaded_model_shard += embedding_size / tensor_model_parallel_size

        # Bytes per param (weights + grads + optimizer)
        bytes_per_param = 18.0  # Default Megatron assumption
        if use_distributed_optimizer:
            bytes_per_param = 6 + (12 / data_parallel_size)

        weight_optimizer_bytes_per_gpu = num_parameters_on_most_loaded_model_shard * bytes_per_param

        # Activation memory (Megatron formula, first PP stage)
        if not sequence_parallel or recompute_granularity != 'selective':
            activation_bytes = 0  # Simplified report
        else:
            # Per transformer layer
            ffn_ratio = ffn_hidden_size / hidden_size if num_experts == 1 else moe_ffn_hidden_size / hidden_size
            activation_per_layer_bytes = (
                seq_length * micro_batch_size * hidden_size *
                (18 + 4 * ffn_ratio)  # From Megatron Table 2, fwd+bwd approx (already in bytes)
            )
            activation_bytes = activation_per_layer_bytes * num_layers

            # Embeddings + dropout
            activation_bytes += (
                8 * seq_length * micro_batch_size * pipeline_model_parallel_size +
                seq_length * micro_batch_size * hidden_size * pipeline_model_parallel_size
            )

            # PP schedule adjustments
            interleaved_penalty = 1.0
            if virtual_pipeline_model_parallel_size is not None and virtual_pipeline_model_parallel_size > 0:
                interleaved_penalty = 1 + ((pipeline_model_parallel_size - 1) / (pipeline_model_parallel_size * virtual_pipeline_model_parallel_size))
                activation_bytes *= interleaved_penalty
            elif pipeline_model_parallel_size > 1 and num_microbatches is not None and num_microbatches > 0:
                activation_bytes *= min(1.0, num_microbatches / pipeline_model_parallel_size)

            if pipeline_model_parallel_size == 1:
                # Output + loss
                activation_bytes += (
                    seq_length * micro_batch_size * hidden_size * 4 *
                    (1 + (vocab_size / hidden_size))
                )

            # Shard by TP
            activation_bytes /= tensor_model_parallel_size

        total_bytes_per_gpu = weight_optimizer_bytes_per_gpu + activation_bytes

        # World size estimate
        world_size = data_parallel_size * tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size * expert_tensor_model_parallel_size

        return {
            'total_params': total_params,
            'total_activated_params': total_activated,
            'activation_ratio': activation_ratio,
            'activated_embed': activated_embed,
            'activated_attn': activated_attn,
            'activated_router': activated_router,
            'activated_moe_ffn': activated_moe_ffn,
            'activated_dense_ffn': activated_dense_ffn,
            'embedding_params': num_parameters_in_embedding_layers,
            'transformer_params': num_parameters_in_transformer_block,
            'mtp_params': num_parameters_in_mtp_block,
            'moe_sharding_factor': moe_sharding_factor,
            'num_local_experts': num_local_experts,
            'params_per_gpu': num_parameters_on_most_loaded_model_shard,
            'bytes_per_param': bytes_per_param,
            'weight_optimizer_gb': weight_optimizer_bytes_per_gpu / NUM_BYTES_IN_GIGABYTE,
            'activation_gb': activation_bytes / NUM_BYTES_IN_GIGABYTE,
            'total_gb_per_gpu': total_bytes_per_gpu / NUM_BYTES_IN_GIGABYTE,
            'world_size': world_size,
            'num_dense_layers': num_dense_layers,
            'num_moe_layers': num_moe_layers,
            'interleaved_penalty': interleaved_penalty if 'interleaved_penalty' in locals() else 1.0
        }

    except Exception as e:
        return {'error': f"Invalid input: {str(e)}"}

def create_output_text(results):
    if 'error' in results:
        return f"❌ Error: {results['error']}"

    output = "🧠 Theoretical Memory Usage Analysis (Megatron-LM)\n\n"

    output += "📊 Model Configuration:\n"
    output += f"• Total Parameters: {format_number(results['total_params'])}\n"
    output += f"• Embedding Parameters: {format_number(results['embedding_params'])}\n"
    output += f"• Transformer Block: {format_number(results['transformer_params'])}\n"
    if results['mtp_params'] > 0:
        output += f"• MTP Block: {format_number(results['mtp_params'])}\n"
    if results['num_moe_layers'] > 0:
        global_experts = results['num_local_experts'] * results.get('world_size', 1) // results.get('expert_model_parallel_size', 1)
        output += f"• Dense Layers: {results['num_dense_layers']}, MoE Layers: {results['num_moe_layers']}\n"
        output += f"• Num Experts (Global/Local): {global_experts} / {results['num_local_experts']}\n"
        output += f"• MoE Sharding Factor: {results['moe_sharding_factor']:.3f}\n\n"
    else:
        output += "• Dense Model\n\n"

    output += "🎯 Activated Parameters (per Forward Pass):\n"
    output += f"• Total Activated: {format_number(results['total_activated_params'])} ({results['activation_ratio']:.2%} of total)\n"
    if results['num_moe_layers'] > 0:
        top_k = results.get('moe_router_topk', 2)
        shared = results.get('shared_experts', 0)
        output += f"• Activated Experts: {top_k + shared} / {global_experts} (sparsity ratio)\n"
        output += f"• Activated MoE FFN: {format_number(results['activated_moe_ffn'])}\n"
        output += f"• Activated Router: {format_number(results['activated_router'])}\n"
    else:
        output += f"• Dense: 100% activated (no sparsity)\n"
    output += f"• Activated Attention: {format_number(results['activated_attn'])}\n"
    output += f"• Activated Embeddings: {format_number(results['activated_embed'])}\n\n"

    output += "🔧 Parallelism Setup:\n"
    output += f"• Estimated World Size: {int(results['world_size'])} GPUs\n"
    output += f"• Params per GPU (sharded): {format_number(results['params_per_gpu'])}\n"
    output += f"• Bytes per Param (incl. opt): {results['bytes_per_param']:.1f}\n\n"

    output += "💾 Memory Breakdown per GPU:\n"
    output += f"• Weights + Optimizer: {format_memory_gb(results['weight_optimizer_gb'] * NUM_BYTES_IN_GIGABYTE)}\n"
    output += f"• Activations: {format_memory_gb(results['activation_gb'] * NUM_BYTES_IN_GIGABYTE)}\n"
    output += f"• Total Memory: {format_memory_gb(results['total_gb_per_gpu'] * NUM_BYTES_IN_GIGABYTE)}\n\n"

    if results['interleaved_penalty'] > 1:
        output += f"• PP Interleaved Penalty: {results['interleaved_penalty']:.2f}x\n"

    output += "⚠️ Assumptions & Notes:\n"
    output += "• Sequence Parallel + Selective Recompute enabled (saves activations)\n"
    output += "• MoE Activations approximated as dense; actual lower (~top-k factor)\n"
    output += "• Excludes overheads (KV cache for inference, comms, temporaries ~10-20% extra)\n"
    output += "• For 1x GPU: Set all parallel sizes=1; for 8x: e.g., DP=8 or TP=8\n"
    output += "• Precision: BF16 params (2B), FP32 activations (4B); adjust if needed\n"

    return output

def main():
    # Defaults from run_pretrain_1024_fineweb_apt4_6_mini_bf16.sh
    defaults_1x = {
        'num_layers': '12',
        'hidden_size': '768',
        'vocab_size': '32000',
        'num_attention_heads': '6',
        'n_kv_heads': '2',  # GQA with num-query-groups=2
        'ffn_hidden_size': '128',
        'moe_ffn_hidden_size': '128',
        'num_experts': '128',
        'moe_router_topk': '2',
        'shared_experts': '0',
        'moe_layer_freq': 'every',
        'shared_expert_intermediate_size': '0',
        'untie_embeddings_and_output_weights': True,
        'mtp_num_layers': '0',
        'gated_linear_multiplier': '1.5',  # SwiGLU
        'data_parallel_size': '1',
        'tensor_model_parallel_size': '1',
        'pipeline_model_parallel_size': '1',
        'expert_model_parallel_size': '1',
        'expert_tensor_model_parallel_size': '1',
        'micro_batch_size': '2',
        'seq_length': '32768',
        'num_microbatches': '16',  # global 32 / micro 2 / DP 1
        'virtual_pipeline_model_parallel_size': '0',
        'sequence_parallel': True,
        'recompute_granularity': 'selective',
        'use_distributed_optimizer': True,
        'precision_bytes_param': '2',  # bf16
        'precision_bytes_activation': '4'
    }

    defaults_8x = defaults_1x.copy()
    defaults_8x.update({
        'data_parallel_size': '8',
        'num_microbatches': '2'  # global 32 / micro 2 / DP 8
    })

    with gr.Blocks(title="Theoretical Memory Usage Calculator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Theoretical Memory Usage Calculator (Megatron-LM)")
        gr.Markdown("Estimates per-GPU memory for training dense/MoE models. Configure for 1x or 8x GPUs via parallelism settings.")

        config_choice = gr.Dropdown(
            choices=["1x GPU (No Parallelism)", "8x GPU (DP=8 Default)"],
            value="1x GPU (No Parallelism)",
            label="Configuration Preset"
        )

        def update_defaults(choice):
            if choice == "1x GPU (No Parallelism)":
                return [defaults_1x[key] for key in key_order]
            else:
                return [defaults_8x[key] for key in key_order]

        with gr.Row():
            # Base Model
            with gr.Column():
                gr.Markdown("### 🏗️ Base Model Parameters")
                num_layers = gr.Textbox(label="Num Layers", value=defaults_1x['num_layers'])
                hidden_size = gr.Textbox(label="Hidden Size", value=defaults_1x['hidden_size'])
                vocab_size = gr.Textbox(label="Vocab Size", value=defaults_1x['vocab_size'])
                num_attention_heads = gr.Textbox(label="Num Attention Heads", value=defaults_1x['num_attention_heads'])
                n_kv_heads = gr.Textbox(label="Num KV Heads (GQA)", value=defaults_1x['n_kv_heads'])
                ffn_hidden_size = gr.Textbox(label="FFN Hidden Size (Dense)", value=defaults_1x['ffn_hidden_size'])

            # MoE
            with gr.Column():
                gr.Markdown("### 🎯 MoE Parameters")
                num_experts = gr.Textbox(label="Num Experts", value=defaults_1x['num_experts'])
                moe_router_topk = gr.Textbox(label="MoE Router Top-K", value=defaults_1x['moe_router_topk'])
                shared_experts = gr.Textbox(label="Shared Experts (always activated)", value=defaults_1x['shared_experts'])
                moe_ffn_hidden_size = gr.Textbox(label="MoE FFN Hidden Size", value=defaults_1x['moe_ffn_hidden_size'])
                moe_layer_freq = gr.Textbox(label="MoE Layer Freq ('every' or 'none')", value=defaults_1x['moe_layer_freq'])
                shared_expert_intermediate_size = gr.Textbox(label="Shared Expert Size", value=defaults_1x['shared_expert_intermediate_size'])
                gated_linear_multiplier = gr.Textbox(label="Gated Linear Mult (1.5 SwiGLU)", value=defaults_1x['gated_linear_multiplier'])
                untie_embeddings_and_output_weights = gr.Checkbox(label="Untie Embed/LM Head", value=defaults_1x['untie_embeddings_and_output_weights'])
                mtp_num_layers = gr.Textbox(label="MTP Num Layers (optional)", value=defaults_1x['mtp_num_layers'], placeholder="0")

            # Parallelism
            with gr.Column():
                gr.Markdown("### 🔧 Parallelism (for 8x: e.g., DP=8)")
                data_parallel_size = gr.Textbox(label="Data Parallel Size (DP)", value=defaults_1x['data_parallel_size'])
                tensor_model_parallel_size = gr.Textbox(label="Tensor Parallel Size (TP)", value=defaults_1x['tensor_model_parallel_size'])
                pipeline_model_parallel_size = gr.Textbox(label="Pipeline Parallel Size (PP)", value=defaults_1x['pipeline_model_parallel_size'])
                expert_model_parallel_size = gr.Textbox(label="Expert Model Parallel (EP)", value=defaults_1x['expert_model_parallel_size'])
                expert_tensor_model_parallel_size = gr.Textbox(label="Expert Tensor Parallel (ETP)", value=defaults_1x['expert_tensor_model_parallel_size'])

            # Training/Activation
            with gr.Column():
                gr.Markdown("### 📈 Training Parameters")
                micro_batch_size = gr.Textbox(label="Micro Batch Size", value=defaults_1x['micro_batch_size'])
                seq_length = gr.Textbox(label="Sequence Length", value=defaults_1x['seq_length'])
                num_microbatches = gr.Textbox(label="Num Microbatches (PP)", value=defaults_1x['num_microbatches'], placeholder="Auto")
                virtual_pipeline_model_parallel_size = gr.Textbox(label="Virtual PP Size (Interleaved)", value=defaults_1x['virtual_pipeline_model_parallel_size'], placeholder="0")
                sequence_parallel = gr.Checkbox(label="Sequence Parallel", value=defaults_1x['sequence_parallel'])
                recompute_granularity = gr.Dropdown(choices=['selective', 'full'], value=defaults_1x['recompute_granularity'], label="Recompute Granularity")

                gr.Markdown("### ⚙️ Optimizer/Memory")
                use_distributed_optimizer = gr.Checkbox(label="Distributed Optimizer", value=defaults_1x['use_distributed_optimizer'])
                precision_bytes_param = gr.Textbox(label="Bytes/Param (e.g., 2 BF16)", value=defaults_1x['precision_bytes_param'])
                precision_bytes_activation = gr.Textbox(label="Bytes/Activation (e.g., 4 FP32)", value=defaults_1x['precision_bytes_activation'])

        output = gr.Textbox(label="📈 Memory Results", lines=20, show_copy_button=True)

        # Key order for defaults lookup (matches input_order)
        key_order = [
            'num_layers', 'hidden_size', 'vocab_size', 'num_attention_heads', 'n_kv_heads', 'ffn_hidden_size',
            'moe_ffn_hidden_size', 'num_experts', 'moe_router_topk', 'shared_experts', 'moe_layer_freq', 'shared_expert_intermediate_size',
            'untie_embeddings_and_output_weights', 'mtp_num_layers', 'gated_linear_multiplier',
            'data_parallel_size', 'tensor_model_parallel_size', 'pipeline_model_parallel_size',
            'expert_model_parallel_size', 'expert_tensor_model_parallel_size',
            'micro_batch_size', 'seq_length', 'num_microbatches', 'virtual_pipeline_model_parallel_size',
            'sequence_parallel', 'recompute_granularity', 'use_distributed_optimizer',
            'precision_bytes_param', 'precision_bytes_activation'
        ]

        input_order = [
            num_layers, hidden_size, vocab_size, num_attention_heads, n_kv_heads, ffn_hidden_size,
            moe_ffn_hidden_size, num_experts, moe_router_topk, shared_experts, moe_layer_freq, shared_expert_intermediate_size,
            untie_embeddings_and_output_weights, mtp_num_layers, gated_linear_multiplier,
            data_parallel_size, tensor_model_parallel_size, pipeline_model_parallel_size,
            expert_model_parallel_size, expert_tensor_model_parallel_size,
            micro_batch_size, seq_length, num_microbatches, virtual_pipeline_model_parallel_size,
            sequence_parallel, recompute_granularity, use_distributed_optimizer,
            precision_bytes_param, precision_bytes_activation
        ]

        # Live update
        def update_output(*args):
            return create_output_text(calculate_memory_usage(*args))

        for inp in input_order:
            inp.change(update_output, inputs=input_order, outputs=output)

        config_choice.change(
            update_defaults,
            inputs=config_choice,
            outputs=input_order
        ).then(update_output, inputs=input_order, outputs=output)

        # Initial load
        demo.load(update_output, inputs=input_order, outputs=output)

    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=7832, share=False)
