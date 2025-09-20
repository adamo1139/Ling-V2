#!/usr/bin/env python3
"""
Model Chooser: Compute & Memory (Ling + Megatron)

Combined tool that:
- Computes MoE model params and training compute (Ling Scaling Laws)
- Reports compute-optimal M/D, EL proxy, optimal LR/Batch from total compute C
- Shows attention vs FFN FLOPs mix per Ling guidance
- Estimates theoretical per‑GPU memory (Megatron-LM theoretical_memory_usage)

Notes:
- Uses the same arithmetic as gradio_optim_model_calculator.py (Ling) and
  gradio_memory_use_calculator.py (Megatron) with the latest fixes
- Shared inputs where reasonable (e.g., sequence length, KV heads)
- Two outputs in tabs: Compute & Scaling, and Memory Usage
"""

import gradio as gr
import math

NUM_BYTES_IN_GIGABYTE = 1024 * 1024 * 1024

def format_number(num):
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

def format_flops(num):
    if num == 0:
        return "0"
    elif num >= 1e18:
        return f"{num / 1e18:.2f}E18"
    elif num >= 1e15:
        return f"{num / 1e15:.2f}E15"
    elif num >= 1e12:
        return f"{num / 1e12:.2f}E12"
    elif num >= 1e9:
        return f"{num / 1e9:.2f}E9"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}E6"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}E3"
    else:
        return f"{num:.2f}"

def format_memory_gb(num_bytes):
    return f"{num_bytes / NUM_BYTES_IN_GIGABYTE:.2f} GB"

def safe_int(val, default=0):
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
    if val is None:
        return default
    if isinstance(val, str):
        if val == '' or val.lower() == 'none':
            return default
    try:
        return float(val)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid float value '{val}' (expected number, got non-numeric)")


# ---------- Compute & Scaling (Ling) ----------
def calculate_model_sizes(
    num_layers,
    hidden_size,
    vocab_size,
    num_attention_heads,
    num_experts,
    moe_ffn_hidden_size,
    moe_router_topk,
    moe_router_topk_scaling_factor,
    moe_router_num_groups,
    moe_router_group_topk,
    shared_experts,
    expert_granularity,
    batch_size,
    sequence_length,
    n_kv,
    num_training_iterations,
    train_flops_multiplier,
    include_embedding_in_flops,
    mfu
):
    try:
        # Convert inputs
        num_layers = safe_int(num_layers, 1)
        hidden_size = safe_int(hidden_size, 1)
        vocab_size = safe_int(vocab_size, 1)
        num_attention_heads = safe_int(num_attention_heads, 1)
        num_experts = safe_int(num_experts, 1)
        moe_ffn_hidden_size = safe_int(moe_ffn_hidden_size, 1)
        moe_router_topk = safe_int(moe_router_topk, 1)
        moe_router_topk_scaling_factor = safe_float(moe_router_topk_scaling_factor, 1.0)
        moe_router_num_groups = safe_int(moe_router_num_groups, 0)
        moe_router_group_topk = safe_int(moe_router_group_topk, 0)
        shared_experts = safe_int(shared_experts, 0)
        expert_granularity = safe_float(expert_granularity, 0.0)
        batch_size = safe_int(batch_size, 1)
        sequence_length = safe_int(sequence_length, 1)
        n_kv = safe_int(n_kv, 1)
        num_training_iterations = safe_int(num_training_iterations, 1)
        train_flops_multiplier = safe_float(train_flops_multiplier, 3.0)
        include_embedding_in_flops = bool(include_embedding_in_flops)
        mfu = safe_float(mfu, 0.4)

        # Validations
        if num_layers <= 0 or hidden_size <= 0:
            raise ValueError("num_layers and hidden_size must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if n_kv <= 0 or n_kv > num_attention_heads:
            raise ValueError("n_kv must be in [1, num_attention_heads]")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if moe_router_topk <= 0 or moe_router_topk > num_experts:
            raise ValueError("moe_router_topk must be in [1, num_experts]")
        if moe_router_num_groups < 0 or moe_router_group_topk < 0:
            raise ValueError("router group values must be non-negative")
        if moe_router_num_groups and (moe_router_group_topk > moe_router_num_groups):
            raise ValueError("moe_router_group_topk must be <= moe_router_num_groups")
        if shared_experts < 0:
            raise ValueError("shared_experts must be >= 0")
        if not (0 < mfu <= 1.0):
            raise ValueError("MFU must be in (0, 1]")

        # Expert hidden dim from granularity if provided; else use moe_ffn_hidden_size
        if expert_granularity and expert_granularity > 0:
            d_expert = int(max(1, round(2 * hidden_size / expert_granularity)))
            d_expert_source = "granularity"
        else:
            d_expert = int(moe_ffn_hidden_size)
            d_expert_source = "ffn_hidden_size"
        if d_expert <= 0:
            raise ValueError("Derived expert hidden size must be positive")
        d_expert = max(64, int(round(d_expert / 8)) * 8)  # kernel-friendly

        # Grouped-query attention factor g = n_kv / n_heads
        g = n_kv / num_attention_heads

        # Effective activated experts per token (routable + shared)
        ea = moe_router_topk
        es = shared_experts
        e_total = num_experts + es
        activation_ratio = (ea + es) / e_total if e_total > 0 else 0.0

        # Embeddings + LM head params (untied): V*H + H*V; include final RMSNorm (H)
        embed_params = vocab_size * hidden_size * 2
        final_norm_params = hidden_size

        # Attention params per layer
        head_dim = hidden_size // num_attention_heads
        attn_qkv_weight = hidden_size * ((num_attention_heads + 2 * n_kv) * head_dim)
        attn_out_weight = (num_attention_heads * head_dim) * hidden_size
        use_qk_norm = True
        attn_qk_norm = (2 * head_dim) if use_qk_norm else 0
        layer_rmsnorms = 2 * hidden_size
        attn_params_per_layer = attn_qkv_weight + attn_out_weight + attn_qk_norm + layer_rmsnorms
        attn_params = num_layers * attn_params_per_layer

        # MoE expert params per layer (SwiGLU: 3 * H * d_expert per expert)
        moe_expert_params_per_layer = num_experts * (3 * hidden_size * d_expert)
        moe_expert_params = num_layers * moe_expert_params_per_layer

        # Router params per layer: H -> E logits (+ bias)
        router_expert_bias = True
        router_params_per_layer = hidden_size * num_experts + (num_experts if router_expert_bias else 0)
        router_params = num_layers * router_params_per_layer

        # Total model params
        total_params = embed_params + final_norm_params + attn_params + moe_expert_params + router_params

        # Activated params per token (heuristic)
        activated_attn = attn_params
        activated_router_per_layer = router_params_per_layer
        activated_ffn_per_layer = (ea + es) * (3 * hidden_size * d_expert)
        activated_ffn = num_layers * activated_ffn_per_layer
        activated_router = num_layers * activated_router_per_layer
        total_activated = activated_attn + activated_router + activated_ffn

        activated_ffn_ratio = (activated_ffn_per_layer * num_layers) / moe_expert_params if moe_expert_params > 0 else 0.0

        # Memory (rough BF16)
        param_memory_gb = total_params * 2 / (1024**3)
        activated_memory_gb = total_activated * 2 / (1024**3)

        # MACs/token/layer
        macs_attn_proj = attn_qkv_weight + attn_out_weight
        macs_attn_core = 2 * sequence_length * hidden_size
        macs_attn_token_layer = macs_attn_proj + macs_attn_core
        macs_router_token_layer = hidden_size * num_experts
        macs_moe_token_layer = 3 * (ea + es) * hidden_size * d_expert

        # FLOPs per token per layer, and across layers (non-embedding M)
        flops_token_layer = 2 * (macs_attn_token_layer + macs_router_token_layer + macs_moe_token_layer)
        M_token = num_layers * flops_token_layer

        # Component-wise totals across all layers
        attn_flops_total = num_layers * 2 * macs_attn_token_layer
        ffn_flops_total = num_layers * 2 * macs_moe_token_layer
        router_flops_total = num_layers * 2 * macs_router_token_layer
        denom_ffn_attn = (attn_flops_total + ffn_flops_total)
        attn_ffn_ratio = (attn_flops_total / denom_ffn_attn) if denom_ffn_attn > 0 else 0.0
        if 0.30 <= attn_ffn_ratio <= 0.40:
            attn_ffn_status = "within 30–40% stable range"
        elif 0.20 <= attn_ffn_ratio <= 0.50:
            attn_ffn_status = "within broader 20–50% tolerance"
        else:
            attn_ffn_status = "outside recommended range"

        # Optional embedding compute/token (logits); not in Ling M
        flops_embed_token = 2 * hidden_size * vocab_size if include_embedding_in_flops else 0

        # Per-step forward FLOPs and training FLOPs multiplier
        c_fwd = batch_size * sequence_length * (M_token + flops_embed_token)
        c_train = train_flops_multiplier * c_fwd

        # Total tokens and training compute
        total_tokens = batch_size * sequence_length * num_training_iterations
        total_training_cost = M_token * total_tokens  # Ling: C = M*D (non-embedding)
        total_train_flops_with_bwd = c_train * num_training_iterations

        # GPU hours estimates at sustained BF16 TFLOPS
        def flops_to_gpu_hours(flops, tflops, mfu):
            return flops / (tflops * 1e12 * mfu) / 3600.0
        hours_3090_ti = flops_to_gpu_hours(total_train_flops_with_bwd, 40.0, mfu)
        hours_h100_sxm5 = flops_to_gpu_hours(total_train_flops_with_bwd, 900.0, mfu)

        # Compute-optimal allocations (Table 1)
        Mopt_dense = 0.0655 * (total_training_cost ** 0.5422)
        Dopt_dense = 15.2582 * (total_training_cost ** 0.4578)
        Mopt_moe = 0.1915 * (total_training_cost ** 0.5095)
        Dopt_moe = 5.2232 * (total_training_cost ** 0.4905)

        # Joint EL scaling law (Eq. 13)
        a_coef = 1.23
        d_coef = -7.61e-2
        gamma_coef = 1.67e-2
        beta_coef = -1.17e-1
        A_start = 1.63e-2
        A_max = 5.28e16
        import math as _math
        A = activation_ratio if activation_ratio > 0 else A_start
        A_hat = max(min(A, A_max), A_start)
        G = (2 * hidden_size) / d_expert
        log10C = _math.log10(total_training_cost) if total_training_cost > 0 else 0.0
        log10G = _math.log10(max(G, 1e-12))
        alpha = a_coef + d_coef * log10C
        exponent_on_A = alpha + gamma_coef * (log10G ** 2) + beta_coef * log10G
        el_joint = (A_hat ** exponent_on_A) if A_hat > 0 else 0.0

        # Optimal hyperparams from C
        if total_training_cost > 0:
            lr_opt = 1.1576 * (total_training_cost ** (-0.1529))
            batch_opt = 0.0694 * (total_training_cost ** 0.3644)
        else:
            lr_opt = 0.0
            batch_opt = 0.0

        return {
            'total_params': total_params,
            'embed_params': embed_params,
            'attn_params': attn_params,
            'moe_expert_params': moe_expert_params,
            'router_params': router_params,
            'activated_attn': activated_attn,
            'activated_router': activated_router,
            'activated_ffn': activated_ffn,
            'total_activated': total_activated,
            'activated_ffn_ratio': activated_ffn_ratio,
            'activation_ratio': activation_ratio,
            'd_expert': d_expert,
            'd_expert_source': d_expert_source,
            'param_memory_gb': param_memory_gb,
            'activated_memory_gb': activated_memory_gb,
            'macs_attn_token_layer': macs_attn_token_layer,
            'macs_router_token_layer': macs_router_token_layer,
            'macs_moe_token_layer': macs_moe_token_layer,
            'flops_token_layer': flops_token_layer,
            'M_token': M_token,
            'flops_embed_token': flops_embed_token,
            'c_fwd': c_fwd,
            'c_train': c_train,
            'total_tokens': total_tokens,
            'total_training_cost': total_training_cost,
            'total_train_flops_with_bwd': total_train_flops_with_bwd,
            'attn_flops_total': attn_flops_total,
            'ffn_flops_total': ffn_flops_total,
            'router_flops_total': router_flops_total,
            'attn_ffn_ratio': attn_ffn_ratio,
            'attn_ffn_status': attn_ffn_status,
            'hours_3090_ti': hours_3090_ti,
            'hours_h100_sxm5': hours_h100_sxm5,
            'Mopt_dense': Mopt_dense,
            'Dopt_dense': Dopt_dense,
            'Mopt_moe': Mopt_moe,
            'Dopt_moe': Dopt_moe,
            'el_joint': el_joint,
            'el_alpha': alpha,
            'el_exponent_on_A': exponent_on_A,
            'el_A_used': A_hat,
            'el_G_used': G,
            'mfu': mfu,
            'lr_opt': lr_opt,
            'batch_opt': batch_opt
        }
    except (ValueError, ZeroDivisionError) as e:
        return {'error': f"Invalid input: {str(e)}"}


def create_compute_output_text(results):
    if 'error' in results:
        return f"❌ Error: {results['error']}"

    output = "🔬 MoE Compute & Size Analysis (Ling)\n\n"
    output += "📊 Parameter Breakdown:\n"
    output += f"• Total Parameters: {format_number(results['total_params'])}\n"
    output += f"• Embedding Parameters: {format_number(results['embed_params'])}\n"
    output += f"• Attention Parameters: {format_number(results['attn_params'])}\n"
    output += f"• MoE Expert Parameters: {format_number(results['moe_expert_params'])}\n"
    output += f"• Router Parameters: {format_number(results['router_params'])}\n\n"

    output += "🎯 Activation & Expert Config:\n"
    output += f"• Expert Hidden Size d_expert: {results['d_expert']} (from {results['d_expert_source']})\n"
    output += f"• Activation Ratio A: {results['activation_ratio']*100:.2f}%\n"
    output += f"• Activated Attention (weights used): {format_number(results['activated_attn'])}\n"
    output += f"• Activated Router (per-token): {format_number(results['activated_router'])}\n"
    output += f"• Activated Experts (per-token): {format_number(results['activated_ffn'])}\n"
    output += f"• Total Activated (per-token): {format_number(results['total_activated'])}\n"
    output += f"• FFN Activation Ratio (params used vs total experts): {results['activated_ffn_ratio']:.3f}\n\n"

    output += "💾 Memory Estimates (BF16):\n"
    output += f"• Total Parameters: {results['param_memory_gb']:.2f} GB\n"
    output += f"• Activated Parameters: {results['activated_memory_gb']:.2f} GB\n\n"

    output += "⚡ FLOPs (Ling non-embedding M):\n"
    output += f"• MACs/token/layer — Attn: {format_flops(results['macs_attn_token_layer'])}\n"
    output += f"• MACs/token/layer — Router: {format_flops(results['macs_router_token_layer'])}\n"
    output += f"• MACs/token/layer — Experts: {format_flops(results['macs_moe_token_layer'])}\n"
    output += f"• FLOPs/token/layer (non-emb): {format_flops(results['flops_token_layer'])}\n"
    output += f"• M = FLOPs/token (all layers, non-emb): {format_flops(results['M_token'])}\n"
    if results['flops_embed_token']:
        output += f"• Embedding FLOPs/token: {format_flops(results['flops_embed_token'])}\n"
    output += f"• Forward FLOPs/step: {format_flops(results['c_fwd'])}\n"
    output += f"• Train FLOPs/step (x multiplier): {format_flops(results['c_train'])}\n"
    output += f"• Total Tokens D = B·L·steps: {format_number(results['total_tokens'])}\n"
    output += f"• Total Training Compute C = M·D: {format_flops(results['total_training_cost'])}\n\n"

    # Attention vs FFN
    attn_share = results['attn_ffn_ratio'] * 100
    ffn_share = 100 - attn_share
    output += "⚖️ Attention vs FFN Compute Mix:\n"
    output += f"• Attention FLOPs share (vs FFN): {attn_share:.1f}%\n"
    output += f"• FFN FLOPs share: {ffn_share:.1f}%\n"
    output += f"• Attention FLOPs (all layers): {format_flops(results['attn_flops_total'])}\n"
    output += f"• FFN FLOPs (all layers): {format_flops(results['ffn_flops_total'])}\n"
    output += f"• Status: {results['attn_ffn_status']} (Ling suggests 30–40% stable; 20–50% ok)\n\n"

    # Compute-optimal
    output += "🧭 Compute-Optimal (Table 1):\n"
    output += f"• Dense Mopt: {format_number(results['Mopt_dense'])} FLOPs/token, Dopt: {format_number(results['Dopt_dense'])} tokens\n"
    output += f"• MoE   Mopt: {format_number(results['Mopt_moe'])} FLOPs/token, Dopt: {format_number(results['Dopt_moe'])} tokens\n"
    output += "• Note: Mopt is non-embedding FLOPs/token (Ling).\n"

    # EL (Eq. 13)
    el_joint = results.get('el_joint', None)
    if el_joint is not None:
        output += "\n📈 Efficiency Leverage (Eq. 13, Ling):\n"
        output += f"• EL(A,G,C) = {el_joint:.2f}x\n"
        output += f"• Using A={results.get('el_A_used'):.4f}, G={results.get('el_G_used'):.2f}, alpha={results.get('el_alpha'):.3f}\n"

    # Optimal hyperparams from C
    output += "\n🛠 Optimal Hyperparameters (from C):\n"
    output += f"• Learning Rate (Nopt): {results.get('lr_opt', 0.0):.6f}\n"
    output += f"• Batch Size (Bopt): {results.get('batch_opt', 0.0):.2f}\n"

    # GPU hours
    hours_3090 = results.get('hours_3090_ti', 0.0)
    hours_h100 = results.get('hours_h100_sxm5', 0.0)
    total_train_flops = results.get('total_train_flops_with_bwd', 0.0)
    output += "\n⏱ GPU Time Estimates (idealized):\n"
    output += f"• Total train FLOPs (fwd+bwd): {format_flops(total_train_flops)}\n"
    output += f"• 3090 Ti hours (40 TFLOPS BF16, single GPU): {hours_3090:.1f} h\n"
    output += f"• H100 SXM5 80GB hours (900 TFLOPS BF16, single GPU): {hours_h100:.1f} h\n"
    output += f"• MFU used: {results.get('mfu', 0.4)*100:.0f}%\n"
    output += "• Notes: single-GPU idealized math; multi-GPU scales ~1/N ignoring comms.\n"

    return output


# ---------- Memory (Megatron) ----------
def calculate_memory_usage(
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
    data_parallel_size,
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    expert_model_parallel_size,
    expert_tensor_model_parallel_size,
    micro_batch_size,
    seq_length,
    num_microbatches,
    virtual_pipeline_model_parallel_size,
    sequence_parallel,
    recompute_granularity,
    use_distributed_optimizer,
    precision_bytes_param,
    precision_bytes_activation
):
    try:
        # Convert and validate inputs
        num_layers = safe_int(num_layers, 1)
        hidden_size = safe_int(hidden_size, 768)
        vocab_size = safe_int(vocab_size, 32000)
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
            moe_layer_pattern = [0] * num_layers

        num_dense_layers = num_layers - sum(moe_layer_pattern)
        num_moe_layers = sum(moe_layer_pattern)
        num_local_experts = num_experts // expert_model_parallel_size if num_experts > 1 else 1

        # Attention projection size (GQA)
        kv_channels = (hidden_size // num_attention_heads)
        query_projection_size = kv_channels * num_attention_heads
        query_projection_to_hidden_size_ratio = query_projection_size / hidden_size
        gqa_factor = n_kv_heads / num_attention_heads

        # Self-attention params per layer (Megatron form)
        self_attn_term = 2 * hidden_size * hidden_size * (
            (1 + gqa_factor) * query_projection_to_hidden_size_ratio
        )

        # Dense FFN per layer
        dense_ffn_term = 2 * hidden_size * (ffn_hidden_size * gated_linear_multiplier + 2)
        num_parameters_in_transformer_layer_dense = dense_ffn_term + self_attn_term

        # MoE FFN per layer (full experts, will shard later)
        moe_ffn_term = 2 * hidden_size * (
            moe_ffn_hidden_size * num_experts * gated_linear_multiplier +
            shared_expert_intermediate_size * gated_linear_multiplier + 2
        )
        num_parameters_in_transformer_layer_moe = moe_ffn_term + self_attn_term

        # Totals
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
        attn_params_per_layer = self_attn_term + 2 * hidden_size
        activated_attn = attn_params_per_layer * (num_layers + (mtp_num_layers or 0))
        activated_router = num_moe_layers * hidden_size * num_experts
        activated_moe_ffn = num_moe_layers * (moe_router_topk + shared_experts) * 3 * hidden_size * moe_ffn_hidden_size
        activated_dense_ffn = num_dense_layers * 2 * hidden_size * ffn_hidden_size * gated_linear_multiplier
        activated_mtp = 0
        if mtp_num_layers:
            activated_mtp = attn_params_per_layer * mtp_num_layers + activated_moe_ffn / num_moe_layers * mtp_num_moe_layers if num_moe_layers > 0 else activated_dense_ffn / num_dense_layers * mtp_num_dense_layers if num_dense_layers > 0 else 0
        total_activated = (
            activated_embed + activated_attn + activated_router + activated_moe_ffn +
            activated_dense_ffn + activated_mtp + final_layernorm
        )
        activation_ratio = total_activated / total_params if total_params > 0 else 1.0

        # Sharding for most loaded shard
        moe_sharding_factor = (num_local_experts / num_experts) / expert_tensor_model_parallel_size if num_experts > 1 else 1.0
        transformer_params_sharded = num_parameters_in_transformer_block * moe_sharding_factor
        num_parameters_on_most_loaded_model_shard = (
            (transformer_params_sharded / pipeline_model_parallel_size) +
            num_parameters_in_mtp_block * moe_sharding_factor +
            embedding_size
        ) / tensor_model_parallel_size
        if untie_embeddings_and_output_weights and pipeline_model_parallel_size == 1:
            num_parameters_on_most_loaded_model_shard += embedding_size / tensor_model_parallel_size

        # Bytes per param
        bytes_per_param = 18.0
        if use_distributed_optimizer:
            bytes_per_param = 6 + (12 / data_parallel_size)
        weight_optimizer_bytes_per_gpu = num_parameters_on_most_loaded_model_shard * bytes_per_param

        # Activation memory (first PP stage)
        if not sequence_parallel or recompute_granularity != 'selective':
            activation_bytes = 0
        else:
            ffn_ratio = ffn_hidden_size / hidden_size if num_experts == 1 else moe_ffn_hidden_size / hidden_size
            activation_per_layer_bytes = (
                seq_length * micro_batch_size * hidden_size *
                (18 + 4 * ffn_ratio)
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
                activation_bytes += (
                    seq_length * micro_batch_size * hidden_size * 4 *
                    (1 + (vocab_size / hidden_size))
                )

            activation_bytes /= tensor_model_parallel_size

        total_bytes_per_gpu = weight_optimizer_bytes_per_gpu + activation_bytes

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
            'interleaved_penalty': interleaved_penalty if 'interleaved_penalty' in locals() else 1.0,
            'expert_model_parallel_size': expert_model_parallel_size
        }
    except Exception as e:
        return {'error': f"Invalid input: {str(e)}"}


def create_memory_output_text(results):
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
    # Single source of truth for defaults (from examples/pretrain/run_pretrain_1024_fineweb_apt4_6_mini_bf16.sh)
    defaults = {
        'num_layers': '12',
        'hidden_size': '768',
        'vocab_size': '32000',
        'num_attention_heads': '6',
        'n_kv_heads': '2',
        'ffn_hidden_size': '128',
        'moe_ffn_hidden_size': '128',
        'num_experts': '128',
        'moe_router_topk': '2',
        'moe_router_topk_scaling_factor': '2.5',
        'moe_router_num_groups': '4',
        'moe_router_group_topk': '2',
        'shared_experts': '0',
        'moe_layer_freq': 'every',
        'shared_expert_intermediate_size': '0',
        'untie_embeddings_and_output_weights': True,
        'mtp_num_layers': '0',
        'gated_linear_multiplier': '1.5',
        # Compute defaults
        'batch_size': '32',  # use global batch size for C = M*D
        'sequence_length': '32768',
        'num_training_iterations': '5000',
        'train_flops_multiplier': '3.0',
        'include_embedding_in_flops': False,
        'mfu': '0.40',
        # Parallelism/memory defaults (1x)
        'data_parallel_size': '1',
        'tensor_model_parallel_size': '1',
        'pipeline_model_parallel_size': '1',
        'expert_model_parallel_size': '1',
        'expert_tensor_model_parallel_size': '1',
        'micro_batch_size': '2',
        'num_microbatches': '16',  # 32 / (2*DP)
        'virtual_pipeline_model_parallel_size': '0',
        'sequence_parallel': True,
        'recompute_granularity': 'selective',
        'use_distributed_optimizer': True,
        'precision_bytes_param': '2',
        'precision_bytes_activation': '4'
    }

    # Preset for 8x: keep global batch 32, so reduce microbatches
    defaults_1x = defaults.copy()
    defaults_8x = defaults.copy()
    defaults_8x.update({
        'data_parallel_size': '8',
        'num_microbatches': '2'
    })

    with gr.Blocks(title="Model Chooser: Compute & Memory", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Model Chooser: Compute & Memory (Ling + Megatron)")
        gr.Markdown("Unified tool to inspect compute scaling and per‑GPU memory. Change any field to update both views.")

        config_choice = gr.Dropdown(
            choices=["1x GPU (No Parallelism)", "8x GPU (DP=8 Default)"],
            value="1x GPU (No Parallelism)",
            label="Configuration Preset"
        )

        with gr.Row():
            # Base Model
            with gr.Column():
                gr.Markdown("### 🏗️ Base Model Parameters")
                num_layers = gr.Textbox(label="Num Layers", value=defaults['num_layers'])
                hidden_size = gr.Textbox(label="Hidden Size", value=defaults['hidden_size'])
                vocab_size = gr.Textbox(label="Vocab Size", value=defaults['vocab_size'])
                num_attention_heads = gr.Textbox(label="Num Attention Heads", value=defaults['num_attention_heads'])
                n_kv_heads = gr.Textbox(label="Num KV Heads (GQA)", value=defaults['n_kv_heads'])
                ffn_hidden_size = gr.Textbox(label="FFN Hidden Size (Dense)", value=defaults['ffn_hidden_size'])

            # MoE
            with gr.Column():
                gr.Markdown("### 🎯 MoE Parameters")
                num_experts = gr.Textbox(label="Num Experts", value=defaults['num_experts'])
                moe_router_topk = gr.Textbox(label="MoE Router Top-K", value=defaults['moe_router_topk'])
                shared_experts = gr.Textbox(label="Shared Experts (always activated)", value=defaults['shared_experts'])
                moe_ffn_hidden_size = gr.Textbox(label="MoE FFN Hidden Size", value=defaults['moe_ffn_hidden_size'])
                expert_granularity = gr.Textbox(label="Expert Granularity G (optional)", value='0')
                moe_router_topk_scaling_factor = gr.Textbox(label="Router Top-K Scaling Factor", value=defaults['moe_router_topk_scaling_factor'])
                moe_router_num_groups = gr.Textbox(label="Router Num Groups", value=defaults['moe_router_num_groups'])
                moe_router_group_topk = gr.Textbox(label="Router Group Top-K", value=defaults['moe_router_group_topk'])
                moe_layer_freq = gr.Textbox(label="MoE Layer Freq ('every' or 'none')", value=defaults['moe_layer_freq'])
                shared_expert_intermediate_size = gr.Textbox(label="Shared Expert Size", value=defaults['shared_expert_intermediate_size'])
                gated_linear_multiplier = gr.Textbox(label="Gated Linear Mult (1.5 SwiGLU)", value=defaults['gated_linear_multiplier'])
                untie_embeddings_and_output_weights = gr.Checkbox(label="Untie Embed/LM Head", value=defaults['untie_embeddings_and_output_weights'])
                mtp_num_layers = gr.Textbox(label="MTP Num Layers (optional)", value=defaults['mtp_num_layers'], placeholder="0")

            # Training/Compute (Ling)
            with gr.Column():
                gr.Markdown("### ⚡ Compute (Ling)")
                batch_size = gr.Textbox(label="Batch Size (global)", value=defaults['batch_size'])
                sequence_length = gr.Textbox(label="Sequence Length", value=defaults['sequence_length'])
                num_training_iterations = gr.Textbox(label="Training Iterations (steps)", value=defaults['num_training_iterations'])
                train_flops_multiplier = gr.Textbox(label="Training FLOPs Mult (≈3)", value=defaults['train_flops_multiplier'])
                include_embedding_in_flops = gr.Checkbox(label="Include embedding/logits FLOPs in per-step forward", value=defaults['include_embedding_in_flops'])
                mfu = gr.Textbox(label="Model FLOP Utilization (MFU)", value=defaults['mfu'])

            # Parallelism/Memory (Megatron)
            with gr.Column():
                gr.Markdown("### 🔧 Parallelism & Memory (Megatron)")
                data_parallel_size = gr.Textbox(label="Data Parallel Size (DP)", value=defaults['data_parallel_size'])
                tensor_model_parallel_size = gr.Textbox(label="Tensor Parallel Size (TP)", value=defaults['tensor_model_parallel_size'])
                pipeline_model_parallel_size = gr.Textbox(label="Pipeline Parallel Size (PP)", value=defaults['pipeline_model_parallel_size'])
                expert_model_parallel_size = gr.Textbox(label="Expert Model Parallel (EP)", value=defaults['expert_model_parallel_size'])
                expert_tensor_model_parallel_size = gr.Textbox(label="Expert Tensor Parallel (ETP)", value=defaults['expert_tensor_model_parallel_size'])
                micro_batch_size = gr.Textbox(label="Micro Batch Size", value=defaults['micro_batch_size'])
                num_microbatches = gr.Textbox(label="Num Microbatches (PP)", value=defaults['num_microbatches'], placeholder="Auto")
                virtual_pipeline_model_parallel_size = gr.Textbox(label="Virtual PP Size (Interleaved)", value=defaults['virtual_pipeline_model_parallel_size'], placeholder="0")
                sequence_parallel = gr.Checkbox(label="Sequence Parallel", value=defaults['sequence_parallel'])
                recompute_granularity = gr.Dropdown(choices=['selective', 'full'], value=defaults['recompute_granularity'], label="Recompute Granularity")
                use_distributed_optimizer = gr.Checkbox(label="Distributed Optimizer", value=defaults['use_distributed_optimizer'])
                precision_bytes_param = gr.Textbox(label="Bytes/Param (e.g., 2 BF16)", value=defaults['precision_bytes_param'])
                precision_bytes_activation = gr.Textbox(label="Bytes/Activation (e.g., 4 FP32)", value=defaults['precision_bytes_activation'])

        # Outputs
        with gr.Tabs():
            with gr.TabItem("Compute & Scaling"):
                compute_output = gr.Textbox(label="📈 Compute & Scaling Results", lines=22, show_copy_button=True)
            with gr.TabItem("Memory Usage"):
                memory_output = gr.Textbox(label="📈 Memory Results", lines=22, show_copy_button=True)

        # Preset updater (only parallel/memory related inputs)
        preset_components = [
            data_parallel_size, tensor_model_parallel_size, pipeline_model_parallel_size,
            expert_model_parallel_size, expert_tensor_model_parallel_size,
            micro_batch_size, num_microbatches, virtual_pipeline_model_parallel_size,
            sequence_parallel, recompute_granularity, use_distributed_optimizer,
            precision_bytes_param, precision_bytes_activation
        ]

        def update_defaults(choice):
            d = defaults_1x if choice == "1x GPU (No Parallelism)" else defaults_8x
            return [
                d['data_parallel_size'], d['tensor_model_parallel_size'], d['pipeline_model_parallel_size'],
                d['expert_model_parallel_size'], d['expert_tensor_model_parallel_size'],
                d['micro_batch_size'], d['num_microbatches'], d['virtual_pipeline_model_parallel_size'],
                d['sequence_parallel'], d['recompute_granularity'], d['use_distributed_optimizer'],
                d['precision_bytes_param'], d['precision_bytes_activation']
            ]

        # Input order for compute
        compute_inputs = [
            num_layers, hidden_size, vocab_size, num_attention_heads,
            num_experts, moe_ffn_hidden_size,
            moe_router_topk, moe_router_topk_scaling_factor, moe_router_num_groups, moe_router_group_topk,
            shared_experts, expert_granularity,
            batch_size, sequence_length, n_kv_heads,
            num_training_iterations, train_flops_multiplier, include_embedding_in_flops, mfu
        ]

        # Input order for memory
        memory_inputs = [
            num_layers, hidden_size, vocab_size, num_attention_heads, n_kv_heads, ffn_hidden_size,
            moe_ffn_hidden_size, num_experts, moe_router_topk, shared_experts, moe_layer_freq, shared_expert_intermediate_size,
            untie_embeddings_and_output_weights, mtp_num_layers, gated_linear_multiplier,
            data_parallel_size, tensor_model_parallel_size, pipeline_model_parallel_size, expert_model_parallel_size, expert_tensor_model_parallel_size,
            micro_batch_size, sequence_length, num_microbatches, virtual_pipeline_model_parallel_size,
            sequence_parallel, recompute_granularity, use_distributed_optimizer,
            precision_bytes_param, precision_bytes_activation
        ]

        all_inputs = list(dict.fromkeys(compute_inputs + memory_inputs))  # preserve order, remove dups

        def combined_update(*args):
            # Map args back to components by order in all_inputs
            vals = {comp._id: val for comp, val in zip(all_inputs, args)}
            def resolve(components):
                return [vals[c._id] for c in components]
            comp_res = calculate_model_sizes(*resolve(compute_inputs))
            mem_res = calculate_memory_usage(*resolve(memory_inputs))
            return create_compute_output_text(comp_res), create_memory_output_text(mem_res)

        for inp in all_inputs:
            inp.change(combined_update, inputs=all_inputs, outputs=[compute_output, memory_output])

        config_choice.change(
            update_defaults,
            inputs=config_choice,
            outputs=preset_components
        ).then(combined_update, inputs=all_inputs, outputs=[compute_output, memory_output])

        # Initial load
        demo.load(combined_update, inputs=all_inputs, outputs=[compute_output, memory_output])

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=7833, share=False)
