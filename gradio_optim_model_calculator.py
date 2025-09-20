#!/usr/bin/env python3
"""
MoE Compute & Size Calculator (Ling Scaling Laws)

This tool computes:
- Parameter counts (embeddings, attention, router, experts)
- Non-embedding FLOPs per token per forward pass M (per Ling definition)
- Training compute C using C = M · D (D = tokens)
- Compute-optimal model/data allocations (Table 1 in paper)
- An EL estimate in compute-optimal regime (proxy): EL ≈ M_dense_opt / M_moe_opt at same C

Notes:
- Uses grouped-query attention factor g = n_kv / n_heads for attention FLOPs.
- Expert granularity G = 2·d_model / d_expert (Ling), so d_expert = 2·H / G.
- Router compute includes H·E logits per token; softmax/top-k overhead is ignored.
- M excludes embeddings per Ling; optional embedding cost is reported separately.
"""

import gradio as gr
import math

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

def format_flops(num):
    """Format computational costs in scientific notation"""
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

def calculate_model_sizes(
    # Base model parameters
    num_layers,
    hidden_size,
    vocab_size,
    num_attention_heads,

    # MoE parameters
    num_experts,
    moe_ffn_hidden_size,
    moe_router_topk,
    moe_router_topk_scaling_factor,
    moe_router_num_groups,
    moe_router_group_topk,
    shared_experts,
    expert_granularity,

    # Computational cost parameters
    batch_size,
    sequence_length,
    n_kv,
    num_training_iterations,
    train_flops_multiplier,
    include_embedding_in_flops,
    mfu
):
    try:
        # Convert inputs to appropriate types
        num_layers = int(num_layers)
        hidden_size = int(hidden_size)
        vocab_size = int(vocab_size)
        num_attention_heads = int(num_attention_heads)
        num_experts = int(num_experts)
        moe_ffn_hidden_size = int(moe_ffn_hidden_size)
        moe_router_topk = int(moe_router_topk)
        moe_router_topk_scaling_factor = float(moe_router_topk_scaling_factor)
        moe_router_num_groups = int(moe_router_num_groups)
        moe_router_group_topk = int(moe_router_group_topk)
        shared_experts = int(shared_experts)
        expert_granularity = float(expert_granularity) if expert_granularity else 0.0
        batch_size = int(batch_size)
        sequence_length = int(sequence_length)
        n_kv = int(n_kv)
        num_training_iterations = int(num_training_iterations)
        train_flops_multiplier = float(train_flops_multiplier)
        include_embedding_in_flops = bool(include_embedding_in_flops)
        mfu = float(mfu)

        # Basic validations
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

        # Derive expert hidden dim from granularity if provided; otherwise use moe_ffn_hidden_size
        if expert_granularity and expert_granularity > 0:
            d_expert = int(max(1, round(2 * hidden_size / expert_granularity)))
            d_expert_source = "granularity"
        else:
            d_expert = int(moe_ffn_hidden_size)
            d_expert_source = "ffn_hidden_size"

        if d_expert <= 0:
            raise ValueError("Derived expert hidden size must be positive")
        # Snap to multiple-of-8 and enforce minimum 64 (kernel-friendly)
        d_expert = max(64, int(round(d_expert / 8)) * 8)

        # Grouped-query attention factor g = n_kv / n_heads
        g = n_kv / num_attention_heads

        # Effective activated experts per token (routable + shared)
        ea = moe_router_topk  # routable activated by top-k
        es = shared_experts   # shared experts always-on
        e_total = num_experts + es
        activation_ratio = (ea + es) / e_total if e_total > 0 else 0.0

        # Embedding + LM head parameters
        # Untied embeddings by default in your setup: word_embeddings (V*H) + lm_head (H*V)
        # Include final model RMSNorm (size H) to match real models
        embed_params = vocab_size * hidden_size * 2
        final_norm_params = hidden_size

        # Attention parameters per layer, matching fused QKV + out in BailingMoeV2Attention
        # query_key_value: H x ((n_heads + 2*n_kv) * d)
        # out: (n_heads * d) x H
        # optional qk norms (2 * d) and two layer RMSNorms per layer (2 * H)
        head_dim = hidden_size // num_attention_heads
        attn_qkv_weight = hidden_size * ((num_attention_heads + 2 * n_kv) * head_dim)
        attn_out_weight = (num_attention_heads * head_dim) * hidden_size
        use_qk_norm = True
        attn_qk_norm = (2 * head_dim) if use_qk_norm else 0
        layer_rmsnorms = 2 * hidden_size
        attn_params_per_layer = attn_qkv_weight + attn_out_weight + attn_qk_norm + layer_rmsnorms
        attn_params = num_layers * attn_params_per_layer

        # MoE expert parameters per layer (per expert is SwiGLU: 3 * H * d_expert)
        moe_expert_params_per_layer = num_experts * (3 * hidden_size * d_expert)
        moe_expert_params = num_layers * moe_expert_params_per_layer

        # Router parameters per layer: H -> E logits (+ expert bias E if enabled)
        router_expert_bias = True
        router_params_per_layer = hidden_size * num_experts + (num_experts if router_expert_bias else 0)
        router_params = num_layers * router_params_per_layer

        # Total model parameters (MoE architecture: Attention + MoE FFN)
        total_params = embed_params + final_norm_params + attn_params + moe_expert_params + router_params

        # Activated parameters per token (heuristic): attention + router + activated experts
        activated_attn = attn_params  # all layers/weights are used each token
        activated_router_per_layer = router_params_per_layer
        # Per-token activated experts params: 3*H*d_expert per expert (SwiGLU)
        activated_ffn_per_layer = (ea + es) * (3 * hidden_size * d_expert)
        activated_ffn = num_layers * activated_ffn_per_layer
        activated_router = num_layers * activated_router_per_layer
        total_activated = activated_attn + activated_router + activated_ffn

        # Ratios
        activated_ffn_ratio = (activated_ffn_per_layer * num_layers) / moe_expert_params if moe_expert_params > 0 else 0.0

        # Calculate memory estimates (rough, in GB for BF16)
        param_memory_gb = total_params * 2 / (1024**3)  # 2 bytes per parameter for BF16
        activated_memory_gb = total_activated * 2 / (1024**3)

        # Exact non-embedding FLOPs per token per layer (MACs*2) per Ling definition
        # Attention MACs/token/layer:
        #   Projections (fused qkv + out): H*((n_h + 2*n_kv)*d) + (n_h*d)*H
        #   Core (scores + weighted sum): ~ 2 * L * (n_h * d) = 2 * L * H
        macs_attn_proj = attn_qkv_weight + attn_out_weight
        macs_attn_core = 2 * sequence_length * hidden_size
        macs_attn_token_layer = macs_attn_proj + macs_attn_core

        # Router MACs per token per layer: H * E (logits)
        macs_router_token_layer = hidden_size * num_experts

        # Activated experts MACs per token per layer (SwiGLU): (ea+es) * (2*H*I + I*H) = 3 * (ea+es) * H * d_expert
        macs_moe_token_layer = 3 * (ea + es) * hidden_size * d_expert

        # FLOPs per token per layer (multiply-add = 2 FLOPs)
        flops_token_layer = 2 * (macs_attn_token_layer + macs_router_token_layer + macs_moe_token_layer)

        # FLOPs per token across all layers (non-embedding M)
        M_token = num_layers * flops_token_layer

        # Component-wise FLOPs across all layers for mix analysis
        attn_flops_total = num_layers * 2 * macs_attn_token_layer
        ffn_flops_total = num_layers * 2 * macs_moe_token_layer
        router_flops_total = num_layers * 2 * macs_router_token_layer
        denom_ffn_attn = (attn_flops_total + ffn_flops_total)
        attn_ffn_ratio = (attn_flops_total / denom_ffn_attn) if denom_ffn_attn > 0 else 0.0
        # Status per Ling: 30–40% stable; 20–50% minor impact
        if 0.30 <= attn_ffn_ratio <= 0.40:
            attn_ffn_status = "within 30–40% stable range"
        elif 0.20 <= attn_ffn_ratio <= 0.50:
            attn_ffn_status = "within broader 20–50% tolerance"
        else:
            attn_ffn_status = "outside recommended range"

        # Optional embedding compute per token (projection to vocab)
        # MACs: H * V (logits); FLOPs_emb = 2 * H * V
        flops_embed_token = 2 * hidden_size * vocab_size if include_embedding_in_flops else 0

        # Forward pass FLOPs per step (multiply by tokens per step)
        c_fwd = batch_size * sequence_length * (M_token + flops_embed_token)

        # Training computation cost multiplier (≈3x fwd for fwd+bwd+update)
        c_train = train_flops_multiplier * c_fwd

        # Total tokens and total training compute
        total_tokens = batch_size * sequence_length * num_training_iterations
        # Per Ling: C = M * D (exclude embeddings)
        total_training_cost = M_token * total_tokens
        total_train_flops_with_bwd = c_train * num_training_iterations

        # Human-readable GPU hours estimates (idealized sustained BF16 TFLOPS)
        def flops_to_gpu_hours(flops, tflops, mfu):
            return flops / (tflops * 1e12 * mfu) / 3600.0

        hours_3090_ti = flops_to_gpu_hours(total_train_flops_with_bwd, 40.0, mfu)   # assume 40 TFLOPS BF16
        hours_h100_sxm5 = flops_to_gpu_hours(total_train_flops_with_bwd, 900.0, mfu)  # assume 900 TFLOPS BF16

        # Compute-optimal model/data allocations (Table 1)
        # Dense
        Mopt_dense = 0.0655 * (total_training_cost ** 0.5422)
        Dopt_dense = 15.2582 * (total_training_cost ** 0.4578)
        # MoE
        Mopt_moe = 0.1915 * (total_training_cost ** 0.5095)
        Dopt_moe = 5.2232 * (total_training_cost ** 0.4905)

        # Joint EL scaling law (Eq. 13, Ling):
        #   EL(A,G,C) = Â^( α + γ*(log10 G)^2 + β*log10 G ), α = a + d*log10(C)
        a_coef = 1.23
        d_coef = -7.61e-2
        gamma_coef = 1.67e-2
        beta_coef = -1.17e-1
        A_start = 1.63e-2
        A_max = 5.28e16

        import math as _math
        A = activation_ratio if activation_ratio > 0 else A_start
        A_hat = max(min(A, A_max), A_start)
        # Effective granularity from snapped d_expert
        G = (2 * hidden_size) / d_expert
        log10C = _math.log10(total_training_cost) if total_training_cost > 0 else 0.0
        log10G = _math.log10(max(G, 1e-12))
        alpha = a_coef + d_coef * log10C
        exponent_on_A = alpha + gamma_coef * (log10G ** 2) + beta_coef * log10G
        el_joint = (A_hat ** exponent_on_A) if A_hat > 0 else 0.0

        # Optimal learning rate and batch size from total compute C
        # Nopt = 1.1576 * C^(-0.1529)
        # Bopt = 0.0694 * C^(0.3644)
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
        return {
            'error': f"Invalid input: {str(e)}"
        }

def create_output_text(results):
    """Format results into readable text output"""
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

    # Attention vs FFN compute mix (Ling guidance)
    if 'attn_flops_total' in results and 'ffn_flops_total' in results:
        attn_share = results['attn_ffn_ratio'] * 100
        ffn_share = 100 - attn_share
        output += "⚖️ Attention vs FFN Compute Mix:\n"
        output += f"• Attention FLOPs share (vs FFN): {attn_share:.1f}%\n"
        output += f"• FFN FLOPs share: {ffn_share:.1f}%\n"
        output += f"• Attention FLOPs (all layers): {format_flops(results['attn_flops_total'])}\n"
        output += f"• FFN FLOPs (all layers): {format_flops(results['ffn_flops_total'])}\n"
        output += f"• Status: {results['attn_ffn_status']} (Ling suggests 30–40% stable; 20–50% ok)\n\n"

    output += "🧭 Compute-Optimal (Table 1):\n"
    output += f"• Dense Mopt: {format_number(results['Mopt_dense'])} FLOPs/token, Dopt: {format_number(results['Dopt_dense'])} tokens\n"
    output += f"• MoE   Mopt: {format_number(results['Mopt_moe'])} FLOPs/token, Dopt: {format_number(results['Dopt_moe'])} tokens\n"
    output += "• Note: Mopt is non-embedding FLOPs/token (Ling).\n"

    # Optimal learning rate and batch size derived from C
    if results.get('lr_opt') is not None and results.get('batch_opt') is not None:
        output += "\n🛠 Optimal Hyperparameters (from C):\n"
        output += f"• Learning Rate (Nopt): {results['lr_opt']:.6f}\n"
        output += f"• Batch Size (Bopt): {results['batch_opt']:.2f}\n"

    # EL (Eq. 13)
    el_joint = results.get('el_joint', None)
    if el_joint is not None:
        output += "\n📈 Efficiency Leverage (Eq. 13, Ling):\n"
        output += f"• EL(A,G,C) = {el_joint:.2f}x\n"
        output += f"• Using A={results.get('el_A_used'):.4f}, G={results.get('el_G_used'):.2f}, alpha={results.get('el_alpha'):.3f}\n"

    # GPU-hour notes
    hours_3090 = results.get('hours_3090_ti', 0.0)
    hours_h100 = results.get('hours_h100_sxm5', 0.0)
    total_train_flops = results.get('total_train_flops_with_bwd', 0.0)
    output += "\n⏱ GPU Time Estimates (idealized):\n"
    output += f"• Total train FLOPs (fwd+bwd): {format_flops(total_train_flops)}\n"
    output += f"• 3090 Ti hours (40 TFLOPS BF16, single GPU): {hours_3090:.1f} h\n"
    output += f"• H100 SXM5 80GB hours (900 TFLOPS BF16, single GPU): {hours_h100:.1f} h\n"
    output += f"• MFU used: {results.get('mfu', 0.4)*100:.0f}%\n"
    output += "• Notes: single-GPU idealized math; multi-GPU scales ~1/N ignoring comms, pipeline, IO.\n"

    return output

def main():
    # Default values from the training script
    defaults = {
        'num_layers': '12',
        'hidden_size': '768',
        'vocab_size': '32000',
        'num_attention_heads': '6',
        'num_experts': '128',
        'moe_ffn_hidden_size': '128',
        'moe_router_topk': '2',
        'moe_router_topk_scaling_factor': '2.5',
        'moe_router_num_groups': '4',
        'moe_router_group_topk': '2',
        'shared_experts': '0',
        'expert_granularity': '0',
        'batch_size': '16',
        'sequence_length': '32768',
        'n_kv': '2',
        'num_training_iterations': '5000',
        'train_flops_multiplier': '3.0',
        'include_embedding_in_flops': False
    }

    with gr.Blocks(title="MoE Compute & Size Calculator (Ling)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 MoE Compute & Size Calculator (Ling)")
        gr.Markdown("Non-embedding FLOPs per token (M), params, and compute based on Ling Scaling Laws. No auto-optimization; just accurate arithmetic.")

        with gr.Row():
            # Base Model Parameters
            with gr.Column():
                gr.Markdown("### 🏗️ Base Model Parameters")
                num_layers = gr.Textbox(
                    label="Number of Layers",
                    value=defaults['num_layers'],
                    placeholder="e.g., 20"
                )
                hidden_size = gr.Textbox(
                    label="Hidden Size",
                    value=defaults['hidden_size'],
                    placeholder="e.g., 1024"
                )
                vocab_size = gr.Textbox(
                    label="Vocabulary Size",
                    value=defaults['vocab_size'],
                    placeholder="e.g., 128000"
                )
                num_attention_heads = gr.Textbox(
                    label="Number of Attention Heads",
                    value=defaults['num_attention_heads'],
                    placeholder="e.g., 32"
                )

            # MoE Parameters
            with gr.Column():
                gr.Markdown("### 🎯 MoE Parameters")
                num_experts = gr.Textbox(
                    label="Number of Experts",
                    value=defaults['num_experts'],
                    placeholder="e.g., 32"
                )
                moe_ffn_hidden_size = gr.Textbox(
                    label="MoE FFN Hidden Size",
                    value=defaults['moe_ffn_hidden_size'],
                    placeholder="e.g., 256"
                )
                expert_granularity = gr.Textbox(
                    label="Expert Granularity G (optional; overrides FFN size if > 0)",
                    value=defaults['expert_granularity'],
                    placeholder="e.g., 12"
                )
                moe_router_topk = gr.Textbox(
                    label="Router Top-K",
                    value=defaults['moe_router_topk'],
                    placeholder="e.g., 2"
                )
                moe_router_topk_scaling_factor = gr.Textbox(
                    label="Router Top-K Scaling Factor",
                    value=defaults['moe_router_topk_scaling_factor'],
                    placeholder="e.g., 2.5"
                )
                moe_router_num_groups = gr.Textbox(
                    label="Router Num Groups",
                    value=defaults['moe_router_num_groups'],
                    placeholder="e.g., 8"
                )
                moe_router_group_topk = gr.Textbox(
                    label="Router Group Top-K",
                    value=defaults['moe_router_group_topk'],
                    placeholder="e.g., 8"
                )
                shared_experts = gr.Textbox(
                    label="Shared Experts (Es)",
                    value=defaults['shared_experts'],
                    placeholder="e.g., 1"
                )

            # Computational Cost Parameters
            with gr.Column():
                gr.Markdown("### ⚡ Computational Cost Parameters")
                batch_size = gr.Textbox(
                    label="Batch Size",
                    value=defaults['batch_size'],
                    placeholder="e.g., 8"
                )
                sequence_length = gr.Textbox(
                    label="Sequence Length",
                    value=defaults['sequence_length'],
                    placeholder="e.g., 1024"
                )
                n_kv = gr.Textbox(
                    label="Number of KV Heads",
                    value=defaults['n_kv'],
                    placeholder="e.g., 8"
                )
                num_training_iterations = gr.Textbox(
                    label="Number of Training Iterations",
                    value=defaults['num_training_iterations'],
                    placeholder="e.g., 1000000"
                )
                train_flops_multiplier = gr.Textbox(
                    label="Training FLOPs Multiplier (≈3 for fwd+bwd+update)",
                    value=defaults['train_flops_multiplier'],
                    placeholder="e.g., 3.0"
                )
                mfu = gr.Textbox(
                    label="Model FLOP Utilization (MFU)",
                    value="0.40",
                    placeholder="e.g., 0.40 for 40%"
                )
                include_embedding_in_flops = gr.Checkbox(
                    label="Include embedding/logits FLOPs in per-step forward",
                    value=defaults['include_embedding_in_flops']
                )

        # Output
        output = gr.Textbox(
            label="📈 Calculation Results",
            lines=20,
            show_copy_button=True
        )

        # Input components list for the function
        inputs = [
            num_layers, hidden_size, vocab_size, num_attention_heads,
            num_experts, moe_ffn_hidden_size,
            moe_router_topk, moe_router_topk_scaling_factor, moe_router_num_groups, moe_router_group_topk,
            shared_experts, expert_granularity,
            batch_size, sequence_length, n_kv, num_training_iterations,
            train_flops_multiplier, include_embedding_in_flops, mfu
        ]

        # Update output when any input changes
        for inp in inputs:
            inp.change(
                fn=lambda *args: create_output_text(calculate_model_sizes(*args)),
                inputs=inputs,
                outputs=output
            )

        # Initial calculation
        demo.load(
            fn=lambda *args: create_output_text(calculate_model_sizes(*args)),
            inputs=inputs,
            outputs=output
        )

    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=7831, share=False)
