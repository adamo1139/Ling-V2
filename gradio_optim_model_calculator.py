#!/usr/bin/env python3
"""
MoE Model Size Calculator - Gradio UI
Calculates total model size, activated parameters, and leverage for Mixture of Experts models
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

    # Computational cost parameters
    batch_size,
    sequence_length,
    n_kv,
    num_training_iterations
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
        batch_size = int(batch_size)
        sequence_length = int(sequence_length)
        n_kv = int(n_kv)
        num_training_iterations = int(num_training_iterations)

        # Calculate embedding parameters (input + output embeddings)
        embed_params = vocab_size * hidden_size * 2

        # Calculate attention parameters per layer
        # QKV projections + output projection
        attn_params_per_layer = 4 * hidden_size * hidden_size
        attn_params = num_layers * attn_params_per_layer

        # Calculate MoE expert parameters
        # Each expert: input projection + output projection
        moe_expert_params_per_layer = num_experts * moe_ffn_hidden_size * hidden_size * 2
        moe_expert_params = num_layers * moe_expert_params_per_layer

        # Calculate router parameters
        # Router network: hidden_size -> num_experts
        router_params_per_layer = hidden_size * num_experts
        router_params = num_layers * router_params_per_layer

        # Total model parameters (MoE architecture: Attention + MoE FFN)
        total_params = embed_params + attn_params + moe_expert_params + router_params

        # Calculate activated parameters
        # Attention is always activated
        activated_attn = attn_params

        # FFN activation: only top-k experts are activated per token
        activated_ffn_per_layer = moe_router_topk * moe_ffn_hidden_size * hidden_size * 2
        activated_ffn = num_layers * activated_ffn_per_layer

        # Total activated parameters (attention + activated FFN)
        total_activated = activated_attn + activated_ffn

        # Calculate ratios and leverage
        activated_ffn_ratio = activated_ffn / moe_expert_params if moe_expert_params > 0 else 0
        model_leverage = total_params / total_activated if total_activated > 0 else 0

        # Calculate memory estimates (rough, in GB for BF16)
        param_memory_gb = total_params * 2 / (1024**3)  # 2 bytes per parameter for BF16
        activated_memory_gb = total_activated * 2 / (1024**3)

        # Calculate computational costs
        # Attention computation cost
        c_attn = 2 * batch_size * sequence_length * (hidden_size ** 2) * (1 + 2 / (num_attention_heads / n_kv)) + 4 * batch_size * (sequence_length ** 2) * hidden_size

        # MoE FFN computation cost
        c_moe_ffn = 6 * batch_size * sequence_length * hidden_size * (moe_router_topk * moe_ffn_hidden_size)

        # Forward pass computation cost (sum over layers + embedding)
        c_fwd = num_layers * (c_attn + c_moe_ffn) + 2 * batch_size * sequence_length * hidden_size * vocab_size

        # Training computation cost (approximately 3x forward pass)
        c_train = 3 * c_fwd

        # Total training cost over all iterations
        total_training_cost = c_train * num_training_iterations

        return {
            'total_params': total_params,
            'embed_params': embed_params,
            'attn_params': attn_params,
            'moe_expert_params': moe_expert_params,
            'router_params': router_params,
            'activated_attn': activated_attn,
            'activated_ffn': activated_ffn,
            'total_activated': total_activated,
            'activated_ffn_ratio': activated_ffn_ratio,
            'model_leverage': model_leverage,
            'param_memory_gb': param_memory_gb,
            'activated_memory_gb': activated_memory_gb,
            'c_attn': c_attn,
            'c_moe_ffn': c_moe_ffn,
            'c_fwd': c_fwd,
            'c_train': c_train,
            'total_training_cost': total_training_cost
        }

    except (ValueError, ZeroDivisionError) as e:
        return {
            'error': f"Invalid input: {str(e)}"
        }

def create_output_text(results):
    """Format results into readable text output"""
    if 'error' in results:
        return f"❌ Error: {results['error']}"

    output = "🔬 **MoE Model Size Analysis**\n\n"

    output += "📊 **Parameter Breakdown:**\n"
    output += f"• Total Parameters: {format_number(results['total_params'])}\n"
    output += f"• Embedding Parameters: {format_number(results['embed_params'])}\n"
    output += f"• Attention Parameters: {format_number(results['attn_params'])}\n"
    output += f"• MoE Expert Parameters: {format_number(results['moe_expert_params'])}\n"
    output += f"• Router Parameters: {format_number(results['router_params'])}\n\n"

    output += "🎯 **Activation Analysis:**\n"
    output += f"• Activated Attention: {format_number(results['activated_attn'])}\n"
    output += f"• Activated FFN: {format_number(results['activated_ffn'])}\n"
    output += f"• Total Activated: {format_number(results['total_activated'])}\n"
    output += f"• FFN Activation Ratio: {results['activated_ffn_ratio']:.3f}\n"
    output += f"• Model Leverage: {results['model_leverage']:.1f}x\n\n"

    output += "💾 **Memory Estimates (BF16):**\n"
    output += f"• Total Parameters: {results['param_memory_gb']:.2f} GB\n"
    output += f"• Activated Parameters: {results['activated_memory_gb']:.2f} GB\n\n"

    output += "⚡ **Computational Cost Analysis:**\n"
    output += f"• Attention Cost: {format_flops(results['c_attn'])}\n"
    output += f"• MoE FFN Cost: {format_flops(results['c_moe_ffn'])}\n"
    output += f"• Forward Pass Cost: {format_flops(results['c_fwd'])}\n"
    output += f"• Training Cost (per step): {format_flops(results['c_train'])}\n"
    output += f"• Total Training Cost: {format_flops(results['total_training_cost'])}\n"

    return output

def main():
    # Default values from the training script
    defaults = {
        'num_layers': '20',
        'hidden_size': '1024',
        'vocab_size': '128000',
        'num_attention_heads': '32',
        'num_experts': '32',
        'moe_ffn_hidden_size': '256',
        'moe_router_topk': '2',
        'moe_router_topk_scaling_factor': '2.5',
        'moe_router_num_groups': '8',
        'moe_router_group_topk': '8',
        'batch_size': '8',
        'sequence_length': '1024',
        'n_kv': '8',
        'num_training_iterations': '1000000'
    }

    with gr.Blocks(title="MoE Model Size Calculator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 MoE Model Size Calculator")
        gr.Markdown("Calculate total model size, activated parameters, and leverage for Mixture of Experts models")

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
            batch_size, sequence_length, n_kv, num_training_iterations
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
