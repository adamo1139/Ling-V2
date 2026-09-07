# Poziomka / APT4

For Poziomka (128 experts, top-32 routing) and the cleaned Polish conversations,
use [the Poziomka SFT guide](../examples/sft/megatron/README_poziomka.md).
It reuses the native Megatron pretraining loop with user+assistant loss masks
and mmap data. The original Ling-mini example below uses a different model
and chat format.

# Environment Preparation
## 1. Clone Ling-V2
## 2. Prepare [Docker Environment](../training/megatron_based_training.dockerfile)
## 3. Apply Patch
```bash
cd Ling-V2
# apply megatron patch
sh training/megatron/apply_patch.sh
```


# Data Preparation

The training data must be in JSON Lines (`.jsonl`) format. Each line should be a JSON object following the structure below.

### Example Data Format

```json
{
  "messages": [
    { "role": "system", "content": "" },
    {
      "role": "user",
      "content": "In a certain country populated predominantly by wizards, there is an impending demonstration that needs careful planning to ensure it meets specific requirements.\n\nThere is a population of n individuals within the city, of which x are wizards committed to attending the demonstration. The remaining (n - x) individuals, non-wizards, have no intention to participate. The city administration will acknowledge and act upon the demonstration only if it engages at least y percent of the total city population. To meet this criterion, wizards are considering the creation of clone puppets that will stand in for the non-wizards at the demonstration.\n\nImportantly, the administration cannot distinguish between real individuals and the clone puppets when calculating participation percentages. They will treat the city's population as n for this calculation, completely unaware of the existence of any clones.\n\nYour task is to determine the minimum number of clone puppets the wizards must create to ensure that the demonstration involves no less than y percent of the original city population n.\n\nInput Format:\n\nThe first line of input includes three space-separated integers, n, x, y (1 ≤ n, x, y ≤ 104, x ≤ n) — representing the total number of city residents, the number of wizards, and the percentage required by the administration, respectively.\n\nIt should be noted that y can be a value exceeding 100 percent, meaning the administration might require a demonstration crowd size greater than the city’s actual population.\n\nOutput Format:\n\nPrint a single integer — the minimum number of clone puppets that need to be created to ensure the demonstration meets or exceeds y percent of n (the city's true population size).\n\nExamples:\n\nInput\n\n10 1 14\n\nOutput\n\n1\n\nInput\n\n20 10 50\n\nOutput\n\n0\n\nInput\n\n1000 352 146\n\nOutput\n\n1108\n\nNote:\n\nIn the first example, to meet the requirement that at least 14% of 10 people (which rounds up to 2) attend, the lone wizard would need to create one clone.\nIn the second example, all 10 required participants are wizards, so no clones are necessary.\nIn the third example, the requirement is for 1460 attendees, so with 352 wizards, an additional 1108 clone puppets are required."
    },
    {
      "role": "assistant",
      "content": "To determine the minimum number of clone puppets required for the demonstration, let's break down the problem step by step.\n\n### Problem Breakdown:\n1. We need to ensure that the total number of participants in the demonstration meets at least \\( y \\% \\) of the original city population \\( n \\).\n2. The wizards, \\( x \\) of them, will definitely participate.\n3. The administration does not want to know about the clone puppets; hence they treat the total city's population as \\( n \\).\n\n### Steps to Solve:\n1. **Calculate Requirement:** Determine the number of participants required to meet \\( y \\% \\) of \\( n \\). This can be done using:\n   \\[\n   \\text{required\\_participants} = \\left\\lceil \\frac{y}{100} \\times n \\right\\rceil\n   \\]\n   We need at least this number of participants in the demonstration.\n   \n2. **Compare with Existing Participants:** Check how many more participants are needed beyond the wizards available:\n   \\[\n   \\text{additional\\_participants\\_needed} = \\max(0, \\text{required\\_participants} - x)\n   \\]\n   If the wizards already meet or exceed the requirement, no additional clones are necessary. Otherwise, calculate the difference.\n\n### Formula Application Using Example:\n- **Example:** For \\( n = 10, x = 1, y = 14 \\):\n  1. Calculate required participants: \\( \\left\\lceil \\frac{14}{100} \\times 10 \\right\\rceil = \\left\\lceil 1.4 \\right\\rceil = 2 \\)\n  2. Calculate clones needed: \\( \\max(0, 2 - 1) = 1 \\)\n\n### Implementation:\nLet's implement this logic in a Python function:\n\n```python\nimport math\n\ndef minimum_clones_needed(n, x, y):\n    # Calculate the required number of participants\n    required_participants = math.ceil((y / 100) * n)\n    \n    # Calculate how many more participants are needed beyond the wizards\n    additional_needed = max(0, required_participants - x)\n    \n    return additional_needed\n\n# Read the input values\nn, x, y = map(int, input().split())\n\n# Calculate the minimum number of clone puppets required\nresult = minimum_clones_needed(n, x, y)\n\n# Print the result\nprint(result)\n```\n\n### Detailed Explanation:\n1. **Input Reading:** Read three integers \\( n, x, y \\).\n2. **Compute Required Participants:** Using \\(\\left\\lceil \\frac{y}{100} \\times n \\right\\rceil\\), calculate how many people need to be at the demonstration.\n3. **Determine Needed Clones:** Compare this number to the number of wizards \\( x \\). If \\( x \\) is less than the required number, calculate the difference; otherwise, zero clones are needed.\n4. **Output Result:** Print the minimum number of clone puppets needed.\n\nThis approach ensures that the demonstration meets the criterion set by the administration efficiently and correctly."
    }
  ]
}
```

# Model Training Configuration

To run training, update the paths in the configuration file located at:

```
Megatron-LM-core_v0.13.0/examples/post_training/modelopt/conf/moe-mini-v2/moe_mini_v2.sh
```

Replace the placeholder paths with your actual dataset paths:

```bash
--train-data-path /path/to/your/train_data.jsonl
--valid-data-path /path/to/your/valid_data.jsonl
--test-data-path /path/to/your/test_data.jsonl
```

# Environment Setup

Set up the environment by exporting the following variables:

```bash
export MLM_MODEL_CKPT=/path/to/your/input_dcp
export MLM_MODEL_SAVE=/path/to/your/output
export PYTHONPATH=/path/to/your/Megatron-LM-core_v0.13.0:$PYTHONPATH
export TOKENIZER_MODEL=/path/to/your/Ling-V2/resource/tokenizer/config_sft
```

# Running the Training

Execute the training script using the following command:

```bash
sh examples/sft/megatron/run.sh
```
