# MoE

MoE stands for **Mixture of Experts**, a model architecture used to scale up the capacity of large language models (LLMs). In MoE, instead of using a single large model with a fixed set of parameters for all tasks, the model activates different subsets of experts (sub-models or layers) depending on the input.

![MoE](../../assets/image/switch_transformer.png)

Here’s how it works in the context of LLMs:

1. **Expert Layers**: An MoE model contains multiple "experts," which are separate neural network modules or layers. Each expert can have its own set of parameters that it uses to make predictions.

2. **Routing Mechanism**: For each input, the model dynamically selects which experts to activate. This is done by a **gating mechanism**, which determines which experts are most relevant for the given input. Typically, only a small fraction of the available experts are activated at any given time. This is often referred to as **sparsity** in MoE models.

3. **Efficiency**: The primary advantage of MoE models is their efficiency. By activating only a few experts for each input, the model can scale to much larger sizes without having to increase the computational cost proportionally. For instance, an MoE model can have hundreds of experts, but only a few of them are used per inference, allowing for more parameters without overwhelming computational resources.

4. **Training**: During training, the gating mechanism learns which experts to activate based on the input, and the rest of the experts are kept inactive. This leads to efficient training and inference, as not all parameters are being updated or used at once.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, top_k=2):
        super(MoE, self).__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Define the experts (e.g., simple feed-forward layers)
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])

        # Define the gating network (outputs logits for each expert)
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Get gating logits (unsqueezed probabilities of expert activation)
        gating_logits = self.gating_network(x)  # Shape: [batch_size, num_experts]

        # Apply softmax to get probabilities
        gating_probs = F.softmax(gating_logits, dim=-1)  # Shape: [batch_size, num_experts]

        # Sort the gating probabilities to select top-k experts
        top_k_values, top_k_indices = torch.topk(gating_probs, self.top_k, dim=-1)

        # Initialize output tensor
        output = torch.zeros(x.size(0), self.expert_dim).to(x.device)

        # Loop over batch size to apply the selected top-k experts
        for i in range(x.size(0)):
            for j in range(self.top_k):
                # For each selected expert, route the input to the expert's output
                expert_idx = top_k_indices[i, j]
                output[i] += self.experts[expert_idx](x[i]) * top_k_values[i, j]

        return output
```
