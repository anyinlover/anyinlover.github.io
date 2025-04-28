# LoRA

LoRA (Low-Rank Adaptation) is a technique used for fine-tuning large pre-trained models by adding trainable low-rank matrices to the existing weights. This approach is particularly useful for reducing the memory and computational costs of fine-tuning large models while maintaining performance.

**Decompose $W$ into low-rank components**:
The core idea is to approximate the weight matrix $W$ by a product of two smaller matrices $A$ and $B$, such that:

$$
W_{\text{new}} = W + \Delta W = W + A \cdot B^T
$$

where:

- $W$ is the original weight matrix (pre-trained and kept frozen).
- $A \in \mathbb{R}^{d_{\text{out}} \times r}$ is a matrix of size $d_{\text{out}} \times r$ (where $r$ is the rank, a hyperparameter that controls the size of the additional trainable parameters).
- $B \in \mathbb{R}^{r \times d_{\text{in}}}$ is a matrix of size $r \times d_{\text{in}}$.
- $\Delta W = A \cdot B^T$ is the low-rank adjustment to the original weight matrix.

The $\alpha$ parameter in the LoRA (Low-Rank Adaptation) technique is a scaling factor that controls the magnitude of the low-rank adaptation's impact on the model.

$$
W_{\text{new}} = W + \Delta W = W + \alpha \cdot A \cdot B^T
$$

A good start value is $\alpha = 2r$.

Excluding the value projection matrix $W_V$​ in LoRA fine-tuning is a practical choice driven by efficiency and the observation that fine-tuning only the query and key projections is often sufficient for improving task performance.

LoRA Dropout is the application of dropout regularization specifically to the low-rank matrices $A$ and $B$ during LoRA fine-tuning.
