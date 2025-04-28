# Zero

In **DeepSpeed**, `zero0`, `zero1`, and `zero2` refer to different stages of **ZeRO (Zero Redundancy Optimizer)**, which is a memory optimization technique used to train large models. These stages help reduce the memory footprint of model parameters by partitioning the optimizer states and gradients across multiple devices (e.g., GPUs). Here's a breakdown of each stage:

1. **ZeRO Stage 0 (zero0)**:
   - In this stage, the model parameters are **partitioned across devices**, but no optimizer state or gradient partitioning is performed.
   - The model's parameters are distributed across the GPUs, so each GPU holds a portion of the parameters, reducing the memory load on each device.
   - This stage does not optimize memory usage beyond the parameter partitioning.

2. **ZeRO Stage 1 (zero1)**:
   - In this stage, **model parameters and gradients are partitioned** across devices.
   - Each GPU holds a portion of the model parameters as well as a portion of the gradients.
   - This stage allows for further memory optimization by partitioning not only the parameters but also the gradients, which helps reduce the overall memory requirements.

3. **ZeRO Stage 2 (zero2)**:
   - In this stage, **model parameters, gradients, and optimizer states are all partitioned** across devices.
   - Each GPU holds a portion of the model parameters, gradients, and optimizer states (such as momentum and variance in Adam).
   - This stage provides significant memory savings by distributing all three major components of the model's training process.

4. **ZeRO Stage 3 (zero3)** (not mentioned, but it's often included):
   - In this stage, in addition to partitioning parameters, gradients, and optimizer states, **communication is further optimized** to reduce data transfer overhead.
   - It also uses techniques like **activation checkpointing** to minimize the memory needed for storing activations during forward/backward passes.

These stages of ZeRO help in training large models that would otherwise not fit into memory by distributing the workload across multiple GPUs and managing memory usage more efficiently.

Let me know if you need more details about any specific stage!
