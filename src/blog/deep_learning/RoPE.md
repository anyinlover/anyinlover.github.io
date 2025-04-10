---
title: Rotary Positional Embedding
pubDate: 2025-04-10 08:56:00
tags:
  - embedding
  - dl
---

# RoPE

The core concept of RoPE is to rotate the embedding vectors based on the token's position. This is achieved by applying a **rotation matrix** to the token's embedding, where the rotation angle is determined by the token's position in the sequence. By rotating the embeddings instead of using fixed position encodings, the model can maintain more flexible and continuous position information.

**The Rotation Mechanism**
RoPE uses a rotation-based encoding mechanism that operates on the embedding space as follows:

#### Step 1: **Sinusoidal Encoding for Rotation Angles**
For each dimension $j$ of the embedding, we compute the rotation angle based on the position $p_i$ of the token:
$$
\theta_{ij} = p_i \cdot \frac{1}{10000^{\frac{2j}{d}}}
$$

where:
- $\theta_{ij}$ is the rotation angle for the $j$-th dimension of the $i$-th token embedding.
- $p_i$ is the position of the token.
- $d$ is the total embedding dimension.

This formula uses a sinusoidal function, similar to the one used in traditional positional encodings (but here it’s used for rotation).

#### Step 2: **Rotation of Embedding Vectors**
For each token embedding $\mathbf{x}_i = [x_{i,1}, x_{i,2}, \dots, x_{i,d}]$, we apply the following transformation to inject the positional information:

For each dimension $j$, we apply a **2D rotation**:

  $$
  \begin{bmatrix} x_{i,2j-1} \\ x_{i,2j} \end{bmatrix} \to
  \begin{bmatrix} \cos(\theta_{ij}) & -\sin(\theta_{ij}) \\ \sin(\theta_{ij}) & \cos(\theta_{ij}) \end{bmatrix}
  \begin{bmatrix} x_{i,2j-1} \\ x_{i,2j} \end{bmatrix}
  $$
  where $\theta_{ij}$ is the rotation angle for the $j$-th dimension, and the cosine and sine functions come from the sinusoidal encoding.

This results in rotating each token’s embedding vector based on its position, ensuring that the embedding captures relative position information between tokens.
