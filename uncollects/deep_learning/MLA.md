# MLA

![mla](../../assets/image/mla.png)

The core of MLA is the low-rank joint compression for keys and values to reduce KV cache:

$$
\begin{align}
    \mathbf{c}_{t}^{KV} &= W^{DKV} \mathbf{h}_{t}, \\
    \mathbf{k}_{t}^{C} &= W^{UK} \mathbf{c}_{t}^{KV}, \\
    \mathbf{v}_{t}^{C} &= W^{UV} \mathbf{c}_{t}^{KV},
\end{align}
$$

where $\mathbf{c}_{t}^{KV} \in \mathbb{R}^{d_c}$ is the compressed latent vector for keys and values;
$d_c (\ll d_h n_h)$ denotes the KV compression dimension;
$W^{DKV} \in \mathbb{R}^{d_c \times d}$ is the down-projection matrix;
and $W^{UK},W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ are the up-projection matrices for keys and values, respectively.
During inference, MLA only needs to cache $\mathbf{c}_{t}^{KV}$, so its KV cache has only $d_{c}l$ elements, where $l$ denotes the number of layers.
In addition, during inference, since $W^{UK}$ can be absorbed into $W^{Q}$, and $W^{UV}$ can be absorbed into $W^{O}$, we even do not need to compute keys and values out for attention.

Moreover, in order to reduce the activation memory during training, we also perform low-rank compression for the queries, even if it cannot reduce the KV cache:

$$
\begin{align}
    \mathbf{c}_{t}^{Q} &= W^{DQ} \mathbf{h}_{t}, \\
    \mathbf{q}_{t}^{C} &= W^{UQ} \mathbf{c}_{t}^{Q},
\end{align}
$$

where $\mathbf{c}_{t}^{Q} \in \mathbb{R}^{d_c^{\prime}}$ is the compressed latent vector for queries;
$d_c^{\prime} (\ll d_h n_h)$ denotes the query compression dimension;
and $W^{DQ} \in \mathbb{R}^{d_c^{\prime} \times d}, W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c^{\prime}}$ are the down-projection and up-projection matrices for queries, respectively.
