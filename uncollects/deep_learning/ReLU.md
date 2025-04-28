# ReLU

**ReLU (Rectified Linear Unit)**:

The **ReLU** activation function is one of the most widely used [[Activation Functions]] in deep learning, particularly because of its simplicity and its ability to mitigate the vanishing gradient problem.

$$
\text{ReLU}(x) = \max(0, x)
$$

![activations](../../assets/image/activations.png)

Here are some reasons why ReLU might be considered better than [[Sigmoid Function]]:

### 1. **Computational Efficiency**

ReLU is computationally more efficient than Sigmoid because it only requires a single operation $max(0,x)$ compared to the exponential calculations involved in Sigmoid. This makes ReLU faster and more scalable for large models.

### 2. **Non-Linearity**

While both functions introduce non-linearity, ReLU does so in a more intuitive way: by mapping all negative values to zero and all positive values to themselves. This leads to simpler gradients during backpropagation and easier optimization.

### 3. **Avoiding Saturation**

Sigmoid has an inherent saturation problem: as the input increases or decreases, the output approaches the limits (0 or 1). This causes issues with gradient flow and can lead to vanishing gradients during backpropagation. ReLU avoids this issue by keeping its output in a more linear range.

### 4. **Better Handling of Input Data**

ReLU is less sensitive to outliers and extreme values, as they are simply clipped at zero. Sigmoid, on the other hand, becomes rapidly saturated for large input values, which can lead to poor gradient flow and reduced model performance.

### 5. **Simplified Backpropagation**

ReLU's non-linearity makes backpropagation easier, as it only requires a simple max operation during forward pass and a simple derivative computation during backward pass. Sigmoid's exponential calculations make backpropagation more complex and computationally expensive.
