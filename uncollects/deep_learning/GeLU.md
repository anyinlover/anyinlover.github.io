# GeLU

The **Gaussian Error Linear Unit** is one of the [[Activation Functions]]. It is computed using the approximation:

$$
   \text{GeLU}(x) = 0.5 x \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

![activations](../../assets/image/activations.png)

Here are some reasons why GeLU might be considered better than [[ReLU]]:

### 1. **Softer Non-Linearity**

GeLU introduces a softer non-linearity compared to ReLU, which can lead to more stable and robust training. The Gaussian component of GeLU helps to smooth out the activation function, reducing the risk of dying neurons (neurons with outputs close to zero).

### 2. **Improved Gradient Flow**

ReLU has a significant drawback: it's not differentiable at $x = 0$. This can cause issues during backpropagation, as gradients are not defined for these points. GeLU, on the other hand, is continuously differentiable, making gradient flow more stable and reliable.

### 3. **Better Handling of Negative Inputs**

GeLU handles negative inputs more effectively than ReLU. In contrast to ReLU's abrupt clipping at $x = 0$, GeLU applies a gentle Gaussian transformation, which can lead to more realistic output distributions.

### 4. **Reduced Vanishing Gradients**

When using ReLU, some neurons may have outputs close to zero, leading to vanishing gradients during backpropagation. GeLU's softer non-linearity helps mitigate this issue by ensuring that gradients are not amplified or diminished excessively.

### 5. **Improved Model Performance**

Experiments have shown that GeLU can lead to improved model performance on certain tasks, such as language modeling and machine translation. This might be due to the fact that GeLU allows for more nuanced representations of the input data.
