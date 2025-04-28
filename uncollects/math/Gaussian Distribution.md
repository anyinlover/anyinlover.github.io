# Gaussian Distribution

The pdf of the Gaussian is given by:

$$
\mathcal{N}(y|\mu, \sigma^2) \triangleq \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{1}{2\sigma^2}(y-\mu)^2}
$$

![gaussian plot](../../assets/image/multiple_gaussians.png)

$$
\begin{aligned}
\mathbb{E}[Y] &= \int_{-\infty}^{\infty} y \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(y-\mu)^2}{2\sigma^2}} \, dy \\
&= \int_{-\infty}^{\infty} (\sigma z + \mu) \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{z^2}{2}} \sigma \, dz \quad (\text{where } z = \frac{y-\mu}{\sigma}, dy = \sigma dz) \\
&= \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} (\sigma z + \mu) e^{-\frac{z^2}{2}} \, dz \\
&= \frac{\sigma}{\sqrt{2\pi}} \int_{-\infty}^{\infty} z e^{-\frac{z^2}{2}} \, dz + \frac{\mu}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{-\frac{z^2}{2}} \, dz \\
&= \frac{\sigma}{\sqrt{2\pi}} \cdot 0 + \frac{\mu}{\sqrt{2\pi}} \cdot \sqrt{2\pi} \\
&= \mu
\end{aligned}
$$

where:

$$
\begin{aligned}
\int_{-\infty}^{\infty} e^{-\frac{z^2}{2}} \, dz &= \sqrt{\int_{-\infty}^{\infty} e^{-\frac{x^2}{2}} \, dx \int_{-\infty}^{\infty} e^{-\frac{y^2}{2}} \, dy} \\
&= \sqrt{\int_{0}^{2\pi} \int_{0}^{\infty} e^{-\frac{r^2}{2}} r \, dr \, d\theta} \\
&= \sqrt{\int_{0}^{2\pi} \left[-e^{-\frac{r^2}{2}}\right]_0^\infty d\theta} \\
&= \sqrt{\int_{0}^{2\pi} 1 \, d\theta} \\
&= \sqrt{2\pi}
\end{aligned}
$$

$$
\begin{aligned}
\mathbb{V}[Y] &= \mathbb{E}[(Y - \mathbb{E}[Y])^2] \\
&= \mathbb{E}[(Y - \mu)^2] \\
&= \int_{-\infty}^{\infty} (y - \mu)^2 \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(y - \mu)^2}{2\sigma^2}} dy \\
&= \frac{1}{\sigma\sqrt{2\pi}} \int_{-\infty}^{\infty} (y - \mu)^2 e^{-\frac{(y - \mu)^2}{2\sigma^2}} dy \\
&\text{Let } z = \frac{y - \mu}{\sigma}, \text{ then } y - \mu = \sigma z \text{ and } dy = \sigma dz \\
&= \frac{1}{\sigma\sqrt{2\pi}} \int_{-\infty}^{\infty} (\sigma z)^2 e^{-\frac{(\sigma z)^2}{2\sigma^2}} \sigma dz \\
&= \frac{\sigma^3}{\sigma\sqrt{2\pi}} \int_{-\infty}^{\infty} z^2 e^{-\frac{z^2}{2}} dz \\
&= \frac{\sigma^2}{\sqrt{2\pi}} \int_{-\infty}^{\infty} z^2 e^{-\frac{z^2}{2}} dz \\
&\text{Using integration by parts: } u = z, dv = ze^{-z^2/2}dz \implies du = dz, v = -e^{-z^2/2} \\
&= \frac{\sigma^2}{\sqrt{2\pi}} \left( \left[-ze^{-z^2/2}\right]_{-\infty}^{\infty} - \int_{-\infty}^{\infty} -e^{-z^2/2} dz \right) \\
&= \frac{\sigma^2}{\sqrt{2\pi}} \left( 0 + \int_{-\infty}^{\infty} e^{-z^2/2} dz \right) \\
&= \frac{\sigma^2}{\sqrt{2\pi}} \sqrt{2\pi} \\
&= \sigma^2
\end{aligned}
$$
