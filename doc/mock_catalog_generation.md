# Lognormal Galaxy Mock Catalog Generation

In this note we detail the approach taken to generate a lognormal galaxy mock catalog, given a power spectrum and linear galaxy bias. Velocities are computed to first order.

The goal is first to describe how to obtain a realization of the matter density field and of the galaxy density field. From these, the drawing of a mock catalog is straightforward, and so will not be described in detail here.

In brief, the number of galaxies in a grid cell is sampled from a Poisson distribution with average number density $(1 + \delta_g)\,\bar n$, and the displacement field is given by
$$\vec\Psi = i \frac{\vec k}{k^2}\,\delta_m(\vec k)$$
assuming vanishing vorticity. Then, the velocity field is
$$\vec v = faH\,\vec\Psi,$$
and the displacement due to RSD is
$$\Delta\vec r=\frac{\hat r\cdot\vec v}{aH}\,\hat r=f\,(\hat r\cdot\vec\Psi)\,\hat r.$$

The goal, now, is to generate a realization for $\delta_m$ and $\delta_g$. We start with the simplest method, and work ourselves up to more complex ways of generating $\delta_m$ and $\delta_g$.

Our first task is to draw a random Gaussian field, and then to add power to it in Fourier space, first for the matter field $\delta_m$, then for the galaxy field $\delta_g$.

## Drawing phases

We first draw a Gaussian random field of unit variance
$$G(\vec k) \sim \mathcal{N_C}(0, 1),$$
where $\mathcal{N_C}$ denotes the complex Gaussian distribution such that when $X=a+ib\sim\mathcal{N_C}(\mu,\sigma^2)$, then both $a$ and $b$ are drawn from a normal distribution with mean $\mu$ and variance $\sigma^2$.

We can reduce variance by choosing fixed phases. That is, by only keeping the phase and not the amplitude. This is often called *fixed* simulations.


## Drawing $\delta_m$

To convert $G(\vec k)$ into the matter density field, we have two options: either we keep the field Gaussian, or we convert it to a lognormal field.

### Gaussian $\delta_m$
Keeping it Gaussian means that
$$\delta_m(\vec k) = \sigma_m(k)\,G(\vec k),$$
where $\sigma^2_m(k)=V P_m(k)/(2\pi)^3$.

### Lognormal $\delta_m$
Alternatively, we can make it lognormal,
$$G_m(\vec k) = \sigma_{m,G}(k)\,G(\vec k),$$
$$\delta_m(\vec r) = \exp\!\big[G_m(\vec r)-\tfrac12\sigma_{m,G}^2(0)\big] - 1,$$
where $\sigma_{m,G}^2(k)=VP^G_m(k)/(2\pi)^3$, and $P^G_m(k)$ is the power spectrum of the Gaussian field calculated from $P_m(k)$.


## Drawing $\delta_g$

Using the same phases $G(\vec k)$, we can now construct the galaxy density field in several ways: linear galaxy bias, separate lognormal distribution, or lognormal linear galaxy bias.

### Linear Galaxy Bias $\delta_g$
The simplest model is to use the definition of the linear galaxy bias,
$$\delta_g(\vec k) = b\,\delta_m(\vec k),$$
where $b$ is the linear galaxy bias.

### Lognormal Field $\delta_g$
Alternatively, we can calculate the Gaussian power spectrum $P^G_g(k)$ from the galaxy power spectrum $P_g(k)=b^2 P_m(k)$, and construct the galaxy density contrast via
$$G_g(\vec k) = \sigma_{g,G}(k)\,G(\vec k),$$
$$\delta_g(\vec r) = \exp\!\big[G_g(\vec r)-\tfrac12\sigma_{g,G}^2(0)\big] - 1.$$
This should give a faithful reconstruction of the input power spectrum, at the expense of an unclear relation between the matter and galaxy density contrasts.

### Linear Lognormal Galaxy Bias $\delta_g$
To satisfy our need for consistency, a different bias relation is proposed that is linear for the Gaussian field $G(\vec k)$, but includes higher-order terms for density contrast itself. That is, we use
$$G_g(\vec k) = b_G\,G_m(\vec k),$$
$$\delta_g(\vec r) = \exp\!\big[G_g(\vec r)-\tfrac12\sigma_{g,G}^2(0)\big] - 1.$$