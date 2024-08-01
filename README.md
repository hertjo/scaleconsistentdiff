# Scale-Consistent Diffusion Modeling via Renormalization Group Principles

This repository explores generative modelling that applies **Renormalization Group (RG)** principles to the **diffusion modeling** framework. I aimed to create a generative process that is explicitly structured around scale-consistent transformations, leveraging RG-inspired hierarchical transformations to improve both theoretical interpretability and practical performance.

### 1. RG Perspective on Generative Modeling

In the Renormalization Group formalism, a physical system described by a high-dimensional variable $ x \in \mathbb{R}^n $ is simplified via successive transformations:

$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \dots \rightarrow x_T,
$$

where each $ x_t $ is a coarser-scale representation obtained through some RG transformation $ R_t $, typically integrating out high-frequency or small-scale degrees of freedom.

In this model, we reinterpret each diffusion timestep as a discrete **scale** in an RG hierarchy, not just a point in a noise schedule. The diffusion process is thus coupled with explicit **rescaling** and **filtering**, rather than merely additive Gaussian noise.

---

### 2. Forward Process: Scale-Resolved Coarse-Graining

Let $ x_0 \sim p_{\text{data}} $ be a data sample (e.g., an image). The forward process generates a sequence of latent representations $ \{x_t\}_{t=1}^T $ via recursive operations:

$$
x_{t+1} = R_t(x_t + \sqrt{\beta_t} \, \epsilon_t), \quad \epsilon_t \sim \mathcal{N}(0, I),
$$

where:

- $ R_t: \mathbb{R}^{n_t} \rightarrow \mathbb{R}^{n_{t+1}} $ is a **renormalization operator**, e.g., block averaging or learnable pooling.
- $ \beta_t $ governs the scale-specific noise level.

Each $ R_t $ implements a map that combines local noise averaging with **spatial coarse-graining**. This transforms the diffusion process from a simple SDE discretization into a **multiscale stochastic flow**.

The cumulative transformation over $ T $ steps defines a forward marginal $ q(x_T \mid x_0) $, which can be analytically or numerically approximated.

---

### 3. Reverse Process: Multiscale Reconstruction

The reverse process attempts to invert this coarse-to-fine flow:

$$
x_t = R_t^{-1}(x_{t+1} - \sqrt{\beta_t} \, \hat{\epsilon}_\theta(x_{t+1}, t)),
$$

where:

- $ \hat{\epsilon}_\theta $ predicts the noise component at each scale $ t $.
- $ R_t^{-1} $ is a learned or structured **upsampling operator** that reconstructs higher-resolution features.

Notably, this process is **not** symmetric to the forward process in a strict sense; instead, it is trained to invert the effect of both stochastic noise and scale compression.

The model effectively learns a sequence of *scale-lifting maps*, which resemble inverse RG flows but are regularized by denoising constraints.

---

### 4. Effective Field Interpretation and Flow Equations

Consider a parameterized effective model $ H_t(x_t) $ at scale $ t $. Each RG step modifies the effective energy landscape:

$$
H_{t+1}(x_{t+1}) = -\log \int \delta(x_{t+1} - R_t(x_t + \sqrt{\beta_t}\epsilon_t)) \, e^{-H_t(x_t)} \, dx_t.
$$

This resembles a **Fokker-Planck flow** followed by a **projection onto coarser variables**. The goal is to capture the evolution of system statistics under such scale transformations.

In the continuous limit (for small $ \beta_t $), this becomes a scale-dependent diffusion equation:

$$
\frac{\partial p(x, \lambda)}{\partial \lambda} = \nabla \cdot \left( D(\lambda) \nabla p(x, \lambda) + F(x, \lambda) p(x, \lambda) \right),
$$

with $ \lambda \sim \log(\text{scale}) $ as the renormalization depth, and $ D, F $ acting as scale-dependent diffusion and drift terms.

---

### 5. Scale-Separated Noise and Frequency Modulation

Unlike classical DDPMs where noise is introduced isotropically at each step, RGDM injects noise **separately at different scales**:

- High-frequency noise is added in early steps (small $ t $), and its effect is suppressed by downsampling.
- Low-frequency noise is added in later steps, targeting global structure.

This scale-localized noise structure can be interpreted in the Fourier domain. Let $ \mathcal{F}[x] $ denote the Fourier transform of $ x $. Then the forward process approximates:

$$
\mathcal{F}[x_{t+1}] \approx \mathcal{F}[R_t(x_t)] + \sqrt{\beta_t} \, \eta_t(\omega), \quad \text{supp}(\eta_t) \subset \Omega_t,
$$

where $ \Omega_t $ defines a band-limited frequency region. The RGDM can thus be viewed as a structured bandpass noise process, akin to a learnable **wavelet cascade**.

---

### 6. Score Estimation and Training Objective

Each step estimates a score function:

$$
s_\theta(x_t, t) \approx \nabla_{x_t} \log q(x_t \mid x_0),
$$

but since $ x_t $ includes both noise and downsampling, $ q(x_t \mid x_0) $ is **scale-marginalized**, and hence not available in closed form.

Instead, we train a noise predictor $ \hat{\epsilon}_\theta $ via:

$$
\mathcal{L}_{\text{RGDM}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \left\| \epsilon - \hat{\epsilon}_\theta(x_t, t) \right\|^2 \right],
$$

where $ x_t $ is generated using the scale-aware forward process. The model is implicitly learning the score of the scale-marginalized latent distribution.

---

## Implications and Future Directions

The proposed RGDM framework suggests a number of theoretical and empirical benefits:

- **Improved generalization** via inductive bias for scale hierarchy.
- **Physical interpretability** through the analogy to RG flow equations.
- **Architectural modularity**, enabling the use of different neural operators per scale.
- **Spectral sparsity**, reducing redundancy by isolating frequency bands during training.

Future work may explore:

- Fully continuous RG diffusion flows via SDEs with scale time.
- Hamiltonian or Lagrangian formulations of the generative process.
- Integration with wavelet or scattering transforms for theoretical grounding.

---
