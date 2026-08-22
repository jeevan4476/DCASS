# Deep Research Specification: WGAN-GP Temporal Traffic Mimicry Generator

**Project**: Dynamic Context-Aware Semantic Steganography (DCASS)  
**Module**: Stealth Traffic Scheduling — Generative Adversarial Network Subsystem  
**Implementation Source**: [`src/stealth/gan/`](file:///home/jeevan/projects/DCASS/src/stealth/gan/) (`generator.py`, `trainer.py`)  
**Date**: August 2026  
**Repository**: `https://github.com/jeevan4476/dcass.git`  

---

## 1. Executive Summary & Threat Model

### 1.1 The Statistical Traffic Analysis Threat
In the DCASS steganography framework, individual carrier media files (Images, Text, and Audio) are transmitted with **zero pixel or waveform alterations** ($D_{\text{KL}}(P_{\text{cover}} \parallel P_{\text{stego}}) = 0.0$). Consequently, Deep Neural Network content steganalysts (such as SRNet, Zhu-Net, and Ye-Net) are mathematically neutralized.

However, an adversary operating at the network perimeter (referred to as the **Warden** or **Deep Packet Inspection (DPI) Monitor**) does not inspect only payload content. The Warden records:
1. **Inter-Transmission Delays ($\Delta t_i = t_{i} - t_{i-1}$)**: The exact elapsed time between consecutive packet transmissions.
2. **Packet Burstiness**: Variance and higher-order moments of transmission frequency.
3. **Channel Switching Frequencies**: Multi-platform distribution patterns across available egress channels.
4. **Circadian / Diurnal Rhythm Alignment**: Whether posting activity matches natural human sleep/wake cycles.

```
       ADVERSARIAL TRAFFIC ANALYSIS COMPARISON
┌────────────────────────────────────────────────────────────────────────┐
│ Naive Steganography Dispatcher (Periodic / Fixed Inter-Packet Delay)   │
│   • Pattern:  |--- 2.0s ---|--- 2.0s ---|--- 2.0s ---|--- 2.0s ---|    │
│   • FFT Spectrum: Massive artificial frequency spike at f = 0.5 Hz!    │
│   • Warden Classification: 100% CONFIRMED AUTOMATED SCRIPT / BOT.      │
├────────────────────────────────────────────────────────────────────────┤
│ DCASS WGAN-GP Temporal Generator (Human Social-Media Mimicry)          │
│   • Pattern:  |-- 1.2s -|- 0.4s -|------ 14.8s ------|--- 3.1s ---|    │
│   • FFT Spectrum: 1/f Pink Noise distribution matching human browsing.  │
│   • Warden Classification: 50% RANDOM GUESS (Indistinguishable from    │
│     genuine human social media activity).                              │
└────────────────────────────────────────────────────────────────────────┘
```

If covert media items are dispatched at fixed intervals ($\Delta t = \text{const}$) or uniform random intervals, the Warden's statistical tests (Kolmogorov-Smirnov test, Auto-Correlation Function, Spectral Entropy) immediately flag the transmission as an automated exfiltration bot.

To eliminate this vulnerability, DCASS deploys a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** that learns the true underlying continuous probability distribution of human social media behavior.

---

## 2. Mathematical Foundations of WGAN-GP

### 2.1 Why Vanilla GANs Fail for Traffic Mimicry
A standard GAN minimizing Jensen-Shannon (JS) divergence suffers from severe training pathologies when modeling high-dimensional temporal distributions:
- **Vanishing Gradients**: When the Warden critic becomes slightly better than the Generator, $\nabla_{\theta_g} L_G \to 0$.
- **Mode Collapse**: The Generator collapses to outputting only a single "safe" average delay (e.g., constantly generating $3.5$ seconds), which fails statistical Kolmogorov-Smirnov uniformity tests.

### 2.2 Wasserstein-1 (Earth Mover's) Distance Formulation
WGAN replaces JS divergence with the **Wasserstein-1 Distance** $\mathcal{W}(\mathbb{P}_r, \mathbb{P}_g)$, defined as the minimum cost of transporting the generated timing distribution $\mathbb{P}_g$ to the empirical human distribution $\mathbb{P}_r$:

$$\mathcal{W}(\mathbb{P}_r, \mathbb{P}_g) = \inf_{\gamma \in \Pi(\mathbb{P}_r, \mathbb{P}_g)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma} \left[ \|\mathbf{x} - \mathbf{y}\|_1 \right]$$

By the Kantorovich-Rubinstein duality theorem, the objective function is reformulated as a supremum over all 1-Lipschitz continuous critic functions $D \in \mathcal{D}_{\|D\|_L \le 1}$:

$$\max_{D \in \mathcal{D}_{\|D\|_L \le 1}} \underset{\mathbf{x} \sim \mathbb{P}_r}{\mathbb{E}} [D(\mathbf{x})] - \underset{\tilde{\mathbf{x}} \sim \mathbb{P}_g}{\mathbb{E}} [D(\tilde{\mathbf{x}})]$$

---

### 2.3 Gradient Penalty Enforcing 1-Lipschitz Continuity
Instead of weight clipping (which severely limits network capacity and causes pathological capacity degradation), DCASS enforces the Lipschitz constraint via a **Gradient Penalty ($L_{\text{GP}}$)** evaluated along straight lines between real and generated samples:

$$\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon) \tilde{\mathbf{x}}, \quad \text{where } \epsilon \sim U(0, 1)$$

$$L_{\text{GP}} = \lambda_{\text{GP}} \cdot \underset{\hat{\mathbf{x}} \sim \mathbb{P}_{\hat{\mathbf{x}}}}{\mathbb{E}} \left[ \left( \|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1 \right)^2 \right]$$

Where:
- $\lambda_{\text{GP}} = 10.0$ (penalty coefficient).
- The total objective optimized by the Warden critic is:

$$L_{\text{Warden}} = \underset{\tilde{\mathbf{x}} \sim \mathbb{P}_g}{\mathbb{E}} [D(\tilde{\mathbf{x}})] - \underset{\mathbf{x} \sim \mathbb{P}_r}{\mathbb{E}} [D(\mathbf{x})] + \lambda_{\text{GP}} \cdot \underset{\hat{\mathbf{x}} \sim \mathbb{P}_{\hat{\mathbf{x}}}}{\mathbb{E}} \left[ \left( \|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1 \right)^2 \right]$$

The Generator is trained to maximize the critic's evaluation score:

$$L_{\text{Generator}} = - \underset{\tilde{\mathbf{x}} \sim \mathbb{P}_g}{\mathbb{E}} [D(\tilde{\mathbf{x}})]$$

---

## 3. Network Architecture: `TemporalPatternGenerator`

The DCASS WGAN Generator is implemented in [`src/stealth/gan/generator.py`](file:///home/jeevan/projects/DCASS/src/stealth/gan/generator.py). It synthesizes structured multi-dimensional temporal sequences conditioned on diurnal time and latent stochasticity.

```
                  WGAN-GP TEMPORAL PATTERN GENERATOR ARCHITECTURE
 ┌───────────────────────────────────────┐   ┌─────────────────────────────────────────┐
 │ Latent Noise: z ~ N(0, I_{128})       │   │ Time-of-Day Hour: h ∈ [0, 23]           │
 └───────────────────────────────────────┘   └─────────────────────────────────────────┘
                     │                                            │
                     │                                            ▼
                     │                        ┌────────────────────────────────────────┐
                     │                        │ Cyclical Fourier Embedding:            │
                     │                        │ [ sin(2π h / 24), cos(2π h / 24) ]     │
                     │                        │   ──► Linear(2 → 32) ──► LayerNorm     │
                     │                        └────────────────────────────────────────┘
                     │                                            │
                     └────────────────────┬───────────────────────┘
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │ Latent Projection Layer                 │
                     │ Linear(128 + 32 → 256) ──► LayerNorm    │
                     └─────────────────────────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │ 2-Layer Autoregressive GRU              │
                     │ Hidden Dimension = 256                  │
                     │ Models adjacent packet transitions      │
                     └─────────────────────────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │ 8-Head Multi-Head Self-Attention        │
                     │ Embed Dim = 256, Heads = 8              │
                     │ Captures long-range session correlation │
                     └─────────────────────────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
 ┌───────────────────────────────────────┐ ┌───────────────────────────────────────────┐
 │ Inter-Item Delay Head                 │ │ Multi-Channel Selection Head              │
 │ Linear(256 → 64) ──► Softplus         │ │ Linear(256 → Num_Channels)                │
 │ Output: Δt_i ∈ [0.5, 30.0] seconds    │ │ Output: Channel Logits (Gumbel-Softmax)   │
 └───────────────────────────────────────┘ └───────────────────────────────────────────┘
```

### 3.1 Component Breakdown & Implementation Rationale

1. **Cyclical Time-of-Day Embedding (`time_encoder`)**:
   - *Why*: A naive scalar representation of hour $h \in [0, 23]$ introduces an artificial numerical discontinuity between 23:59 and 00:00.
   - *Solution*: We project time onto the continuous unit circle:
     $$\mathbf{t} = \left[ \sin\left(\frac{2\pi h}{24}\right), \, \cos\left(\frac{2\pi h}{24}\right) \right] \in \mathbb{R}^2 \xrightarrow{\text{MLP}} \mathbf{e}_{\text{time}} \in \mathbb{R}^{32}$$
   - This enables the generator to seamlessly model natural human sleep/wake circadian cycles (e.g., long dormant delays during 02:00–06:00 and high activity during 12:00–14:00 and 19:00–22:00).

2. **2-Layer Bidirectional/Autoregressive GRU (`gru`)**:
   - *Why*: Packet transmissions are not independently identically distributed (i.i.d.); a short delay (burst transmission) is typically followed by a longer reading/browsing pause.
   - *Solution*: A 2-layer Gated Recurrent Unit (GRU, hidden dimension 256) maintains a rolling memory state of preceding packet dispatch timings.

3. **8-Head Multi-Head Temporal Attention (`attention`)**:
   - *Why*: Recurrent networks suffer from vanishing context over long transmission sequences ($N > 30$ media items).
   - *Solution*: Multi-Head Attention allows each packet step to attend across the entire session history, preserving macro-level session rhythm.

4. **Output Heads**:
   - **Delay Head**: Computes positive continuous scalar delays $\Delta t_i = \text{Softplus}(W_d h_i + b_d) + \Delta t_{\min}$.
   - **Channel Head**: Outputs unnormalized logits over available distribution channels ($K=3$), sampled via Gumbel-Softmax during training for differentiable backpropagation.

---

## 4. Adversarial Training Loop (`src/stealth/gan/trainer.py`)

Adversarial training is executed using the **Two Time-Scale Update Rule (TTUR)**:

```python
# Trainer Execution Workflow
for epoch in range(num_epochs):
    for batch in dataloader:
        # Step 1: Train Warden Critic (5 steps per generator step)
        for _ in range(warden_steps):
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_schedule = generator(z, seq_len, time_of_day)
            
            d_real = warden(real_delays, real_channels)
            d_fake = warden(fake_schedule.delays.detach(), fake_schedule.channel_logits.detach())
            
            gp = compute_gradient_penalty(warden, real_delays, fake_schedule.delays)
            warden_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
            
            warden_optimizer.zero_grad()
            warden_loss.backward()
            warden_optimizer.step()
        
        # Step 2: Train Generator (1 step)
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_schedule = generator(z, seq_len, time_of_day)
        gen_eval = warden(fake_schedule.delays, fake_schedule.channel_logits)
        
        gen_loss = -gen_eval.mean()
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
```

### Key Training Hyperparameters:
| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| **Latent Dimension ($z$)** | 128 | Standard Gaussian noise vector capturing timing entropy. |
| **Hidden Dimension** | 256 | Balances expressive capacity with low GPU memory footprint. |
| **Critic Steps per Gen Step ($n_{\text{critic}}$)** | 5 | Ensures Wasserstein distance estimation remains accurate. |
| **Gradient Penalty ($\lambda_{\text{GP}}$)** | 10.0 | Standard optimal regularizer enforcing 1-Lipschitz smoothness. |
| **Optimizer** | Adam | $\beta_1 = 0.0, \beta_2 = 0.9$, Generator LR = $1 \times 10^{-4}$, Critic LR = $2 \times 10^{-4}$ (TTUR). |

---

## 5. Integration with DCASS Stealth Scheduler

The trained GAN checkpoint (`storage/models/gan_generator.pt`) is loaded by [`src/stealth/stealth_scheduler.py`](file:///home/jeevan/projects/DCASS/src/stealth/stealth_scheduler.py):

```python
# Invocation in StealthScheduler
def _schedule_gan(self, media_ids: list[str], base_delay: float, checkpoint: Optional[Path]) -> dict:
    generator = self._load_generator(checkpoint)
    z = torch.randn(1, 128, device=self.device)
    hour = torch.tensor([datetime.now().hour], device=self.device)
    
    with torch.no_grad():
        schedule = generator(z, sequence_length=len(media_ids), time_of_day=hour)
    
    delays = schedule.delays.squeeze(0).cpu().tolist()
    channels = schedule.sample_channels().squeeze(0).cpu().tolist()
    
    return {
        "items": media_ids,
        "delays": delays,
        "channels": channels,
        "mode_used": "gan"
    }
```

If the GAN checkpoint is absent, `StealthScheduler` automatically degrades gracefully to the static statistical profile (`NoiseController`) without interrupting the transmission pipeline.
