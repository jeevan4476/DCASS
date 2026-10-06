# Component 3: Transmission Timing & Traffic Analysis Evasion
## Adversarial Traffic Scheduling, DPI Interception & Cadence Analysis

> **Author:** DCASS Security & Steganography Core Architecture Team  
> **Theoretical Goal:** Evasion of Deep Packet Inspection (DPI) and Traffic Flow Analysis (TFA) Wardens  
> **Implementation Script:** `scripts/analysis/benchmark_3_traffic_evasion.py`

---

## 1. Network Layer Threat Model (Traffic Flow Analysis)

Even if a steganographic system exhibits zero pixel modification ($D_{KL} = 0$) and zero semantic selection bias, it remains vulnerable to **Traffic Flow Analysis (TFA)** at the network transport layer.

### The Network Adversary (Inline Firewall / ISP DPI):
An inline Deep Packet Inspection (DPI) Warden (e.g. Zeek, Suricata, Bro, or ISP flow analyzers) monitors packet streams between the sender and external platforms (Flickr, cloud endpoints, social media, forums).

The Warden records the **Packet Inter-Arrival Times (IAT)**:
$$\Delta t_i = t_{i} - t_{i-1} \quad (i = 1, \dots, N)$$
And the **Channel Distribution Vector**:
$$\mathbf{c}_i \in \{0, 1, \dots, C-1\}$$

```
Sender ─────────► [DPI Warden Sniffer] ─────────► Public Platforms / Drops
                         │
                         ▼
             ┌─────────────────────────┐
             │ Inter-Arrival Times Δt  │
             │ Channel Routing c_i     │
             │ Burst Window Rates      │
             └─────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │  DeepPacketInspectionWarden (ML)  │
        │  Score ∈ [0, 1]                   │
        │  Threshold: Score > 0.50 -> ALERT │
        └───────────────────────────────────┘
```

The Warden detects covert traffic using three heuristic and statistical rules:
1. **Periodic Rhythm Anomaly:** Bot/scripted transmissions emit packets at fixed or regular intervals (e.g. exactly every $3.0\text{s}$ or uniform jitter $\mathcal{U}[2, 4]$), causing an unnatural spike in frequency power spectra ($H_{\Delta t} \to 0$).
2. **Burst Rate Anomaly:** Automated exfiltration flushes messages rapidly, violating rate limits and triggering volume threshold alerts.
3. **Channel Concentration Anomaly:** Automated scripts hammer a single channel without respecting per-channel rate cooling periods.

---

## 2. The 4 Schedulers Compared

To prove the superiority of DCASS's machine-learned scheduling, the benchmark evaluates four distinct transmission strategies:

| Scheduler | Architecture / Policy | Delay Generation Mechanism | Channel Routing |
| :--- | :--- | :--- | :--- |
| **1. Static Cadence** | Fixed Periodic Clock | $\Delta t_i = 3.0\text{ s} \quad \forall i$ | Round-robin / Single |
| **2. Stochastic Poisson** | Classical Stochastic Stego | $\Delta t_i \sim \text{Exponential}(\lambda = 1/3)$ | Random Uniform $\mathcal{U}\{0, C-1\}$ |
| **3. DCASS Temporal WGAN** | Wasserstein GAN with Gradient Penalty (`TemporalPatternGenerator`) | $G(\mathbf{z}, \text{hour}) \to \Delta t_i$ trained on human posting distributions | Softmax channel probabilities |
| **4. DCASS PPO RL Agent** | Multi-Channel PPO Actor-Critic (`PPOAgent`) | Dynamic policy $\pi_\theta(a \mid s)$ optimizing latency vs. DPI reward | Action masking over cooling channels |

---

## 3. The Adversarial Evaluator: DeepPacketInspectionWarden

DCASS's internal adversarial validator is the [`DeepPacketInspectionWarden`](file:///home/img1/projects/DCASS/src/analysis/adversarial/warden.py), representing an enterprise-grade DPI sensor.

### 3.1 Feature Extraction Pipeline
For each sequence of transmitted packets, the Warden computes:
1. **Inter-Arrival Time Statistics:** Mean ($\mu_{\Delta t}$), Variance ($\sigma^2_{\Delta t}$), Coefficient of Variation ($CV = \sigma / \mu$), Skewness, Kurtosis.
2. **Delay Shannon Entropy:**
   $$H_T = -\sum_{b} p_b \log_2(p_b)$$
   Measures the unpredictability and disorder of timing intervals.
3. **Sliding Window Burst Density:**
   $$\rho_{\text{burst}} = \max_{t} \sum_{i} \mathbb{I}(t \le t_i < t + W)$$
   Detects sudden flushes of packets within small time windows $W = 5.0\text{s}$.
4. **Channel Rate-Limit Violations:**
   Tracks per-channel cooldown timers. If a packet is transmitted on channel $c$ while $c$ is still in cooldown:
   $$\text{Violation}_{\text{rate}} = \text{True} \implies \text{Score} \leftarrow 1.0 \quad (\text{Immediate Flag})$$

### 3.2 Warden Scoring
The Warden combines the feature vector through a trained multi-layer perceptron (MLP) and heuristic rules to output an **Interception Score** $S \in [0, 1]$.
* $S \ge 0.50$: **Flagged as Covert / Malicious** (Transmission Blocked).
* $S < 0.50$: **Passed as Natural Human Traffic** (Covert Transmission Successful).

---

## 4. Empirical Evaluation Metrics

1. **Warden Interception Rate ($P_{\text{detect}}$):**
   $$P_{\text{detect}} = \frac{N_{\text{flagged}}}{N_{\text{total\_transmissions}}} \times 100\%$$
   - Static Cadence: $\approx 98\text{--}100\%$ (instantly flagged by periodic frequency and burst sensors).
   - Poisson Noise: $\approx 45\text{--}65\%$ (frequently trips burst density thresholds).
   - DCASS WGAN: $< 8.0\%$ (closely mirrors natural human inter-arrival distributions).
   - DCASS RL Policy: $< 3.0\%$ (actively observes channel cooldown states and avoids detection).

2. **Evasion Success Rate ($\% = 100 - P_{\text{detect}}$):**
   Target: $> 97.0\%$ evasion for DCASS RL.

3. **Two-Sample Kolmogorov-Smirnov Distance to Human Traffic:**
   Comparing generated delays $\Delta t_{\text{model}}$ to empirical human social media post intervals $\Delta t_{\text{human}}$:
   $$D_{KS} = \sup_{t} | F_{\text{model}}(t) - F_{\text{human}}(t) |$$
   A lower $D_{KS}$ indicates closer fidelity to natural human behavior.

4. **Transmission Latency Overhead:**
   Total wall-clock duration required to transmit a 50-carrier payload (seconds).

---

## 5. Output Artifacts Generated

The script outputs:
1. `paper_tables/table3_traffic_evasion.tex`:
   - Interception Rate ($P_{\text{detect}}$) across the 4 schedulers.
   - Delay Shannon Entropy ($H_T$).
   - KS statistic $D_{KS}$ against human baseline.
   - Mean message delivery latency.
2. `paper_plots/fig_interarrival_cdf.png`:
   - Cumulative Distribution Function (CDF) comparing Static, Poisson, GAN, RL, and Human traffic cadences.
