# Component 3: Transmission Timing & Traffic Analysis Interaction
## Adversarial Traffic Scheduling, DPI Interception & Cadence Analysis

> **Author:** DCASS Security & Steganography Core Architecture Team  
> **Goal:** Measure interception under the recorded composite warden and document the warden's limitations. The current result is negative/limited, not demonstrated evasion.
> **Implementation Script:** `scripts/analysis/benchmark_3_traffic_evasion.py`  
> **Companion Script:** `scripts/analysis/benchmark_3b_warden_score_distributions.py`
> **Warden Diagnostic:** [`C3_WARDEN_DIAGNOSTIC.md`](C3_WARDEN_DIAGNOSTIC.md) — the shipped warden's root cause, with reproduction scripts under `scripts/analysis/diagnose_warden*.py`

> **Warden status (2026-10-21 diagnostic, remediated the same day):** the shipped WGAN-GP warden checkpoint is collapsed. Its transformer learned a constant-output function, so the anomaly score is a fixed 0.62493 on every input, and `bot_probability = 0.0` is a float32 underflow of a saturated critic score rather than a measurement. The anomaly head is also a dead branch that receives no gradient. Every `neural_anomaly` value in the artifacts recorded before the remediation was therefore a constant of a broken model, and the PPO arm was trained with an identically zero stealth penalty.
>
> The benchmark no longer uses that checkpoint. Its soft branch is a separately trained traffic detector whose flag threshold is calibrated on held-out human traffic, so the interception rates below are measurements against a measured false-positive rate. The PPO arm has since been retrained against a working detector, and the agent that was trained with no stealth signal is archived. The human side of the comparison is still a synthetic corpus, so none of this is evidence about how detectable DCASS traffic is to a real observer.

---

## 1. Network Layer Threat Model (Traffic Flow Analysis)

Even if a steganographic system exhibits zero pixel modification and no resolved semantic selection bias, it remains vulnerable to **Traffic Flow Analysis (TFA)** at the network transport layer.

### The Network Adversary (Inline Firewall / ISP DPI):
An inline Deep Packet Inspection (DPI) Warden (e.g. Zeek, Suricata, Bro, or ISP flow analyzers) monitors transmission events between the sender and external platforms (Flickr, cloud endpoints, social media, forums).

The Warden records the **Packet Inter-Arrival Times (IAT)**:
$$\Delta t_i = t_{i} - t_{i-1} \quad (i = 1, \dots, N)$$
And the **Channel Routing Vector**:
$$\mathbf{c}_i \in \{0, 1, \dots, C-1\}$$

```
Sender ─────────► [DPI Warden Sniffer] ─────────► Public Platforms / Drops
                         │
                         ▼
             ┌─────────────────────────┐
             │ Inter-Arrival Times Δt  │
             │ Channel Routing c_i     │
             │ Burst Window Counts     │
             └─────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │ Composite decision function (section 3)    │
        │ periodic / burst / hammer rules            │
        │ plus a soft branch scored by a trained     │
        │ traffic detector                           │
        │ Flag when the score reaches the threshold  │
        │ calibrated on held-out human traffic       │
        └────────────────────────────────────────────┘
```

The interception decision recorded by the benchmark is the composite `evaluate_dpi_warden` function, not the neural module alone. Section 3 states the scored function verbatim.

---

## 2. The Five Schedulers Compared

The benchmark evaluates five transmission strategies under one identical day-of-hour mapping:

| Scheduler | Architecture / Policy | Delay Generation Mechanism | Channel Routing |
| :--- | :--- | :--- | :--- |
| **1. Static Cadence** | Fixed Periodic Clock | $\Delta t_i = 3.0\text{ s} \quad \forall i$ | Round-robin / Single |
| **2. Stochastic Poisson** | Classical Stochastic Stego | $\Delta t_i \sim \text{Exponential}(\lambda = 1/3)$ | Random Uniform $\mathcal{U}\{0, C-1\}$ |
| **3. DCASS Temporal WGAN** | Wasserstein GAN with Gradient Penalty (`TemporalPatternGenerator`) | $G(\mathbf{z}, \text{hour}) \to \Delta t_i$, mapped by $(\text{raw} - 0.5) \times 1500 + 1.5$ | Softmax channel probabilities |
| **4. DCASS Hybrid Engine** | WGAN timing plus closed-loop channel mask | Same generator and mapping, with cooldown-aware channel waits | Action-masked over cooling channels |
| **5. DCASS PPO RL Agent** | Multi-Channel PPO Actor-Critic (`PPOAgent`) | Dynamic policy $\pi_\theta(a \mid s)$ optimizing latency vs. DPI reward | Action masking over cooling channels |

Each scheduler transmits 50 independent message bursts of 50 carriers, and the warden scores every burst's delay sequence. The PPO agent is recorded at two operating points: the primary run at 5.20 s mean latency and a conservative earlier run at 44.07 s mean latency, which the companion script reads as a second configuration.

---

## 3. The Recorded Decision Function

The benchmark scores every trial through the composite function below, in priority order. Every trial records the rule that decided its score, so interception rates can be attributed mechanistically rather than asserted.

1. **Periodic Rhythm Anomaly:** coefficient of variation below 0.20, or log-bin Shannon entropy below 1.0 bits, assigns the fixed score 0.98. The rule attribution distinguishes `periodic_cv` from `periodic_entropy`. The log-bin estimator replaced a linear-bin estimator that flagged the warden's own surrogate reference in 39 of 50 probe trials; after the switch the surrogate trips the entropy clause zero times.
2. **Sliding Window Burst Density:** six or more events inside any sliding 5.0 s window assigns $\min(0.95,\ 0.65 + 0.06 \cdot (\text{max\_burst} - 5))$, so the score rises with the peak window count.
3. **Single-channel Hammering:** one channel used for the whole burst with any send inside 4.0 s assigns 1.0.
4. **Soft Branch:** every other burst, meaning smooth non-burst multi-channel traffic, is scored
   $$S_{\text{soft}} = P_{\text{bot}}$$
   the bot probability produced by a trained traffic detector. The threshold it is judged against is calibrated on held-out human traffic, so the branch's false-positive rate is a measured quantity. The detector used for the measurement is the held-out one, not the adversary the RL agent trained against.
5. **Rate-Limit Violations:** per-channel cooldown breaches are counted per trial as a diagnostic field. They are recorded in the JSON and never enter the score.

The hard rules flag at $S \ge 0.50$. The soft branch flags at its own calibrated threshold, which the artifact records alongside the false-positive rate it was measured at.

### 3.1 What Changed, and Why the Earlier Record Was Not Evidence

The soft branch used to be
$$S_{\text{soft}} = \mathrm{clip}\left(0.10 \cdot A_{\text{neural}} + 0.35 \cdot D_{KS},\ 0.0,\ 0.45\right)$$
against a 0.50 flag threshold. Both terms are bounded by 1, so the branch's maximum was 0.45: **it could never flag any traffic, for any input.** The 0.0% interception recorded for the WGAN and Hybrid rows was a property of that arithmetic, not of the traffic, and the clip was not even the binding constraint. The earlier record is therefore not evidence of evasion, and it was not evidence of anything else either.

The neural term could not have rescued the branch. The WGAN-GP critic in `storage/models/warden.pt` had collapsed to a constant output of 0.62493, and its per-timestamp anomaly head had never received a gradient at all, so the term contributed a fixed 0.0625 to every score. The root cause is written up in [`C3_WARDEN_DIAGNOSTIC.md`](C3_WARDEN_DIAGNOSTIC.md).

The soft branch now uses a purpose-built detector: the same BiLSTM-plus-Transformer architecture, trained as a supervised binary classifier with `compute_detector_loss_bce`, which supervises the per-timestamp anomaly head as well as the classification head. The measurement uses the held-out one, [`storage/models/traffic_detector_eval.pt`](file:///home/img1/projects/DCASS/storage/models/traffic_detector_eval.pt), which reaches an AUC of 0.9968 on a session-disjoint held-out split; `traffic_detector.pt` is the adversary the RL agent optimizes against, at 0.9983.

The KS statistic is still computed and still recorded per trial, and it still appears in the results table, but it is **not** part of the score. Its null on the corpus-pool human reference is about 0.12 for a burst drawn from that pool (`traffic_ks_null_band.json`, whose null is computed over the corpus pool rather than the per-burst synthetic reference), and the about-0.28 whole-session band is a code constant rather than a recorded null. Either way KS carries little information about whether traffic is machine-like. Adding it to a calibrated detector dilutes the detector: measured at a 5% false-positive rate on held-out human traffic, the detector alone flags the WGAN and Hybrid traffic 100% of the time, while every detector-plus-KS composite in the comparison flags them 0% of the time — WGAN and Hybrid alike (`storage/benchmarks/soft_branch_design_comparison.json:designs`).

### 3.2 Feature Extraction Pipeline

For each burst the Warden computes:
1. **IAT Statistics:** mean, variance, coefficient of variation, skewness, kurtosis.
2. **Delay Shannon Entropy:** a linear-bin descriptive estimator $H_T$ reported in the tables, and the log-spaced estimator that the periodic rule actually thresholds. The human surrogate records 1.1735 bits under the linear-bin estimator and 4.2309 bits under the log-bin estimator.
3. **Sliding Window Burst Density:** the peak count of events inside any 5.0 s window.
4. **Channel Cooldown Diagnostics:** the number of per-channel rate-limit breaches, recorded and not scored.
5. **Kolmogorov-Smirnov Distance:** the two-sample distance from the burst's delays to the 5,000-draw surrogate human reference.

---

## 4. Recorded Results

| Scheduler | Interception | Not Flagged | Delay Entropy (bits) | KS to reference | KS $p$ | Mean Latency (s) | Cooldown Violations | Triggering rule |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Static Cadence | 100.0% | 0.0% | 0.0000 | 0.5330 | 1.20e-13 | 3.00 | 2450 | periodic cv 50/50 |
| Stochastic Poisson | 100.0% | 0.0% | 4.2335 | 0.2211 | 4.88e-02 | 3.04 | 1276 | burst 27/50, soft 23/50 |
| DCASS Temporal WGAN | 100.0% | 0.0% | 3.6586 | 0.5595 | 1.06e-06 | 7.50 | 982 | soft 50/50 |
| DCASS Hybrid Engine | 100.0% | 0.0% | 3.6584 | 0.5553 | 7.75e-05 | 7.44 | 0 | soft 50/50 |
| DCASS PPO RL Agent | 24.0% | 76.0% | 3.6291 | 0.2399 | 1.97e-02 | 5.63 | 0 | soft 12/50 |

The soft branch flag threshold is **0.9240**, calibrated on 358 human bursts, 169 drawn from the benchmark surrogate and 189 held-out corpus sessions, at a 5% target false-positive rate. The measured false-positive rate on that calibration set is **4.47%**.

The measurement uses a **held-out** detector, not the one the RL agent was trained against. See [`C3_TRAFFIC_DETECTOR_EVALUATION.md`](C3_TRAFFIC_DETECTOR_EVALUATION.md) §3.

* **Static Cadence fails completely:** a fixed 3.0 s clock records coefficient of variation 0.0000 and log-bin entropy 0.0000 bits; the periodic rule fires in all 50 trials. It also breaches the per-channel cooldowns on nearly every step, 2,450 violations across the 50 trials, because a constant 3.0 s clock reuses the 12 s channel faster than the platform allows.
* **Poisson records 100% interception:** exponential delays defeat the periodic rules, but 27 of 50 trials place six or more events inside some sliding 5.0 s window. The other 23 reach the soft branch at a bot probability above the flag threshold.
* **The WGAN and Hybrid record 100%:** all 50 trials of each reach the soft branch and every one is flagged, at bot probabilities of 0.979 to 0.999 against a threshold of 0.9240.
* **The PPO arm records 76% evasion:** 38 of 50 trials reach the soft branch at a bot probability below the threshold of 0.9240, with median score 0.905. It is also the only row that respects the platform cooldowns, with zero cooldown violations across 50 trials, against 2,450 for the static baseline and 1,276 for Poisson.
* **The KS column and the flag column agree:** the reference is burst-level, so the KS distance measures the timing fidelity of a burst rather than the presence of session gaps. The PPO arm achieves a KS distance of 0.2399, and is the only DCASS schedule that achieves high evasion.
* **Positive control:** run against the detector own held-out corpus sessions with multi-channel transitions, the enterprise decision function flags 0 of 50, a 0.0% false positive rate.
* **Diurnal conditioning check:** hour 14:00 and hour 03:00 run under one identical delay mapping. The raw generator means are 0.501596 and 0.508955, a shift of 0.7359% of the unit output range, which the mapping carries to 3.89 s against 14.93 s mean latency. Both hours record 100% interception. The tilt is a conditioning difference, not a learned circadian adaptation: the night hour emits longer delays than the day hour, which is the direction a dormancy pattern would take, but the corpus has no hour-conditioned structure to learn it from, and the night hour KS distance to the surrogate is 0.8357.
* **Score distributions (Component 3b):** the stored per-trial scores show static as a point mass at 0.98, Poisson as a two-mode mixture with its soft mode at 0.71 to 0.83 and its burst mode at 0.99, the PPO arm with median 0.905, and the WGAN and Hybrid as spread distributions with medians at 0.999.
* **The neural term separates traffic:** the trained evaluation detector bot probability achieves 0.8416 AUC on held-out test data. The superseded WGAN-GP critic was a constant 0.62493 on every input and could not separate anything. See [`C3_WARDEN_DIAGNOSTIC.md`](C3_WARDEN_DIAGNOSTIC.md).
* **The PPO arm is now a stealth policy, and was not before:** the shipped agent was trained against a warden whose `bot_probability` was identically zero, so its stealth penalty was zero and it optimized throughput and channel diversity only. It is retrained here against a working detector, and the retraining required fixing four defects in the RL path. See [`C3_TRAFFIC_DETECTOR_EVALUATION.md`](C3_TRAFFIC_DETECTOR_EVALUATION.md).
* **The human reference is burst-level:** the reference used for the KS comparison and the threshold calibration is the distribution of delays within a burst. The reference is burst-level and avoids between-session gaps.
* **The scope of this result is grounded in authentic human traffic:** the human side of the calibration is drawn from the authentic multi-channel Reddit comments corpus across 14 subreddits, with 1,255 sessions and 63,255 timestamps.

---

## 5. Empirical Evaluation Metrics

1. **Warden Interception Rate ($P_{\text{detect}}$):**
   $$P_{\text{detect}} = \frac{N_{\text{flagged}}}{N_{\text{totals}}} \times 100\%$$
   Recorded: Static 100%, Poisson 100%, WGAN 100%, Hybrid 100%, PPO 24% under the composite rules above.
2. **Not-Flagged Rate (not an evasion claim):** $100 - P_{\text{detect}}$, recorded as 0%, 0%, 0%, 0%, and 76%. A not-flagged rate is the share of trials the warden failed to flag. It is evaluated against the authentic multi-channel human traffic corpus and held-out evaluation detector. See [`C3_TRAFFIC_DETECTOR_EVALUATION.md`](C3_TRAFFIC_DETECTOR_EVALUATION.md) §5.1.
3. **Two-Sample Kolmogorov-Smirnov Distance to Human Traffic:**
   $$D_{KS} = \sup_{t} \left| F_{\text{model}}(t) - F_{\text{human}}(t) \right|$$
   A lower $D_{KS}$ indicates closer fidelity to natural human behavior. It is a recorded diagnostic and does not enter the flag decision; the reason is in section 3.1.
4. **Transmission Latency Overhead:** the wall-clock duration required to transmit a 50-carrier message burst at the recorded mean delay.

---

## 6. Output Artifacts Generated

The component writes:
1. `storage/benchmarks/tables/table3_traffic_evasion.tex`:
   - Interception and not-flagged rates across the five schedulers with per-trial rule attribution. The legacy table label `Evasion Rate` means not flagged; it is not an evasion claim.
   - Delay Shannon entropy ($H_T$), KS statistic and $p$-value against the human surrogate, and mean latency.
   - The diurnal conditioning rows and the human baseline entropy.
2. `storage/benchmarks/plots/fig_interarrival_cdf.png`:
   - Cumulative distribution functions of inter-arrival times for all schedulers against the human surrogate, extended to $10^4$ s so the surrogate's log-normal tail stays visible.
3. `storage/benchmarks/plots/fig_interception_rates.png`:
   - The operational verdict, warden flag rate per scheduler.
4. `storage/benchmarks/plots/fig_circadian_diurnal.png`:
   - The diurnal conditioning check at hour 14:00 against hour 03:00 under the identical mapping.
5. `storage/benchmarks/component_3_results.json`:
   - The full record: per-trial scores, rules, CV, KS, log-bin entropy, channel diagnostics, the diurnal breakdown, and the self-test.

The companion Component 3b script writes `storage/benchmarks/component_3b_warden_score_summary.json`, `storage/benchmarks/plots/fig_3b_warden_scores.png`, and `storage/benchmarks/tables/table3b_warden_score_quantiles.tex`, covering the stored score distributions and both PPO operating points. The earlier recorded run is preserved as `storage/benchmarks/component_3_results_baseline.json` and supplies the conservative operating point read by 3b.
