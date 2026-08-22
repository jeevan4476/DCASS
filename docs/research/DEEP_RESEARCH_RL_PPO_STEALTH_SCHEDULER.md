# Deep Research Specification: PPO Adaptive Stealth Scheduler (Reinforcement Learning)
---

## 1. Executive Summary & Problem Rationale

### 1.1 Why Reinforcement Learning in Addition to GAN?
The WGAN-GP generator synthesizes open-loop, realistic human timing profiles based on historical training distributions. However, a live operational network environment is **dynamic, adversarial, and non-stationary**:

1. **Active Warden Behavior**: The perimeter monitor may dynamically increase inspection sensitivity when suspicious bursts occur.
2. **Channel Rate Limits**: Egress platforms (e.g., social media APIs, forum endpoints) enforce dynamic rate-limiting thresholds (e.g., max 10 posts/minute); exceeding these triggers immediate account throttling or IP banning.
3. **Queue Backlog Pressure**: Alice's outbound message queue fluctuates in real-time. A purely slow stealth profile will create infinite queue latency, while a rushed transmission risks detection.

```
       OPEN-LOOP (GAN) vs. CLOSED-LOOP (RL) CONTROL
┌────────────────────────────────────────────────────────────────────────┐
│ Open-Loop Scheduling (WGAN-GP Only)                                    │
│   • Generates static schedule upfront: [Δt_1, Δt_2, ..., Δt_N]         │
│   • Ignores real-time channel feedback or dynamic warden alert states. │
│   • Risk: Channel gets rate-limited mid-session -> Packets drop!       │
├────────────────────────────────────────────────────────────────────────┤
│ Closed-Loop Adaptive Control (PPO Reinforcement Learning)              │
│   • Continuously observes State s_t (Warden score, queue, rate limits) │
│   • Selects Action a_t = (Delay Δt, Channel k) dynamically per packet. │
│   • Maximizes throughput while maintaining Warden suspicion below      │
│     alert thresholds.                                                  │
└────────────────────────────────────────────────────────────────────────┘
```

The **PPO Adaptive Stealth Scheduler** implements a closed-loop Markov Decision Process (MDP) agent that dynamically balances the fundamental trade-off: **Transmission Throughput vs. Evasion Probability**.

---

## 2. Markov Decision Process (MDP) Formulation

The transmission scheduling problem is formulated as an infinite-horizon discounted Markov Decision Process $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.

### 2.1 State Space ($\mathcal{S} \in \mathbb{R}^{16}$)
At decision step $t$, the environment presents a 16-dimensional continuous state vector $\mathbf{s}_t$:

| State Index | Feature Name | Range | Description |
| :---: | :--- | :---: | :--- |
| `0` | `queue_ratio` | $[0.0, 1.0]$ | Ratio of remaining queue size to maximum capacity: $Q_t / Q_{\max}$. |
| `1` | `time_sin` | $[-1.0, 1.0]$ | Sinusoidal diurnal encoding: $\sin(2\pi h / 24)$. |
| `2` | `time_cos` | $[-1.0, 1.0]$ | Cosine diurnal encoding: $\cos(2\pi h / 24)$. |
| `3..5` | `channel_availability` | $[0.0, 1.0]^3$ | Rate-limit availability across channels $k \in \{0, 1, 2\}$. |
| `6..8` | `channel_usage_ratio` | $[0.0, 1.0]^3$ | Proportion of recent packets sent on each channel. |
| `9` | `recent_warden_score` | $[0.0, 1.0]$ | Rolling average suspicion score from the Warden over the last 20 packets. |
| `10` | `warden_trend` | $[-1.0, 1.0]$ | Derivative of Warden suspicion ($\Delta W = W_t - W_{t-1}$). |
| `11` | `mean_recent_delay` | $[0.0, 1.0]$ | Normalized average inter-packet delay over the last 5 steps. |
| `12` | `delay_variance` | $[0.0, 1.0]$ | Variance of recent delays (measures burstiness). |
| `13` | `session_progress` | $[0.0, 1.0]$ | Elapsed episode time relative to maximum time: $t_{\text{elapsed}} / T_{\max}$. |
| `14..15`| `channel_congestion` | $[0.0, 1.0]^2$ | Estimated network jitter/drop rate on primary links. |

---

### 2.2 Action Space ($\mathcal{A}$)
The agent executes a hybrid action $\mathbf{a}_t = (\Delta t_t, k_t)$:
1. **Continuous Delay ($\Delta t_t \in [0.5, 30.0]$ s)**: The sleep interval before transmitting the next packet.
2. **Discrete Channel ($k_t \in \{0, 1, \dots, K-1\}$)**: The egress distribution channel selected for dispatch.

---

### 2.3 Multi-Objective Reward Function ($R_t$)
The reward function mathematically aligns the agent's policy with operational goals:

$$R_t = w_{\text{tp}} \cdot r_{\text{throughput}} - w_{\text{stealth}} \cdot r_{\text{warden}} - w_{\text{burst}} \cdot r_{\text{burst}} - r_{\text{penalty}}$$

Where:
1. **Throughput Reward ($r_{\text{throughput}}$)**:
   $$r_{\text{throughput}} = \frac{1}{\Delta t_t} \quad (\text{Encourages lower latency and faster message delivery})$$
2. **Warden Suspicion Penalty ($r_{\text{warden}}$)**:
   $$r_{\text{warden}} = \left( \text{WardenScore}(t) \right)^2 \quad (\text{Quadratic penalty strongly penalizing high-risk actions})$$
3. **Burstiness Penalty ($r_{\text{burst}}$)**:
   $$r_{\text{burst}} = |\Delta t_t - \bar{\Delta t}| \quad (\text{Penalizes erratic unnatural timing deviations})$$
4. **Hard Constraint Penalties ($r_{\text{penalty}}$)**:
   - Rate-limit violation: $-10.0$ penalty if the selected channel is currently throttled.
   - Detection tripwire: $-100.0$ terminal penalty if Warden suspicion exceeds $0.85$.

---

## 3. Proximal Policy Optimization (PPO) Mathematical Framework

DCASS utilizes **PPO with Generalized Advantage Estimation (GAE)**, implemented in [`src/stealth/rl/agent.py`](file:///home/jeevan/projects/DCASS/src/stealth/rl/agent.py).

```
                      PPO ACTOR-CRITIC AGENT ARCHITECTURE
                                State Vector s_t ∈ R^16
                                          │
                                          ▼
                      ┌─────────────────────────────────────────┐
                      │ Shared Feature Extractor MLP            │
                      │ Linear(16 → 256) ──► LayerNorm ──► ELU  │
                      │ Linear(256 → 256) ──► LayerNorm ──► ELU │
                      └─────────────────────────────────────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
        ┌─────────────────────────┐ ┌───────────┐ ┌─────────────────────────┐
        │ Continuous Delay Actor  │ │ Discrete  │ │ Value Critic Head       │
        │ Linear(256 → 2)         │ │ Channel   │ │ Linear(256 → 1)         │
        │ μ_delay, σ_delay        │ │ Head      │ │ V(s_t) ∈ R              │
        │ (Gaussian Distribution) │ │ Logits k  │ │ (State Value Function)  │
        └─────────────────────────┘ └───────────┘ └─────────────────────────┘
```

### 3.1 Probability Ratio & Clipped Surrogate Objective
Let $r_t(\theta)$ be the probability ratio between the new policy $\pi_\theta$ and the old policy $\pi_{\theta_{\text{old}}}$:

$$r_t(\theta) = \frac{\pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t)}{\pi_{\theta_{\text{old}}}(\mathbf{a}_t \mid \mathbf{s}_t)} = \frac{\pi_\theta^{\text{delay}}(\Delta t_t \mid \mathbf{s}_t) \cdot \pi_\theta^{\text{channel}}(k_t \mid \mathbf{s}_t)}{\pi_{\theta_{\text{old}}}^{\text{delay}}(\Delta t_t \mid \mathbf{s}_t) \cdot \pi_{\theta_{\text{old}}}^{\text{channel}}(k_t \mid \mathbf{s}_t)}$$

To prevent destructively large policy updates, PPO clips the surrogate objective:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Where $\epsilon = 0.2$ is the clipping hyperparameter.

---

### 3.2 Generalized Advantage Estimation (GAE-$\lambda$)
Advantages $\hat{A}_t$ are computed using temporal difference residuals $\delta_t^V$:

$$\delta_t^V = R_t + \gamma V_\phi(\mathbf{s}_{t+1}) - V_\phi(\mathbf{s}_t)$$

$$\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

Where:
- $\gamma = 0.99$ (discount factor).
- $\lambda = 0.95$ (GAE bias-variance trade-off parameter).

---

### 3.3 Total Multi-Task Loss Function
The agent optimizes the combined loss:

$$L^{\text{TOTAL}}(\theta, \phi) = - L^{\text{CLIP}}(\theta) + c_1 \cdot L^{\text{VF}}(\phi) - c_2 \cdot \mathcal{H}(\pi_\theta)$$

Where:
- $L^{\text{VF}}(\phi) = \frac{1}{2} \mathbb{E}_t \left[ (V_\phi(\mathbf{s}_t) - \hat{R}_t)^2 \right]$ (Critic mean-squared error loss, $c_1 = 0.5$).
- $\mathcal{H}(\pi_\theta) = \mathcal{H}(\pi^{\text{delay}}) + \mathcal{H}(\pi^{\text{channel}})$ (Entropy bonus promoting exploration, $c_2 = 0.01$).

---

## 4. Simulation Environment: `StealthEnvironment`

Implemented in [`src/stealth/rl/environment.py`](file:///home/jeevan/projects/DCASS/src/stealth/rl/environment.py), the environment simulates realistic multi-channel egress constraints and active Warden surveillance:

```python
# Stepping the Environment
def step(self, action: dict) -> Tuple[np.ndarray, float, bool, dict]:
    delay = float(np.clip(action["delay"], 0.5, 30.0))
    channel_id = int(action["channel"])
    
    # 1. Update simulation clock
    self.current_time += delay
    
    # 2. Check channel rate limits
    channel = self.channels[channel_id]
    rate_violated = not channel.can_send(self.current_time)
    
    # 3. Query Warden detection critic on recent packet window (W = 20)
    warden_suspicion = self.warden.evaluate(self.recent_delays, self.recent_channels)
    
    # 4. Compute composite reward
    reward = (1.0 / delay) - self.lambda_stealth * (warden_suspicion ** 2)
    if rate_violated:
        reward -= 10.0
        
    # 5. Check termination
    done = (len(self.queue) == 0) or (self.current_time >= self.max_episode_time)
    
    return self._get_state(), reward, done, {"warden_score": warden_suspicion}
```

---

## 5. Integration into `StealthScheduler`

When `StealthScheduler.schedule(media_ids, mode='rl')` is invoked in [`src/stealth/stealth_scheduler.py`](file:///home/jeevan/projects/DCASS/src/stealth/stealth_scheduler.py):

```python
def _schedule_rl(self, media_ids: list[str], base_delay: float, checkpoint: Optional[Path]) -> dict:
    agent = self._load_rl_agent(checkpoint)
    env = StealthEnvironment(num_channels=self.num_channels, warden=self._warden)
    
    state = env.reset(media_sequence=media_ids)
    delays, channels = [], []
    done = False
    
    while not done:
        action, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        
        delays.append(action["delay"])
        channels.append(action["channel"])
        state = next_state
        
    return {
        "items": media_ids,
        "delays": delays,
        "channels": channels,
        "mode_used": "rl"
    }
```

If the RL agent checkpoint (`storage/models/rl_agent.pt`) is not present, `StealthScheduler` automatically falls back to `mode='gan'` or `mode='static'` seamlessly.
