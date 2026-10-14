# Component 5: API Concurrency & System Stress
## Multi-Threaded Load Testing, Per-Session State Isolation & Latency Percentiles

> **Author:** DCASS Security & Steganography Core Architecture Team  
> **Theoretical Goal:** Verify that the FastAPI endpoints and scheduling pipelines survive high-load concurrent production workloads with zero state corruption.  
> **Implementation Script:** `scripts/analysis/benchmark_5_system_concurrency.py`

---

## 1. Concurrency Architecture & Thread-Safety

In a distributed multi-agent deployment, multiple sender and receiver agents interact with the DCASS node simultaneously.

### 1.1 Historic Concurrency Bottlenecks (Identified & Resolved in Tier-1)
1. **Global Transmission Lock:**
   * *Problem:* A single boolean flag `_transmission_active` blocked all concurrent `/transmit` requests, causing HTTP 409 errors.
   * *Fix:* Replaced with per-session state mapping:
     $$\mathcal{S}_{\text{active}}: \text{session\_id} \to \{\text{thread}, \text{status}, \text{packet\_count}, \text{recipient}\}$$
2. **Scheduler Non-Reentrancy:**
   * *Problem:* `StealthScheduler` mutated `_rl_agent.env` and called `_rl_agent.buffer.clear()` in place. Concurrent requests would race on the RL rollout state, corrupting generated delays.
   * *Fix:* Added `threading.Lock()` and serialized `_schedule_unlocked()`, isolating rollout episodes under a critical section.
3. **Wire Metadata Leaks in Packet Names:**
   * *Problem:* Early versions named packet files `session_id__media_id_channel_seq.json`, leaking transmission sequence and channel routing to anyone observing the file system.
   * *Fix:* Filenames are strictly opaque `session_id__media_id.json`. Sequence numbers, channel indices, and delays are stored **only inside encrypted/authenticated JSON payloads**.

---

## 2. Multi-Threaded Stress Test Methodology

```
                   ┌───────────────────────────────────────────────┐
                   │    Multi-Threaded Workload Generator (Pool)   │
                   │    (10, 20, 50 Concurrent Worker Threads)     │
                   └───────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
           Session 1 (Client A)   Session 2 (Client B)   Session N (Client N)
            Key: K_1, Msg: M_1     Key: K_2, Msg: M_2     Key: K_N, Msg: M_N
                    │                      │                      │
                    ▼                      ▼                      ▼
           POST /api/encode       POST /api/encode       POST /api/encode
           POST /api/transmit     POST /api/transmit     POST /api/transmit
           POST /api/decode       POST /api/decode       POST /api/decode
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           ▼
                   ┌───────────────────────────────────────────────┐
                   │      Verification & Isolation Analysis        │
                   │  • 100% Correct Plaintext Reconstruction      │
                   │  • Zero Session Cross-Talk or State Leakage   │
                   │  • Latency Distribution (p50, p95, p99)       │
                   └───────────────────────────────────────────────┘
```

The benchmark uses Python's `concurrent.futures.ThreadPoolExecutor` to launch concurrent client sessions against the local API engine.

### Test Matrix:
* **Concurrency Levels:** 10, 20, and 50 simultaneous worker threads.
* **Payloads:** Random UTF-8 secret messages generated per worker.
* **Keys:** Unique 32-byte cryptographic session keys generated per worker (`os.urandom(32)`).
* **Endpoints Tested:**
  1. `/encode` (DSSC mode, multi-modal balanced candidate selection).
  2. `/transmit` (Asynchronous threaded transmission with scheduler delays).
  3. `/decode` (DSSC mode, session-keyed state inversion).

---

## 3. Quantitative Performance Metrics

1. **Throughput (Requests Per Second — RPS):**
   $$\text{RPS} = \frac{N_{\text{total\_requests}}}{\Delta t_{\text{total\_wallclock}}}$$
2. **Latency Percentiles ($p_{50}, p_{95}, p_{99}$):**
   - $p_{50}$ (Median): Expected $< 25\text{ms}$ per request.
   - $p_{95}$: Expected $< 75\text{ms}$.
   - $p_{99}$: Expected $< 120\text{ms}$.
3. **Session Cross-Talk Rate:**
   $$\text{CrossTalkRate} = \frac{N_{\text{corrupted\_or\_swapped\_payloads}}}{N_{\text{total\_sessions}}} \equiv 0.000\%$$
   Every client must recover their own message with 100.0% verification; no client may receive another client's plaintext.
4. **Error Rate:**
   $$\text{ErrorRate} = \frac{N_{\text{failed\_HTTP\_requests}}}{N_{\text{total\_requests}}} \equiv 0.000\%$$

---

## 4. Output Artifacts Generated

The script outputs:
1. `paper_tables/table5_system_concurrency.tex`:
   - Concurrency level ($10, 20, 50$ threads).
   - Throughput (RPS).
   - $p_{50}, p_{95}, p_{99}$ latency (ms).
   - Success rate ($\% = 100.0\%$).
2. `paper_plots/fig_latency_percentiles.png`:
   - Histogram and cumulative latency curves under increasing concurrent load.
