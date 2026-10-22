# Component 5: Codec Concurrency & System Stress
## Multi-Threaded Encode/Decode Load Testing, Latency Percentiles & Payload Isolation

> **Author:** DCASS Security & Steganography Core Architecture Team  
> **Theoretical Goal:** Measure direct-engine DSSC encode/decode concurrency and payload isolation under multithreaded load, and disclose the API transmit and wire-metadata limitations that this benchmark does not exercise.
> **Implementation Script:** `scripts/analysis/benchmark_5_system_concurrency.py`

The filename retains the original scope label. The recorded benchmark is a direct-engine codec test. It calls `SemanticEngine.encode` and `SemanticEngine.decode`; it does not call the FastAPI routes, `/api/transmit`, `StealthScheduler`, or the API session map.

---

## 1. Concurrency-Relevant Code Paths and Current Status

In a distributed deployment, multiple sender and receiver agents can interact with a DCASS node simultaneously. The code paths below matter for that claim, but not all of them are covered by this benchmark.

### 1.1 Current Status

1. **API transmission admission is bounded, not exclusive.**
   * *Current behavior:* `src/api/server.py::_claim_transmission` admits a session while fewer than `MAX_CONCURRENT_TRANSMISSIONS` are active, and `/api/transmit` returns HTTP 429 once that bound is reached. The default bound is 4 and it is overridable with the `DCASS_MAX_CONCURRENT_TRANSMISSIONS` environment variable. The sessions are independent: each writes its packets under its own session-id-prefixed filename and its own manifest, so two sessions to the same recipient do not collide. The per-session map tracks `active`, `stop_requested`, and `progress` for status and stop handling.
   * *Scope:* The benchmark does not call `/api/transmit`, so the 100 percent success and zero-swap figures below do not testify about concurrent API transmissions.

2. **Scheduler locking is per instance.**
   * *Current behavior:* `StealthScheduler` has a `threading.Lock()` and serializes `_schedule_unlocked()`. The API creates one scheduler per transmit request.
   * *Scope:* The benchmark does not exercise the scheduler, so concurrent scheduler behavior is not measured here.

3. **Packet filenames are opaque; packet bodies are plaintext.**
   * *Current behavior:* Filenames are `session_id__media_id.json` and omit sequence and channel. The packet JSON body contains `channel_id`, `sequence_number`, `delay_seconds`, `timestamp`, `mode_used`, and `session_id`; the manifest contains the plaintext `message`.
   * *Scope:* A filesystem observer can read the JSON metadata and manifest even though the filename is opaque. This is a disclosed limitation, not a benchmark result.

4. **The engine singleton is initialized under a lock; encode and decode calls are not globally serialized.**
   * *Current behavior:* `src/api/server.py` guards lazy engine creation with `_engine_lock`. The benchmark calls the same engine object directly from worker threads.

---

## 2. Direct-Engine Stress Test Methodology

```
ThreadPoolExecutor (10, 20, 50 worker threads)
  -> SemanticEngine.encode(secret, mode="dssc", session_key=key)
  -> SemanticEngine.decode(media_ids, mode="dssc", session_key=key)
  -> compare reconstructed_message to the session's own secret
```

The benchmark uses Python's `concurrent.futures.ThreadPoolExecutor`. It does not launch HTTP clients and does not call the FastAPI application.

### Test Matrix

* **Concurrency levels:** 10, 20, and 50 worker threads.
* **Sessions per level:** 100. The same 100 session identifiers are reused at each level, so the run records 300 sessions in total.
* **Payloads:** Fixed marker messages of the form `SECRET_PAYLOAD_FOR_CLIENT_<id>_VERIFY_ISOLATION!`.
* **Keys:** 32-byte keys derived as `SHA-256("session_key_<id>")`. They are unique per session identifier and deterministic across levels.
* **Calls:** `SemanticEngine.encode` and `.decode` in DSSC mode with ECC parity 8 and image, text, and audio modalities.
* **Latency scope:** The timer starts inside the worker after the task begins. It excludes queue wait, HTTP, scheduler, and transmit delays.
* **Cross-talk check:** A decoded payload counts as a swap only when it differs from the session's own secret and contains the shared `SECRET_PAYLOAD` marker.
* **Not exercised:** FastAPI `/api/encode`, `/api/transmit`, `/api/decode`, `/api/inbox/{session_id}/decode`, `StealthScheduler`, the API session map, and packet-file isolation.

---

## 3. Quantitative Performance Metrics

1. **Throughput (sessions per second):**
   $$\text{Sessions/s} = \frac{N_{\text{completed sessions}}}{\Delta t_{\text{total\_wallclock}}}$$
   Each session performs one encode and one decode, so this is not an HTTP request rate.

2. **Latency percentiles ($p_{50}, p_{90}, p_{95}, p_{99}$):**
   - In-task encode-plus-decode service time in milliseconds.
   - The 100-session sample makes $p_{99}$ close to the maximum. Treat it as a tail indicator, not a precise population percentile.
   - The recorded run gives $p_{50}=4975.6$ ms and $p_{99}=5978.9$ ms at 10 threads, and $p_{50}=24367.5$ ms and $p_{99}=27948.6$ ms at 50 threads.
   - The earlier sub-100 ms end-to-end target is not met under this workload. The growth is queueing and contention, not codec arithmetic.

3. **Observed payload swap rate:**
   $$\text{SwapRate} = \frac{N_{\text{swapped marker payloads}}}{N_{\text{sessions}}}$$
   The recorded run reports 0.000 percent at all three levels. Corrupted payloads without the marker count as failures, not swaps.

4. **Decode success rate:**
   $$\text{SuccessRate} = \frac{N_{\text{exact plaintext reconstructions}}}{N_{\text{sessions}}}$$
   The recorded run reports 100.0 percent at all three levels. This is a decode-correctness check, not an HTTP error rate.

5. **Stage attribution caveat:**
   - The FAISS row is computed as full encode time minus ECC framing and permutation time.
   - The centroid row is computed as full decode time minus ECC syndrome and inverse permutation time.
   - These are residual attributions, not direct instrumentation of the FAISS call.

---

## 4. Output Artifacts Generated

The script writes:

1. `storage/benchmarks/tables/table5_system_performance.tex`:
   - Part A: payload size, carrier count, encode and decode time, total latency, encode rate, carrier speed.
   - Part B: worker threads, total sessions, sessions per second, $p_{50}$, $p_{90}$, $p_{95}$, $p_{99}$, success rate, observed swap rate.
2. `storage/benchmarks/plots/fig5_latency_breakdown.png`:
   - Stacked encode and decode stage breakdown per payload size.
3. `storage/benchmarks/plots/fig5_concurrency_scaling.png`:
   - Throughput and latency percentile curves under increasing worker-thread load.
4. `storage/benchmarks/component_5_results.json`:
   - Schema `component5-results-v2`.
   - `provenance`, `metric_definitions`, `payload_scaling`, and `concurrency_stress`.
   - The artifact carries a full provenance block recording the commit, the timestamp, the device, and the package versions. New runs record the git commit, timestamp, torch version, device, GPU name, platform, script, and command.

The report block in `BENCHMARKING_REPORT.md` Section 7.3 is generated from the table file and must remain byte-identical to it. The script can regenerate the table and both figures without rerunning the benchmark:

```bash
.venv/bin/python scripts/analysis/benchmark_5_system_concurrency.py --artifacts-only
```

---

## 5. Scope and Disclosed Limitations

1. **Direct-engine only.** The benchmark does not exercise the FastAPI layer, HTTP status handling, or the API session map.
2. **Bounded API transmit.** The server admits several concurrent `/api/transmit` sessions up to `MAX_CONCURRENT_TRANSMISSIONS` and returns 429 beyond that bound. The benchmark does not test that path.
3. **Plaintext packet bodies.** Filenames are opaque, but packet metadata and the manifest message are readable on disk. This needs a separate security fix if the threat model includes filesystem observers.
4. **Narrow swap metric.** The cross-talk check only detects a different payload that carries the shared marker. It does not test wrong-key rejection, cross-session decode by media identifier, or inbox isolation.
5. **In-task latency.** The percentiles exclude queue wait and API overhead. They are not client-observed end-to-end latencies.
6. **Small tail sample.** Each configuration records 100 sessions. The $p_{99}$ estimate is close to the maximum and should not be over-interpreted.
7. **Residual stage attribution.** The FAISS and centroid rows are inferred by subtraction, not measured at the call site.
