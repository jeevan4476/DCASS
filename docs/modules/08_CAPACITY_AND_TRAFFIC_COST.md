# 08 · Capacity & Traffic Cost of the Exact Channel

> Audit item M-18: the honest cost of losslessness must be stated alongside
> the stealth claims, because they pull against each other.

## The core trade-off

`exact_vcp` buys **exact recovery** by spending **one carrier per payload byte**.
Every byte is a separate media item - a separate observable event on the wire.
The mode that guarantees message integrity is therefore also the mode that
generates the most traffic per secret bit.

## Capacity math

| Quantity | Value |
|---|---|
| Payload unit | 1 byte per media item |
| Reed-Solomon parity | 8 bytes (default) → corrects up to t = 4 byte errors |
| Message `"Meet me at the cafe at noon"` | 26 bytes |
| + parity | 34 bytes |
| **Media items transmitted** | **34** |
| Observable events on the wire | 34 posts/packets (+ injected idle gaps) |

General formula:

```
carriers = ceil(message_utf8_bytes * (data + parity) / data)
         = len(utf8(message)) + parity_bytes          # RS appends parity 1:1
events   = carriers + idle_gap_count
duration ≈ sum(delays)  ≈ carriers * base_delay + gaps
```

With the default `base_delay=3s`, that 26-character message takes roughly
100+ seconds of traffic to deliver. A 1 KiB document would take ~1030 media
items and ~52 minutes at the same profile.

## Stealth implications

1. **Detection surface scales with message length.** The warden observes a
   time series; longer messages give it more samples. Short covert messages
   are the intended use; bulk transfer is out of scope for this channel.
2. **ECC doubles effective error resilience but adds fixed overhead** of 8
   carriers per message regardless of length - negligible for sentences,
   meaningful for very short payloads.
3. **Idle gaps and jitter add cover but also duration.** The NoiseController's
   human-like pauses stretch total transmission time, increasing the window
   in which an observer can correlate endpoints.
4. **Dynamic context keys (`ContextKeyManager`) do not change capacity**, only
   the mapping security. Traffic volume is identical with or without keying.

## Design guidance

- Keep messages sentence-sized (< ~200 bytes). Above that, the carrier count
  itself becomes an anomaly signal regardless of timing camouflage.
- Prefer higher `base_delay` profiles for longer messages; prefer shorter
  messages over aggressive timing tricks when possible.
- Report detection results as a function of all three variables -
  carriers per message, delay distribution, and warden training size - not as
  a single headline number.

## What to claim (and what not to)

✅ "Exact reconstruction within RS correction capacity (t=4 byte errors)"
✅ "Carrier files are never modified - content-based steganalysis cannot apply"
✅ "Traffic mimicked via [static jitter / GAN / PPO] scheduling"

❌ "Perfectly secure" or "Zero Detection" - see module 06 for scoped language
❌ Any throughput claim without stating carriers-per-byte cost
