# data/

## `human_traffic.json` (GAN training data, Phase A)

Expected by `scripts/stealth/train_gan.py --data <path>` (default:
`data/human_traffic.json`). Format:

```json
[
  {
    "delays": [5.2, 3.1, 12.4, ...],
    "channels": [0, 1, 0, 2, ...],
    "time_of_day": 14
  },
  ...
]
```

Each entry is one "session" — a sequence of inter-transmission delays
(seconds) and the channel used for each transmission, plus the hour of day
(0-23) the session started.

### Where this comes from

- **Real capture (preferred for a paper result):** instrument a real social
  app / messaging usage log and extract `(delay, channel, hour)` tuples per
  session. Not provided in this repo — bring your own.
- **Synthetic fallback (works out of the box):** if the file at `--data`
  doesn't exist, `HumanTrafficDataset._generate_synthetic_data()`
  ([src/stealth/gan/trainer.py](../src/stealth/gan/trainer.py)) generates
  1000 synthetic sessions with Poisson-like exponential delays and random
  channel switching. This is enough to exercise and test the full pipeline
  (see `docs/GAN_RL_INTEGRATION_PLAN.md`), but a Warden trained only on this
  synthetic distribution is learning to distinguish "exponential delays"
  from the Generator's output — not real human behavior. Treat results
  trained only on the synthetic fallback as a pipeline sanity check, not a
  paper claim about real traffic.
- `docker-compose.yml`'s `dcass-gen-traffic` service calls
  `scripts/stealth/generate_traffic_dataset.py --num-sessions ... --output
  /app/data/behavioral/human_traffic.json`, but **that script does not
  exist in this tree** (same class of gap as the missing `train_gan.py` /
  `train_rl.py` this integration pass added — it just wasn't in scope here).
  Until it's written, `docker compose --profile training run
  dcass-gen-traffic` will fail; use the synthetic fallback above instead, or
  write your own generator matching the JSON schema at the top of this file.

### Train/validation split

`train_gan.py --val-fraction 0.1` (default) holds out 10% of the dataset
(via `HumanTrafficDataset.split`, seeded for reproducibility) for
`scripts/stealth/eval_warden.py`'s held-out AUC measurement. Use the same
`--data` and `--val-fraction` for both scripts so the split lines up.
