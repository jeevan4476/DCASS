# Timestamp GAN

Generates `n` randomised, realistic-looking timestamps using a GAN (WGAN-GP) trained on a
synthetic event log. The generator learns *when* events tend to happen (business-hour peaks,
lunch dip, quiet nights, quieter weekends); the noise fed into it supplies the randomness, so
every run gives fresh, non-repeating timestamps.

## Setup

Uses the virtual environment at `C:\Users\2024591\OneDrive\projects\.venv` (has torch, numpy,
pandas already installed). Either activate it once per shell:

```powershell
& "C:\Users\2024591\OneDrive\projects\.venv\Scripts\Activate.ps1"
cd "C:\Users\2024591\OneDrive\projects\gan"
python gan.py ...
```

or call the venv's python directly without activating:

```powershell
& "C:\Users\2024591\OneDrive\projects\.venv\Scripts\python.exe" "C:\Users\2024591\OneDrive\projects\gan\gan.py" ...
```

The examples below assume you're `cd`'d into the project folder and use `python` — substitute
the full venv path if you haven't activated it.

## Quick start

```powershell
python gan.py train              # train once, writes gan_ckpt.pt (~1 min on GPU)
python gan.py sample -n 20        # generate 20 timestamps
```

`sample` auto-trains a checkpoint on first use if `gan_ckpt.pt` doesn't exist yet, so a fresh
clone works with just the `sample` command.

## Commands

### `train` — train the GAN

```powershell
python gan.py train [--steps 3000] [--samples 20000] [--batch 256] [--n-critic 5] [--lr 1e-4] [--seed 0] [--ckpt gan_ckpt.pt]
```

Trains on a synthesized "real" event log and saves weights to `gan_ckpt.pt`. Re-run any time you
want to retrain (e.g. with more `--steps` for a sharper fit).

### `sample` — generate n timestamps

```powershell
python gan.py sample -n <N> [--start DATE] [--end DATE] [--csv out.csv] [--seed N] [--ckpt gan_ckpt.pt]
```

| Flag | Default | Meaning |
|---|---|---|
| `-n` | *required* | how many timestamps to generate |
| `--start` | `2026-09-01` | inclusive start of the window |
| `--end` | `2026-10-01` | exclusive end of the window |
| `--csv` | — | also write results to this CSV file |
| `--seed` | random | fix for reproducible output |

The window can be any size — a full year, a single day, or a single hour — and results always
follow the learned time-of-day/weekday shape within it.

**Examples**

```powershell
# 20 timestamps in the default month-long window
python gan.py sample -n 20

# 10 timestamps on one specific day
python gan.py sample -n 10 --start 2026-09-15 --end 2026-09-16

# 10 timestamps inside a single hour (e.g. 2pm-3pm)
python gan.py sample -n 10 --start "2026-09-15 14:00:00" --end "2026-09-15 15:00:00"

# write 100 timestamps to a CSV, reproducibly
python gan.py sample -n 100 --csv out.csv --seed 42
```

### `eval` — check the learned distribution

```powershell
python gan.py eval [-n 20000] [--ckpt gan_ckpt.pt]
```

Prints ASCII histograms comparing the real (synthetic) data vs. generated timestamps by hour-of-day
and weekday, plus a total variation distance (0 = identical distributions). Use this after
training to sanity-check the GAN actually learned the pattern.

## Using it from Python

```python
from gan import generate_timestamps

timestamps = generate_timestamps(20, start="2026-09-15 14:00:00", end="2026-09-15 15:00:00")
```

Returns a sorted `pandas.DatetimeIndex`. Trains a checkpoint automatically on first use if none
exists.

## How it works (short version)

1. A synthetic "real" event log is generated with realistic rhythm: weekday peaks around
   10:00/15:00 with a lunch dip, near-silent nights, and lower/later weekend activity.
2. Each timestamp is encoded as points on two unit circles — one for position-in-week, one for
   time-of-day — so the model can represent both weekday and hourly rhythm at the same scale.
3. A WGAN-GP (Wasserstein GAN with gradient penalty) trains a generator to produce points on
   those circles matching the real distribution.
4. `sample` decodes generator output into (weekday, time-of-day), picks an actual calendar date
   from your requested range that matches, and uses rejection sampling (redrawing rejects) so
   even a narrow window (e.g. one hour) gets timestamps distributed by the learned time-of-day
   shape instead of being clamped to the window's edges.

## Troubleshooting

- **`RuntimeError: could not draw enough timestamps in this window`** — the requested window is
  extremely narrow or falls in an almost-never-occurring part of the pattern (e.g. a few seconds
  at 3am). Widen the window slightly.
- **VSCode shows import warnings for torch/numpy/pandas** — cosmetic; VSCode is pointed at the
  system Python instead of the venv. Set the interpreter to
  `C:\Users\2024591\OneDrive\projects\.venv\Scripts\python.exe`.
- **Want to retrain from scratch** — delete `gan_ckpt.pt` and re-run `train` or `sample`.
