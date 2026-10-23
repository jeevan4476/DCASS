# 2026-10-23 post-remediation audit (re-run)

Two-analyst audit — one **research analyst** (claims verified against artifacts) and one
**technical analyst** (code behavior at HEAD) — re-run after the original /tmp-only reports were
lost in a restart cleanup. Everything here was re-derived against the current tree.

- **State at audit:** clean tree, HEAD `0afd9a9`. **Line references are valid at this commit** and
  may drift as approved fixes land.
- **Method:** every number recomputed from its artifact during the session; claims whose inputs
  were not stored are recorded as *unverifiable*, never corrected by guess; frozen artifacts were
  not rewritten; both analysts were read-only, no training runs.
- **Verdict schemes:** research = S / MF / Q / UB with P0–P2; technical = CONFIRMED / REFUTED /
  RISK / DESIGN with P0–P2.

## Files

| File | Contents |
|---|---|
| `SYNTHESIS.md` | Merged findings, batch mapping (B3 / B2.5 / B4), open-item list |
| `ganrl-c3-technical.md` | 6 standing candidates (all CONFIRMED; the RL tail-pad upgraded to P1 via `/api/transmit` reachability), new findings N1–N7, proposed FIX-A…FIX-I |
| `ganrl-c3-research.md` | 44 claim rows (S 30 / UB 1 / MF 3 / Q 10); P1 = regenerate `C3_GENERATOR_ADVERSARY.md` §8 from `critic_sensitivity_final_run.json` |
| `api-codec-infra-research.md` | 28 rows; the only P0 cluster (QUICKSTART/SCRIPTS command defects), JWT mechanism, `REAL_TRAFFIC_CAPTURE.md` corpus-table mixture |
| `semantic-c2-research.md` | 17 rows; C2 code follow-ups **closed** (no P0/P1); 2 MF doc fixes remain |

Read `SYNTHESIS.md` first. These reports document and propose; only explicitly approved batches
turn their recommendations into commits.
