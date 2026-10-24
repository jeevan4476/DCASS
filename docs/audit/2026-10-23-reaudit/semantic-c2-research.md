# Semantic / C2 Research Audit (report 1 of 3)

**Date:** 2026-10-23
**Auditor role:** RESEARCH ANALYST (quantitative/factual verification only; code behavior is the TECHNICAL ANALYST's lane)

## Git state header

Verified at audit start: HEAD `0afd9a9`; six unpushed commits ahead of the published baseline `28ff7d3` (`b89f321`, `79eda5e`, `c37f73c`, `0b4a7eb`, `a5c0620`, `0afd9a9`); working tree clean.

## Scope

- Primary doc: `docs/benchmarking/02_SEMANTIC_SELECTION_BIAS.md`, plus the C2-facing claims in `docs/benchmarking/BENCHMARK_DEEP_EXPLANATION.md` and the `CLAIM_LEDGER.md` C2 rows.
- Backing artifacts: `storage/benchmarks/component_2_results.json`, `component_2_results_manifest.json`, `component_2b_cluster_usage_results.json` (+ manifest), `storage/data/indices/voronoi_codebook.meta.json` / `voronoi_codebook.npz`, and mechanism code in `src/engine/dssc_encoder.py`, `src/engine/dssc_state_space.py`.
- Not in scope: C1 media results, codec/ECC (C4), concurrency (C5), GAN/RL (C3) — covered in reports 2/3.

## Tool limitations

- All quoted numbers were recomputed from artifact JSON (and the codebook NPZ) with the repository's `.venv/bin/python` in read-only mode; no scrimping on precision — full-precision values are listed before the doc rounding.
- No benchmark scripts were re-executed (no training, no benchmark runs). `torch` was used only where a checkpoint had to be read (`torch.load`, read-only).
- grep-based retrieval cannot rank frequencies or verify prose context, so every numeric cell in the table below was recomputed programmatically rather than pattern-matched.
- The `BENCHMARK_DEEP_EXPLANATION.md` analytic null-center claim is recomputed from the analytic formula, not from any stored artifact, because the KS null simulations are seeded in-script and their outputs were not archived.

## Executive summary

- The C2 story is numerically solid: every headline, table, and mechanism-null number in `02_SEMANTIC_SELECTION_BIAS.md` reconciles exactly with `component_2_results.json` / `component_2b_cluster_usage_results.json` — 14 of 17 rows verdict S.
- Two minor-truth findings (MF/P2): (a) `02_SEMANTIC_SELECTION_BIAS.md` names a code symbol `MODALITY_SEQUENCE` that does not exist — the real mechanism is the round-robin at `dssc_encoder.py:201`; (b) `BENCHMARK_DEEP_EXPLANATION.md:346` quotes an analytic two-sample KS null center of 0.0155 while the formula recomputes to ≈0.01526 (the accompanying 5% critical value 0.0238 is correct).
- One scope-wording qualify (Q/P2): the §-level "projection KS" claim needs to say it is a **single pooled PC1** test (one projection per modality, `benchmark_2_semantic_distribution.py:749–758`), not a family over dimensions; the multiplicity story (0/1536 BH rejection) is the per-dimension family and is separately exact.
- Verdict counts: S 14, MF 2, Q 1, UB 0. Priority: P0 0, P1 0, P2 3.
- Ledger cross-check: `C2-ENERGY-01`, `C2-KS-02`, `C2-MECH-04` are `supported` and match recomputation; `C2-UNIFORM-03` stays correctly `retracted`.
- ALREADY SETTLED (cited, not re-litigated): paper §V-C diurnal split (3.8936/3.2067, 14.932/4.1463, KS 0.430264/0.835704, 100%), C2 Table-2 PC1 shares + projection-KS multiplicity handling, and the report §7.2 ρ rounding disclosures are covered by B1/B2. No verified B1/B2 item was re-opened.

## Claim rows

| # | Claim (doc:line or ID) | Artifacts | Recomputed value (full precision → doc rounding) | Verdict | Pri | Fix note |
|---|---|---|---|---|---|---|
| 1 | Corpus + generator mixture decomposition (weighted avg 6.2242, mixing 1.5847, sum 7.8090) | `component_2_results.json`, `component_2b_...json` | 6.224224880074541 → 6.2242; 1.5847350248901924 → 1.5847; sum 7.808959904964734 → 7.8090; residual ≈ 8.88e-16 (exact-up-to-fp) | S | — | None; identity H = w·H_m + H(w) verified numerically for both campaigns |
| 2 | Modality shares image/text/audio 33.21/34.11/32.68%; corpus proportions 12.96/81.54/5.50% | `component_2_results.json` | 0.33213644524 → 33.21%; 0.34111310592 → 34.11%; 0.32675044883 → 32.68%; proportions 0.12958/0.81539/0.05502 → 12.96/81.54/5.50% | S | — | — |
| 3 | Codebook density min 59 / max 8814 / mean 958.1 / median 341 | `voronoi_codebook.npz` + `.meta.json` | min 59, max 8814, mean 958.121, median 341.0 (median recomputed fresh this pass) | S | — | — |
| 4 | Carrier reuse factor 1.1658 | `component_2_results.json` | 1.16575973 → 1.1658 | S | — | — |
| 5 | Unique-media entropy 7.7814 | `component_2_results.json` | 7.781401775776322 | S | — | — |
| 6 | Uniform null rejected: χ² 3253.4, perm p 0.0005 (ledger C2-UNIFORM-03 retraction) | `component_2b_...json` | chi2_vs_uniform 3253.4044444 → 3253.4; perm p 0.0004997501 → 0.0005 | S | — | — |
| 7 | Mechanism null not rejected: χ² 268.6 on 253 dof, perm p 0.208 (C2-MECH-04) | `component_2b_...json` | 268.55983457004186 → 268.6; dof 253; perm p 0.20789605197401298 → 0.208 | S | — | — |
| 8 | Entropy interval [7.7837, 7.8098]; uniform-null mean 7.9829 band [7.9799, 7.9857]; mech-null center 7.7969; z 0.51 vs mech / 127.35 vs uniform; Cramér's V 0.0099; asymptotic p 0.240; max_z 2.62 p 0.796 | `component_2b_...json` | band [7.783659920259019, 7.809756153264083]; uniform mean 7.982884; band [7.9799, 7.9857]; mech center 7.796853; z recomputed from (7.7935−7.7969) vs sqrt of band variances (0.00337 / 0.00666 → 0.51); uniform z 127.35; V 0.009913988 → 0.0099; p 0.2396 → 0.240; max_z 2.617286 → 2.62, p 0.796 | S | — | — |
| 9 | Headline p-values: image energy 0.009 / image MMD 0.012 / text 0.004 (both) / audio 0.574, 0.642 (C2-ENERGY-01) | `component_2_results.json` headline | 0.008991 → 0.009; 0.011988 → 0.012; 0.003996 → 0.004 (both text tests); 0.574426 → 0.574; 0.642358 → 0.642 | S | — | — |
| 10 | 0/1,536 KS coordinates reject under BH; familywise p 0.315/0.148/0.092 (C2-KS-02) | `component_2_results.json` per-modality `ks_family` | 512×3 = 1,536 families, bh_rejections = 0 each; familywise 0.315369/0.147705/0.091816 → 0.315/0.148/0.092 | S | — | — |
| 11 | Observed mean D_KS 0.016–0.017, max 0.0363; familywise null p95 ≈0.0378–0.0391 | `component_2_results.json` | means 0.017000/0.016275/0.015918 (→ "0.016–0.017" ✓); max 0.036347 → 0.0363 ✓; null p95 0.039122/0.038822/0.037787 | S | — | — |
| 12 | Packaging residuals ±33/+217/−183; binomial SD ≈79 | `component_2b_...json` | residuals match recorded fields; binomial SD 78.669… → 79 | S | — | — |
| 13 | Mechanism code: deterministic round-robin chunk→modality; HMAC-seeded PCG64 permutation; 2 families per modality; fresh session key per message | `dssc_encoder.py:201`, `dssc_state_space.py:90–110`, `dssc_encoder.py:181–185`, benchmark-2 script line ~448 | `available_modalities[chunk_idx % len(available_modalities)]` (text-first with text ctx offset); HMAC digest → PCG64; MODALITY_FAMILY_MAP = 2 families per modality; per-message key derivation confirmed | S | — | — |
| 14 | Doc names symbol `MODALITY_SEQUENCE` for the chunk→modality rule | `dssc_encoder.py` (no such symbol tree-wide) | Symbol `MODALITY_SEQUENCE` does not exist anywhere in `src/`; the rule is the inline round-robin at `dssc_encoder.py:201` | MF | P2 | Replace the symbol name with the actual construct (round-robin with text-context offset) or add/alias the symbol in code |
| 15 | "Projection KS" scope wording (per-modality coordinate family claim) | benchmark-2 script lines 749–758 | The test computes DS on the single pooled PC1 projection of the KS statistic set (one projection per modality), not a per-dimension family | Q | P2 | Add scope wording: "pooled-PC1 projection KS (one test per modality)"; familywise multiplicity claims live in the per-dimension KS families (row 10) |
| 16 | table2b modality weights 1/3, 0.34074, 0.32593; mixing 1.58473; residual 0.0 | `component_2b_...json` | weights reproduce exactly (1/3, 0.34074, 0.32593); mixing 1.58473; residual 0.0 exactly | S | — | — |
| 17 | Analytic two-sample KS null center "0.0155" (BENCHMARK_DEEP_EXPLANATION.md:346) | analytic formula (0.8687·sqrt(1/n+1/m), n=9200, m=5000) | 0.8687 · sqrt(1.08696e-4 + 2.0e-4) = 0.8687 × 0.0175698 ≈ 0.015263 → doc says 0.0155 (≈1.5% high). Companion critical value: c(0.05)=1.3581 × 0.0175698 = 0.023862 → doc 0.0238 ✓ | MF | P2 | Correct the null center to ≈0.0153 (or quote the empirical sim mean 0.01713 explicitly if that is the intended referent); keep 0.0238 as-is |

## ALREADY SETTLED — cited, not re-verified (per audit protocol)

- Paper §V-C diurnal split (3.8936 / 3.2067; 14.932 / 4.1463; KS 0.430264 / 0.835704; 100% of hours) — covered by B1/B2.
- C2 Table-2 PC1 carrier-share rows and the projection-KS multiplicity treatment in the report — covered by B1/B2 (and independently re-confirmed by rows 10/15's artifacts above).
- Report §7.2 ρ disclosures (N = 35,000 exchanges, +0.0054, ~616× estimated) — covered by B1/B2.
- `ENV-CORPUS-02` corpus-density figures — supported again by row 3 recomputation.

## Prioritized follow-ups (P1 first)

- (no P0/P1 in scope)
- P2 — Rename or add-code-symbol: align `02_SEMANTIC_SELECTION_BIAS.md` mechanism wording with the real round-robin construct (`dssc_encoder.py:201`), or introduce a named constant `MODALITY_SEQUENCE` in code so the doc's symbol reference becomes true.
- P2 — Fix `BENCHMARK_DEEP_EXPLANATION.md:346` analytic KS null center 0.0155 → ≈0.0153 (recomputed 0.015263); note explicitly whether the number is meant to be the analytic mean or the empirical simulation mean (0.01713, separate quantity).
- P2 — Add scope wording that the "projection KS" is a single pooled-PC1 test per modality, and that the 0/1536 multiplicity result is a different (per-dimension) test family.
