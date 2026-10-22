# Archived benchmark artifacts

## Component 4c pre-hardening run

Files:

- `component_4c_ecc_failure_results_pre_hardening.json`
- `table4c_ecc_failure_modes_pre_hardening.tex`
- `fig_4c_ecc_failure_modes_pre_hardening.png`

This is the Component 4c artifact recorded before the DSSC decoder and
exact_vcp decoder were changed to require framed payloads whenever ECC is
enabled. The dropout condition recorded 41 CRC-passing wrong-plaintext
decodes across the 320 trials at 2 through 5 percent deletion. The cause was the legacy unframed fallback:
a deletion moved the frame marker away from byte zero, RS still reported a
successful decode, and the decoder accepted the resulting raw bytes as a
legacy message.

The current `component_4c_ecc_failure_results.json` is the post-hardening
artifact and records zero false acceptances. Keep this archived run as
evidence for the finding and for the before/after comparison in the
benchmark documentation.
