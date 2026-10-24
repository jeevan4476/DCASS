# Dependency Security Advisories and Dispositions

This document records the disposition of third-party package vulnerabilities identified during the Python and frontend dependency security audits.

Last audited: October 2026.
Audit tools: `pip-audit` 2.10.1 against PyPI and OSV databases, and `npm audit` against the npm registry.

---

## 1. Resolved vulnerabilities

The following packages were updated to secure releases, eliminating 53 known CVEs:

- `pillow`: Upgraded to 13.0.0, resolving arbitrary code execution, buffer overflow, and memory disclosure vulnerabilities in legacy image codecs. The test suite verified all image loading, transform, and CLIP encoding paths with 419 passing tests.
- `brotli`: Upgraded to 1.2.1, addressing decompression buffer overflow risks.
- `certifi`: Upgraded to 2026.10.23, updating trusted root certificates.
- `filelock`: Upgraded to 4.2.4, fixing race conditions in inter-process lock files.
- `opentelemetry-api`: Upgraded to 1.45.1, patching telemetry context leakage.
- `setuptools`: Upgraded to 84.2.1, patching build-time package archive traversal risks.
- `axios` in `frontend/package-lock.json`: Patched to current secure patch level, eliminating SSRF and prototype pollution advisories.

---

## 2. Dispositions for unpatched upstream advisories

Three packages report open advisories with no patched upstream release available at this time. Each item was audited against the DCASS codebase to verify absence of exploitable attack surface.

### 2.1 ecdsa 0.19.2

- **Advisory identifier:** PYSEC-2026-1325, GHSA-wj6h-64fc-37mp, CVE-2024-23342.
- **Description:** Minerva timing side-channel attack on P-256 curve during `ecdsa.SigningKey.sign_digest()`. An attacker observing signing latency can recover private nonce bits.
- **Upstream status:** The upstream maintainers consider side-channel attacks out of scope, and no fix is planned.
- **DCASS call-site analysis:** Zero exposure. `ecdsa` is an indirect dependency introduced transitively through `python-jose[cryptography]`. DCASS does not instantiate ECDSA keys, does not sign digests using elliptic curves, and does not perform ECDH key exchange. Authentication in `src/api/auth.py` uses symmetric HMAC-SHA256 tokens only.
- **Disposition:** Accepted risk. The vulnerable code paths are never called by any DCASS module.

### 2.2 nltk 3.10.3

- **Advisory identifier:** PYSEC-2026-3740, GHSA-8mgp-746c-j5xp, CVE-2026-81726.
- **Description:** File sandbox bypass in model persistence helpers, including `TransitionParser.train`, `TransitionParser.parse`, `AveragedPerceptron.save`, `AveragedPerceptron.load`, and `PerceptronTagger.save_to_json`. These functions call built-in `open()` on caller-provided model paths without checking sandbox boundaries.
- **Upstream status:** No patched release published yet.
- **DCASS call-site analysis:** Zero exposure. DCASS imports only `nltk.tokenize.sent_tokenize` and `nltk.tokenize.word_tokenize` in `src/engine/chunker.py` to divide user secret text into semantic phrases. DCASS never invokes parsers, taggers, or model persistence APIs. No user-supplied file path is passed to any NLTK function.
- **Disposition:** Accepted risk. The vulnerable model-loading submodules are not imported or reachable in DCASS.

### 2.3 python-jose 3.5.0

- **Advisory identifier:** PYSEC-2026-4298, GHSA-3qf3-8w2g-rqmx, CVE-2026-85394.
- **Description:** Failure to properly validate asymmetric keys in HMAC initialization. An attacker holding a service public key can forge HS256 tokens if algorithm validation is omitted during token verification.
- **Upstream status:** No patched release published yet.
- **DCASS call-site analysis:** Mitigated by application architecture. In `src/api/auth.py`, `jwt.decode` explicitly pins `algorithms=["HS256"]`. The backend does not use asymmetric public keys for token authentication. The signing key is a private symmetric secret, `JWT_SECRET`, generated randomly on server startup or read from an environment variable. Because the algorithm is pinned to HS256 and no public keys exist, key-confusion attacks cannot succeed.
- **Disposition:** Mitigated in code through strict algorithm pinning in `src/api/auth.py`.
