# Dependency Security Advisories and Dispositions

This document records resolved vulnerabilities and dependency dispositions across the Python and frontend stacks.

Last audited: November 2026.

---

## 1. Python dependencies

- **`PyJWT` (`>=2.9.0`) replaces `python-jose[cryptography]`**: Eliminates the unpatched `python-jose` algorithm-confusion advisory (`CVE-2026-85394`) and transitive `ecdsa` side-channel advisory (`CVE-2024-23342`). `src/api/auth.py` continues to enforce `algorithms=["HS256"]` with symmetric HMAC secrets.
- **`nltk` (`>=3.10.4`)**: Upgraded from `3.10.3` to patch all 33 NLTK advisories, including critical `CVE-2026-128138`.
- **`opentelemetry-api` (`>=1.45.2`)**: Upgraded from `1.45.1`.
- **`pillow` (`>=13.0.0`), `brotli` (`>=1.2.1`), `certifi` (`>=2026.10.23`), `filelock` (`>=4.2.4`), `setuptools` (`>=84.2.1`)**: Kept at patched floor versions in `requirements.txt`.
- **Optional notebook stack and unused lockfile removed**: Dropped `jupyter` and `ipykernel` from `requirements.txt` and removed `uv.lock`, clearing all stale lockfile alerts (`aiohttp`, `tornado`, `jupyterlab`, `notebook`, `nbconvert`, `pyasn1`, `soupsieve`, `urllib3`, `black`, `multidict`).

---

## 2. Frontend dependencies

- **`next` and `eslint-config-next` (`15.5.29`)**: Upgraded from `14.2.x`, resolving critical Image Optimization RCEs, Server Actions DoS (`CVE-2026-94482`), and cross-client response injection (`CVE-2026-119348`).
- **`follow-redirects` (`>=1.16.3`) and `postcss` (`>=8.5.33`)**: Pinned via `overrides` in `frontend/package.json` to resolve cross-realm header/credential leakage (`CVE-2026-116689`) and PostCSS path/XSS advisories.
- **`minimatch`**: Updated to patched release (`11.0.9` in `@typescript-eslint`) resolving `CVE-2026-118764`.
- **`axios`**: Upgraded to `>=1.18.0` via `bun install`, resolving all prototype pollution, SSRF, and ReDoS advisories.
