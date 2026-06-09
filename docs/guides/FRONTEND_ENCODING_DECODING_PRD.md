# Frontend Encoding/Decoding Integration PRD

## Purpose

This PRD captures the current state of the DCASS frontend integration with the backend encode/decode flow and lists the product gaps that should be addressed before additional implementation work begins.

## Current Verified State

- Backend exposes working REST surfaces for encode, decode, search, status, readiness, wire packet monitoring, and transmission in [src/api/server.py](/home/jeevan/Documents/uni/dcass/src/api/server.py:155).
- Frontend API client is already wired to `encode`, `decode`, `search`, `status`, `ready`, and `benchmark/latest` in [frontend/src/lib/api.ts](/home/jeevan/Documents/uni/dcass/frontend/src/lib/api.ts:102).
- Frontend has implemented pages for home, status, encode, and wire view in [frontend/src/app/page.tsx](/home/jeevan/Documents/uni/dcass/frontend/src/app/page.tsx:1), [frontend/src/app/status/page.tsx](/home/jeevan/Documents/uni/dcass/frontend/src/app/status/page.tsx:1), [frontend/src/app/encode/page.tsx](/home/jeevan/Documents/uni/dcass/frontend/src/app/encode/page.tsx:1), and [frontend/src/app/wire/page.tsx](/home/jeevan/Documents/uni/dcass/frontend/src/app/wire/page.tsx:1).
- Frontend production build succeeds with `npm run build`, so the current UI compiles and type-checks.

## Problem Statement

The encode flow is partially integrated with the frontend, but the full frontend round-trip is incomplete. Users can encode and transmit a message from the UI, but they cannot decode a sequence from the UI, and some screens describe capabilities that are not currently selectable or visible in the frontend.

## Confirmed Gaps

### 1. No decode page in the frontend

- The backend provides `POST /api/decode` in [src/api/server.py](/home/jeevan/Documents/uni/dcass/src/api/server.py:187).
- The frontend API client exposes `decodeSequence()` in [frontend/src/lib/api.ts](/home/jeevan/Documents/uni/dcass/frontend/src/lib/api.ts:117).
- There is no `frontend/src/app/decode/page.tsx`.
- Navigation does not include a decode entry in [frontend/src/components/Navigation.tsx](/home/jeevan/Documents/uni/dcass/frontend/src/components/Navigation.tsx:9).
- Home page cards also omit decode in [frontend/src/app/page.tsx](/home/jeevan/Documents/uni/dcass/frontend/src/app/page.tsx:22).

Impact:
- Frontend users cannot verify that encoded media IDs reconstruct correctly.
- The product does not currently demonstrate the full encode -> transmit -> decode loop in the browser.

### 2. Encode page always transmits in static mode

- The transmit request sent from the encode page hardcodes `mode: 'static'` in [frontend/src/app/encode/page.tsx](/home/jeevan/Documents/uni/dcass/frontend/src/app/encode/page.tsx:109).
- The backend supports `static`, `rl`, `gan`, and `auto` in [src/api/server.py](/home/jeevan/Documents/uni/dcass/src/api/server.py:345).

Impact:
- The UI does not expose the backend’s fallback-aware stealth scheduling modes.
- The home page’s messaging about dynamic fallback is stronger than the actual encode experience.

### 3. Frontend exposes API helpers that are not surfaced in UI

- `decodeSequence`, `searchCorpus`, and `getLatestBenchmark` exist in [frontend/src/lib/api.ts](/home/jeevan/Documents/uni/dcass/frontend/src/lib/api.ts:117).
- There are no corresponding decode, search, or benchmark pages in `frontend/src/app/`.

Impact:
- The frontend capability set is narrower than the API client suggests.
- This can confuse maintainers and create drift between implementation and product expectations.

### 4. Documentation currently overstates frontend integration

- Frontend README lists `POST /api/decode` as part of frontend API integration in [frontend/README.md](/home/jeevan/Documents/uni/dcass/frontend/README.md:86).
- Separate project docs already note `/decode` is not implemented yet in [docs/guides/RUNNING_FRONTEND_BACKEND.md](/home/jeevan/Documents/uni/dcass/docs/guides/RUNNING_FRONTEND_BACKEND.md:152).

Impact:
- Team members may assume decode is already usable from the browser when it is only available at the backend/API layer.

## Out of Scope

- Rewriting the encoder or decoder core logic in [src/engine/encoder.py](/home/jeevan/Documents/uni/dcass/src/engine/encoder.py:1) or [src/engine/decoder.py](/home/jeevan/Documents/uni/dcass/src/engine/decoder.py:1).
- Training or changing RL/GAN stealth models.
- Changing corpus quality, FAISS indexing behavior, or semantic ranking logic.

## Product Requirements

### Requirement 1: Add a browser-based decode workflow

The frontend must provide a dedicated decode screen where a user can:

- Paste or load a list of `media_ids`
- Submit them to `/api/decode`
- View reconstructed meaning
- View per-item verification state
- See verification rate and elapsed time

### Requirement 2: Complete the round-trip experience

After encoding, the user should be able to continue into decode with minimal friction. Preferred UX:

- Allow copying media IDs from encode results
- Provide a “Send to Decode” action from encode results or wire view
- Preserve sequence ordering visibly

### Requirement 3: Expose transmission mode honestly

If the backend supports transmission modes beyond `static`, the encode UI should either:

- Let the user select `static`, `rl`, `gan`, or `auto`, or
- Clearly label the feature as static-only in the UI and docs

### Requirement 4: Align docs with actual UI coverage

Frontend-facing docs must clearly separate:

- Backend APIs that exist
- Frontend pages that exist
- Planned but not yet implemented pages

## Suggested Implementation Order

1. Add `/decode` page and navigation entry.
2. Wire decode page to `decodeSequence()`.
3. Add “Use encoded result in decode” handoff from the encode page.
4. Expose transmit mode selection or simplify product messaging to match static-only behavior.
5. Update docs to reflect the implemented frontend scope.

## Acceptance Criteria

- A user can encode a message from `/encode`.
- A user can take the resulting `media_ids` into `/decode` without using external tools.
- `/decode` shows reconstructed meaning, item list, verification flags, verification rate, and elapsed time.
- Navigation and landing page expose the decode workflow.
- Transmit mode behavior shown in the UI matches what is actually sent to the backend.
- Frontend docs no longer imply decode is already available unless the page exists.

## Notes From Review

- I did not validate live semantic results end-to-end against real indices/models in a running backend session here.
- I did validate the integration statically from the codebase and confirmed the frontend build succeeds.
