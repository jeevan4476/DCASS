# DCASS Dynamic Semantic State-Space Coding (DSSC)
## Proposed Next-Generation Encoding Architecture

**Status:** Architecture proposal  
**Purpose:** Replace flat byte-to-cluster encoding with semantic-first, high-capacity, exact-recovery carrier coding  
**Target:** DCASS (Distributed Cross-modal Adaptive Steganography System)

---

# 1. Executive Summary

The current DCASS `exact_vcp` pipeline has a fundamental abstraction problem:

```text
Message
  ↓
UTF-8 bytes
  ↓
RS-ECC
  ↓
Each byte selects one VCP cluster
  ↓
One media carrier per byte
```

This provides deterministic recovery, but it wastes the structure of the media corpus.

For example:

```text
49-character message
    ↓
49 plaintext bytes
+ 5 framing bytes
+ 8 RS parity bytes
    ↓
62 encoded bytes
    ↓
62 media carriers
```

The proposed architecture replaces this with:

```text
Message
  ↓
Frame + RS-ECC
  ↓
Payload bitstream
  ↓
Semantic chunk analysis
  ↓
Semantic cluster families
  ↓
Session-specific carrier state spaces
  ↓
Payload bits encoded inside valid semantic carrier choices
  ↓
Media sequence
```

The key principle is:

> **Semantic context chooses where DCASS is allowed to encode. Payload bits choose which carrier inside that semantic space is selected.**

This reverses the problematic ordering used by the current system.

---

# 2. The Core Design Principle

## Old architecture

```text
Payload bits
    ↓
VCP cluster
    ↓
Find media
```

Problem:

```text
Payload = random-looking bits
        ↓
Random cluster
        ↓
Potentially unrelated media
```

Example:

```text
Message meaning:
"communication is not working"

Payload bits happen to select:
Food cluster

Selected carrier:
Pizza / restaurant / cooking media
```

No amount of ranking inside a food cluster makes the carrier naturally represent communication.

---

## New architecture

```text
Message semantics
       ↓
Semantic topic
       ↓
Allowed semantic cluster family
       ↓
Valid carrier state space
       ↓
Payload chooses a carrier inside that space
```

Example:

```text
Message:
"communication is not working"

Semantic topics:
communication
failure
technology

Allowed cluster family:
C12 = computers
C43 = phones
C88 = networks
C102 = troubleshooting

Candidate media:
network cables
phone screens
computer errors
server infrastructure
people communicating

Payload bits:
        ↓
Select one valid candidate deterministically
```

Every carrier can remain semantically plausible while still encoding exact data.

---

# 3. Proposed Architecture Name

Recommended name:

# DSSC — Dynamic Semantic State-Space Coding

Alternative names:

- Dynamic Semantic Carrier Coding (DSCC)
- Semantic State-Space Steganography (S4)
- Dynamic Semantic Carrier Codebook (DSCCB)
- Multi-Dimensional Semantic Carrier Coding (MDSCC)

For the rest of this document:

```text
DSSC
=
Dynamic Semantic State-Space Coding
```

---

# 4. Complete System Architecture

```text
┌───────────────────────────────────────────────────────────────┐
│                         SENDER                                │
└───────────────────────────────────────────────────────────────┘

                    Original Message
                           │
                           ▼
                   UTF-8 Serialization
                           │
                           ▼
                    Payload Framing
                  (version + length + CRC)
                           │
                           ▼
                    Reed-Solomon ECC
                           │
                           ▼
                     Payload Bitstream
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      Semantic Analyzer             Session Key
             │                           │
             ▼                           ▼
       Semantic Chunks         Dynamic Codebook Seed
             │                           │
             ▼                           ▼
       Topic / Intent          Keyed Permutations
             │                           │
             └─────────────┬─────────────┘
                           ▼
              Semantic Cluster Family
                           │
                           ▼
              Candidate Carrier Discovery
                           │
                           ▼
             Valid Carrier State Space
                           │
                           ▼
            Enumerative / Variable-Length
                 Payload Encoding
                           │
                           ▼
                   Media Carrier IDs
                           │
                           ▼
                    Carrier Sequence


┌───────────────────────────────────────────────────────────────┐
│                        RECEIVER                               │
└───────────────────────────────────────────────────────────────┘

                    Carrier Sequence
                           │
                           ▼
                Session Manifest / Key
                           │
                           ▼
              Reconstruct Semantic Space
                           │
                           ▼
              Reconstruct Candidate Sets
                           │
                           ▼
              Reconstruct State Ordering
                           │
                           ▼
               Carrier ID → State Index
                           │
                           ▼
                Recover Payload Bitstream
                           │
                           ▼
                   Reed-Solomon Decode
                           │
                           ▼
                    Frame Validation
                           │
                           ▼
                    Original Message
```

---

# 5. Main Components

## 5.1 Payload Layer

The payload layer remains mostly unchanged.

```text
Message
   ↓
UTF-8
   ↓
Frame
   ↓
RS-ECC
   ↓
Bitstream
```

Example:

```text
"hello"
   ↓
68 65 6c 6c 6f
   ↓
frame:
[version | length | CRC | payload]
   ↓
RS parity
   ↓
protected codeword
```

Why preserve this layer?

Because it provides:

- exact byte recovery
- message integrity
- corruption detection
- error correction
- language independence
- arbitrary binary payload support

DSSC changes the **carrier encoding layer**, not the fundamental payload reliability layer.

---

## 5.2 Semantic Analysis Layer

The original message is analyzed before carrier selection.

Example:

```text
"My message is to convey that this is not working."
```

Possible semantic decomposition:

```text
Block 1:
"my message"

Topic:
communication / expression

Block 2:
"is to convey"

Topic:
communication / delivery

Block 3:
"this is not working"

Topic:
failure / malfunction / troubleshooting
```

Important:

> Semantic analysis does NOT replace the exact payload.

The original bytes remain protected in the payload stream.

Semantic analysis only determines:

```text
What kinds of carriers are acceptable?
```

---

## 5.3 Semantic Cluster Families

Instead of forcing one payload byte to equal one VCP cluster ID:

```text
byte 0x6D → cluster 109
```

we build higher-level semantic families.

Example:

```text
COMMUNICATION FAMILY
├── C12: phones
├── C43: people talking
├── C88: messaging
├── C102: documents
└── C177: networks

TECHNOLOGY FAMILY
├── C8: computers
├── C55: servers
├── C91: programming
└── C203: hardware

FAILURE FAMILY
├── C31: errors
├── C67: broken objects
├── C119: troubleshooting
└── C151: warning states
```

A semantic block may activate one or more families.

For:

```text
"this is not working"
```

the active space might be:

```text
TECHNOLOGY
+
FAILURE
```

The available carrier alphabet is then:

```text
All valid carriers inside those semantic families
```

---

# 6. Carrier State Space

This is the most important technical component.

Each candidate carrier has multiple deterministic properties:

```text
Carrier State =
(
    semantic_family,
    vcp_cluster,
    modality,
    carrier_identity
)
```

Example:

```text
Carrier:
media_48392

State:

semantic family = TECHNOLOGY
cluster         = 55
modality        = image
carrier rank    = 37
```

Instead of manually assuming:

```text
8 bits cluster
+ 2 bits modality
+ 6 bits rank
= 16 bits
```

we define the complete set of valid states.

Example:

```text
Semantic block: TECHNOLOGY + FAILURE

Valid candidates:

state 0   → media_101
state 1   → media_245
state 2   → media_331
...
state 1023 → media_99021
```

If there are:

```text
1024 valid states
```

then the block has:

```text
log2(1024) = 10 bits
```

of exact capacity.

If there are:

```text
1500 valid states
```

the system should not simply waste the extra states.

This is why DSSC should use variable-capacity state-space coding.

---

# 7. State-Space Capacity

For each semantic block:

```text
N = number of valid carrier states
```

The theoretical capacity is:

```text
capacity ≈ log2(N) bits
```

Examples:

| Candidate states | Approximate capacity |
|---:|---:|
| 16 | 4 bits |
| 32 | 5 bits |
| 64 | 6 bits |
| 128 | 7 bits |
| 256 | 8 bits |
| 512 | 9 bits |
| 1,024 | 10 bits |
| 2,048 | 11 bits |
| 4,096 | 12 bits |
| 65,536 | 16 bits |

This leads to an important observation:

> Carrier capacity should be determined by the number of valid, deterministic carrier choices — not artificially limited to the 256 VCP cluster IDs.

---

# 8. Variable-Length / Enumerative Coding

The candidate sets will not always have powers of two.

Example:

```text
Semantic block:
COMMUNICATION

Candidate carriers = 300
```

A naïve implementation could use:

```text
floor(log2(300)) = 8 bits
```

which uses only 256 carriers and wastes 44 states.

Or:

```text
ceil(log2(300)) = 9 bits
```

which creates 512 possible bit patterns but only 300 valid states.

The better solution is to treat:

```text
0 ... 299
```

as a deterministic symbol alphabet and use an enumerative or arithmetic-style coding strategy.

Conceptually:

```text
Payload stream
      ↓
Map next payload value into [0, N)
      ↓
Select carrier state
```

This allows DSSC to use the available semantic carrier state space efficiently.

Implementation can begin with a simpler fixed-capacity version:

```text
bits_per_carrier = floor(log2(N))
```

Then evolve toward full variable-length state-space coding.

---

# 9. Dynamic Session Codebook

A static public ordering is weak.

Suppose every user knows:

```text
state 0 → media A
state 1 → media B
state 2 → media C
```

Then anyone with the corpus can decode.

Instead, create a session-specific ordering.

```text
Corpus candidates
       ↓
Canonical deterministic ordering
       ↓
Session key K
       ↓
PRF / keyed hash
       ↓
Session permutation
       ↓
Dynamic carrier codebook
```

Example:

Base ordering:

```text
0 → media_A
1 → media_B
2 → media_C
3 → media_D
```

Session 1:

```text
0 → media_C
1 → media_A
2 → media_D
3 → media_B
```

Session 2:

```text
0 → media_B
1 → media_D
2 → media_A
3 → media_C
```

The same media item can therefore encode different symbols in different sessions.

This borrows the important conceptual lesson from dynamic-codebook steganography:

> The sender and receiver should independently reconstruct a shared, session-specific mapping rather than permanently exposing one static encoding table.

---

# 10. Encoding Algorithm

## Step 1 — Serialize the message

```python
payload = message.encode("utf-8")
```

---

## Step 2 — Frame the payload

Example:

```text
[version]
[length]
[CRC]
[payload]
```

The frame ensures the receiver can verify exact reconstruction.

---

## Step 3 — Add ECC

```text
frame
   ↓
RS encoder
   ↓
codeword
```

Example:

```text
54 bytes
+
8 parity bytes
=
62-byte codeword
```

---

## Step 4 — Convert to bitstream

```text
62 bytes
× 8
=
496 bits
```

---

## Step 5 — Semantic chunking

Analyze the original message.

```text
Message
   ↓
Chunks
   ↓
Topics / semantic intents
```

Example:

```text
"Please send the document because the system is failing."

Block A:
"Please send the document"
→ communication / documents

Block B:
"because the system is failing"
→ technology / failure
```

---

## Step 6 — Build semantic families

For each block:

```text
Topic
   ↓
Find relevant VCP clusters
   ↓
Build candidate family
```

Example:

```text
communication:
C12, C43, C88

documents:
C102, C114

technology:
C8, C55

failure:
C31, C67
```

---

## Step 7 — Collect valid carriers

For each semantic block:

```text
candidate_set =
all corpus items belonging to allowed clusters
```

Then optionally filter:

```text
minimum CLIP similarity
+
quality threshold
+
deduplication
+
modality constraints
+
availability constraints
```

---

## Step 8 — Build session-specific ordering

```text
candidate_set
      ↓
canonical ordering
      ↓
keyed permutation
      ↓
dynamic state indices
```

Now:

```text
payload symbol 137
```

does not mean:

```text
the 137th carrier in the public corpus
```

It means:

```text
the carrier assigned state 137 under this session
```

---

## Step 9 — Encode payload bits

Consume payload information according to the capacity of the current semantic candidate set.

```text
Current block:
N = 1024 valid states

Capacity:
10 bits

Take:
next 10 payload bits

Example:
1011010011

Binary:
723

Select:
state 723
```

The selected carrier is guaranteed to belong to the semantic space.

---

## Step 10 — Continue

```text
Payload bits
      ↓
Block A carrier states
      ↓
Block B carrier states
      ↓
Block C carrier states
      ↓
...
```

The encoder continues until the complete protected payload is represented.

---

# 11. Decoding Algorithm

The receiver must have:

- carrier IDs
- corpus version
- VCP codebook version
- clustering model version
- semantic-family construction rules
- session key or equivalent shared session parameters

The receiver performs:

```text
Carrier ID
   ↓
Lookup metadata
   ↓
Identify semantic family
   ↓
Reconstruct candidate set
   ↓
Reconstruct session permutation
   ↓
Carrier → state index
   ↓
State index → payload information
   ↓
Rebuild bitstream
   ↓
RS decode
   ↓
Frame validation
   ↓
Original UTF-8 bytes
   ↓
Original message
```

---

# 12. The Critical Manifest

Exact decoding depends on deterministic reconstruction.

Every session should therefore have a manifest concept.

Example:

```json
{
  "protocol_version": 2,
  "corpus_id": "dcass-v1",
  "corpus_hash": "...",
  "vcp_model_version": "...",
  "cluster_codebook_hash": "...",
  "embedding_model": "...",
  "semantic_family_version": "...",
  "ordering_algorithm": "blake3-keyed-v1",
  "ecc_scheme": "rs",
  "frame_version": 1
}
```

The manifest does not necessarily need to be publicly transmitted in full.

Some values can be pre-shared or implied by protocol version.

The purpose is:

```text
Sender and receiver must reconstruct exactly the same state spaces.
```

---

# 13. Error Correction Design

DSSC should retain Reed-Solomon, but with one important correction to the previous MDCS design:

> Do not claim that one corrupted carrier automatically equals one RS symbol error.

A carrier may encode several bits.

If those bits span multiple bytes, one carrier substitution may corrupt multiple RS symbols.

Therefore, DSSC should explicitly define an ECC alignment layer.

Recommended:

```text
Payload bytes
     ↓
RS codeword
     ↓
Interleaving / symbol layout
     ↓
Carrier state encoder
```

Possible strategies:

### Strategy A — Byte-aligned carrier symbols

Each carrier encodes an integer number of bytes.

Advantages:

- easy RS accounting
- predictable error model

Disadvantages:

- may waste capacity

### Strategy B — Bitstream interleaving

Spread bits from adjacent RS symbols across different carriers.

Advantages:

- a single carrier loss is distributed
- burst errors may be reduced

Disadvantages:

- more complex

### Strategy C — Carrier-level erasures

If a carrier is missing, mark its associated payload positions as erasures.

Advantages:

- RS can correct more known-location erasures than unknown symbol errors

Disadvantages:

- requires exact mapping from carrier loss to affected symbols

Recommended initial implementation:

```text
RS
+
deterministic bit interleaver
+
carrier mapping
```

Then experimentally measure carrier substitutions versus recovered payload success.

---

# 14. Accuracy: Can DSSC Achieve 95–100%?

## Short answer

Yes — but the meaning of "accuracy" must be defined correctly.

### Under deterministic, controlled conditions:

```text
100% exact recovery is achievable in principle.
```

### In a realistic system:

```text
95–100% end-to-end recovery is a reasonable engineering target,
but it must be measured experimentally.
```

Do not promise 100% universally before testing.

---

# 15. Accuracy Breakdown

## Case 1 — Perfect transmission

Conditions:

- same corpus
- same corpus version
- same cluster assignments
- same semantic-family algorithm
- same session key
- same ordering
- all carrier IDs received correctly

Expected result:

```text
100% exact message recovery
```

Why?

Because semantic analysis does not reconstruct the message.

The original bytes are reconstructed from deterministic carrier states.

The semantic layer only constrains the allowed carrier alphabet.

---

## Case 2 — Small number of carrier errors

Expected result:

```text
Potentially 100%
```

if:

```text
RS parity capacity
≥
actual resulting RS symbol errors
```

Important:

The exact number of correctable carrier substitutions depends on the carrier-to-ECC bit mapping.

This must be measured.

---

## Case 3 — Missing carriers

Expected result:

```text
Potentially very high with erasure-aware decoding
```

if the receiver knows which carriers are missing and can map the missing payload positions to erasures.

RS codes can generally handle more known-location erasures than unknown symbol errors for the same parity budget.

---

## Case 4 — Corpus mismatch

Example:

```text
Sender:
corpus version 1

Receiver:
corpus version 2
```

Expected result without strict version control:

```text
Potentially catastrophic failure
```

Therefore:

```text
Corpus fingerprint validation is mandatory.
```

The system should fail explicitly rather than silently output incorrect text.

---

## Case 5 — Semantic model disagreement

If sender and receiver independently run a nondeterministic LLM or semantic classifier:

```text
sender chooses family A
receiver chooses family B
```

decoding may fail.

Therefore:

### Recommended rule

The semantic-family sequence should either be:

1. deterministically reproducible from shared inputs, or
2. encoded/protected as protocol metadata.

For exact decoding, never rely on two uncontrolled LLM runs producing exactly the same semantic decisions.

---

# 16. Expected Accuracy Table

| Scenario | Expected exact recovery |
|---|---:|
| Controlled environment, no transmission errors | 100% target |
| Controlled environment + correctable errors | Up to 100% |
| Small corpus drift detected before decode | Safe rejection |
| Undetected corpus drift | Unreliable |
| Semantic classifier mismatch without metadata | Unreliable |
| Versioned deterministic semantic mapping | Near 100% target |
| Real-world production pipeline | Must be experimentally measured |

The important distinction is:

```text
Semantic similarity ≠ decoding accuracy
```

DSSC can have:

```text
100% exact payload recovery
+
high semantic plausibility
```

because those are separate layers.

---

# 17. Why DSSC Is Better Than Current exact_vcp

## Current

```text
Byte
  ↓
Cluster
  ↓
Media
```

Problems:

- 8 bits per carrier maximum from cluster identity
- semantic meaning determined by payload
- 1 byte often costs 1 carrier
- many unrelated carriers
- poor visual/narrative coherence

---

## DSSC

```text
Semantic block
       ↓
Semantic candidate space
       ↓
Session-specific carrier states
       ↓
Payload selects state
```

Benefits:

- semantic context comes first
- carrier identity provides more capacity
- VCP clusters become semantic infrastructure
- payload does not blindly dictate topic
- exact byte payload remains intact
- dynamic codebooks can improve security
- candidate space size determines capacity

---

# 18. Why DSSC Is Better Than the Previous MDCS Proposal

Previous MDCS:

```text
next 8 payload bits
       ↓
cluster ID
       ↓
modality
       ↓
rank
```

DSSC:

```text
semantic topic
       ↓
allowed cluster family
       ↓
candidate carrier states
       ↓
payload selects one state
```

The critical improvement:

> **MDCS uses semantics after payload-to-cluster selection. DSSC uses semantics before payload-to-carrier selection.**

That is the main architectural difference.

---

# 19. Example Walkthrough

Message:

```text
"My communication system is not working."
```

## Semantic analysis

```text
Block A:
"My communication system"

Families:
communication
technology

Block B:
"is not working"

Families:
failure
technology
```

---

## Candidate spaces

Block A:

```text
communication + technology
=
1,024 valid carriers
```

Capacity:

```text
10 bits per carrier
```

Block B:

```text
failure + technology
=
512 valid carriers
```

Capacity:

```text
9 bits per carrier
```

---

## Payload

Suppose protected payload contains:

```text
496 bits
```

If average usable capacity is approximately:

```text
10 bits per carrier
```

roughly:

```text
ceil(496 / 10)
=
50 carriers
```

But this is only an example.

The actual carrier count depends heavily on:

- corpus size
- semantic candidate set size
- number of semantic blocks
- filtering thresholds
- coding efficiency

A larger corpus can provide larger semantic candidate spaces and therefore more capacity per carrier.

---

# 20. Important Reality Check: Carrier Count

DSSC does NOT automatically guarantee:

```text
62 carriers → 4 carriers
```

That would violate the practical information-capacity constraints unless each selected carrier can represent a very large number of distinguishable states.

The actual formula is approximately:

```text
carrier_count
≈
payload_bits
/
average_bits_per_carrier
```

Example:

```text
496 payload bits

Average state capacity = 12 bits/carrier

≈ 42 carriers
```

If average capacity is:

```text
16 bits/carrier

≈ 31 carriers
```

Therefore the best way to improve capacity is:

```text
larger corpus
+
larger semantic candidate spaces
+
efficient variable-length coding
+
more deterministic carrier states
```

---

# 21. Research Contribution

The strongest research claim is NOT:

> We use VCP clusters instead of bytes.

That alone is not enough.

The stronger contribution is:

> **DSSC separates semantic carrier admissibility from payload state selection. A semantic analysis layer first constructs topic-constrained carrier state spaces, after which a dynamic session-specific codebook maps exact protected payload symbols to individual carriers inside those spaces. This enables exact payload reconstruction while maintaining semantic constraints on the carrier sequence.**

Potential contributions:

1. Semantic-first coverless encoding
2. Dynamic session-specific carrier codebooks
3. Multi-dimensional carrier state spaces
4. Exact payload recovery independent of semantic reconstruction
5. Variable-capacity carrier coding
6. Cross-modal semantic families
7. Carrier-level robustness analysis

---

# 22. Experimental Evaluation Plan

You should evaluate DSSC against:

- current `exact_vcp`
- previous MDCS
- semantic-only baseline
- random carrier baseline

## Metrics

### 1. Exact recovery rate

```text
decoded message == original message
```

### 2. Bit error rate

```text
incorrect bits / total bits
```

### 3. Carrier efficiency

```text
payload bits / number of carriers
```

### 4. Carrier count

```text
number of carriers per message
```

### 5. Semantic coherence

Measure:

- adjacent carrier similarity
- chunk-to-carrier CLIP similarity
- overall narrative coherence

### 6. Robustness

Test:

- carrier substitutions
- missing carriers
- reordered carriers
- corpus mismatch
- cluster mismatch

### 7. Detectability

Compare:

```text
normal corpus sequences
vs
DSSC carrier sequences
```

using statistical and ML-based detection where possible.

---

# 23. Implementation Plan

## Phase 1 — Deterministic Prototype

Implement:

```text
Semantic block
→ manually defined cluster family
→ candidate set
→ fixed floor(log2(N)) capacity
→ deterministic ordering
→ exact decoding
```

Goal:

```text
Prove the architecture works.
```

---

## Phase 2 — Session Dynamic Codebook

Add:

```text
shared key
+
keyed permutation
```

Goal:

```text
same corpus
+
different session
=
different encoding mapping
```

---

## Phase 3 — Variable Capacity Coding

Replace:

```text
floor(log2(N))
```

with:

```text
enumerative / arithmetic-style state-space coding
```

Goal:

```text
use more of the available candidate states
```

---

## Phase 4 — ECC Alignment

Add:

```text
RS-aware interleaving
```

Goal:

```text
measure exactly how carrier failures translate to RS symbol errors
```

---

## Phase 5 — Cross-Modal Expansion

Allow:

```text
image
text
audio
```

inside semantic families.

Goal:

```text
increase candidate state space
without sacrificing semantic constraints
```

---

# 24. Recommended Codebase Changes

Suggested modules:

```text
src/
├── engine/
│   ├── dssc_encoder.py
│   ├── dssc_decoder.py
│   ├── semantic_family.py
│   ├── state_space.py
│   ├── dynamic_codebook.py
│   ├── carrier_ordering.py
│   └── ecc_interleaver.py
│
├── vcp/
│   ├── cluster_index.py
│   ├── family_index.py
│   └── corpus_manifest.py
│
├── security/
│   ├── session_key.py
│   └── keyed_permutation.py
│
├── protocol/
│   ├── payload_framing.py
│   ├── manifest.py
│   └── versioning.py
```

---

# 25. Core Pseudocode

## Encoding

```python
def encode_dssc(message, session_key, corpus):
    # Exact payload path
    framed = frame_payload(message.encode("utf-8"))
    codeword = rs_encode(framed)
    bitstream = BitStream(codeword)

    # Semantic planning path
    chunks = semantic_chunk(message)

    carriers = []

    for chunk in chunks:
        families = select_semantic_families(chunk, corpus)

        candidates = collect_candidates(
            corpus=corpus,
            families=families,
        )

        states = build_dynamic_state_space(
            candidates=candidates,
            session_key=session_key,
            chunk=chunk,
        )

        while chunk_needs_capacity(chunk, bitstream):
            symbol = consume_payload_symbol(
                bitstream,
                state_count=len(states),
            )

            carrier = states[symbol]
            carriers.append(carrier)

    return carriers
```

## Decoding

```python
def decode_dssc(carriers, session_key, corpus, manifest):
    validate_manifest(corpus, manifest)

    bitstream = BitStream()

    state_spaces = reconstruct_state_spaces(
        carriers=carriers,
        session_key=session_key,
        corpus=corpus,
        manifest=manifest,
    )

    for carrier, states in zip(carriers, state_spaces):
        symbol = states.index(carrier)
        bitstream.append_symbol(symbol, state_count=len(states))

    codeword = bitstream.to_bytes()

    framed = rs_decode(codeword)
    payload = unframe_payload(framed)

    return payload.decode("utf-8")
```

Note:

The real implementation must define unambiguously:

- how many bits/symbol information each state contributes
- how chunk boundaries are communicated or reconstructed
- how the final partial symbol is padded
- how semantic state spaces are reproduced by the decoder

These are protocol details, not optional implementation details.

---

# 26. Final Recommendation

Build DSSC, but do it incrementally.

## Version 1

```text
Semantic families
+
deterministic candidate sets
+
fixed-capacity state indices
+
RS-ECC
```

Target:

```text
100% exact recovery in controlled tests
```

## Version 2

```text
+
session-specific dynamic codebook
```

Target:

```text
better security and non-static mappings
```

## Version 3

```text
+
variable-capacity state-space coding
```

Target:

```text
better carrier efficiency
```

## Version 4

```text
+
ECC-aware interleaving
+
carrier erasure handling
```

Target:

```text
95–100% recovery under realistic corruption scenarios,
depending on the tested error model and ECC configuration
```

---

# 27. Final Answer on Accuracy

## Can this architecture achieve 95–100%?

**Yes, with an important distinction.**

### Exact deterministic environment

If:

- corpus is identical
- manifest matches
- state-space construction is deterministic
- session key is correct
- carrier sequence is received correctly

then the architecture should target:

```text
100% exact recovery
```

### Real-world environment

With:

- missing carriers
- substitutions
- transmission errors
- corpus changes
- metadata drift

the target should be:

```text
95–100% exact recovery
```

depending on:

- Reed-Solomon parameters
- interleaving strategy
- erasure support
- carrier error rate
- corpus synchronization

The correct scientific claim is therefore:

> **DSSC is designed for 100% exact recovery under synchronized deterministic conditions and should target 95–100% end-to-end recovery under bounded corruption, with the final robustness demonstrated empirically rather than assumed.**

---

# 28. Final Architecture in One Diagram

```text
                         SECRET MESSAGE
                                │
                                ▼
                      ┌─────────────────┐
                      │ Frame + RS-ECC  │
                      └────────┬────────┘
                               │
                               ▼
                        PAYLOAD BITSTREAM
                               │
             ┌─────────────────┴─────────────────┐
             │                                   │
             ▼                                   ▼
      SEMANTIC ANALYSIS                    SESSION KEY
             │                                   │
             ▼                                   ▼
       TOPIC BLOCKS                     KEYED CODEBOOK
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
                  SEMANTIC CLUSTER FAMILIES
                               │
                               ▼
                    VALID CARRIER CANDIDATES
                               │
                               ▼
                    DYNAMIC STATE SPACE
                               │
                               ▼
                PAYLOAD SELECTS VALID STATE
                               │
                               ▼
                       MEDIA CARRIER ID
                               │
                               ▼
                       MEDIA SEQUENCE


RECEIVER:

MEDIA SEQUENCE
       │
       ▼
RECONSTRUCT STATE SPACES
       │
       ▼
CARRIER → PAYLOAD SYMBOLS
       │
       ▼
RECOVER BITSTREAM
       │
       ▼
RS DECODE
       │
       ▼
FRAME VALIDATION
       │
       ▼
EXACT ORIGINAL MESSAGE
```

---

# Conclusion

The proposed DSSC architecture fixes the fundamental conceptual problem of the current DCASS design:

```text
OLD:
Payload chooses meaning

NEW:
Meaning constrains the carrier space;
payload chooses a state inside that meaningful space.
```

That separation is the core innovation.

The payload layer remains exact and error-correctable.

The semantic layer determines carrier plausibility.

The dynamic state-space layer provides capacity.

The session-specific codebook provides a path toward security.

The resulting system is a much stronger foundation for DCASS than either:

```text
byte → cluster → media
```

or:

```text
semantic chunk → approximate semantic reconstruction
```

because it attempts to preserve both:

```text
Exact information recovery
+
Semantic carrier coherence
```
