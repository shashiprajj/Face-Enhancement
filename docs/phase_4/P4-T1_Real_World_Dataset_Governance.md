# P4-T1 — Real-World Dataset Acquisition & Governance

| Field | Value |
|---|---|
| Phase / Task | Phase 4 / P4-T1 |
| Date | 2026-10-03 |
| Status | **Governance framework, freeze tooling and tests complete. No real-world data: G-3 BLOCKED.** Final real-world evaluation is stopped at this boundary (Phase 4 rule 17). |
| Register | `data/registry/p4_realworld_sources.csv` |
| Freeze tool | `scripts/p4_freeze_realworld.py` (+ `tests/test_p4_freeze_realworld.py`, 8 tests) |
| Upstream spec | `configs/phase2/realworld_collection_v1.yaml` (`rw-spec-1.0.0`), validator `scripts/p2_validate_collection.py` |
| Blocked evidence | `data/manifests/phase4/realworld/rw-v1-{DS03,VENDOR,DS05,DS02}/BLOCKED.json` |
| User decision (2026-10-03) | Stop at the G-3 boundary. Acquire no new source and do not curate DS02. |

---

## 1. Intended Production Image Conditions

The Phase 0 spec (master §6, P0-T3) defines the conditions the evaluation population must cover:

| Condition | Phase 0 target | rw-spec code | Notes |
|---|---|---|---|
| Low resolution / distance | Mild–Moderate primary, Severe secondary, Extreme flag/decline | DIST 3/6/10 m, LCDV | |
| Defocus blur | same | DFOC | |
| Motion blur | Mild–Moderate, Severe best effort | MOTN | |
| Sensor noise / low light | same | LOWL 50/10 lux | |
| Compression / re-encode / screenshots | same | COMP messaging ×1/×3, 480p stream | Screenshots: add as a COMP level in the next spec revision |
| Exposure / contrast / glare / backlight | Conditional (C7) | LOWL, BKLT | No candidate corrects these (Phase 3) |
| Occlusion | Keep the occluder; never invent | OCCL (optional) | |
| Poor crop / pose | Process and warn | pose yaw ±30° on HQ_REF; crop mode | |
| Mixed degradation | Primary | natural combinations of the above | |

**Conflict (recorded, not resolved here):** the Phase 4 plan lists "ID/document captures" as a candidate condition, but P0 master §3 puts PAN/KYC/**document processing** out of scope. A face photographed *from* a document could still be an FES input, while document *processing* is not. Product must confirm whether document-sourced faces are in the input domain (Q-1). Until then they are **not** part of the population spec.

## 2. Candidate Sources and Legal Status (register)

`data/registry/p4_realworld_sources.csv`. Re-confirmed 2026-10-03 against P1-T6 and P2-T5. No source was newly downloaded.

| Source | Real capture | Same-identity HQ ref | Legal status | Blocking reason |
|---|---|---|---|---|
| DS03 in-house consent collection | ✅ | ✅ | **PENDING_LEGAL_REVIEW** | No reviewer; consent form + DPDP review not approved |
| Vendor collection to rw-spec | ✅ | ✅ | NOT_EVALUATED | No procurement |
| DS05 Casual Conversations v2 | ✅ | ✅ | NOT_REVIEWED | Agreement unread; leaning NC |
| DS02 public-domain portraits | ✅ (mostly HQ) | ✅ | NOT_APPROVED_FOR_P4 | User decision; publicity/privacy + biometric processing unreviewed |
| DS12 SCface | ✅ | ✅ | REJECTED | Non-commercial agreement |
| DS14 WIDER FACE | ✅ | ❌ | REJECTED | CC BY-NC-ND |
| DS09/10, DS11, DS13 | ✅ | partial | NOT_REVIEWED | No licence grant |

**Public availability is not treated as commercial permission.** Only `legal_status = APPROVED` with an approval reference, a named reviewer, a date and `evaluation` in the approved uses passes the gate (§5).

## 3. Identity / Reference Requirements

| Requirement | Rule | Enforced by |
|---|---|---|
| Identity provenance | Consent-linked participant id (`P-xxxxxx`, random, not derived from identity) | p2 validator |
| Same-identity HQ reference | Every LQ capture links to an HQ_REF of the **same participant**, inter-ocular distance ≥ 120 px | p2 validator + freeze (pairing re-checked) |
| Cross-image genuine pairs | HQ_REF of other sessions/poses of the same participant (TAR) | data model |
| Impostor pairs | HQ_REFs of other participants (FAR operating point) **from the dev split only** | P4-T4 |
| Degradation label | `condition` + `level` as captured (observed, not synthetic) + P4-T2 measured features | p2 validator, P4-T2 |
| No PII | Forbidden columns, regex value scan, GPS EXIF rejected, age bands only | p2 validator |
| Withdrawal | `withdrawn` participants never enter the frozen manifest | p2 validator + freeze |

## 4. Dev/Test Separation and Leakage Checks

- **Split unit = participant:** `sha256("fes-ds03-v1" + participant_id) mod 100 < 30 → dev`, else test. No training split.
- **Dev** is used for every threshold: quality gate, routing, the TAR/FAR operating point and the evaluator agreement band. **Test** is opened once, after `freeze.json` exists and the P4-T6 gate is frozen.
- Leakage checks in `p4_freeze_realworld.py`. Any hit refuses the freeze.

| Check | Method |
|---|---|
| Participant in > 1 split | set check |
| Exact duplicates | SHA-256 groups |
| Near duplicates | 64-bit pHash, Hamming ≤ 3 (P2-T2 rule), union with exact groups |
| Duplicate group spanning participants or splits | refuse |
| Same person enrolled twice | optional SFace identity screen on HQ_REFs across participants, cos ≥ 0.59 (P2-T7 data-derived 1:N value); embeddings in memory only |
| Population separation | Every frozen row carries `data_kind = REAL_WORLD`. Synthetic manifests live under `phase3/` and `phase4/population_*`, are labelled SYNTHETIC, and are never merged. |

## 5. Freeze Procedure (executable)

```
1. scripts/dock.sh python scripts/p2_validate_collection.py --root <col> --manifest <acq.csv.gz>   # spec + PII + pairing
2. Legal reviewer sets the register row: legal_status=APPROVED, approval_ref, reviewer, approval_date, approved_uses
3. scripts/dock.sh python scripts/p4_freeze_realworld.py --source-id DS03 --acq <acq.csv.gz> --root <col> \
       --version rw-v1 --identity-screen
   → data/manifests/phase4/realworld/rw-v1/{manifest.csv.gz, freeze.json}
```

`freeze.json` records the SHA-256 of the manifest, the acquisition manifest and the register row, plus the spec version, git head, leakage results and thresholds. A version is **immutable**: re-freezing with different content is refused (exit 3). Without legal evidence the tool exits 2 **before opening any image** and writes `BLOCKED.json`.

## 6. Validation (tests)

`tests/test_p4_freeze_realworld.py`, 8 passed in `fes-dev:cpu-p2`. Fixtures are generated in a temp directory from a FairFace reference. They are not data.

| Test | Asserts |
|---|---|
| legal gate × 4 | PENDING status, missing approval_ref, missing reviewer, or a use other than `evaluation` → exit 2 + BLOCKED_G3 |
| committed register | **every** current source is blocked |
| cross-participant duplicate | identical images under two participants → refused |
| clean collection | freezes; identical re-freeze is a no-op; tampered file → integrity failure |
| immutability | the same version with different content → exit 3 |

## 7. Dataset Card — `rw-v1` (real-world evaluation population)

| Field | Value |
|---|---|
| Status | **NOT CREATED. No approved source (G-3).** |
| Intended use | Evaluation only: identity preservation, fabrication, quality gate and routing on real degraded captures |
| Out of scope | Training, publication of faces (unless the `publication_of_samples` scope is granted), any recognition use |
| Composition (planned) | 30-participant pilot (rw-spec), DIST/LOWL/MOTN/COMP/LCDV required, DFOC/BKLT/OCCL optional; final N by power analysis on pilot variance (TH-11) |
| Splits | participant-level, 30 % dev / 70 % test |
| Demographics | Optional self-reported (age band, gender, Monk skin tone). **No fairness claim below a powered sample** (Phase 4 rule 14). |
| Provenance & hashes | `freeze.json` (on creation) |
| Retention | Project end + 90 days, or within 30 days of withdrawal |
| Known limitations | One organisation's capture devices; pilot size not powered for strata |

## 8. Decision

**Final real-world evaluation is STOPPED at this boundary.** P4-T4's real-world arm is blocked. Every other Phase 4 task proceeds on the clearly labelled **SYNTHETIC** `fes-syn-v1` populations, and none of its results are presented as real-world evidence.

Unblocking needs a named licence/compliance reviewer who approves the DS03 consent form and spec (or a vendor contract), records the approval in the register, collects the pilot, and then runs §5.

## 9. Files

`data/registry/p4_realworld_sources.csv` · `scripts/p4_freeze_realworld.py` · `tests/test_p4_freeze_realworld.py` · `data/manifests/phase4/realworld/*/BLOCKED.json` · `configs/phase4/p4_protocol.yaml`
