# P1-T6 — Dataset Survey & License Screening

| Field | Value |
|---|---|
| Phase / Task | Phase 1 / P1-T6 |
| Status | Complete |
| Date checked | 2026-09-28 |
| Depends on | P1-T1 … P1-T5 |
| Matrix | [`dataset_license_matrix.csv`](dataset_license_matrix.csv) (21 entries) |
| Downloads | **None** in this task |

---

## 1. Screening Basis

- **Intended use:** internal evaluation that selects a model for a **commercial** product. Using data to decide on a commercial product is treated as **commercial activity**. So "non-commercial research only" datasets are **INELIGIBLE**, even for evaluation (conservative; reviewer may revisit, P1-T3 §6 Q5).
- **A public download is not a license.** A missing license means no grant (P1-T3 R5).
- Statuses and rules are as in P1-T3 (ELIGIBLE / INELIGIBLE / REVIEW_REQUIRED; commercial use confirmed, so no BLOCKED).
- **No private or sensitive data.** Demographic labels (FairFace) are used only for **aggregate** fairness stratification (P0-T3 R-D7).

## 2. The Core Requirement: Same Identity at Different Quality

Three ways to get it, in order of license feasibility:

| Route | How | Datasets |
|---|---|---|
| **R-A Synthetic degradation** of a clean reference | The clean image *is* the same identity. Degrade it at controlled levels (P1-T7); compare restored output vs clean reference. | DS01 FairFace, DS02 public-domain portraits, (DS04 Unsplash Lite, DS06 FFHQ subset — review) |
| **R-B Cross-image same identity** | Several different photos of the same person. Degrade one, keep another clean as the identity reference (genuine pairs), plus impostor pairs. | DS02 public-domain portraits (multi-year portraits), DS03 in-house, (DS05, DS09/10 — review) |
| **R-C Real paired HQ/LQ** | The same person captured by good and bad devices/conditions | DS03 in-house consented (gold standard), (DS12 SCface — review) |

## 3. Dataset Candidates by Category

### 3.1 Synthetic-degradation source data (clean references)

| ID | Dataset | Resolution | Identities | License finding | Status |
|---|---|---|---|---|---|
| **DS01** | **FairFace** | 224×224 aligned | none linked | **CC BY 4.0** (README; HF card) | **ELIGIBLE** |
| **DS02** | **Public-domain official portraits** (to curate: US federal works under 17 U.S.C. §105; verified PD/CC0 Commons items) | High | multi-image per person possible | Public domain / CC0 **per item** | **ELIGIBLE** (per-item verification) |
| DS04 | Unsplash Lite 1.4.0 | High | none | Lite grants commercial *download, store, internal ML training*. Evaluation is not named; no disclosure; publicity/privacy clause | REVIEW_REQUIRED |
| DS06 | FFHQ license-filtered subset (PD/CC0/US-Gov/CC BY images) | 1024 | none | Per-image licenses include PD/CC BY, but the package/metadata is CC BY-NC-SA | REVIEW_REQUIRED |
| DS07 | FFHQ (full) | 1024 | none | CC BY-NC-SA 4.0 (dataset); NC images | INELIGIBLE (+ training contamination) |
| DS08 | CelebA / CelebA-HQ / CelebA-Test | var./1024 | 10,177 | "non-commercial research purposes only" | INELIGIBLE |
| DS15 | UTKFace | aligned | none | "non-commercial research purposes only" | INELIGIBLE |
| DS16 | DigiFace-1M (synthetic) | rendered | 110k | Research Use of Data Agreement; "non-commercial" | INELIGIBLE |
| DS17 | Face Synthetics | rendered | none | "non-commercial research" | INELIGIBLE |
| DS18 | CelebRef-HQ | HQ | 1,005 | "non-commercial research purposes only" | INELIGIBLE |
| DS21 | AuthFace HQ set | HQ | none | Non-commercial | INELIGIBLE |

### 3.2 Paired ground-truth / same-identity data

| ID | Dataset | Structure | License finding | Status |
|---|---|---|---|---|
| **DS03** | **In-house consented capture** (to be created) | Real HQ vs LQ per participant | Company-owned under consent | **ELIGIBLE once collected** under approved consent (Q-5) |
| DS12 | SCface | Mugshot HQ vs 5 surveillance cameras × 3 distances, 130 subjects | Signed release agreement required; PDF not machine-readable here | REVIEW_REQUIRED |
| DS10 | XQLFW | Cross-quality LFW pairs | Code MIT; data "All Rights Reserved"; inherits LFW | REVIEW_REQUIRED |
| DS09 | LFW | 5,749 identities, news photos | No explicit license; official site unreachable today | REVIEW_REQUIRED |
| DS05 | Casual Conversations v1/v2 (Meta) | Consented participants, multiple videos | Agreement requires sign-up; not reviewed | REVIEW_REQUIRED (**promising**: consented, built for fairness evaluation) |

### 3.3 Direct real-world low-quality data

| ID | Dataset | Nature | License finding | Status |
|---|---|---|---|---|
| DS11 | TinyFace | Native LR (≈20×16), 5,139 identities | No terms stated | REVIEW_REQUIRED |
| DS13 | QMUL-SurvFace | Native surveillance LR, 15,573 identities | No terms seen | REVIEW_REQUIRED |
| DS14 | WIDER FACE / WIDER-Test | In-the-wild, small/occluded | CC BY-NC-ND 4.0 | INELIGIBLE |
| DS19 | WebPhoto-Test / CelebChild-Test / LFW-Test | BFR real-world test sets | Derived from web/CelebA/LFW; no own license | REVIEW_REQUIRED (CelebChild likely INELIGIBLE via CelebA) |
| DS20 | VGGFace2 / MS-Celeb-1M | Identity sets | Withdrawn by owners | INELIGIBLE |

> **Gap:** there is **no ELIGIBLE public source of real-world degraded faces today.** Real-world evaluation (P0-T4 S-REAL, mandatory per master §8) depends on DS03 (in-house) or on reviewer approval of DS05/DS11/DS12/DS13.

## 4. Candidates That Can Proceed to Selection (P1-T7)

| Status | Datasets |
|---|---|
| **ELIGIBLE → selection & acquisition** | DS01 FairFace · DS02 public-domain portraits (curation, per-item) · DS03 in-house (once consent program exists) |
| **REVIEW_REQUIRED → planned, not downloaded** | DS05 Casual Conversations · DS12 SCface · DS04 Unsplash Lite · DS06 FFHQ PD/CC-BY subset · DS10 XQLFW · DS11 TinyFace · DS13 QMUL-SurvFace · DS09 LFW · DS19 |
| **INELIGIBLE** | DS07, DS08, DS14, DS15, DS16, DS17, DS18, DS20, DS21 |

## 5. Reviewer Questions (datasets)

1. Is internal **evaluation** of a commercial product's model on "non-commercial research" data prohibited? (This screening assumes yes.)
2. Does Unsplash Lite's "internally use… to train ML models for internal business purposes" cover **evaluation**?
3. Priority: review the **Casual Conversations v2** and **SCface** agreements. They are the most valuable real-world and paired sources.
4. FFHQ per-image PD/CC BY subset: is selecting images via NC-SA metadata acceptable?
5. Legal basis and consent design for **DS03** (DPDP obligations; P0 Q-5).

## 6. Acceptance Criteria

| Criterion | Status |
|---|---|
| Candidates cover needed scenarios (synthetic, paired, real LR, same-identity, fairness) | ✅ §2–§3 |
| Same-identity/different-quality explicitly addressed | ✅ §2 routes R-A/R-B/R-C |
| Licensing documented from primary sources | ✅ matrix (sources + date). Unreachable official pages are recorded as such. |
| Synthetic degradation as fallback/companion | ✅ R-A (DS01, DS02) |
| No unlicensed dataset downloaded | ✅ nothing downloaded |
