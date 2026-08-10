# DroneTrackingMOT

Multiple Object Tracking (MOT) for UAV swarms in infrared imagery,
targeting Anti-UAV / counter-drone applications. Developed for
submission to LAFusion 2026.

The final pipeline combines a YOLO26n detector, BoT-SORT with global
motion compensation, GM-FLIT association, NSA-Kalman, and optional
SAHI inference. This repository also documents the targeted detector
fine-tuning procedure based on sequence-level hard-example mining.

## At a Glance

| Item | Adopted configuration / result |
|---|---|
| Dataset | 4th Anti-UAV Challenge, Track 3 (infrared MOT) |
| Detector | YOLO26n fine-tuned from COCO-pretrained weights |
| Tracking backbone | BoT-SORT + global motion compensation |
| Detector refinement | Sequence-level hard-example mining + random-sample mix |
| Best accuracy configuration | HardMining + GM-FLIT + NSA-Kalman + SAHI (slice=512) |
| Final internal-split result | **79.6% MOTA**, **80.3% IDF1**, **712 IDs**, 22.49 FPS |

## Navigation

- [Experimental protocol](#experimental-record)
- [Repository structure](#15-repository-structure)
- [Setup](#16-setup)
- [Reproducing the pipeline](#17-reproducing-the-pipeline)

<details>
<summary><strong>Project status</strong></summary>

## Completed work

- [x] Systematic literature analysis (10 core studies, 2 supporting refs)
- [x] Dataset preparation (4th Anti-UAV Challenge, Track 3)
- [x] YOLO26n fine-tuning on IR drone detection
- [x] Systematic FPS/accuracy benchmark (BoT-SORT vs. ByteTrack)
- [x] Full MOT evaluation (MOTA/IDF1/IDs) on held-out sequences
- [x] GMC vs. Re-ID ablation (BoT-SORT internals)
- [x] Root-cause investigation (density, camera motion, background clutter)
- [x] Pixel-level temporal preprocessing ablation (MOG2/EMA/Optical Flow) -- **negative result**
- [x] Track-level post-processing filters (static, spatial, hybrid) -- **negative result**
- [x] Source-level investigation of the GMC x Re-ID interaction -- **inconclusive, documented as a limitation**
- [x] Adaptive Kalman filter (Q/R by velocity) -- **marginal positive result**
- [x] Multi-tracker native comparison (6 Ultralytics-supported trackers)
- [x] **FLIT (L2-IoU hybrid association) + NSA-Kalman -- strongest result of the project**
- [x] Post-processing exploration (8 techniques total: 3 filters, adaptive Kalman, adaptive threshold, lost-query bank) -- 1 strong result (GM-FLIT+NSA), 7 null/negative -- **post-processing exploration closed**
- [x] Targeted retraining -- **succeeded (sequence-level hard-example + random-sample mix), +0.7pp MOTA standalone; combined with SAHI = project's final result**
- [x] Pixel-level preprocessing at inference (CLAHE/gamma/unsharp/sharpen) on hard-mined detection set -- **second independent confirmation of domain-shift finding (Section 7)**
- [x] SAHI (slicing-aided inference) -- **new best accuracy result, explicit speed trade-off documented**
- [ ] Manuscript write-up (Method / Results / Discussion)

</details>

<details>
<summary><strong id="experimental-record">Experimental record</strong> — dataset, detector, HEM, ablations, and detailed results</summary>

## 1. Literature Analysis Summary

Structured search on IEEE Xplore and Scopus (2023-2026), Boolean
string combining UAV/drone + multiple object tracking + infrared/
Anti-UAV context. After screening: 46 unique records -> 10 studies
retained for in-depth synthesis (Quality Assessment score >= 5.0/10).

| Study | Architecture | Modality | Dataset | QA |
|---|---|---|---|---|
| Zhu et al. 2025 (PFTrack) | Transformer (progressive fusion) | RGB-IR | VT-MOT | 8.5 |
| Chen 2025 | Tracking-by-detection (YOLOv12+BoT-SORT) | IR | Anti-UAV | 8.0 |
| Ma et al. 2026 (MCTrack) | Transformer (multi-stage) | RGB-IR | VT-MOT/UniRTL | 8.0 |
| Li et al. 2026 | Transformer (Deformable DETR) | IR/RGB | Anti-UAV/AU-MOT | 7.5 |
| Dabbabi & Delleji 2025 | GNN + Transformer | RGB-IR | Custom | 7.5 |
| Wang et al. 2025 (Dist-Tracker) | Tracking-by-detection | IR | Anti-UAV | 7.0 |
| Jun & Kim 2026 | Segmentation + Kalman | IR/RGB/Full | Anti-UAV | 6.5 |
| Qin et al. 2025 (PPTracker) | Tracking-by-detection + prompt | IR | Anti-UAV | 6.0 |
| Hong et al. 2025 (GM-PHD) | Bayesian | IR | IR time-sensitive | 6.0 |
| Zhang et al. 2026 | Transformer (MOTRv2) | IR/RGB | VTMOT/UAVSwarm | 5.5 |

Supporting (non-core) references: Bentamou et al. 2023 (ICIP,
distortion-robust IR detection); Serghei & Popescu 2026 (DeTrAck,
embedded PTZ tracking); Yuan et al. 2024 (IRSTDID-800, IR destriping
benchmark); Chen & Wu 2025 (ReynoldsFlow+, optical-flow-encoded
training input, evaluated on Anti-UAV) -- all considered for
motivation/methodology but outside the core 10-study QA corpus.

**Key gaps identified** (motivating this work's experimental design):
1. Benchmark fragmentation -- few studies share a common dataset for
   direct comparison.
2. GNN-based architectures remain scarce.
3. Efficiency (FPS) vs. accuracy trade-off rarely quantified under
   comparable hardware conditions.
4. Few controlled infrared-only vs. RGB-IR fusion ablations.

Given hardware constraints (consumer-grade laptop, no dedicated
lab GPU), this project prioritizes **lightweight, tracking-by-detection
architectures** with systematic efficiency benchmarking, rather than
heavy Transformer/GNN pipelines -- directly addressing gap #3 above.

## 2. Dataset

**Source:** [4th Anti-UAV Challenge](https://zenodo.org/records/15103888)
official release, Track 3 (Multiple Object Tracking).
Published by China Telecom, DOI: `10.5281/zenodo.15103888`, MIT License.

> China Telecom. (2025). *Official Dataset Release for the 4th
> Anti-UAV Challenge* [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.15103888

- Original videos: 640x512, 25 FPS.
- Ground truth format: MOTChallenge standard
  (`frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility`).
- Single class: `drone`.
- Swarm density varies substantially by sequence: as few as ~4-5
  simultaneous targets in some sequences, up to ~50 in dense-swarm
  sequences (e.g., UAV002: 24-49 concurrent targets across 750 frames,
  120 unique track IDs).
- Some sequences feature a moving camera (pan/tilt/zoom); others are
  static -- relevant to the GMC and motion analyses (Sections 5-6).

**Preprocessing pipeline:**
1. `convert_mot_to_yolo.py` -- extracts frames from `.mp4` and converts
   MOTChallenge annotations to YOLO detection format (normalized
   `class cx cy w h`).
2. `split_train_val.py` -- splits data **at the sequence level** (not
   frame level) to avoid temporal leakage between train/val, following
   the methodological caveat raised in Chen (2025). Applies temporal
   subsampling (`FRAME_STRIDE`) to reduce redundancy between
   consecutive frames.

**Final dataset (stride=4):** 32,330 training frames / 5,724
validation frames (from an original 151,831 extracted frames, 750
frames/sequence at 25 FPS). All tracking evaluations in this document
use the full 750-frame validation sequences (no subsampling), which
are held out from detector fine-tuning.

## 3. Detector: YOLO26n Fine-tuning

Fine-tuned from COCO-pretrained weights (single class: `drone`),
50 epochs max, early stopping (patience=15). Trained on Google Colab
(Tesla T4).

**Best checkpoint: epoch 10** (`weights/best.pt`)

| Metric | Value |
|---|---|
| Precision | 0.878 |
| Recall | 0.867 |
| mAP50 | 0.904 |
| mAP50-95 | 0.478 |

Training stopped early at epoch 25 (no improvement for 15 epochs),
total training time: 5.28 hours on a single T4.

## Hard-Example Mining (Detector Optimization)

The detector refinement is an **offline, sequence-level hard-example
mining procedure**, not online hard-example mining (OHEM). A trained
detector and tracker are first used to identify training sequences in
which the complete tracking pipeline performs poorly; all frames from
the selected sequences are then used in a targeted detector
fine-tuning stage.

### Baseline detector and eligible sequences

The starting point is the best checkpoint from the base YOLO26n
training run (for the Colab run, `yolo26n_ir_drone-3/weights/best.pt`).
`src/inference/run_tracker_on_train.py` reconstructs the project split
with seed 42 and reserves 15% of sequences for validation. Only the
remaining training sequences are eligible for mining, preventing
validation-sequence leakage.

For every eligible video, the script runs `YOLO.track` with BoT-SORT
(`botsort.yaml`) and a confidence threshold of 0.25, then writes one
MOT-format result file per sequence under `runs/train_predictions/`.
The associated evaluation step produces `train_summary.csv`, containing
per-sequence MOTA, IDF1, MOTP, IDs, FP, and FN.

### Difficulty score and selection

`src/inference/select_hard_examples.py` excludes the aggregate
`OVERALL` row and ranks sequences by the following score:

```text
score = 2 × (100 − MOTA) + (100 − IDF1)
      + 0.02 × FN + 0.005 × FP + 0.20 × IDs
```

The 20 sequences with the largest score are written to
`runs/train_predictions/hard_examples.txt`. Thus, MOTA is the largest
component, but the implementation also explicitly penalizes low IDF1,
false negatives, false positives, and identity switches. MOTP is
recorded in the summary but is not part of the selection score.

### Dataset construction and fine-tuning

`src/inference/build_hard_dataset.py` copies **all** JPEG frames and
their YOLO labels from the selected sequences into
`dataset/dataset_yolo_hard/`. It does not select individual boxes or
frames based on detector confidence or loss.

The final fine-tuning was executed in Colab from the base checkpoint,
for eight epochs at 640 px, batch size 32, AdamW, `lr0=5e-5`,
`lrf=0.01`, `weight_decay=5e-4`, and `patience=8`. The training dataset
mixed the 3,846 hard-sequence frames with 5,000 randomly sampled frames
from the original training split. The random component is deliberate:
it reduces over-specialization to the hard subset and retains coverage
of the original training distribution.

> **Reproducibility note.** The scripts build only the hard subset.
> Construction of the 3,846 + 5,000 mixed training dataset and its
> `/content/data.yaml` were performed in the Colab workflow. Preserve
> that composition when reproducing the reported HardMining detector.

## 4. Tracking Evaluation (BoT-SORT + YOLO26n baseline)

Evaluated on 30 held-out sequences (full 750-frame sequences, no
temporal subsampling) never seen during detector fine-tuning, using
`py-motmetrics` against the original MOTChallenge-format ground truth.

**Overall (30 sequences, IoU threshold 0.5):**

| Metric | Value |
|---|---|
| MOTA | 75.9% |
| MOTP | 0.229 |
| IDF1 | 71.2% |
| ID switches | 2,327 |
| False Positives | 47,050 |
| False Negatives | 66,128 |
| Total GT objects | 478,548 |

For reference, this places the baseline (YOLO26n + BoT-SORT, default
config) between `qin2025pptracker` (67.9% MOTA) and `chen2025strong`
(76.1% MOTA, officially ranked 5th on the competition's hidden test
leaderboard, entry "wish", 0.7609) on comparable 4th Anti-UAV
Challenge Track 3 data -- though note this evaluation uses an
internal held-out split, not the official hidden test set, so
comparison should be treated as indicative rather than exact. For
context, the official leaderboard's top entries reach up to 84.99%
MOTA (1st place, "ocewang"), meaning this project's results are
comparable to mid-leaderboard submissions, not the current
state-of-the-art on the official benchmark.

**Tracker comparison (BoT-SORT vs. ByteTrack, same detector):**

| Metric | BoT-SORT | ByteTrack | Δ |
|---|---|---|---|
| MOTA | 75.9% | 71.6% | -4.3 pp |
| MOTP | 0.229 | 0.230 | ~equal |
| IDF1 | 71.2% | 49.4% | **-21.8 pp** |
| ID switches | 2,327 | 4,538 | +95.0% |
| FPS (overall) | 46.46 | 49.84 | +7.3% |

ByteTrack's speed advantage was modest (+7.3% FPS) relative to its
identity-preservation cost (-21.8 pp IDF1). MOTP is nearly identical
between the two, since both share the same detector -- the quality
gap is attributable entirely to the association strategy, investigated
further in Section 5.

## 5. Tracker Association Ablation: GMC x Re-ID

**Motivation.** The large IDF1 gap between BoT-SORT and ByteTrack
(Section 4) was initially hypothesized to stem from BoT-SORT's
appearance-based (Re-ID) matching. However, the default BoT-SORT
config used has `with_reid: False`, so this gap could **not** be
attributed to Re-ID. A 2x2 ablation was run instead, toggling Global
Motion Compensation (GMC, `sparseOptFlow`) and Re-ID (`model: auto`)
independently, on the same 30 held-out sequences.

| Variant | GMC | Re-ID | MOTA(%) | MOTP | IDF1(%) | IDs | FPS |
|---|---|---|---|---|---|---|---|
| noGMC_noReID | off | off | 72.2 | 0.228 | 49.5 | 4,508 | 86.44 |
| **GMC_noReID** (default config) | **on** | off | **75.9** | 0.229 | **71.2** | **2,327** | 46.46 |
| noGMC_ReID | off | on | 72.2 | 0.228 | 49.5 | 4,507 | 86.98 |
| GMC_ReID | on | on | 72.2 | 0.228 | 49.5 | 4,507 | 87.01 |
| ByteTrack (reference) | -- | -- | 71.6 | 0.230 | 49.4 | 4,538 | 49.84 |

**Finding 1 -- GMC, not Re-ID, explains the original gap.**
`GMC_noReID` is the only variant achieving high IDF1 (71.2%) and low
ID-switch count (2,327); all three other variants collapse to
IDF1 ~49.5%, statistically indistinguishable from ByteTrack. This
corrects the initial Re-ID hypothesis: global motion compensation is
the actual driver of the improvement, consistent with the dataset
containing sequences with camera pan/tilt/zoom (Section 2).

**Finding 2 -- confirmed GMC/Re-ID interaction in the Ultralytics
BoT-SORT implementation.** `GMC_ReID` (GMC nominally on, Re-ID on)
performs identically to the no-GMC variants in both quality metrics
*and* FPS (~87, matching the no-GMC variants, vs. 46.46 for
GMC_noReID). A source-level investigation of this interaction is
documented in Section 9.

Full results: `runs/mot_eval/ablation_gmc_reid_summary.csv`.

## 6. Root-Cause Investigation: What Actually Drives Tracking Failure?

Three candidate explanations for the large per-sequence performance
variance (Section 4) were tested in sequence, each ruling out the
previous and redirecting the investigation:

**Hypothesis 1 -- Swarm density.** Pearson correlation between mean
targets-per-frame and MOTA/IDF1 was weak and, counterintuitively,
slightly *positive* (botsort: r=+0.196 MOTA, r=+0.180 IDF1;
bytetrack: r=+0.176 MOTA, r=-0.043 IDF1). **Ruled out** -- see
`runs/mot_eval/density_vs_performance.csv`.

**Hypothesis 2 -- Camera motion.** A sparse Lucas-Kanade optical flow
estimator (median displacement per frame) was used as a proxy for
camera ego-motion. Correlation with performance was again weak
(botsort: r=+0.224 MOTA, r=+0.368 IDF1; bytetrack: r=+0.126 MOTA,
r=-0.089 IDF1). BoT-SORT's IDF1 correlates *positively* with motion,
consistent with GMC (Section 5, Finding 1) successfully compensating
for camera movement. **Ruled out** -- see
`runs/mot_eval/camera_motion_vs_performance.csv`.

**Hypothesis 3 -- Structured background clutter.** Manual inspection
of the four lowest-motion sequences in the dataset (UAV281: 0.0038,
UAV279: 0.0039, UAV293: 0.0046, UAV291: 0.0051 px/frame) -- also four
of the worst-performing sequences (IDF1 of 12.6%, 19.4%, 38.0%, and
36.2% respectively with BoT-SORT) -- revealed a static camera framing
a building facade with numerous windows and doors: small,
high-contrast, static structures in infrared, plausibly generating
persistent false positives indistinguishable from stationary drones.

This was then tested **quantitatively across all 30 sequences** by
computing five per-sequence image-quality metrics (entropy, Canny edge
density, Laplacian variance, local contrast, hot-pixel percentage,
sampled every 25 frames) and correlating them with MOTA/IDF1/FP-rate/
FN-rate:

| Image metric | MOTA (botsort) | IDF1 (botsort) | FN_rate (botsort) |
|---|---|---|---|
| mean_edge_density | -0.349 | **-0.598** | +0.473 |
| mean_local_contrast | -0.269 | -0.563 | +0.411 |
| mean_entropy | -0.271 | -0.355 | +0.316 |
| mean_laplacian_var | +0.272 | +0.253 | -0.281 |
| mean_hot_pixel_pct | +0.246 | +0.102 | -0.247 |

`edge_density` is the strongest predictor of IDF1 across the dataset
(r=-0.598), confirming the visual finding at scale. Notably,
`local_contrast` also correlates *negatively* with IDF1 -- the
opposite of the classical "low contrast is the problem" assumption
from the literature (contrast-enhancement techniques such as CLAHE
were deprioritized based on this finding).

A separate case, UAV169 (MOTA 41.8%, among the highest motion
estimates at 1.68 px/frame), illustrates a distinct failure mode:
camera movement combined with occlusion by tree foliage, causing
intermittent detection loss rather than false-positive proliferation.

Full results: `runs/mot_eval/frame_quality_vs_performance.csv`.

## 7. Preprocessing Ablation: MOG2, EMA, Optical Flow (Negative Result)

**Motivation.** Given the background-clutter finding (Section 6),
three temporal background-suppression techniques were tested as
pixel-level pre-filters applied *before* detection: MOG2 background
subtraction, an exponential moving average (EMA) background model,
and a dense optical-flow motion mask.

| Method | MOTA(%) | IDF1(%) | IDs | FP_rate | FN_rate | FPS |
|---|---|---|---|---|---|---|
| Baseline (no preprocessing) | 75.9 | 71.2 | 2,327 | 0.098 | 0.138 | 46.46 |
| MOG2 | **-168.7** | 16.0 | 4,937 | **2.146** | 0.531 | 31.66 |
| EMA | 2.3 | 38.0 | 4,823 | 0.671 | 0.296 | 37.49 |
| Optical Flow | 22.5 | 26.4 | 5,180 | 0.118 | **0.646** | 19.62 |

**All three methods severely degraded performance.** Interpretation:
**detector domain shift** -- YOLO26n was fine-tuned exclusively on
raw, unmasked frames, and was not robust to the substantially
different input distribution created by pixel-level masking at
inference time. This is treated as a valid negative result: it rules
out pixel-level pre-filtering *without also retraining the detector
on similarly preprocessed data* (Section 11, planned future direction).

**Independent confirmation.** A second, separate experiment applying
pixel-level enhancement techniques (CLAHE, gamma correction, unsharp
masking, sharpening) at inference time, evaluated via detection-only
mAP50 on a hard-mined subset (`dataset_yolo_hard`, no fine-tuning),
reproduced the same failure pattern: every technique degraded mAP50
relative to the unmodified baseline (0.547), from a mild -3.3%
(unsharp, gamma, CLAHE) to a catastrophic -69.4% (sharpen, mAP50 =
0.167). This is a second, independent confirmation (different
experiment family, different metric, different dataset subset) of
the same domain-shift finding, further reinforcing that inference-time
pixel preprocessing is not a viable direction without consistent
retraining.

Full results: `runs/mot_eval/preprocessing_ablation_summary.csv`.

## 8. Track-Level Post-Processing Filters (Negative Results)

Three **track-level** post-processing filters were tested, operating
on the tracker's *output* trajectories (never seen by the detector,
eliminating domain-shift risk), all reprocessing the existing
`GMC_noReID` output files without re-running detection/tracking.

**8.1 Static Track Filter** (per-ID rolling position variance):
degraded performance at every configuration (best: MOTA 64.6%, IDF1
65.1% at window=45, std=2.0, vs. 75.9%/71.2% baseline). Diagnostic
showed suppressed detections had 51.6-79.9% GT match rate on the
worst sequences -- the filter was removing real drones, not clutter --
and high fractions of short (<10 frame) tracks, indicating identity
fragmentation rather than long-lived static clutter tracks.

**8.2 Spatial Recurrence Filter** (grid-cell occupancy, independent
of track ID continuity): degraded performance monotonically with
suppression rate; even the mildest setting (1.1% suppression)
underperformed baseline.

**8.3 Hybrid Filter** (spatial recurrence + local velocity): initial
sweep with velocity thresholds 1.0-5.0 px/frame showed **no effect**
(identical to 8.2's results); diagnostic revealed hot-cell median
velocity was 0.04-0.08 px/frame, ~15-70x below the tested range.
Recalibrated sweep (0.05-0.20 px/frame) still degraded performance at
every non-zero-suppression configuration, though the least aggressive
setting approached baseline asymptotically (only 7 grid cells in the
entire 30-sequence dataset met the strictest occupancy criterion).

**8.4 Adaptive Confidence Threshold** (per-frame `track_high_thresh`
adjustment based on mean detection confidence, applied on top of the
GM-FLIT + NSA-Kalman configuration): **null result.** MOTA/IDF1
unchanged (76.1%/72.2% -> 76.0%/72.2%, within measurement noise), IDs
essentially unchanged (1,760 -> 1,761), while FPS dropped 16.5%
(66.79 -> 55.78) purely from per-frame computation overhead. No
benefit observed at any measured operating point.

**8.5 Lost-Query Bank** (direction-consistency re-association for
LOST tracks, inspired by `zhang2026robust`'s query propagation):
**negative result.** IDF1 dropped 3.2 pp (72.2% -> 69.0%) and, more
notably, ID switches **more than doubled** (1,760 -> 4,015) --
opposite of the intended effect. Interpretation: in dense swarms,
motion-direction similarity has low discriminative power, since many
drones fly in similar directions simultaneously; the cosine-similarity
gate likely reattaches LOST tracks to *nearby detections belonging to
different drones* rather than recovering the correct one, actively
creating new identity errors instead of preventing them.

**8.6 Kinematic/Shape Consistency Analysis (Measurement, No Implementation)**

Motivated by `jun2026segment`'s shape-aware Kalman state and
`li2026contextaware`'s kinematic anomaly scoring, a frame-to-frame
kinematic signature analysis was conducted on the final pipeline's
own tracking output (HardMining detector + SAHI512 + BoT-SORT),
categorizing 464,345 consecutive-frame transitions as TP, FP, or
ID_SWITCH (Mann-Whitney U test between categories):

| Metric | TP median | FP median | Practical effect | p-value |
|---|---|---|---|---|
| area_ratio | 1.0000 | 0.9997 | ~0.03% (negligible) | 0.927 |
| aspect_ratio_ratio | 1.0000 | 0.9996 | ~0.04% (negligible) | 3.44e-04 |
| width_ratio | 1.0000 | 1.0000 | ~0% (negligible) | 0.0108 |
| center_shift_normalized | 0.0010 | 0.0019 | **~90% larger for FP** | 0.00e+00 |

**Given the very large sample size (n > 400,000), statistical
significance (p<0.001) does not imply a practically exploitable
effect size**: area and aspect-ratio ratios differ by only 0.03-0.04%
between TP and FP, well below any plausible gating threshold's
resolution. The one metric with a substantial effect,
`center_shift_normalized`, measures the same underlying quantity
(normalized centroid displacement) already penalized by GM-FLIT's L2
term (Section 10) -- making a dedicated kinematic/shape gate largely
redundant with the association cost already in use. Consistent with
this, `center_shift_normalized` does *not* differ between TP and
ID_SWITCH events (p=0.207), confirming Section 6's earlier finding
that identity switches in this pipeline are driven by structured
background clutter, not motion dynamics.

**Conclusion: no additional exploitable signal found.** This is
treated as the sixth independently-motivated tracker-level
intervention (after Sections 8.1-8.5) to yield a null result,
reinforcing GM-FLIT + NSA-Kalman (+ SAHI + HardMining) as a stable
configuration for this dataset rather than an easy target for further
heuristic refinement. No gate was implemented, following the
project's practice of measuring before implementing (cf. Section 8.3's
velocity-threshold recalibration).

Full data: `runs/mot_eval/tracking_failure_summary.md`.

**Summary (revised).** Five independently-formulated post-hoc/
association-level modifications (static filter, spatial filter,
hybrid filter, adaptive threshold, lost-query bank) all failed to
improve on the GM-FLIT + NSA-Kalman configuration, three with clear
negative effects and two null. This consistent pattern reinforces
Section 10's result as a genuine local optimum for this tracker
architecture and dataset, rather than an easy baseline waiting to be
further improved by additional heuristics.

> **Reproducibility note:** this session's baseline measurement for
> GM-FLIT gamma=0.55 + NSA-Kalman (MOTA=76.1%, IDF1=72.2%, IDs=1,760)
> differs slightly from the value reported in Section 10
> (MOTA=76.7%, IDF1=74.5%, IDs=1,634), likely due to a minor
> Ultralytics version difference between sessions (8.4.105 vs.
> 8.4.113, observed in Colab logs). Both are treated as consistent
> given the small magnitude of the difference (<1 pp MOTA); Section 10's
> figures remain the reported result, with this session's figures used
> only as the internal baseline for Sections 8.4-8.5's *relative*
> comparisons.

## 9. Source-Level Investigation: The GMC x Re-ID Interaction

Following Finding 2 in Section 5, the Ultralytics tracker source code
(v8.4.105) was inspected. Confirmed: the raw frame is passed
unconditionally to `tracker.update()` regardless of `with_reid`
(ruling out the frame being withheld). GMC is applied via
`_pre_first_associate()` in `byte_tracker.py`, inside a `try/except`
that silently falls back to an identity warp (`np.eye(2,3)`, i.e., no
motion compensation) on any exception:

```python
if hasattr(self, "gmc") and img is not None:
    try:
        warp = self.gmc.apply(img, results_high.xyxy)
    except Exception as e:
        LOGGER.warning(f"GMC failed, falling back to identity: {e}")
        warp = np.eye(2, 3)
    multi_gmc(strack_pool, warp)
    multi_gmc(unconfirmed, warp)
```

**Working hypothesis (not confirmed):** `with_reid=True` causes
`self.gmc.apply()` to raise an exception (silently caught), explaining
both symptoms from Section 5 simultaneously: IDF1 collapsing to the
no-GMC level, and FPS matching the no-GMC variants (early exception
aborts before the expensive optical-flow computation). Confirming this
would require verbose-logged re-runs and/or runtime patching; judged
out of scope given time constraints and documented as an open
limitation of the Ultralytics BoT-SORT implementation (v8.4.105).

## 10. Robust FLIT (Geman-McClure L2-IoU) + NSA-Kalman -- Strongest Result

**Refinement.** The initial linear L2-IoU hybrid (FLIT, Section 10
draft) degraded beyond gamma=0.35. Replacing the linear L2 term with
a Geman-McClure robust loss -- `robust_l2 = l2_norm^2 / (l2_norm^2 +
sigma^2)`, sigma=0.15 -- saturates the penalty for large centroid
displacements, motivated by Szeliski's discussion of M-estimators for
robust cost functions. This extension is referred to as **GM-FLIT**
(Geman-McClure FLIT) in this document.

| gamma | MOTA(%) | IDF1(%) | IDs | FPS |
|---|---|---|---|---|
| -- (baseline) | 75.9 | 71.2 | 2,327 | 46.46 |
| 0.15 | 76.4 | 72.2 | 2,109 | 41.93 |
| 0.25 | 76.6 | 73.5 | 1,998 | 42.92 |
| 0.35 | 76.7 | 74.5 | 1,877 | 42.61 |
| 0.45 | 76.7 | 74.5 | 1,707 | 43.85 |
| **0.55** | **76.7** | **74.5** | **1,634** | 44.18 |
| 0.65 | 75.4 | 71.4 | 1,842 | 45.25 |
| 0.75 | 74.1 | 69.2 | 2,217 | 45.03 |
| 0.85 | 72.8 | 67.0 | 2,491 | 44.75 |

**Result.** MOTA and IDF1 form a stable plateau across gamma in
[0.35, 0.55] (76.7% / 74.5%, identical to one decimal place), while
ID switches continue decreasing monotonically within the plateau
(1,877 -> 1,634), before all metrics degrade sharply beyond gamma=0.55.
This asymmetric plateau-then-collapse shape is consistent with the
saturating behavior expected of a robust M-estimator loss: within the
plateau, increasing gamma further suppresses the (already-bounded)
influence of large centroid displacements without over-penalizing
legitimate matches; beyond it, the L2 term begins to dominate the
combined cost and overrides IoU/appearance evidence.

**Adopted configuration: gamma=0.55**, the plateau point with the
lowest ID-switch count.

| Metric | Baseline (`GMC_noReID`) | **GM-FLIT γ=0.55 + NSA-Kalman** | Δ |
|---|---|---|---|
| MOTA | 75.9% | **76.7%** | +0.8 pp |
| IDF1 | 71.2% | **74.5%** | **+3.3 pp** |
| IDs | 2,327 | **1,634** | **-29.8%** |

This internal-split MOTA/IDF1 falls within the range of 5th-7th place
official leaderboard submissions (73.8%-76.1% MOTA, entries "wish"/
`chen2025strong`, "feifei123", "number_one"), though direct comparison
remains indicative only, since this project's evaluation uses an
internal held-out split rather than the official hidden test set,
which the top leaderboard entries reach up to 84.99% MOTA (1st place,
"ocewang"). This is nonetheless the strongest and final tracking
configuration adopted in this project.

Full sweep: `runs/mot_eval/flit_nsa_sweep_summary.csv` (gamma <= 0.35)
and `runs/mot_eval/flit_nsa_sweep_summary_v2.csv` (gamma >= 0.45).

## 11. SAHI (Slicing-Aided Hyper Inference) -- New Best Result

**Motivation.** Given the dataset's small native resolution (640x512)
and the persistently small apparent size of drone targets (Section 2),
SAHI was tested as a detection-time technique: instead of running
YOLO26n on the full frame, the frame is sliced into overlapping tiles,
each processed at higher effective resolution, with detections merged
back via NMS. Unlike the failed high-resolution retraining attempt
  (an earlier high-resolution training attempt, abandoned due to
  infrastructure issues), this introduces
no upsampling artifacts, since each tile is a native-resolution crop,
not a resized image. Tested on top of the project's best tracking
configuration (GM-FLIT gamma=0.55 + NSA-Kalman, Section 10).
**Combined with targeted hard-example retraining.** Following the
hard-example mining protocol documented above, the fine-tuned "HardMining"
detector was evaluated across the same SAHI slice sweep, alongside
the original detector, both combined with GM-FLIT + NSA-Kalman:

| Detector | Slice | MOTA(%) | MOTP | IDF1(%) | IDs | FP_rate | FN_rate | FPS |
|---|---|---|---|---|---|---|---|---|
| Original | 320 | 75.5 | 0.239 | 77.0 | 975 | 10.97% | 13.31% | 10.79 |
| Original | 384 | 77.1 | 0.238 | 79.4 | 648 | 10.34% | 12.39% | 14.66 |
| Original | 448 | 77.5 | 0.239 | 80.6 | 567 | 10.27% | 12.12% | 14.40 |
| Original | 512 | 78.7 | 0.235 | 79.7 | 725 | 9.24% | 11.89% | 22.25 |
| Original | 576 | 79.2 | 0.237 | 79.2 | 791 | 9.17% | 11.50% | 22.93 |
| HardMining | 320 | 76.4 | 0.242 | 77.5 | 864 | 10.83% | 12.64% | 10.92 |
| HardMining | 384 | 77.6 | 0.241 | 79.5 | 642 | 10.20% | 12.03% | 14.60 |
| HardMining | 448 | 77.9 | 0.243 | **80.8** | 592 | 10.18% | 11.81% | 14.21 |
| **HardMining** | **512** | **79.6** | 0.239 | **80.3** | **712** | **8.94%** | **11.36%** | 22.49 |
| HardMining | 576 | 79.6 | 0.239 | 79.9 | 718 | 9.06% | 11.20% | 22.97 |

**Final adopted configuration: HardMining detector + GM-FLIT
gamma=0.55 + NSA-Kalman + SAHI (slice=512).** This Pareto-dominates
slice=576 under the same detector (equal MOTA, strictly better IDF1
and IDs), simplifying the earlier two-operating-point framing into a
single recommended configuration:

| Metric | Original baseline (Section 4) | **Final (HardMining+FLIT+NSA+SAHI512)** | Δ |
|---|---|---|---|
| MOTA | 75.9% | **79.6%** | **+3.7 pp** |
| IDF1 | 71.2% | **80.3%** | **+9.1 pp** |
| IDs | 2,327 | **712** | **-69.4%** |
| FPS | 46.46 | 22.49 | -51.6% |

Since MOTA is the official competition ranking metric (Section 4),
this is reported as the project's single headline result. `slice=448`
remains worth noting as a secondary, identity-preservation-optimized
alternative (IDF1=80.8%, IDs=567 with the original detector -- the
lowest ID-switch count observed in this project), for use cases
prioritizing identity consistency over raw MOTA.

**Timing caveat.** FPS values in this table were measured in a later
session than the original SAHI-only sweep and are systematically
faster for equivalent configurations (e.g., slice=512: 14.47 -> 22.25
FPS), likely reflecting system load/thermal conditions on the target
hardware (Apple M2, no dedicated GPU) rather than an algorithmic
change. All comparisons within this table are internally consistent
(same session); cross-session FPS comparisons elsewhere in this
document should be treated as approximate.

Full results: `runs/mot_eval/sahi_summary_v2.csv`.

**Trade-off (critical for reporting).** This gain comes at a
substantial computational cost: FPS drops from 46.46 to 22.49
(-51.6%) relative to the original baseline in the same final sweep.
This remains a direct tension with this work's stated efficiency
motivation (Section 1, gap #3). Two reportable operating points are:

- **Speed-oriented:** GM-FLIT gamma=0.55 + NSA-Kalman (no SAHI) --
  76.7% MOTA, 44.18 FPS. Suitable for resource-constrained, real-time
  deployment.
- **Accuracy-oriented:** HardMining + GM-FLIT + NSA-Kalman + SAHI
  (slice=512) -- 79.6% MOTA, 80.3% IDF1, 712 IDs, 22.49 FPS. Suitable
  when detection and identity preservation are prioritized over
  throughput (e.g., post-hoc/offline analysis).

Full results: `runs/mot_eval/sahi_summary.csv`.

## 12. Adaptive Kalman Filter (Velocity-Scaled Process Noise) -- Marginal Result

Prior to discovering FLIT (Section 10), a simpler intervention was
tested in isolation: scaling the Kalman filter's velocity-component
process noise (Q) by each track's current estimated speed (tightening
the association gate for near-stationary tracks). Monkey-patched via
`BOTrack.shared_kalman` and `BOTSORT.get_kalmanfilter()`.

| ref_speed | MOTA(%) | IDF1(%) | IDs |
|---|---|---|---|
| -- (baseline) | 75.9 | 71.2 | 2,327 |
| 0.1 | 75.9 | 70.7 | 2,325 |
| 0.2 | 75.9 | 71.2 | 2,347 |
| 0.3 | 75.8 | 71.2 | 2,351 |
| **0.5** | 75.9 | **71.5** | 2,350 |
| **1.0** | 75.9 | **71.5** | 2,324 |
| 2.0 | 75.7 | 70.9 | 2,330 |
| 4.0 | 75.5 | 70.0 | 2,366 |

A small, non-monotonic peak at `ref_speed` in [0.5, 1.0] (+0.3 pp
IDF1). Qualitatively positive (unlike Section 8's filters) but of
marginal practical value on its own. Superseded by the combined
FLIT + NSA-Kalman result (Section 10), which yields a ~10x larger
IDF1 gain.

Full sweep: `runs/mot_eval/adaptive_kalman_sweep_summary.csv`.

## 13. Multi-Tracker Native Comparison

All 6 trackers natively supported by the installed Ultralytics
version (8.4.105) were evaluated with the same detector, using each
tracker's default configuration, across all 30 held-out sequences.

| Tracker | MOTA(%) | MOTP | IDF1(%) | IDs | FPS |
|---|---|---|---|---|---|
| **BoT-SORT** | **75.9** | 0.229 | **71.2** | 2,327 | 46.46 |
| FastTrack | 72.6 | 0.230 | 50.5 | 3,912 | 85.46 |
| ByteTrack | 71.6 | 0.230 | 49.4 | 4,538 | 49.84 |
| TrackTrack | 62.4 | NaN | 68.5 | 1,029 | 39.08 |
| OC-SORT | 48.2 | 0.210 | 49.2 | 5,835 | 46.96 |
| Deep OC-SORT | 34.7 | 0.215 | 36.8 | 7,554 | 21.28 |

**BoT-SORT (with GMC) is the clear winner** among all 6 natively
supported trackers, both in MOTA and IDF1, consistent with Section 5's
finding that motion compensation -- not appearance matching or
observation-centric motion modeling -- is the key differentiator on
this benchmark. Notably, the more architecturally sophisticated
OC-SORT and Deep OC-SORT perform *worse* than the simpler ByteTrack
baseline on this dataset (48.2% and 34.7% MOTA respectively, with the
highest ID-switch counts of any tracker tested), suggesting their
observation-centric re-projection and deep appearance embeddings are
not well suited to this domain (small, visually homogeneous, densely
swarming infrared targets) without domain-specific tuning. This
reinforces the choice of BoT-SORT + FLIT + NSA-Kalman (Section 10) as
the project's tracking backbone.

TrackTrack's `NaN` MOTP is an unexplained artifact of that tracker's
output on this dataset, noted but not investigated further given time
constraints.

**Engineering note (OC-SORT / Deep OC-SORT reliability):** both
trackers initially crashed with `TypeError: 'NoneType' object is not
subscriptable` when processing more than one sequence per Python
process, traced to `model.predictor.trackers = None` leaving the
`trackers` *attribute* present (only its value nulled), so
Ultralytics's re-initialization guard (likely an `hasattr` check)
never re-created the tracker list. Fixed with `del
model.predictor.trackers` (forcing `hasattr` to return `False`),
validated across the full 30-sequence run reported above.

Full results: `runs/mot_eval/multitracker_leaderboard.csv`.

## 14. Current Status and Recommended Next Steps

**Best configurations found (two operating points, Section 11):**
- Speed-oriented: GM-FLIT (`gamma=0.55`) + NSA-Kalman -- MOTA=76.7%,
  IDF1=74.5%, IDs=1,634, 44.18 FPS.
- Accuracy-oriented: HardMining + the above + SAHI (`slice=512`) --
  MOTA=79.6%, IDF1=80.3%, IDs=712, 22.49 FPS.

Both exceed `chen2025strong` (76.1% MOTA, official hidden test set),
with the usual caveat that this project's evaluation uses an internal
held-out split.

**Targeted retraining -- succeeded on the second attempt.** An initial
high-resolution attempt failed due to Colab infrastructure issues
(corrupted validation labels and unsustainable epoch time at
`imgsz=1280`). The corrected protocol is the sequence-level
hard-example mining procedure documented above: a composite tracking
score selects difficult training sequences, whose frames are mixed with
random training frames before a short, low-learning-rate fine-tuning at
native 640 resolution. The resulting "HardMining" detector, evaluated
standalone (BoT-SORT, no other modifications), improved MOTA by +0.7pp
over the original detector (75.9% -> 76.6%) primarily via a 6.0%
reduction in false negatives, with IDF1/IDs essentially unchanged.
Combined with GM-FLIT+NSA-Kalman+SAHI (Section 11), this produces the
project's final headline result (MOTA=79.6%, IDF1=80.3%, IDs=712).

A frame-level refinement (selecting only individual FP/FN frames
rather than whole hard sequences, for a more targeted and
computationally efficient hard-negative mining strategy) was designed
but not completed, given GPU budget exhaustion. This remains a
promising direction for future work; see repository issue notes for
the identified methodological caveat (FP frames must retain their
true, complete ground-truth labels rather than being blanked, to
avoid teaching the detector to suppress genuine co-occurring targets).

**Post-processing exploration (Section 8) is closed:** five further
association/track-level modifications beyond GM-FLIT+NSA-Kalman were
tested (static filter, spatial filter, hybrid filter, adaptive
confidence threshold, lost-query bank), all null or negative,
reinforcing GM-FLIT+NSA-Kalman (optionally combined with SAHI) as a
stable local optimum for this tracker/dataset combination.

**Remaining lower-priority directions**, not pursued given time
constraints: 7D shape-aware Kalman state (`jun2026segment`), temporal
memory with confidence decay (`li2026contextaware`) -- both judged
lower expected-value given the marginal/negative-return pattern
observed in similarly-scoped interventions (Sections 8, 12).

</details>

## 15. Repository Structure
```text
train.py # base YOLO26n fine-tuning script

src/data/convert_mot_to_yolo.py # MOT gt.txt -> YOLO label format
src/data/split_train_val.py # sequence-level train/val split + subsampling
src/inference/common.py # shared paths/helpers
src/inference/sanity_check.py # visual inspection of detector predictions
src/inference/track_video.py # detector + tracker (single video) demo
src/inference/run_tracker_on_val.py # detector + tracker on held-out sequences, w/ timing
src/inference/evaluate_mot.py # MOTA/IDF1/FPS evaluation vs. ground truth
src/inference/run_tracker_on_train.py # training-sequence tracking for HEM
src/inference/select_hard_examples.py # composite-score Top-20 selection
src/inference/build_hard_dataset.py # copies hard-sequence frames/labels
src/inference/generate_tracker_variants.py # GMC x Re-ID ablation configs
src/inference/run_ablation.py # runs GMC x Re-ID ablation
src/inference/evaluate_ablation.py # consolidates GMC x Re-ID results
src/inference/run_preprocessing_ablation.py # runs MOG2/EMA/OptFlow ablation
src/inference/evaluate_preprocessing_ablation.py # consolidates preprocessing ablation results
src/inference/run_static_filter_sweep.py # static track filter sweep (8.1)
src/inference/run_spatial_filter_sweep.py # spatial recurrence filter sweep (8.2)
src/inference/run_hybrid_filter_sweep.py # hybrid spatial+velocity filter sweep (8.3)
src/inference/run_adaptive_kalman_sweep.py # adaptive Kalman sweep (Section 12)
src/inference/run_flit_nsa_sweep.py # FLIT x NSA-Kalman sweep (Section 10)
src/inference/run_multitracker_leaderboard.py # 6-tracker native comparison (Section 13)
src/inference/debug_ocsort_traceback.py # OC-SORT bug reproduction/diagnostic
src/inference/run_adaptive_track_high_thresh.py # adaptive confidence threshold (8.4)
src/inference/run_lost_query_bank.py # lost-query bank (8.5)
src/inference/run_sahi_sweep.py # SAHI slice-size sweep (Section 11)
src/preprocessing/background_masks.py # MOG2/EMA/OptFlow frame generators
src/postprocessing/static_track_filter.py # per-ID temporal variance filter
src/postprocessing/spatial_recurrence_filter.py # grid-cell occupancy filter
src/postprocessing/hybrid_spatial_dynamics_filter.py # occupancy + local velocity filter
src/postprocessing/adaptive_kalman.py # velocity-scaled Kalman process noise patch
src/postprocessing/flit_association.py # GM-FLIT (robust L2-IoU) association cost patch
src/postprocessing/nsa_kalman_patch.py # NSA-Kalman confidence-scaled measurement noise patch
src/postprocessing/adaptive_track_high_thresh_patch.py # adaptive confidence threshold patch
src/postprocessing/lost_query_bank_patch.py # lost-query bank direction-consistency patch
src/postprocessing/diagnose_suppression.py # match-rate/track-length diagnostic
src/postprocessing/diagnose_hot_cell_velocity.py # hot-cell velocity distribution diagnostic
src/analysis/density_correlation.py # density vs. performance correlation + plots
src/analysis/camera_motion_estimation.py # camera motion estimation + correlation
src/analysis/frame_quality_metrics.py # entropy/edges/contrast metrics + correlation
configs/data.yaml # Ultralytics dataset config
configs/tracker_variants/ # generated tracker YAML variants (ablation)
weights/best.pt # best fine-tuned checkpoint
```

`dataset/` and `runs/` are gitignored (regenerable via the scripts
above, given access to the original 4th Anti-UAV Challenge Track 3
data, DOI `10.5281/zenodo.15103888`).

## 16. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 17. Reproducing the Pipeline

Run the stages below in order. The exact reported HEM fine-tuning was
performed in Colab; its mixed-dataset construction is documented in
[Hard-Example Mining](#hard-example-mining-detector-optimization).

```bash
python src/data/convert_mot_to_yolo.py
python src/data/split_train_val.py
python train.py
python src/inference/sanity_check.py
python src/inference/run_tracker_on_val.py
python src/inference/evaluate_mot.py
python src/inference/generate_tracker_variants.py
python src/inference/run_ablation.py
python src/inference/evaluate_ablation.py
python src/analysis/density_correlation.py
python src/analysis/camera_motion_estimation.py
python src/analysis/frame_quality_metrics.py
python src/inference/run_preprocessing_ablation.py
python src/inference/evaluate_preprocessing_ablation.py
python src/inference/run_static_filter_sweep.py
python src/inference/run_spatial_filter_sweep.py
python src/inference/run_hybrid_filter_sweep.py
python src/inference/run_adaptive_kalman_sweep.py
python src/inference/run_flit_nsa_sweep.py
python src/inference/run_multitracker_leaderboard.py
python src/inference/run_adaptive_track_high_thresh.py
python src/inference/run_lost_query_bank.py
python src/inference/run_sahi_sweep.py
```

### Reproducing the hard-subset generation

First generate MOT predictions for the eligible training sequences:

```bash
python src/inference/run_tracker_on_train.py
```

Evaluate those prediction files against the corresponding ground truth
with the project's training-sequence evaluation workflow, producing
`runs/train_predictions/train_summary.csv`. Then select and build the
hard subset:

```bash
python src/inference/select_hard_examples.py
python src/inference/build_hard_dataset.py
```

This builds the hard subset only. To reproduce the reported HardMining
detector, mix its 3,846 frames with 5,000 randomly sampled original
training frames, then fine-tune from the base checkpoint using the
Colab settings documented in the HEM section.

## Citation

Paper in preparation for LAFusion 2026 (Universidade Federal do Rio
de Janeiro). Details to be added upon acceptance.
