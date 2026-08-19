# CWM-mini → Research Protocol (v1)

Synthesized from a 4-expert panel (NVIDIA training-efficiency · ElevenLabs evaluation ·
Fish Audio codec/objective · NeurIPS methods-rigor). Target: a rigorous, measured
speech-ML paper. Realistic home **Interspeech/ICASSP main track**; **NeurIPS/ICLR main =
borderline stretch** (needs effect across 2 scales + 2 languages + a general framing).
A negative result is still publishable (D&B / benchmark framing) — that is the insurance.

Medical framing is **dropped**. Hardware = **one 70 GB H200 MIG** (no multi-GPU NCCL).

---

## 1. The reframe (what makes it more than "we added a loss")

Not *"add JEPA → turn-taking emerges"* (incremental SSL, desk-reject risk). Instead a
**factorial mechanism decomposition** over three orthogonal factors:

- **target ROLE** ∈ {partner-future, self-future, **shuffled-partner (placebo)**}
- **objective SPACE** ∈ {latent-JEPA, token-CE}
- **COUPLING** ∈ {trunk-coupled, side-tower}

**Falsifiable claim:** *only* the `partner × latent × trunk-coupled` cell improves
conversational competence, at matched params/FLOPs. This turns "we added a loss" into
"we isolated **which property** of the loss matters" — a mechanistic, general finding.

**Decisive control = the shuffled-partner placebo:** bit-identical arch/params/FLOPs/
schedule, but the target's time index is permuted → zero predictable structure. If
`partner-JEPA > placebo`, the gain provably comes from *predictable partner structure*,
not extra compute / params / VICReg noise. The self-future role-swap then isolates
"partner-ness" specifically.

---

## 2. Model (shrink for the grid)

- **Workhorse CWM-S ≈ 175M**: `d_model=768, n_layers=16, n_heads=12 (head_dim=64), n_kv=3
  (GQA 4:1), depth_layers=2, jepa_pred_layers=2`. **Tie** text_emb↔text_head. Big enough
  for real text-ppl + turn-taking emergence (dGSLM-class), small enough that a ~30–50-run
  grid finishes in weeks on one MIG.
- **XS fallback ≈ 60–118M** (`d640/14`) if the grid exceeds ~40 runs.
- **Hero ≈ 370M** (`d1024/24`) as **one** 2× scale-confirmation of the two decisive arms
  (+ its B0 control). ~4 GPU-days, no larger. Satisfies "effect survives a second scale."
- Rationale (all 4 experts agree): the claim is a **relative/mechanistic** effect, standard
  to establish at small scale; **data, not params, is the binding constraint**.

---

## 3. Objective redesign (fixes the verified flaws)

- **Couple JEPA to the trunk** (fixes flaw #1): the predictor `P_φ` (3 blocks) consumes the
  **core hidden `h`** (the listening state that already ingests both channels). **Delete** the
  side-tower `ctx`/`inp`. JEPA gradient now shapes the shared decoder. Keep the legacy
  side-tower as the explicit **COUPLING ablation arm** (turns the prior null into a finding).
- **Target = partner's future continuous latent** (fixes flaws #2/#3): Mimi's dequantized
  **all-8-codebook** continuous latent of the **partner** stream, lifted by a **data2vec-style
  EMA teacher** (EMA of a small trainable twin + stop-grad + light VICReg residual → no
  collapse, and *not* a deterministic function of the cb0 CE head). Carries prosody/energy/
  timbre/VAD — the turn-taking signal.
- **Horizons** {2,4,8,16} frames (160 ms–1.28 s), **listen-masked** by partner VAD so the loss
  concentrates while the partner speaks / approaches a boundary.
- **Two distinct speakers per dialogue** (fixes flaw #2) → actually predicts the interlocutor.
- **Inner-monologue text CE head** from the **previously-unused transcripts** (fixes flaw #5):
  forced-aligned word tokens interleaved on the timeline (text leads own-audio by ~2 frames).
  One delayed-streams model does **STT** (audio leads) and **TTS** (text leads). Held **constant
  across all arms** so it isn't confounded with the JEPA effect. Own acoustic delayed τ=1 frame
  (Moshi), undone at decode.

---

## 4. Data

- **Training = synthetic dyadic in the CODE domain** (no waveform mixing): place two speakers'
  Mimi code streams on parallel tracks, fill gaps with a cached silence frame; overlaps/
  backchannels are free; **VAD/turn labels are EXACT by construction**. **Gap/overlap
  distribution is drawn from REAL data (CANDOR-measured), not hand-set** — this defeats the
  circularity attack.
- **Primary turn-taking EVAL on REAL stereo dialog: CANDOR** (850 h, 1,657 dyads, per-speaker
  channels, free for research; English). Measuring turn-taking on data whose statistics you
  injected is circular — a real held-out corpus is the non-negotiable credibility fix.
- **Turkish generality replication**: synthetic-overlapped Common Voice tr (+ YODAS2 tr / FLEURS
  / MediaSpeech as license-safe scale). **No YouTube.**
- **Frozen splits by speaker AND sentence** (deterministic hash → 80/10/10), written once to
  `splits.json`; checkpoint selected on VAL, **TEST touched once**.

---

## 5. Metrics & statistics (pre-registered)

- **PRIMARY (single, no multiplicity penalty):** turn-shift **anticipation AUROC** from a
  **frozen-backbone linear probe** on trunk `h_t` at **lead L=500 ms** (6 frames), on the
  **real** held-out test split. Effect = **ΔAUROC(partner-JEPA − B0/placebo)** with 95%
  **conversation-cluster bootstrap** CI (B=10k) + DeLong cross-check. Frozen linear probe =
  the I-JEPA/V-JEPA standard, identical capacity across arms → defends the "emergence" claim.
- **Secondary (BH-FDR q=0.05):** VAP balanced-accuracy + macro-F1 (Ekstedt–Skantze 256 joint
  states, 2 s future); AUROC(L) curve for L∈{0,0.25,0.5,1,2}s (proves *anticipation* not
  reaction); **future-partner-token probe** (top-1/5 at horizons {2,4,8}) vs a raw-Mimi-latent
  baseline; **intelligibility CER** round-trip (Whisper-Turkish + a wav2vec2/MMS second
  decoder), reported as **CER_gap = CER_model − CER_resynth** (Mimi-resynthesis topline = the
  ASR floor); **MOS proxy** UTMOS + TorchAudio SQUIM, only relative to real + resynth anchors.
- **Coupling proof:** gradient-norm into `Core.blocks` from `L_jepa` must be > 0 and
  non-trivial (report ‖∂L_jepa/∂W_core‖ / ‖∂L_sem/∂W_core‖ per layer) — verifies flaw #1 is
  actually fixed, not just refactored.
- **Stats hygiene:** ≥3 seeds/cell; effect sizes (Cohen's d) with 95% CI always, never bare p;
  Holm/BH across the ~6-row ablation; speaker-disjoint test for all ppl/CER/probe numbers.

**Data-efficiency (the NeurIPS angle) = a CURVE:** train B0 vs partner-JEPA at
{1,3,10,30,100} h at ≥3 seeds; report **hours-to-threshold** and **area-between-curves**;
primary x-axis = unique conversational hours; secondary = FLOP-matched. "Left-shifted frontier."

**Matched compute via a FLOP-meter, not fixed steps:** stop each run at a fixed cumulative
training-FLOP budget C. Coupled-JEPA adds ~+30–40 % FLOPs/audio-step, so B0 processes *more*
audio at equal C — the honest comparison. Report on **both** matched-FLOP and matched-data axes.

---

## 6. Efficiency (make the grid feasible)

Tie embeddings; **static shapes** per stage; `torch.compile(core, depth,
mode='max-autotune-no-cudagraphs')` (keep jepa/rollout + self-heal eager); **fp32 master
weights + autocast(bf16)** (removes bf16-optimizer noise that would confound a data-efficiency
claim — trivial at 175M on 70 GB); backend flags (TF32, flash SDPA); **hold the whole audio
corpus in RAM** (I/O, not compute, is the audio bottleneck); **shared text-pretrained backbone**
reused across every grid run (grid varies only the duplex phase → ~5–10× cheaper). Target
**~6–10× tokens/sec** vs today's ~31.5k. Text-stage MFU must clear the ≥0.30 bench gate post-compile.

---

## 7. Implementation phases (single-file trainer)

- **P0 — Efficiency + workhorse config** (mechanical, keeps b3 path working): CWM-S env config,
  tied embeddings (flag), backend flags, guarded `torch.compile`, fp32-master+autocast, audio-in-RAM.
- **P1 — Objective redesign**: couple JEPA to core `h`; data2vec EMA-teacher target on partner
  continuous latent; factorial arm switches {role, space, coupling}; coupling-proof gradient check.
- **P2 — Duplex data (code-domain synthetic)**: shard schema v2 (own/ptn codes+lat, vad, aligned
  text, exact turns), `build_duplex_shards`, `_duplex_batches`, two-speaker sampling, silence frame,
  CANDOR-measured gap distribution; inner-monologue forced alignment (WhisperX/MFA).
- **P3 — Eval suite**: `eval-suite` subcommand → frozen splits, linear-probe AUROC (primary), VAP,
  CER round-trip, MOS proxies, stats helpers (`_bootstrap_ci`, `_delong`, `_bh_fdr`, `_turkish_norm`).
- **P4 — Real-data eval**: CANDOR ingest (32 k→24 k, Mimi-encode per channel); Turkish synthetic replication.
- **P5 — The grid**: factorial arms × ≥3 seeds × data budgets, FLOP-meter matched-compute, shared
  backbone; then the hero 2× scale-check + negative-result writeup path.

---

## 8. Honest venue odds (if the grid lands cleanly)

- Interspeech / ICASSP main: **high** (squarely their scope — VAP, dGSLM, Moshi live there).
- NeurIPS / ICLR main: **modest/borderline** — only if effect holds across 2 scales + 2
  languages + framed as a general *"what to predict while listening"* SSL principle.
- Medical venues: **~0 % for years** (separate clinical program). Framing dropped.
- **Insurance:** ship the synthetic-duplex benchmark + linear-probe-VAP protocol as a reusable
  contribution; a clean negative ("interlocutor-predictive latent objectives do **not** yield
  emergent turn-taking beyond compute-matched controls") is publishable regardless of sign.
