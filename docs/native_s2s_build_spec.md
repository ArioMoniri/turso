# Native Turkish‑Medical Full‑Duplex S2S — Self‑Build Spec

Verified engineering spec (6 module engineers + 4 adversarial reviewers, web‑sourced, Aug 2026).
Goal: **build**, not just serve, a native Turkish speech‑to‑speech + text model with **emotion**,
**zero‑shot voice cloning**, and **interruptible streaming**, on **open weights only**
(Qwen3.5‑Omni is API‑only → excluded; we build on the open Qwen3‑Omni‑30B).

## Verdict

- **Feasible.** GPU is *not* the binding constraint (green): the trained parts are a 0.1–0.5 B head
  + a 0.6 B duplex sidecar; the 30 B backbone and Mimi are **frozen** and their outputs are
  **precomputed offline**, so head training runs on your **one 71 GB MIG** for the prototype.
- **The one real blocker (red‑team, HIGH‑RISK):** there is **no legally‑usable, natively‑Turkish,
  expressive teacher or corpus.** Every clean Turkish set (Common Voice, YODAS, MediaSpeech) is
  *neutral read speech*; the only expressive zero‑shot clone teacher (XTTS‑v2) is non‑commercial +
  its vendor is defunct. So **M5 (emotion + expressive cloning) — the differentiator — has no
  supervision source and will not converge as‑spec** without new data. **De‑risk this FIRST.**

## Architecture (frozen = ❄, trained = 🔥)

```
mic 16k ─▶ ❄ Qwen3-Omni-30B Thinker (AWQ 4-bit)  ── hidden_states[layer 24] (2048-d) ─┐
            (Turkish speech-IN + text reasoning)   ── inner-monologue text tokens ─────┤
                                                                                       ▼
  ❄ CAM++ timbre (192) + ❄ emotion2vec (768) ──▶ 🔥 M3 AUDIO HEAD (CSM depth-transformer over Mimi)
                                                     └─ predicts 8 Mimi codebooks @ 12.5 Hz
  🔥 M4 SoulX-Duplug-0.6B (listen/speak/interrupt) ──▶ barge-in cancels M3            │
                                                                                       ▼
                                                        ❄ Mimi decoder ──▶ 24 kHz Turkish speech out
```

---

## M0 — Repo scaffold & the offline‑cache trick (the feasibility enabler)

**Decision:** fresh repo `medvoice/`. Do **not** fork moshi‑finetune wholesale; do **not** extend the
current `scripts/turkish_medvoice.py` (its Whisper‑enc → Qwen2.5‑QLoRA → external‑TTS cascade is the
wrong core). Harvest: (a) pip‑vendor `moshi` (Mimi + streaming); (b) fork **SesameAILabs/csm**
`models.py` as the M3 head skeleton; (c) copy the CER‑gate + eval utils from `turkish_medvoice.py`.

**Two‑pass offline cache** (so head training never re‑runs the 30 B or Mimi):
- **Pass A (venv‑backbone):** load `Qwen3OmniMoeThinkerForConditionalGeneration` (AWQ), teacher‑force
  each `(Turkish speech‑IN, target text)`, capture `hidden_states[24]` at assistant‑text positions →
  `qwen_hidden float16[T_text,2048]` + `text_ids`.
- **Pass B (venv‑mimi):** Mimi‑encode each target 24 kHz wav → `mimi_codes int16[8,T_audio]`.
- Store shards keyed by utterance id. Prototype cache = 20–50 h (~1–2 GB); full = 300–1000 h (~20 GB).
- **Then the head trainer loads only cached tensors** — the 30 B is *absent* from the training loop.

**Biggest M0 risk:** rate/alignment mismatch — the Thinker emits hidden states at **text‑token cadence
(~word rate)**, not Mimi's fixed **12.5 Hz**. You must define the delay/interleave that maps
inner‑monologue text steps → audio frames (copy Moshi's delay pattern; validate on 2–5 h first).

---

## M1 — Frozen backbone hidden states  ❄  (no training)

- **Model:** `Qwen/Qwen3-Omni-30B-A3B-Instruct` Thinker, 4‑bit AWQ
  (`cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`), `transformers>=4.57`.
- **Tap:** `hidden_states[24]` (Qwen's own Talker uses `accept_hidden_layer=24` per config.json — the
  class *default is 18, which is wrong*; **use 24**). Shape `[B,T,2048]` bf16, **detached**.
- **Load:** `Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(..., device_map={"":0})`;
  `del model.visual` (drop ViT, keep `audio_tower` for Turkish speech‑IN); `requires_grad_(False)`.
- **Interface to M3:** M3 owns `proj = nn.Linear(2048, d_head)`; input at step t =
  `proj(h_t) + embed_tokens(text_token_t)`. Streaming = a forward hook on `model.model.layers[23]`.
- **VRAM:** ~17–20 GB weights + audio encoder ~2–4 GB + KV ~0.4 GB. **Risk:** off‑by‑one on the tap
  layer silently trains the head against the wrong signal — assert layer 24.

## M2 — Mimi codec  ❄  (Kyutai, CC‑BY‑4.0)

24 kHz → 12.5 Hz, **8 RVQ codebooks** (cb0 = WavLM‑distilled semantic, cb1‑7 acoustic), fully causal,
80 ms frame. ~1–2 GB. **Risk:** cb0 is English‑heavy → weaker Turkish semantics (optional Turkish
semantic‑distill finetune later; not required for v1).

## M3 — Streaming audio head  🔥  (the core new model)

- **Type:** **depth‑transformer (CSM/Moshi RQ), not Orpheus‑flatten** (flattening 8 tok/frame ×8's the
  sequence length and kills streaming). Fork **SesameAILabs/csm** `models.py` — its backbone dim is
  **2048 == Qwen's**, so the projection reuses as‑is.
- **Sub‑models:** temporal transformer (~1 B, llama‑ish 16L/2048d) predicts frame‑level semantic token
  from `proj(h_t)+text`; depth decoder (~100 M, 4L/1024d) predicts the 7 residual acoustic codebooks.
- **Loss:** `Σ_frames Σ_{k=0..7} w_k · CE(logits_{s,k}, mimi_code[k,s])`, semantic weight **w₀=100**
  (Moshi `alpha=100`), per Moshi's delay pattern.
- **Data/GPU:** cached `(qwen_hidden, text_ids, mimi_codes)`; prototype on one MIG (backbone absent),
  full run scales on 4–8× H100 but is cheap (0.1–0.5 B trainable).
- **Risk:** Qwen never learned Turkish *speech‑out*, so the head bears the whole acoustic burden;
  Turkish's phonemic orthography (the text stream) mitigates it. Watch for repetition/mode‑collapse.

## M4 — Duplex state head  🔥  (interruptible pseudo‑duplex)

- **Model:** fork **Soul-AILab/SoulX-Duplug-0.6B** (Apache‑2.0, arXiv 2603.14877) — Qwen3‑0.6B + frozen
  12.5 Hz speech tokenizer; LoRA r=32 for Turkish. 160 ms chunk; states {idle, nonidle, backchannel,
  complete, incomplete}. On `complete`/`interrupt` → cancel M3 generation + flush.
- **Train:** weighted CE over interleaved {audio, ASR‑text, state}, up‑weight sparse state tokens.
  ~1–3 GPU‑days Turkish ASR adapt + ~10–20 k steps state LoRA. Fits the MIG.
- **Honest ceiling:** *interruptible pseudo‑duplex*, **not** true overlap. **Echo risk:** turn‑F1
  collapses when it hears its own output — run the classifier on the **user mic stream only** + add
  AEC (WebRTC/`speexdsp`), since neither SoulX nor Freeze‑Omni ships echo cancellation.

## M5 — Emotion + cloning  🔥  (the differentiator — and the blocker)

- **Timbre:** frozen **CAM++** (`iic/speech_campplus_sv…`, the encoder CosyVoice2 uses) → 192‑d.
- **Emotion:** frozen **emotion2vec_plus_large** → 768‑d, + inline `<emo:x intensity=y>` tags.
- **Disentangle (so a cloned voice can be re‑emoted):** IndexTTS‑2 recipe (arXiv 2506.21619) —
  GRL + speaker‑classifier adversarial loss: `L = L_AR + α·L_emo‑adv + β·L_SCL + γ·L_GRL`.
- **Encoders run offline → ~0 GPU during head training** (only small `W_t/W_e` projections train).
- **🔴 BLOCKER:** disentanglement needs **multi‑speaker, emotion‑labeled Turkish** — which does not
  exist license‑clean. Without it, M5 won't converge. **This is the project's critical path.**

## M6 — Data & staged curriculum

- **Licensed Turkish (verified):** Common Voice tr v26 **130 h CC0** (spine) + MediaSpeech tr **10 h
  CC‑BY** + YODAS2 tr000 manual‑caption (CC‑BY, filter). CER‑gate everything (keep low caption↔ASR CER).
- **Medical terms:** a **pronunciation lexicon** for Latin/EN drug names in Turkish orthography
  (grapheme handling), upsampled — reuse the repo's drug gazetteer + leading‑space tokenizer work.
- **Harness:** clone **kyutai-labs/moshi-finetune** (already implements the delayed multi‑codebook loss
  + Mimi tokenization + LoRA).
- **Curriculum:** cache → M3 warmup (neutral read speech, intelligible Turkish) → M4 duplex → **M5
  emotion/clone (gated on expressive data)** → medical‑term pass. Each stage has a MIG go/no‑go gate.
- **🔴 Binding risk:** *expressive‑data starvation* — neutral read speech gets you intelligible Turkish
  but **not** emotion/expressive cloning.

---

## GPU plan

| Stage | Where | Cost |
|---|---|---|
| Pass‑A/B offline cache (50 h) | one 71 GB MIG | 6–15 GPU‑h (one‑time) |
| M3 head **prototype** (20–50 h) | one 71 GB MIG | days |
| M4 duplex LoRA | one 71 GB MIG | 1–3 GPU‑days |
| **Full run** M3+M5 (300–1000 h) | rented **4–8× H100/H200‑80 GB** | ~1–3 weeks, ~$2–8 k |
| Serve the assembled model | one 71 GB MIG (or 1× H200 FP8) | — |

**No GPU purchase to prototype or to serve.** The cluster is only for the full‑scale head/emotion run —
and you should **not** rent it until the prototype passes its gates *and* the expressive‑data problem is solved.

## Eval gates (all inference‑only, fit the MIG)

- Turkish speech‑out: **CER** via `whisper-large-v3` round‑trip, **reported as a delta above the ASR
  floor** on ground‑truth human audio (not absolute).
- Medical‑term pronunciation: a drug‑name test set (accuracy).
- Clone similarity: **ECAPA** (speechbrain) cosine.
- Emotion controllability; barge‑in **turn‑F1**; first‑audio latency.
- **MIG go/no‑go before any cluster spend:** intelligible neutral Turkish (CER within ~X of floor) +
  clone similarity above threshold on a 20–50 h prototype.

## De‑risk order (do the blocker first)

1. **Week 0 (parallel with M0/M1):** *expressive‑supervision existence proof* — confirm/commission a
   small **license‑clean expressive Turkish (ideally medical) corpus**, ~2–4 consented voice actors,
   emotion‑labeled. **This is the long pole; start it before writing head code.**
2. Build M0 scaffold + Pass‑A/B cache on 2–5 h; **validate the tap layer (24), the 12.5 Hz alignment,
   and the delay pattern** (freeze the cache key) before scaling.
3. M3 head prototype on 20–50 h neutral Turkish → hit the intelligibility gate on the MIG.
4. Add M4 duplex (+ AEC). Add M5 only once (1) yields real expressive data.
5. Only then rent 4–8× H100 for the full 300–1000 h run.

## Key repos

- Qwen3‑Omni: <https://github.com/QwenLM/Qwen3-Omni> · AWQ: <https://huggingface.co/cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit>
- Mimi/Moshi + finetune: <https://github.com/kyutai-labs/moshi> · <https://github.com/kyutai-labs/moshi-finetune>
- CSM head (fork this): <https://github.com/SesameAILabs/csm>
- Duplex: <https://github.com/Soul-AILab/SoulX-Duplug> · <https://huggingface.co/Soul-AILab/SoulX-Duplug-0.6B>
- Emotion/timbre: IndexTTS‑2 <https://github.com/index-tts/index-tts> · CAM++ (3D‑Speaker) · emotion2vec
- Data: Common Voice tr (CC0) · YODAS2 (CC‑BY) · MediaSpeech tr (OpenSLR‑108)
