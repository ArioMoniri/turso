#!/usr/bin/env python3
# =============================================================================
#  CWM-mini — Conversational World Model (from scratch, single H200 MIG slice)
# =============================================================================
#  A small full-duplex speech-text model trained FROM SCRATCH whose novelty is a
#  JEPA-style INTERLOCUTOR-PREDICTION objective: while listening, predict the
#  other speaker's FUTURE audio latents in embedding space, and roll an internal
#  latent state forward ("silent thought") instead of emitting padding.
#
#  Frozen (allowed — same idea as a pretrained BPE): Mimi codec (12.5 Hz, 8 RVQ).
#  Trained from scratch (~500M): SentencePiece tok + dual-stream core + depth
#  transformer (acoustic codebooks) + JEPA module + silent-thought rollout.
#
#  ONE self-contained file. Runs from /data on the server, in tmux. It is
#  SELF-HEALING (OOM/transient -> re-exec with --resume), RESUMABLE (every step is
#  checkpointed), RESOURCE-GUARDED (auto-detects free disk/RAM/VRAM and scales /
#  prunes / stops BEFORE maxing out), and REPORTING (RUNLOG.md + REPORT_*.md).
#
#  GATES (honored automatically by `auto`): smoke overfit (tiny model, 32 samples,
#  must reach near-zero loss) -> throughput/MFU bench (>= min MFU) -> stage runs.
#
#  SUBCOMMANDS:
#    setup      venv/deps check, HF login prompt, print hardware + resource plan
#    data       download + clean + PACK text/audio (resumable, license-logged)
#    tokenizer  train the SentencePiece unigram tokenizer + report fertility
#    smoke      overfit a tiny model on 32 samples (the trust gate)
#    bench      throughput + MFU + ETA on random data
#    train      --stage {jepa,b0,b1,b2,b3,duplex,medical}  (resumable)
#    eval       --stage <ckpt-stage>   loss/ppl + JEPA probe + throughput
#    auto       run the WHOLE roadmap unattended, gated + self-healing + resumable
#    doctor     preflight + resource report
#
#  USAGE (server, in tmux):
#    export CWM_ROOT=/data/cwm
#    python cwm.py auto            # runs S0 -> smoke gate -> bench gate -> S1 -> ...
#
#  NOT a clinical tool. Medical stage is documentation-support framing only.
# =============================================================================
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Only stdlib at import time. Heavy libs (torch, datasets, sentencepiece,     #
#  moshi/mimi) are imported lazily inside the functions that use them, so the  #
#  CLI + doctor + resource planning work even before deps are installed.       #
# --------------------------------------------------------------------------- #


def _env(k, d):
    return os.environ.get(k, d)


# =========================================================================== #
#  CONFIG — every path/dim/knob; override via CWM_* env vars.                  #
# =========================================================================== #
@dataclass
class Config:
    # ---- layout (everything lives under one root on /data) ------------------
    root: str = _env("CWM_ROOT", "/data/cwm")
    seed: int = int(_env("CWM_SEED", "1234"))

    # ---- model dims (the paper model ~500M) ---------------------------------
    vocab_size: int = int(_env("CWM_VOCAB", "32000"))
    n_codebooks: int = 8               # Mimi: cb0 semantic + cb1..7 acoustic
    codebook_size: int = 2048
    frame_hz: float = 12.5
    d_model: int = int(_env("CWM_DMODEL", "1024"))
    n_layers: int = int(_env("CWM_LAYERS", "24"))
    n_heads: int = int(_env("CWM_HEADS", "16"))
    n_kv_heads: int = int(_env("CWM_KV", "4"))
    depth_layers: int = int(_env("CWM_DEPTH", "6"))
    jepa_ctx_layers: int = int(_env("CWM_JEPA_CTX", "12"))
    jepa_pred_layers: int = int(_env("CWM_JEPA_PRED", "3"))
    jepa_lat_dim: int = int(_env("CWM_JEPA_LAT", "128"))   # JEPA target-space dim
    seq_len: int = int(_env("CWM_SEQ", "4096"))
    rope_theta: float = 1e4
    rms_eps: float = 1e-5

    # ---- objective weights + curriculum -------------------------------------
    lambda_audio: float = float(_env("CWM_LA", "0.5"))
    lambda_jepa: float = float(_env("CWM_LJ", "0.5"))
    lambda_rollout: float = float(_env("CWM_LR", "0.25"))
    aux_warmup_frac: float = 0.10      # λj, λr = 0 for first 10% of steps
    jepa_horizons: tuple = (2, 4, 8)   # frames ahead (160/320/640 ms)
    ema_base: float = 0.996
    ema_final: float = 0.9999
    jepa_var_w: float = float(_env("CWM_JEPA_VAR", "1.0"))    # VICReg variance (anti-collapse)
    jepa_cov_w: float = float(_env("CWM_JEPA_COV", "0.04"))   # VICReg covariance (decorrelate)
    jepa_collapse_std: float = 0.10    # warn (at log cadence) if online std stays below this

    # ---- optimization -------------------------------------------------------
    micro_batch: int = int(_env("CWM_MBS", "4"))
    grad_accum: int = int(_env("CWM_GA", "8"))
    lr: float = float(_env("CWM_LR_PEAK", "3e-4"))
    warmup_steps: int = int(_env("CWM_WARMUP", "2000"))
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    total_steps: int = int(_env("CWM_STEPS", "60000"))
    log_steps: int = int(_env("CWM_LOG_STEPS", "20"))
    save_secs: int = int(_env("CWM_SAVE_SECS", "1800"))   # checkpoint every 30 min
    val_steps: int = int(_env("CWM_VAL_STEPS", "1000"))
    keep_ckpts: int = int(_env("CWM_KEEP", "3"))

    # ---- data budgets (auto-scaled to free disk at runtime) -----------------
    text_datasets: tuple = ("HuggingFaceFW/fineweb-2",)  # config 'tur_Latn'
    # Audio sources tried in order (first that yields decodable clips wins). The
    # mozilla-foundation CV repos are gated loading-scripts that modern `datasets`
    # can't stream, so we prefer the ungated parquet mirror + FLEURS (wav) and only
    # fall back to gated CV (needs HF_TOKEN + accepted terms). Override with
    # CWM_AUDIO_SPECS="repo:config:textcol,repo:config:textcol".
    audio_specs: str = _env("CWM_AUDIO_SPECS",
        "ysdede/commonvoice_17_tr_fixed:default:transcription"  # native parquet, ungated, TR
        ",google/fleurs:tr_tr:transcription"                    # parquet fallback (~10h)
        ",mozilla-foundation/common_voice_17_0:tr:sentence")    # gated: needs HF_TOKEN
    target_audio_hours: int = int(_env("CWM_AUDIO_H", "2000"))
    target_text_tokens: int = int(_env("CWM_TEXT_TOK", str(30_000_000_000)))
    disk_reserve_gb: int = int(_env("CWM_DISK_RESERVE", "150"))  # shared SSD: keep headroom
    ckpt_budget_gb: int = int(_env("CWM_CKPT_BUDGET", "300"))

    # ---- dynamic resource guard thresholds (percent) ------------------------
    disk_warn: int = int(_env("CWM_DISK_WARN", "85"))
    disk_stop: int = int(_env("CWM_DISK_STOP", "93"))
    ram_warn: int = int(_env("CWM_RAM_WARN", "88"))
    ram_stop: int = int(_env("CWM_RAM_STOP", "95"))
    vram_guard: float = float(_env("CWM_VRAM_GUARD", "0.90"))
    vram_shrink: float = float(_env("CWM_VRAM_SHRINK", "0.94"))
    min_mfu: float = float(_env("CWM_MIN_MFU", "0.20"))

    # ---- Mimi / tokenizer ---------------------------------------------------
    mimi_repo: str = _env("CWM_MIMI", "kyutai/moshiko-pytorch-bf16")  # hosts moshi-format Mimi

    def __post_init__(self):
        for d in (self.data_dir, self.shard_dir, self.ckpt_dir, self.log_dir,
                  self.tok_dir, self.hf_cache):
            Path(d).mkdir(parents=True, exist_ok=True)

    # derived paths
    @property
    def data_dir(self):  return str(Path(self.root) / "data")
    @property
    def shard_dir(self): return str(Path(self.root) / "data" / "shards")
    @property
    def ckpt_dir(self):  return str(Path(self.root) / "checkpoints")
    @property
    def log_dir(self):   return str(Path(self.root) / "logs")
    @property
    def tok_dir(self):   return str(Path(self.root) / "tokenizer")
    @property
    def hf_cache(self):  return _env("HF_HOME", str(Path(self.root) / "hf_cache"))

    def stage_ckpt(self, stage):
        return str(Path(self.ckpt_dir) / stage)


CFG = Config()


# =========================================================================== #
#  LOGGING + RUNLOG + REPORT                                                    #
# =========================================================================== #
def log(msg, err=False):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, file=(sys.stderr if err else sys.stdout), flush=True)
    try:
        Path(CFG.log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(CFG.log_dir) / "cwm.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def runlog(rec: dict):
    try:
        rec = dict(rec); rec["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(Path(CFG.log_dir) / "RUNLOG.md", "a", encoding="utf-8") as fh:
            fh.write("- " + json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def alert(msg):
    log("ALERT: " + msg, err=True)
    try:
        with open(Path(CFG.log_dir) / "ALERT.md", "a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def write_report(name, obj):
    try:
        p = Path(CFG.log_dir) / f"REPORT_{name}.md"
        p.write_text("# REPORT: " + name + "\n\n```json\n"
                     + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```\n",
                     encoding="utf-8")
        log(f"report -> {p}")
    except Exception as e:
        log(f"report write failed: {e}", err=True)


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


# =========================================================================== #
#  RESOURCE PROBES + DYNAMIC GUARD (disk / RAM / VRAM)                          #
# =========================================================================== #
def disk_pct(path):
    try:
        u = shutil.disk_usage(path); return round(100 * u.used / u.total, 1)
    except Exception:
        return 0.0


def disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return 0.0


def ram_pct():
    try:
        with open("/proc/meminfo") as f:
            m = {}
            for ln in f:
                k, _, v = ln.partition(":")
                m[k.strip()] = int(v.strip().split()[0])  # kB
        total = m.get("MemTotal", 1); avail = m.get("MemAvailable", total)
        return round(100 * (total - avail) / total, 1)
    except Exception:
        return 0.0


def vram_used_total():
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return (total - free) / 1e9, total / 1e9
    except Exception:
        pass
    return 0.0, 0.0


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def _prune_dir(d, keep_globs=("*.tmp", "*.part"), max_log_mb=512):
    """Reclaim disk WITHOUT touching anything the run still needs."""
    try:
        for pat in keep_globs:
            for f in Path(d).rglob(pat):
                try: f.unlink()
                except Exception: pass
        for f in Path(CFG.log_dir).glob("*.log"):
            if f.stat().st_size > max_log_mb * 1e6:
                with open(f, "rb") as fh:
                    fh.seek(-64 * 1024 * 1024, os.SEEK_END); tail = fh.read()
                f.write_bytes(tail)
    except Exception:
        pass


class ResourceGuard:
    """Called each optimizer step. DYNAMIC: relieves pressure BEFORE a crash.
      VRAM >= vram_guard  -> empty_cache; still >= vram_shrink -> caller shrinks.
      DISK >= disk_warn   -> prune tmp/oversized logs; >= disk_stop -> STOP phase.
      RAM  >= ram_stop    -> STOP phase (checkpoint first).
    Returns 'ok' | 'shrink' | 'stop'."""
    def __init__(self, cfg):
        self.cfg = cfg; self.last = 0.0

    def step(self, root):
        import torch
        now = time.time()
        if now - self.last < 15:          # cheap: at most every 15s
            return "ok"
        self.last = now
        act = "ok"
        used, total = vram_used_total()
        if total and used / total >= self.cfg.vram_guard:
            try: torch.cuda.empty_cache()
            except Exception: pass
            used, total = vram_used_total()
            if total and used / total >= self.cfg.vram_shrink:
                act = "shrink"
        d = disk_pct(root)
        if d >= self.cfg.disk_warn:
            _prune_dir(self.cfg.shard_dir); _prune_dir(self.cfg.ckpt_dir)
            d = disk_pct(root)
        r = ram_pct()
        if d >= self.cfg.disk_stop or r >= self.cfg.ram_stop:
            alert(f"CRITICAL disk={d}% ram={r}% -> STOP phase (checkpoint first)")
            return "stop"
        return act


# =========================================================================== #
#  SELF-HEALING SUPERVISOR (bounded re-exec with --resume on OOM/transient)     #
# =========================================================================== #
MAX_HEAL = int(_env("CWM_MAX_HEAL", "10"))
_FATAL = ("FileNotFoundError", "AssertionError", "KeyboardInterrupt", "SystemExit",
          "ModuleNotFoundError", "ImportError")


def classify_error(e):
    name = type(e).__name__
    s = str(e).lower()
    if "out of memory" in s or "cuda error" in s or name == "OutOfMemoryError":
        return "oom"
    if name in _FATAL:
        return "fatal"
    if any(t in s for t in ("timed out", "connection", "temporarily", "5xx", "429")):
        return "transient"
    return "unknown"


def _reexec(extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    n = int(env.get("CWM_HEAL_N", "0")) + 1
    if n > MAX_HEAL:
        log(f"[heal] exhausted {MAX_HEAL} re-execs; surfacing.", err=True)
        return False
    env["CWM_HEAL_N"] = str(n)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # --resume is a TOP-LEVEL flag: it must sit BEFORE the subcommand token, else
    # the subparser rejects it and the re-exec'd process dies at argparse (exit 2),
    # which would silently defeat the entire self-heal/OOM-ladder/resume mechanism.
    rest = sys.argv[1:]
    if "--resume" not in rest:
        rest = ["--resume"] + rest
    argv = [sys.executable, sys.argv[0]] + rest
    log(f"[heal] re-exec #{n}/{MAX_HEAL}: {' '.join(argv[1:])}", err=True)
    runlog({"event": "reexec", "n": n})
    try:
        os.execve(sys.executable, argv, env)
    except Exception as e:
        log(f"[heal] execve failed: {e}", err=True)
        return False


def supervise(fn):
    """Run fn(); on OOM shrink batch+seq and re-exec; on transient re-exec; else raise."""
    try:
        return fn()
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        kind = classify_error(e)
        log(f"[heal] caught {type(e).__name__} ({kind}): {str(e)[:160]}", err=True)
        if kind == "oom":
            lvl = int(_env("CWM_HEAL_LVL", "0")) + 1
            new = {"CWM_HEAL_LVL": lvl}
            # degradation ladder: micro-batch -> grad-accum -> seq
            if lvl == 1:   new["CWM_MBS"] = max(1, CFG.micro_batch // 2)
            elif lvl == 2: new["CWM_MBS"] = "1"; new["CWM_GA"] = CFG.grad_accum * 2
            elif lvl >= 3: new["CWM_MBS"] = "1"; new["CWM_SEQ"] = max(1024, CFG.seq_len // 2)
            import gc
            try:
                import torch; gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass
            if _reexec(new):
                return
            raise
        if kind == "transient":
            time.sleep(20)
            if _reexec():
                return
            raise
        raise


# =========================================================================== #
#  MIMI CODEC WRAPPER (frozen)                                                  #
# =========================================================================== #
class MimiWrap:
    def __init__(self, cfg=CFG):
        self.cfg = cfg; self._m = None; self.sr = 24000; self._proj = None
        self._load_err = None

    def load(self):
        if self._m is not None:
            return self._m
        if self._load_err is not None:      # don't re-hammer HF on a hard failure
            raise self._load_err
        import torch
        from moshi.models import loaders
        from huggingface_hub import hf_hub_download
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = _hf_token()
        # The moshi-format Mimi weight lives in the moshiko repo (DEFAULT_REPO),
        # NOT kyutai/mimi (that's the transformers-format repo with model.safetensors).
        name = getattr(loaders, "MIMI_NAME", "tokenizer-e351c8d8-checkpoint125.safetensors")
        default_repo = getattr(loaders, "DEFAULT_REPO", "kyutai/moshiko-pytorch-bf16")
        repos = []
        for r in (self.cfg.mimi_repo, default_repo):
            if r and r not in repos:
                repos.append(r)
        log(f"[mimi] loading moshi-format Mimi ({name}) ...")
        m = None; errs = []
        for r in repos:
            try:
                w = hf_hub_download(r, name, token=tok) if tok else hf_hub_download(r, name)
                try:
                    m = loaders.get_mimi(w, device=dev, num_codebooks=self.cfg.n_codebooks)
                except TypeError:           # older moshi: no num_codebooks kwarg
                    m = loaders.get_mimi(w, device=dev)
                log(f"[mimi] loaded from {r}")
                break
            except Exception as e:
                errs.append(f"{r}: {str(e)[:140]}")
        if m is None:
            self._load_err = RuntimeError("Mimi load failed (moshi weight not found). "
                "The moshi-format weight is in kyutai/moshiko-pytorch-bf16, not "
                "kyutai/mimi. Tried: " + " | ".join(errs))
            raise self._load_err
        try:
            m.set_num_codebooks(self.cfg.n_codebooks)
        except Exception as e:
            log(f"[mimi] set_num_codebooks failed ({e}); using default.", err=True)
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        self._m = m; self.sr = int(getattr(m, "sample_rate", 24000))
        return m

    @property
    def embed_dim(self):
        return self.cfg.jepa_lat_dim

    def _projection(self):
        """Fixed, seeded [codebook_size, jepa_lat_dim] matrix that maps a frozen
        semantic token to a stable continuous vector. Deterministic across runs
        and shards so the JEPA target space never changes dimension mid-training."""
        import numpy as np
        if self._proj is None:
            rng = np.random.default_rng(self.cfg.seed)
            P = rng.standard_normal((self.cfg.codebook_size, self.cfg.jepa_lat_dim))
            P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-6)
            self._proj = P.astype("float16")
        return self._proj

    def encode(self, wav16k_or_24k, in_sr):
        """wav (1D float32) -> Mimi codes int16[n_codebooks, T]."""
        import torch
        import numpy as np
        m = self.load()
        x = np.asarray(wav16k_or_24k, dtype="float32").reshape(-1)
        if in_sr != self.sr:
            import librosa
            x = librosa.resample(x, orig_sr=in_sr, target_sr=self.sr)
        t = torch.tensor(x).view(1, 1, -1).to(next(m.parameters()).device)
        with torch.no_grad():
            codes = m.encode(t)               # [1, K, T]
        return codes[0].to("cpu").numpy().astype("int16")

    def semantic_latent(self, codes):
        """JEPA target [D, T]: fixed projection of the frozen semantic codes
        (codebook 0). Depends only on the documented encode() path."""
        import numpy as np
        P = self._projection()
        sem = np.asarray(codes)[0].astype("int64") % self.cfg.codebook_size
        return P[sem].T.astype("float16")     # [D, T]

    def decode(self, codes):
        import torch
        import numpy as np
        m = self.load()
        c = torch.tensor(np.asarray(codes)).long().unsqueeze(0).to(next(m.parameters()).device)
        with torch.no_grad():
            wav = m.decode(c)
        return wav[0, 0].to("cpu").numpy(), self.sr


# =========================================================================== #
#  TOKENIZER (SentencePiece unigram 32k)                                        #
# =========================================================================== #
def train_tokenizer(cfg=CFG):
    import sentencepiece as spm
    corpus = Path(cfg.tok_dir) / "tok_corpus.txt"
    model_prefix = str(Path(cfg.tok_dir) / "spm")
    if Path(model_prefix + ".model").exists():
        log("[tok] already trained.")
        return model_prefix + ".model"
    if not corpus.exists() or corpus.stat().st_size < 1_000_000:
        _build_tokenizer_corpus(cfg, corpus)
    log(f"[tok] training SentencePiece unigram vocab={cfg.vocab_size} ...")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=model_prefix, vocab_size=cfg.vocab_size,
        model_type="unigram", character_coverage=1.0,
        input_sentence_size=5_000_000, shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 8, byte_fallback=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3)
    _tok_fertility_report(cfg, model_prefix + ".model")
    return model_prefix + ".model"


def _build_tokenizer_corpus(cfg, out, max_lines=6_000_000):
    """~70% TR / 20% EN / 10% TR-medical text sampled to a flat corpus file."""
    log("[tok] sampling tokenizer corpus (70 TR / 20 EN / 10 TR-med) ...")
    from datasets import load_dataset
    mix = [("HuggingFaceFW/fineweb-2", "tur_Latn", "text", 0.70),
           ("HuggingFaceFW/fineweb", None, "text", 0.20)]
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for repo, conf, col, frac in mix:
            budget = int(max_lines * frac); c = 0
            try:
                ds = load_dataset(repo, conf, split="train", streaming=True,
                                  cache_dir=cfg.hf_cache)
                for ex in ds:
                    t = str(ex.get(col, "")).strip()
                    if len(t) < 40:
                        continue
                    for line in t.split("\n"):
                        line = line.strip()
                        if len(line) >= 20:
                            fh.write(line + "\n"); c += 1; n += 1
                    if c >= budget:
                        break
            except Exception as e:
                log(f"[tok] corpus {repo} skipped ({e}).", err=True)
    log(f"[tok] corpus lines: {n}")


def _tok_fertility_report(cfg, model_path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    sample = ["merhaba doktor bey başım çok ağrıyor ne yapmalıyım",
              "hastaya parasetamol ve amoksisilin reçete edildi",
              "the patient was prescribed amoxicillin for the infection"]
    fert = sum(len(sp.encode(s)) for s in sample) / sum(len(s.split()) for s in sample)
    write_report("tokenizer", {"vocab": sp.get_piece_size(),
                               "fertility_tok_per_word": round(fert, 3)})


def load_tokenizer(cfg=CFG):
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=str(Path(cfg.tok_dir) / "spm.model"))


# =========================================================================== #
#  DATA — download + clean + PACK to memmapped shards (resumable)              #
# =========================================================================== #
def _shard_writer(prefix, dtype, cols):
    """Append-only sharded writer -> .npy shards + manifest.json (resumable)."""
    import numpy as np
    man_path = Path(CFG.shard_dir) / f"{prefix}_manifest.json"
    man = read_json(man_path, {"shards": [], "rows": 0}) or {"shards": [], "rows": 0}
    return man, man_path


def build_text_shards(cfg=CFG, target_tokens=None):
    """Tokenize text corpora into packed uint16 token shards of seq_len."""
    import numpy as np
    sp = load_tokenizer(cfg)
    target = target_tokens or cfg.target_text_tokens
    man_path = Path(cfg.shard_dir) / "text_manifest.json"
    man = read_json(man_path, None) or {"tokens": 0, "shards": [], "consumed": 0}
    man.setdefault("consumed", 0)
    if man["tokens"] >= target:
        log(f"[data] text shards done ({man['tokens']} tokens)."); return
    # dynamic disk cap
    free = disk_free_gb(cfg.data_dir)
    cap_tokens = int(max(0, free - cfg.disk_reserve_gb) * 1e9 / 2)  # uint16=2B/token
    target = min(target, man["tokens"] + cap_tokens)
    log(f"[data] building text shards up to {target} tokens (free {free:.0f}GB, "
        f"resume-skip {man['consumed']} docs) ...")
    from datasets import load_dataset
    eos = sp.eos_id()
    buf = []; sidx = len(man["shards"]); seen = 0
    SHARD = 2_000_000  # tokens/shard (~4MB)
    try:
        ds = load_dataset(cfg.text_datasets[0], "tur_Latn", split="train",
                          streaming=True, cache_dir=cfg.hf_cache)
        for ex in ds:
            if man["tokens"] >= target:
                break
            seen += 1
            if seen <= man["consumed"]:       # exact cursor: already packed
                continue
            man["consumed"] = seen
            ids = sp.encode(str(ex.get("text", ""))) + [eos]
            buf.extend(ids)
            if len(buf) >= SHARD:
                arr = np.asarray(buf[:SHARD], dtype="uint16"); buf = buf[SHARD:]
                fp = Path(cfg.shard_dir) / f"text_{sidx:06d}.npy"
                np.save(fp, arr); man["shards"].append(fp.name)
                man["tokens"] += len(arr); sidx += 1
                write_json(man_path, man)
                if sidx % 20 == 0:
                    log(f"[data] text tokens: {man['tokens']}")
                    if disk_pct(cfg.data_dir) >= cfg.disk_stop:
                        alert("disk near full during text packing -> stopping early"); break
    except Exception as e:
        log(f"[data] text build interrupted ({e}); progress saved.", err=True)
    log(f"[data] text shards: {man['tokens']} tokens in {len(man['shards'])} shards.")


def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _audio_specs(cfg):
    # format: repo[:config[:textcol[:split]]]   (config 'default'/'' -> no-config)
    specs = []
    for tok in cfg.audio_specs.split(","):
        p = tok.strip().split(":")
        if not p or not p[0]:
            continue
        specs.append({"repo": p[0], "config": p[1] if len(p) > 1 else "default",
                      "text": p[2] if len(p) > 2 else None,
                      "split": p[3] if len(p) > 3 else "train"})
    return specs


def _text_of(ex, key):
    for k in ([key] if key else []) + ["sentence", "transcription", "text", "raw_transcription"]:
        if k and ex.get(k):
            return str(ex[k])
    return ""


def _open_audio_stream(cfg, spec):
    """Return a streaming dataset for `spec` if some variant yields a DECODABLE
    clip. Robust to config vs no-config and script-vs-parquet: tries the given
    config and no-config, over the default revision then the parquet-converted
    ref; passes an HF token if present (for gated repos)."""
    from datasets import load_dataset, Audio
    tok = _hf_token(); last = None; split = spec.get("split", "train")
    conf = spec.get("config")
    conf_variants = ["default", None] if conf in (None, "", "default") else [conf, None]
    for conf_v in conf_variants:
        for rev in (None, "refs/convert/parquet"):
            try:
                pos = [spec["repo"]] + ([conf_v] if conf_v else [])
                kw = dict(split=split, streaming=True, cache_dir=cfg.hf_cache)
                if rev: kw["revision"] = rev
                if tok: kw["token"] = tok
                ds = load_dataset(*pos, **kw)
                # datasets v4+ decodes audio via torchcodec+ffmpeg; avoid that dep
                # entirely by taking RAW bytes and decoding them ourselves.
                try:
                    ds = ds.cast_column("audio", Audio(decode=False))
                except Exception:
                    pass
                ex = next(iter(ds))                 # probe: must exist AND decode
                wav, sr = _decode_audio(ex.get("audio"))
                if wav is not None and len(wav) > 0:
                    log(f"[data] audio source OK: {spec['repo']}"
                        f"[{conf_v or '-'}/{split}]{' @'+rev if rev else ''}")
                    return ds, spec
                last = "probe clip did not decode"
            except Exception as e:
                last = e
    log(f"[data] audio source unavailable {spec['repo']}:{conf} ({last}).", err=True)
    return None, None


def build_audio_shards(cfg=CFG, target_hours=None):
    """Mimi-encode Turkish speech into int16 code shards [K, T] (resumable).
    Robust to gated/script datasets: picks the first source that streams AND
    decodes; records it so resume stays on the same source."""
    import numpy as np
    target_h = target_hours or cfg.target_audio_hours
    man_path = Path(cfg.shard_dir) / "audio_manifest.json"
    man = read_json(man_path, None) or {"hours": 0.0, "shards": [], "consumed": 0, "source": ""}
    man.setdefault("consumed", 0); man.setdefault("source", "")
    if man["hours"] >= target_h:
        log(f"[data] audio shards done ({man['hours']:.1f} h)."); return
    free = disk_free_gb(cfg.data_dir)
    cap_h = int(max(0, free - cfg.disk_reserve_gb) * 1e9 / (5 * 1024 * 1024))
    target_h = min(target_h, man["hours"] + cap_h)
    log(f"[data] Mimi-encoding audio up to {target_h} h (free {free:.0f}GB, "
        f"resume-skip {man['consumed']} clips) ...")
    # pick a working source; prefer the previously-chosen one on resume
    specs = _audio_specs(cfg)
    specs.sort(key=lambda s: 0 if f"{s['repo']}:{s['config']}" == man["source"] else 1)
    ds = None; spec = None
    for cand in specs:
        ds, spec = _open_audio_stream(cfg, cand)
        if ds is not None:
            break
    if ds is None:
        alert("no audio source is reachable/decodable. Set HF_TOKEN (and accept the "
              "Common Voice terms on huggingface.co), or set CWM_AUDIO_SPECS to a "
              "dataset you can stream. Audio shards left empty.")
        write_json(man_path, man); return
    man["source"] = f"{spec['repo']}:{spec['config']}"; text_key = spec.get("text")
    write_json(man_path, man)
    mimi = MimiWrap(cfg)
    try:                                    # fail fast: load Mimi ONCE up front
        mimi.load()
    except Exception as e:
        alert(f"Mimi codec load failed: {e}"); write_json(man_path, man); return
    sidx = len(man["shards"]); seen = 0
    try:
        for ex in ds:
            if man["hours"] >= target_h:
                break
            seen += 1
            if seen <= man["consumed"]:       # exact cursor: already processed
                continue
            man["consumed"] = seen            # count it as consumed even if skipped
            wav, sr = _decode_audio(ex.get("audio"))
            if wav is None or len(wav) < sr * 0.5:
                continue
            try:
                codes = mimi.encode(wav, sr)               # [K, T]
                lat = mimi.semantic_latent(codes)          # [D, T] JEPA target
            except Exception as e:
                log(f"[data] encode skip ({e}).", err=True); continue
            fp = Path(cfg.shard_dir) / f"audio_{sidx:07d}.npz"
            np.savez_compressed(fp, codes=codes, lat=lat, text=_text_of(ex, text_key))
            man["shards"].append(fp.name)
            man["hours"] += codes.shape[1] / cfg.frame_hz / 3600.0
            sidx += 1
            if sidx % 200 == 0:
                write_json(man_path, man)
                log(f"[data] audio: {man['hours']:.1f} h")
                if disk_pct(cfg.data_dir) >= cfg.disk_stop:
                    alert("disk near full during audio packing -> stopping early"); break
        write_json(man_path, man)
    except Exception as e:
        log(f"[data] audio build interrupted ({e}); progress saved.", err=True)
        write_json(man_path, man)
    log(f"[data] audio shards: {man['hours']:.1f} h in {len(man['shards'])} clips.")


def _sf_or_librosa(src):
    """Decode a file/BytesIO to (mono float32, sr). soundfile first (wav/flac and,
    with a modern libsndfile, mp3); librosa/audioread fallback for mp3 if not."""
    import io
    import numpy as np
    import soundfile as sf
    try:
        a, sr = sf.read(src, dtype="float32")
        return (a.mean(1) if a.ndim > 1 else a), sr
    except Exception:
        try:
            import librosa
            if isinstance(src, io.BytesIO):
                src.seek(0)
            a, sr = librosa.load(src, sr=None, mono=True)
            return np.asarray(a, dtype="float32"), sr
        except Exception:
            return None, 0


def _decode_audio(field):
    import io
    import numpy as np
    if field is None:
        return None, 0
    try:
        if isinstance(field, dict):
            if field.get("array") is not None:      # already-decoded array
                a = np.asarray(field["array"], dtype="float32")
                return (a.mean(1) if a.ndim > 1 else a), int(field.get("sampling_rate", 16000))
            if field.get("bytes"):
                return _sf_or_librosa(io.BytesIO(field["bytes"]))
            if field.get("path"):
                return _sf_or_librosa(field["path"])
    except Exception:
        return None, 0
    return None, 0


# =========================================================================== #
#  MODEL — dual-stream core + depth transformer + JEPA + rollout               #
# =========================================================================== #
def _build_modules(cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        def __init__(self, d, eps): super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
        def forward(self, x):
            return self.w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

    def rope(x, theta):
        # x: [B,H,T,D] -> rotary
        B, H, T, D = x.shape
        half = D // 2
        freqs = 1.0 / (theta ** (torch.arange(0, half, device=x.device).float() / half))
        t = torch.arange(T, device=x.device).float()
        ang = torch.outer(t, freqs)                       # [T, half]
        # keep rotary math in x.dtype so bf16 q/k stay bf16 (else SDPA sees
        # fp32 q/k vs bf16 v and raises a dtype-mismatch on the first GPU forward)
        cos = ang.cos()[None, None].to(x.dtype); sin = ang.sin()[None, None].to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    class Attn(nn.Module):
        def __init__(self, d, nh, nkv, theta):
            super().__init__()
            self.nh, self.nkv, self.hd, self.theta = nh, nkv, d // nh, theta
            self.q = nn.Linear(d, nh * self.hd, bias=False)
            self.k = nn.Linear(d, nkv * self.hd, bias=False)
            self.v = nn.Linear(d, nkv * self.hd, bias=False)
            self.o = nn.Linear(nh * self.hd, d, bias=False)

        def forward(self, x, causal=True):
            B, T, _ = x.shape
            q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
            k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
            v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
            q, k = rope(q, self.theta), rope(k, self.theta)
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            return self.o(y.transpose(1, 2).reshape(B, T, -1))

    class SwiGLU(nn.Module):
        def __init__(self, d, h=None):
            super().__init__(); h = h or int(d * 8 / 3 // 64 * 64)
            self.w1 = nn.Linear(d, h, bias=False); self.w3 = nn.Linear(d, h, bias=False)
            self.w2 = nn.Linear(h, d, bias=False)
        def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class Block(nn.Module):
        def __init__(self, d, nh, nkv, eps, theta):
            super().__init__()
            self.n1 = RMSNorm(d, eps); self.at = Attn(d, nh, nkv, theta)
            self.n2 = RMSNorm(d, eps); self.mlp = SwiGLU(d)
        def forward(self, x, causal=True):
            x = x + self.at(self.n1(x), causal)
            return x + self.mlp(self.n2(x))

    class Core(nn.Module):
        """Decoder-only over interleaved TEXT + SEM(cb0) tokens. Stream-typed
        heads: TEXT positions -> vocab; SEM positions -> codebook0."""
        def __init__(self, cfg):
            super().__init__()
            V, S = cfg.vocab_size, cfg.codebook_size
            d = cfg.d_model
            self.text_emb = nn.Embedding(V, d)
            self.sem_emb = nn.Embedding(S, d)
            self.type_emb = nn.Embedding(2, d)        # 0=text 1=sem
            self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.n_kv_heads,
                                               cfg.rms_eps, cfg.rope_theta)
                                         for _ in range(cfg.n_layers)])
            self.norm = RMSNorm(d, cfg.rms_eps)
            self.text_head = nn.Linear(d, V, bias=False)
            self.sem_head = nn.Linear(d, S, bias=False)

        def embed(self, tok, typ):
            e = torch.where(typ.unsqueeze(-1).bool(),
                            self.sem_emb((tok * typ).clamp(0, self.sem_emb.num_embeddings - 1)),
                            self.text_emb((tok * (1 - typ)).clamp(0, self.text_emb.num_embeddings - 1)))
            return e + self.type_emb(typ)

        def forward(self, tok, typ):
            h = self.embed(tok, typ)
            for b in self.blocks:
                h = b(h, causal=True)
            h = self.norm(h)
            return h, self.text_head(h), self.sem_head(h)

    class Depth(nn.Module):
        """RQ-transformer: predict acoustic codebooks 1..7 at each SEM frame,
        conditioned on the core hidden state for that frame."""
        def __init__(self, cfg):
            super().__init__()
            d = cfg.d_model; self.K = cfg.n_codebooks - 1; S = cfg.codebook_size
            self.proj = nn.Linear(d, d)
            self.cb_emb = nn.ModuleList([nn.Embedding(S, d) for _ in range(self.K)])
            self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.n_kv_heads,
                                               cfg.rms_eps, cfg.rope_theta)
                                         for _ in range(cfg.depth_layers)])
            self.norm = RMSNorm(d, cfg.rms_eps)
            self.heads = nn.ModuleList([nn.Linear(d, S, bias=False) for _ in range(self.K)])

        def forward(self, hframe, acoustic):
            # hframe: [N, d] core hidden at SEM frames; acoustic: [N, K] targets
            N = hframe.shape[0]
            seq = [self.proj(hframe).unsqueeze(1)]
            for k in range(self.K - 1):
                seq.append(self.cb_emb[k](acoustic[:, k]).unsqueeze(1))
            x = torch.cat(seq, dim=1)                 # [N, K, d]
            for b in self.blocks:
                x = b(x, causal=True)
            x = self.norm(x)
            return [self.heads[k](x[:, k]) for k in range(self.K)]

    class JEPA(nn.Module):
        """Interlocutor prediction: from user-stream Mimi latents, predict FUTURE
        latents (horizons k) in an EMA-target embedding space (I-JEPA style)."""
        def __init__(self, cfg, in_dim):
            super().__init__()
            d = cfg.d_model
            self.inp = nn.Linear(in_dim, d)
            self.ctx = nn.ModuleList([Block(d, cfg.n_heads, cfg.n_kv_heads,
                                            cfg.rms_eps, cfg.rope_theta)
                                      for _ in range(cfg.jepa_ctx_layers)])
            self.tgt = nn.ModuleList([Block(d, cfg.n_heads, cfg.n_kv_heads,
                                            cfg.rms_eps, cfg.rope_theta)
                                      for _ in range(cfg.jepa_ctx_layers)])
            self.pred = nn.ModuleList([Block(d, cfg.n_heads, cfg.n_kv_heads,
                                             cfg.rms_eps, cfg.rope_theta)
                                       for _ in range(cfg.jepa_pred_layers)])
            self.horizons = cfg.jepa_horizons
            self.var_w = cfg.jepa_var_w; self.cov_w = cfg.jepa_cov_w
            for p in self.tgt.parameters():
                p.requires_grad_(False)
            self.inp_t = nn.Linear(in_dim, d)
            for p in self.inp_t.parameters():
                p.requires_grad_(False)

        @torch.no_grad()
        def ema_update(self, m):
            for pt, ps in zip(self.tgt.parameters(), self.ctx.parameters()):
                pt.mul_(m).add_(ps, alpha=1 - m)
            for pt, ps in zip(self.inp_t.parameters(), self.inp.parameters()):
                pt.mul_(m).add_(ps, alpha=1 - m)

        def _vicreg(self, z):
            # VICReg on the ONLINE embedding: variance hinge pushes per-dim std
            # toward 1 (prevents representation collapse); covariance decorrelates
            # dims. Computed in fp32. z: [B,T,D].
            z = z.float().reshape(-1, z.shape[-1])          # [N, D]
            std = torch.sqrt(z.var(dim=0) + 1e-4)           # per-dim std over N
            var_loss = torch.mean(F.relu(1.0 - std))        # hinge -> std>=1
            zc = z - z.mean(dim=0, keepdim=True)
            N, D = zc.shape
            cov = (zc.T @ zc) / max(1, N - 1)               # [D, D]
            cov_off = cov - torch.diag_embed(torch.diagonal(cov))
            cov_loss = cov_off.pow(2).sum() / D
            return var_loss, cov_loss, std.mean().item()

        def forward(self, lat):
            # lat: [B, T, in_dim] user-stream continuous latents.
            # Match the module's own (bf16 on GPU) dtype — the latents arrive as
            # fp32 from numpy, and the Linear would otherwise dtype-mismatch.
            lat = lat.to(self.inp.weight.dtype)
            c = self.inp(lat)
            for b in self.ctx: c = b(c, causal=True)
            with torch.no_grad():
                t = self.inp_t(lat)
                for b in self.tgt: t = b(t, causal=False)
            inv = c.new_zeros(()); T = lat.shape[1]
            for k in self.horizons:
                if T <= k: continue
                p = c[:, :T - k]
                for b in self.pred: p = b(p, causal=True)
                inv = inv + F.smooth_l1_loss(p, t[:, k:].detach())
            # VICReg variance+covariance keeps the invariance term from being
            # minimized by collapsing the representation to a constant.
            var_loss, cov_loss, std = self._vicreg(c)
            loss = inv + self.var_w * var_loss + self.cov_w * cov_loss
            return loss, std

    class Rollout(nn.Module):
        """Silent-thought: feed prev top hidden through a learned adapter; aux
        consistency loss = rollout state at t predicts core hidden at t+k."""
        def __init__(self, cfg):
            super().__init__()
            self.adapter = nn.Linear(cfg.d_model, cfg.d_model)
        def forward(self, hidden):
            # hidden: [B,T,d]; predict future hidden from adapted current.
            hidden = hidden.to(self.adapter.weight.dtype)
            r = self.adapter(hidden[:, :-4])
            return F.smooth_l1_loss(r, hidden[:, 4:].detach())

    class CWM(nn.Module):
        def __init__(self, cfg, jepa_in):
            super().__init__()
            self.cfg = cfg
            self.core = Core(cfg); self.depth = Depth(cfg)
            self.jepa = JEPA(cfg, jepa_in); self.rollout = Rollout(cfg)

    return {"CWM": CWM, "modules": locals()}


def build_model(cfg, jepa_in=512):
    import torch
    parts = _build_modules(cfg)
    model = parts["CWM"](cfg, jepa_in).to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.bfloat16) if torch.cuda.is_available() else model
    n = sum(p.numel() for p in model.parameters())
    log(f"[model] CWM built: {n/1e6:.1f}M params (jepa_in={jepa_in})")
    return model


# =========================================================================== #
#  CHECKPOINTING (resumable)                                                    #
# =========================================================================== #
def save_ckpt(cfg, stage, model, opt, sched, step, best, extra=None):
    import torch
    d = Path(cfg.stage_ckpt(stage)); d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"step_{step}.pt.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched else None,
                "step": step, "best": best, "cfg": asdict(cfg),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "extra": extra or {}}, tmp)
    os.replace(tmp, d / "last.pt")
    write_json(d / "state.json", {"step": step, "best": best})
    # prune old + honor ckpt disk budget
    _cap_ckpts(cfg)
    log(f"[ckpt] {stage} @ step {step} -> {d/'last.pt'}")


def _cap_ckpts(cfg):
    tot = 0
    files = sorted(Path(cfg.ckpt_dir).rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        tot += f.stat().st_size
        if tot > cfg.ckpt_budget_gb * 1e9 and f.name not in ("last.pt", "best.pt"):
            try: f.unlink()
            except Exception: pass


def load_ckpt(cfg, stage, model, opt=None, sched=None):
    import torch
    p = Path(cfg.stage_ckpt(stage)) / "last.pt"
    if not p.exists():
        return 0, float("inf")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if opt and ck.get("opt"): opt.load_state_dict(ck["opt"])
    if sched and ck.get("sched"): sched.load_state_dict(ck["sched"])
    try:
        torch.set_rng_state(ck["rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["cuda_rng"])
    except Exception:
        pass
    log(f"[ckpt] resumed {stage} @ step {ck['step']} (best {ck['best']:.4f})")
    return ck["step"], ck["best"]


# =========================================================================== #
#  DATA LOADING (packed shards -> batches)                                      #
# =========================================================================== #
def _text_batches(cfg, mbs, seq):
    import numpy as np
    man = read_json(Path(cfg.shard_dir) / "text_manifest.json", {"shards": []})
    shards = [Path(cfg.shard_dir) / s for s in man.get("shards", [])]
    rng = np.random.default_rng(cfg.seed)
    while shards:
        rng.shuffle(shards)
        for sp in shards:
            try:
                arr = np.load(sp)
            except Exception:
                continue
            n = (len(arr) // seq) * seq
            if n < seq:
                continue
            arr = arr[:n].reshape(-1, seq)
            idx = rng.permutation(len(arr))
            for i in range(0, len(idx) - mbs, mbs):
                yield arr[idx[i:i + mbs]]


def _audio_batches(cfg, mbs, seq_frames=256):
    import numpy as np
    man = read_json(Path(cfg.shard_dir) / "audio_manifest.json", {"shards": []})
    shards = [Path(cfg.shard_dir) / s for s in man.get("shards", [])]
    rng = np.random.default_rng(cfg.seed + 1)
    buf_codes, buf_lat = [], []
    while shards:
        rng.shuffle(shards)
        for sp in shards:
            try:
                z = np.load(sp, allow_pickle=True)
                codes, lat = z["codes"], z["lat"]
            except Exception:
                continue
            # codes [K,Tc], lat [D,Tl] can differ by a frame -> clamp to the shorter
            T = min(codes.shape[1], lat.shape[1], seq_frames)
            if T < 16:
                continue
            buf_codes.append(codes[:, :T]); buf_lat.append(lat[:, :T])
            if len(buf_codes) >= mbs:
                yield buf_codes[:mbs], buf_lat[:mbs]
                buf_codes, buf_lat = buf_codes[mbs:], buf_lat[mbs:]


def _infer_jepa_dim(cfg, fallback):
    """JEPA input dim = the stored Mimi-latent channel count. Read it from a real
    shard so a shape mismatch can't hard-crash training after hours of run."""
    import numpy as np
    man = read_json(Path(cfg.shard_dir) / "audio_manifest.json", {"shards": []})
    for s in man.get("shards", [])[:5]:
        try:
            z = np.load(Path(cfg.shard_dir) / s, allow_pickle=True)
            return int(z["lat"].shape[0])
        except Exception:
            continue
    return int(fallback)


# =========================================================================== #
#  TRAINING LOOP (self-healing, resumable, resource-guarded)                    #
# =========================================================================== #
def _stage_flags(stage):
    # which aux objectives are ON for each ablation variant
    return {
        "jepa":    dict(text=False, audio=True, jepa=True, rollout=False),
        "b0":      dict(text=True, audio=True, jepa=False, rollout=False),
        "b1":      dict(text=True, audio=True, jepa=False, rollout=True),
        "b2":      dict(text=True, audio=True, jepa=True, rollout=False),
        "b3":      dict(text=True, audio=True, jepa=True, rollout=True),
        "duplex":  dict(text=True, audio=True, jepa=True, rollout=True),
        "medical": dict(text=True, audio=True, jepa=True, rollout=True),
    }.get(stage, dict(text=True, audio=True, jepa=True, rollout=True))


def train_stage(cfg, stage, resume=False, max_seconds=None):
    import torch
    import numpy as np
    set_seed(cfg.seed)
    flags = _stage_flags(stage)
    jepa_in = _infer_jepa_dim(cfg, cfg.jepa_lat_dim) if flags["audio"] else cfg.jepa_lat_dim
    model = build_model(cfg, jepa_in)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(s):
        if s < cfg.warmup_steps:
            return s / max(1, cfg.warmup_steps)
        prog = (s - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = LambdaLR(opt, lr_lambda)

    step, best = (load_ckpt(cfg, stage, model, opt, sched) if resume else (0, float("inf")))
    guard = ResourceGuard(cfg)
    tgen = _text_batches(cfg, cfg.micro_batch, cfg.seq_len) if flags["text"] else None
    agen = _audio_batches(cfg, cfg.micro_batch) if flags["audio"] else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    warm_aux = int(cfg.aux_warmup_frac * cfg.total_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]
    # per-stage tokens/step for an honest MFU (audio stages run 256 frames, not seq_len)
    stage_tok = cfg.seq_len if flags["text"] else 256
    t0 = time.time(); last_save = time.time(); micro = 0; nonfinite_window = False
    start_step = step   # for a correct it/s on RESUME (steps done THIS session)
    opt.zero_grad(set_to_none=True)
    log(f"=== TRAIN stage={stage} flags={flags} from step {step} ===")
    runlog({"event": "train_start", "stage": stage, "step": step,
            "commit": _git_commit(), "cfg_hash": _cfg_hash(cfg)})

    while step < cfg.total_steps:
        # ------- assemble one micro-batch loss (text and/or audio) -----------
        loss = torch.zeros((), device=dev, dtype=torch.float32)
        parts = {}
        try:
            if flags["text"] and tgen is not None:
                tb = next(tgen)
                tok = torch.tensor(tb.astype("int64"), device=dev)
                typ = torch.zeros_like(tok)
                _, tlogits, _ = model.core(tok, typ)
                lt = torch.nn.functional.cross_entropy(
                    tlogits[:, :-1].reshape(-1, cfg.vocab_size).float(),
                    tok[:, 1:].reshape(-1))
                loss = loss + lt; parts["text"] = lt.item()
            if flags["audio"] and agen is not None:
                cb, lat = next(agen)
                T = min(x.shape[1] for x in cb)
                sem = torch.tensor(np.stack([c[0, :T] for c in cb]).astype("int64"), device=dev)
                typ = torch.ones_like(sem)
                h, _, slogits = model.core(sem, typ)
                la = torch.nn.functional.cross_entropy(
                    slogits[:, :-1].reshape(-1, cfg.codebook_size).float(),
                    sem[:, 1:].reshape(-1))
                # depth: acoustic codebooks 1..7 at each frame
                ac = torch.tensor(np.stack([c[1:, :T] for c in cb]).astype("int64"),
                                  device=dev).permute(0, 2, 1).reshape(-1, cfg.n_codebooks - 1)
                dlog = model.depth(h.reshape(-1, cfg.d_model), ac)
                ld = sum(torch.nn.functional.cross_entropy(dlog[k].float(), ac[:, k])
                         for k in range(cfg.n_codebooks - 1)) / (cfg.n_codebooks - 1)
                loss = loss + cfg.lambda_audio * (la + ld)
                parts["audio"] = (la + ld).item()
                aux_on = step >= warm_aux
                if flags["jepa"] and aux_on:
                    L = torch.tensor(np.stack([l[:, :T].T for l in lat]).astype("float32"),
                                     device=dev)   # [B,T,D]
                    lj, jstd = model.jepa(L)
                    loss = loss + cfg.lambda_jepa * lj
                    parts["jepa"] = float(lj.detach()); parts["jepa_std"] = round(jstd, 3)
                if flags["rollout"] and aux_on:
                    lr_ = model.rollout(h)
                    loss = loss + cfg.lambda_rollout * lr_
                    parts["rollout"] = float(lr_.detach())
        except StopIteration:
            log("[data] exhausted a stream; recreating iterators.")
            tgen = _text_batches(cfg, cfg.micro_batch, cfg.seq_len) if flags["text"] else None
            agen = _audio_batches(cfg, cfg.micro_batch) if flags["audio"] else None
            continue

        # accumulate only finite micro-losses; a single bad micro-batch marks the
        # whole window so we skip the optimizer step rather than apply poison grads
        if torch.isfinite(loss):
            (loss / cfg.grad_accum).backward()
        else:
            nonfinite_window = True
            alert("non-finite micro-loss -> dropped from this accumulation window")
        micro += 1
        if micro % cfg.grad_accum == 0:
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            if nonfinite_window or not torch.isfinite(gnorm):
                alert(f"non-finite in accumulation window (gnorm={float(gnorm):.3g}) -> skip step")
                opt.zero_grad(set_to_none=True); nonfinite_window = False; continue
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if flags["jepa"]:
                m = cfg.ema_final - (cfg.ema_final - cfg.ema_base) * max(0.0, 1 - step / cfg.total_steps)
                model.jepa.ema_update(m)
            step += 1
            g = guard.step(cfg.root)
            if g == "stop":
                save_ckpt(cfg, stage, model, opt, sched, step, best)
                log("[guard] stopping phase to protect the box; resume with --resume."); return "stopped"
            if step % cfg.log_steps == 0:
                used, tot = vram_used_total()
                its = (step - start_step) / max(1e-6, time.time() - t0)
                mfu = _mfu(cfg, its, stage_tok)
                eta_h = (cfg.total_steps - step) / max(1e-6, its) / 3600
                log(f"stage={stage} step={step}/{cfg.total_steps} loss={loss.item():.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e} vram={used:.1f}/{tot:.1f} {its:.3f}it/s "
                    f"mfu={mfu:.2f} eta={eta_h:.1f}h "
                    + " ".join(f"{k}={v}" for k, v in parts.items()))
                write_json(Path(cfg.log_dir) / f"metrics_{stage}.jsonl.last",
                           {"step": step, "loss": loss.item(), **parts, "mfu": mfu})
                # throttled collapse check: only after aux is on, once per log step
                js = parts.get("jepa_std")
                if js is not None and step >= warm_aux + 500 and js < cfg.jepa_collapse_std:
                    alert(f"JEPA online std {js} < {cfg.jepa_collapse_std} despite VICReg "
                          f"— raise CWM_JEPA_VAR (now {cfg.jepa_var_w})")
                # MFU note is expected-low on audio stages (short 256-frame seqs)
                if mfu < cfg.min_mfu and step > 200 and flags["text"]:
                    alert(f"MFU {mfu:.2f} < {cfg.min_mfu} — throughput low")
            if time.time() - last_save > cfg.save_secs:
                save_ckpt(cfg, stage, model, opt, sched, step, best); last_save = time.time()
            if max_seconds and time.time() - t0 > max_seconds:
                save_ckpt(cfg, stage, model, opt, sched, step, best)
                log(f"[gate] hit max_seconds={max_seconds}; checkpointed."); return "timebox"
    save_ckpt(cfg, stage, model, opt, sched, step, best)
    log(f"=== TRAIN stage={stage} DONE @ step {step} ===")
    runlog({"event": "train_done", "stage": stage, "step": step})
    return "done"


def _mfu(cfg, its, tokens_per_seq=None):
    # rough model-flops-util vs ~150 TFLOPS sustained on the MIG.
    # tokens_per_seq defaults to seq_len (text) but audio stages run fewer frames,
    # so callers pass the real per-sequence length to avoid an inflated MFU.
    tps = cfg.seq_len if tokens_per_seq is None else tokens_per_seq
    N = cfg.d_model * cfg.d_model * 12 * cfg.n_layers  # crude param proxy
    flops_tok = 6 * N
    toks = its * cfg.micro_batch * cfg.grad_accum * tps
    return round(flops_tok * toks / 150e12, 3)


# =========================================================================== #
#  SMOKE + BENCH GATES                                                          #
# =========================================================================== #
def cmd_smoke(cfg=CFG):
    """Overfit a TINY model on 32 random samples -> loss must go near zero."""
    import torch
    log("=== SMOKE (tiny model overfit 32 samples) ===")
    tiny = Config(**{**asdict_base(cfg), "d_model": 128, "n_layers": 2, "depth_layers": 2,
                     "jepa_ctx_layers": 2, "jepa_pred_layers": 1, "seq_len": 128,
                     "vocab_size": 512, "codebook_size": 256})
    model = build_model(tiny, jepa_in=64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(0)
    tok = torch.randint(0, tiny.vocab_size, (32, tiny.seq_len), generator=g).to(dev)
    typ = torch.zeros_like(tok)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    t0 = time.time(); loss = None
    for it in range(300):
        _, logits, _ = model.core(tok, typ)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, tiny.vocab_size).float(), tok[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0:
            log(f"[smoke] it={it} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
        if time.time() - t0 > 900:
            alert("smoke exceeded 15min"); break
    ok = loss is not None and loss.item() < 0.5
    write_report("smoke", {"final_loss": float(loss.item()), "pass": ok,
                           "secs": round(time.time() - t0)})
    log(f"[smoke] {'PASS' if ok else 'FAIL'} final_loss={loss.item():.4f}")
    return ok


def cmd_bench(cfg=CFG, steps=30):
    import torch
    import numpy as np
    log("=== BENCH (throughput + MFU + ETA) ===")
    model = build_model(cfg, jepa_in=512)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=torch.cuda.is_available())
    tok = torch.randint(0, cfg.vocab_size, (cfg.micro_batch, cfg.seq_len)).to(dev)
    typ = torch.zeros_like(tok)
    for i in range(3):    # warmup
        _, lg, _ = model.core(tok, typ)
        torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size).float(),
                                          tok[:, 1:].reshape(-1)).backward()
        opt.step(); opt.zero_grad()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    for i in range(steps):
        _, lg, _ = model.core(tok, typ)
        torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size).float(),
                                          tok[:, 1:].reshape(-1)).backward()
        opt.step(); opt.zero_grad()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    its = steps / (time.time() - t0)
    mfu = _mfu(cfg, its)
    eta_h = cfg.total_steps / its / 3600
    rep = {"it_per_s": round(its, 3), "mfu": mfu, "eta_full_hours": round(eta_h, 1),
           "gpu": gpu_name(), "min_mfu": cfg.min_mfu, "pass": mfu >= cfg.min_mfu}
    write_report("bench", rep)
    log(f"[bench] {its:.3f} it/s  MFU {mfu:.2f}  ETA(full)={eta_h:.1f}h  "
        f"{'PASS' if rep['pass'] else 'BELOW MIN'}")
    return rep


# =========================================================================== #
#  EVAL (loss/ppl + JEPA probe + throughput)                                    #
# =========================================================================== #
def cmd_eval(cfg, stage):
    import torch
    log(f"=== EVAL stage={stage} ===")
    model = build_model(cfg, _infer_jepa_dim(cfg, cfg.jepa_lat_dim))
    s, _ = load_ckpt(cfg, stage, model)
    if s == 0:
        log("[eval] no checkpoint; train first.", err=True)
        write_report(f"eval_{stage}", {"error": "no checkpoint"}); return
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    losses = []
    with torch.no_grad():
        for i, tb in zip(range(50), _text_batches(cfg, cfg.micro_batch, cfg.seq_len)):
            import numpy as np
            tok = torch.tensor(tb.astype("int64"), device=dev); typ = torch.zeros_like(tok)
            _, lg, _ = model.core(tok, typ)
            losses.append(torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, cfg.vocab_size).float(), tok[:, 1:].reshape(-1)).item())
    ppl = math.exp(sum(losses) / max(1, len(losses)))
    rep = {"text_ppl": round(ppl, 2), "n": len(losses), "step": s,
           "note": "CER/duplex-bench/turn-AUROC need synth duplex data + whisper -> S3+"}
    write_report(f"eval_{stage}", rep)
    log(f"[eval] {stage}: text ppl={ppl:.2f}")


# =========================================================================== #
#  GENERATE (qualitative sanity check: TR text + speech continuation)          #
# =========================================================================== #
def _sample_logits(logits, temperature=0.9, top_k=50):
    import torch
    logits = logits.float() / max(1e-5, temperature)
    if top_k and top_k < logits.shape[-1]:
        v, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < v[..., -1:], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)[..., 0]


def _depth_generate(depth, hframe, temperature, top_k):
    """Autoregressively sample acoustic codebooks cb1..cb7 for one frame from the
    core hidden, mirroring the teacher-forced training layout."""
    import torch
    x = depth.proj(hframe).unsqueeze(1)                 # [1,1,d]
    out = []
    for k in range(depth.K):
        xk = x
        for b in depth.blocks:
            xk = b(xk, causal=True)
        xk = depth.norm(xk)
        a = _sample_logits(depth.heads[k](xk[:, -1]), temperature, top_k)  # [1]
        out.append(a)
        if k < depth.K - 1:
            x = torch.cat([x, depth.cb_emb[k](a).unsqueeze(1)], dim=1)
    return torch.stack(out, dim=1)                      # [1, K]


def cmd_generate(cfg, stage, prompt, max_new, do_audio, seconds, temperature, top_k):
    import torch
    import numpy as np
    log(f"=== GENERATE stage={stage} ===")
    model = build_model(cfg, _infer_jepa_dim(cfg, cfg.jepa_lat_dim))
    s, _ = load_ckpt(cfg, stage, model)
    if s == 0:
        log("[gen] no checkpoint for that stage.", err=True); return
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.root) / "samples"; out_dir.mkdir(parents=True, exist_ok=True)

    # ---- text ---------------------------------------------------------------
    sp = load_tokenizer(cfg)
    ids = sp.encode(prompt) if prompt else [sp.bos_id() if sp.bos_id() >= 0 else 2]
    tok = torch.tensor([ids], device=dev)
    with torch.no_grad():
        for _ in range(max_new):
            typ = torch.zeros_like(tok)
            _, tl, _ = model.core(tok[:, -cfg.seq_len:], typ[:, -cfg.seq_len:])
            nxt = _sample_logits(tl[0, -1], temperature, top_k).view(1, 1)
            tok = torch.cat([tok, nxt], dim=1)
            if int(nxt) == sp.eos_id():
                break
    text = sp.decode(tok[0].tolist())
    (out_dir / f"gen_{stage}_text.txt").write_text(text, encoding="utf-8")
    log(f"[gen] TEXT ({stage}):\n{text}\n")

    if not do_audio:
        return
    # ---- speech continuation (prime on a real clip, let the model continue) --
    mimi = MimiWrap(cfg)
    try:
        mimi.load()
    except Exception as e:
        log(f"[gen] Mimi unavailable, skipping audio ({e}).", err=True); return
    man = read_json(Path(cfg.shard_dir) / "audio_manifest.json", {"shards": []})
    shards = man.get("shards", [])
    n_prime = 12                                        # ~1s of real audio to prime
    prime_codes = None
    if shards:
        z = np.load(Path(cfg.shard_dir) / shards[len(shards) // 2], allow_pickle=True)
        prime_codes = np.asarray(z["codes"])[:, :n_prime]   # [8, n_prime]
        sem = torch.tensor(prime_codes[0][None, :].astype("int64"), device=dev)
        log(f"[gen] priming on {n_prime} real frames from {shards[len(shards)//2]}")
    else:
        sem = torch.tensor([[np.random.randint(0, cfg.codebook_size)]], device=dev)
    n_frames = max(1, int(seconds * cfg.frame_hz))
    gen_frames = []
    with torch.no_grad():
        for _ in range(n_frames):
            typ = torch.ones_like(sem)
            h, _, sl = model.core(sem[:, -cfg.seq_len:], typ[:, -cfg.seq_len:])
            cb17 = _depth_generate(model.depth, h[:, -1], temperature, top_k)  # [1,7]
            cb0 = sem[:, -1:]                            # current-frame semantic
            gen_frames.append(torch.cat([cb0, cb17], dim=1)[0].to("cpu").numpy())
            nxt = _sample_logits(sl[0, -1], temperature, top_k).view(1, 1)
            sem = torch.cat([sem, nxt], dim=1)
    gen = np.stack(gen_frames, axis=1).astype("int64")  # [8, n_frames]
    codes = np.concatenate([prime_codes, gen], axis=1) if prime_codes is not None else gen
    try:
        wav, sr = mimi.decode(codes)
        import soundfile as sf
        wpath = out_dir / f"gen_{stage}_audio.wav"
        sf.write(str(wpath), np.asarray(wav, dtype="float32"), sr)
        secs = codes.shape[1] / cfg.frame_hz
        log(f"[gen] AUDIO ({stage}): {secs:.1f}s ({'primed+' if prime_codes is not None else ''}"
            f"generated) -> {wpath}")
    except Exception as e:
        log(f"[gen] audio decode failed: {e}", err=True)


# =========================================================================== #
#  AUTO ROADMAP (gated, self-healing, resumable)                               #
# =========================================================================== #
def cmd_auto(cfg, args):
    state_p = Path(cfg.root) / "roadmap_state.json"
    st = read_json(state_p, {}) or {}

    def done(k): return st.get(k, {}).get("done")
    def mark(k): st[k] = {"done": True, "t": time.time()}; write_json(state_p, st); os.environ["CWM_HEAL_N"] = "0"

    plan = ["setup", "tokenizer", "data", "smoke", "bench",
            "train:jepa", "train:b3", "eval:b3"]
    log(f"=== AUTO roadmap: {plan} ===")
    for step in plan:
        if done(step):
            log(f"[auto] {step} already done — skip."); continue
        log(f"[auto] >>>>>> {step}")
        res = _run_step(cfg, step, args)
        if res == "done":
            mark(step)
        elif res == "pause":
            # stage checkpointed itself early (timebox / resource-stop) — NOT a
            # failure and NOT complete. Leave it un-marked and exit cleanly so the
            # next `auto` run resumes this same stage from its checkpoint.
            log(f"[auto] step '{step}' PAUSED (checkpointed). Re-run "
                "`python cwm.py auto` (or run_cwm.sh start) to resume."); return
        else:
            die(f"[auto] step '{step}' did not pass its gate — see REPORT/ALERT. "
                "Fix the cause, then re-run: python cwm.py auto")
    log("=== AUTO roadmap complete (S0 + JEPA + B3). Extend plan for B0/B1/B2/duplex/medical. ===")


def _run_step(cfg, step, args):
    """Returns 'done' (mark + advance) | 'pause' (clean resume-later) | 'fail' (gate)."""
    if step == "setup":   return "done" if cmd_setup(cfg) else "fail"
    if step == "tokenizer":
        train_tokenizer(cfg)
        return "done" if Path(cfg.tok_dir, "spm.model").exists() else "fail"
    if step == "data":
        build_text_shards(cfg); build_audio_shards(cfg)
        tm = read_json(Path(cfg.shard_dir) / "text_manifest.json", {"tokens": 0})
        am = read_json(Path(cfg.shard_dir) / "audio_manifest.json", {"hours": 0})
        ok = tm.get("tokens", 0) > 1e6 and am.get("hours", 0) > 1.0
        if not ok:
            alert("data step produced too little (network/disk?) — not marking done")
        return "done" if ok else "fail"
    if step == "smoke":   return "done" if cmd_smoke(cfg) else "fail"
    if step == "bench":   return "done" if cmd_bench(cfg).get("pass", False) else "fail"
    if step.startswith("train:"):
        stage = step.split(":", 1)[1]
        r = train_stage(cfg, stage, resume=True, max_seconds=getattr(args, "max_seconds", None))
        # only a true completion advances the roadmap; timebox/stop -> resume later
        return "done" if r == "done" else "pause"
    if step.startswith("eval:"):
        cmd_eval(cfg, step.split(":", 1)[1]); return "done"
    return "done"


# =========================================================================== #
#  SETUP / DOCTOR / UTIL                                                        #
# =========================================================================== #
PIP = ["torch", "numpy", "sentencepiece", "soundfile", "librosa", "datasets>=2.18",
       "huggingface_hub", "moshi", "einops"]


def cmd_setup(cfg=CFG):
    log("=== SETUP ===")
    cfg.__post_init__()
    for mod in ("torch", "numpy", "sentencepiece", "soundfile", "datasets", "moshi"):
        try:
            __import__(mod)
        except Exception:
            log(f"  installing (missing {mod}) ...", err=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + PIP)
            break
    ok = doctor_report(cfg)
    log("=== SETUP DONE ===")
    return ok


def cmd_doctor(cfg=CFG):
    return doctor_report(cfg)


def doctor_report(cfg):
    healthy = True
    try:
        import torch
        log(f"  torch {torch.__version__} cuda={torch.cuda.is_available()} gpu={gpu_name()}")
        if not torch.cuda.is_available():
            alert("CUDA not available"); healthy = False
    except Exception as e:
        log(f"  torch missing ({e})", err=True); healthy = False
    free = disk_free_gb(cfg.root)
    log(f"  disk free @ {cfg.root}: {free:.0f} GB (reserve {cfg.disk_reserve_gb}, "
        f"ckpt budget {cfg.ckpt_budget_gb})")
    log(f"  ram used: {ram_pct()}%   disk used: {disk_pct(cfg.root)}%")
    if free < cfg.disk_reserve_gb + 50:
        alert(f"low disk ({free:.0f}GB) — data plan will auto-shrink")
    # auto-scale data targets to disk
    usable = max(0, free - cfg.disk_reserve_gb)
    log(f"  auto data cap: ~{int(usable*1e9/2/1e9)}B text tok OR ~{int(usable*1e9/(5*1024*1024))}h audio")
    log(f"=== DOCTOR (healthy={healthy}) ===")
    return healthy


def set_seed(s):
    import random
    import numpy as np
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def asdict_base(cfg):
    return {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _cfg_hash(cfg):
    import hashlib
    return hashlib.sha1(json.dumps(asdict_base(cfg), sort_keys=True, default=str)
                        .encode()).hexdigest()[:8]


def _clean_exit(code):
    """Exit WITHOUT running interpreter finalization. torch/moshi background
    threads can crash the GIL teardown (PyGILState_Release core dump) when a
    normal SystemExit unwinds through them, so we flush and hard-exit instead."""
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


def die(msg):
    log(msg, err=True); runlog({"event": "die", "msg": msg[:200]}); _clean_exit(1)


# =========================================================================== #
#  MAIN                                                                         #
# =========================================================================== #
def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    p = argparse.ArgumentParser(description="CWM-mini from-scratch trainer")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-seconds", type=int, default=None, dest="max_seconds")
    sub = p.add_subparsers(dest="cmd")
    for c in ("setup", "tokenizer", "data", "smoke", "bench", "doctor"):
        sub.add_parser(c)
    t = sub.add_parser("train"); t.add_argument("--stage", required=True)
    e = sub.add_parser("eval"); e.add_argument("--stage", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--stage", default="b3")
    g.add_argument("--prompt", default="Merhaba, bugün hava çok güzel ve")
    g.add_argument("--max-new", type=int, default=80, dest="max_new")
    g.add_argument("--audio", action="store_true")
    g.add_argument("--seconds", type=float, default=4.0)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--top-k", type=int, default=50, dest="top_k")
    sub.add_parser("auto")
    args = p.parse_args()
    cmd = args.cmd or "auto"

    def run():
        if cmd == "setup":      return cmd_setup(CFG)
        if cmd == "doctor":     return cmd_doctor(CFG)
        if cmd == "tokenizer":  return train_tokenizer(CFG)
        if cmd == "data":       build_text_shards(CFG); build_audio_shards(CFG); return
        if cmd == "smoke":      return cmd_smoke(CFG)
        if cmd == "bench":      return cmd_bench(CFG)
        if cmd == "train":      return train_stage(CFG, args.stage, resume=args.resume,
                                                   max_seconds=args.max_seconds)
        if cmd == "eval":       return cmd_eval(CFG, args.stage)
        if cmd == "generate":   return cmd_generate(CFG, args.stage, args.prompt, args.max_new,
                                                    args.audio, args.seconds, args.temperature, args.top_k)
        if cmd == "auto":       return cmd_auto(CFG, args)

    try:
        if "--no-heal" in sys.argv or cmd in ("doctor", "setup", "smoke", "bench", "generate", "eval"):
            run()
        else:
            supervise(run)
    except SystemExit as e:
        _clean_exit(int(e.code) if isinstance(e.code, int) else (0 if e.code in (None, 0) else 1))
    except KeyboardInterrupt:
        log("interrupted (Ctrl-C).", err=True); _clean_exit(130)
    except BaseException as e:
        log(f"[fatal] {type(e).__name__}: {str(e)[:300]}", err=True)
        runlog({"event": "fatal", "err": type(e).__name__}); _clean_exit(1)
    _clean_exit(0)


if __name__ == "__main__":
    main()
