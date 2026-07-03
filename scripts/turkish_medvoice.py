#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 turkish_medvoice.py  —  Native Turkish (medical) Speech-to-Speech + Text model
================================================================================

ONE self-contained file that: sets up the environment, builds/synthesizes data,
TRAINS/fine-tunes a NATIVE low-latency speech-to-speech model for Turkish that is
optimized for medical terminology (Turkish + English code-switch), and RUNS a
comprehensive "best-in-era" Turkish native-voice benchmark against baselines.

WHY "native" (and not a cascade)?
---------------------------------
A cascade (STT -> text-LLM -> TTS) waits for the *whole* transcript, then the
*whole* response, then the *whole* synthesis before the first audio plays. This
model (a) skips the Whisper ASR *decoder* entirely — the frozen Whisper *encoder*
features feed the LLM directly — and (b) STREAMS: the LLM emits Turkish response
text incrementally and the OmniVoice tail starts speaking the first chunk while
the LLM is still generating. First-audio latency drops from "sum of three full
stages" to "encoder pass + first text chunk + one OmniVoice chunk (RTF ~0.025)".

ARCHITECTURE  (all generative parts are Apache/MIT so the product is shippable)
--------------------------------------------------------------------------------
  [16 kHz Turkish speech in]
        │
        ▼  whisper-ft2 ENCODER ONLY (frozen)            # your Turkish Whisper-large-v3-turbo FT
     1280-d @ 50 Hz features
        │
        ▼  Conv1d(1280->d_llm, k=2, s=2) + LayerNorm    # projector, 50 Hz -> 25 Hz (TRAINED)
     speech soft-tokens
        │  spliced into the LLM input-embedding stream at a <speech> marker
        ▼  Qwen2.5-7B-Instruct  (QLoRA NF4 4-bit + LoRA)  # the "brain" (TRAINED, tiny adapters)
     incremental Turkish TEXT tokens ───────────────► [text stream: logging / audit trail]
        │  chunk every ~N tokens (R:W streaming policy)
        ▼  omnivoice-ft1 (frozen, text-conditioned, voice-clone, RTF 0.025)  # your Turkish TTS FT
  [24 kHz Turkish speech out]

  TEACHER (offline, text only): Qwen3-Omni-30B-A3B (AWQ-4bit) understands Turkish
  speech input and is a strong Turkish/medical text reasoner. It CANNOT speak
  Turkish (Turkish is not in its speech-output languages) so it is used ONLY as a
  sequence-level-KD text teacher; target AUDIO is always synthesized by OmniVoice.

SUBCOMMANDS
-----------
  setup   env/pip check, interactive HF login (once), prefetch models, smoke test
  data    download TR + medical datasets, build the medical gazetteer, synthesize
          (instruction_speech, response_text, response_speech) triples  [cached]
  train   --stage {align,s2s,medical,distill} [--resume]   QLoRA, checkpoint/resume
  eval    --suite {asr,tts,s2s,medical,all} [--baselines ...]   the full benchmark
  serve   streaming native S2S endpoint  (--cascade for the safe fallback path)
  doctor  preflight + AUTO-FIX (CUDA/MIG env, missing deps, low disk, asset paths,
          TTS reachability with autostart, HF auth)
  auto    run the WHOLE roadmap unattended (doctor->data->train*->eval), resumable

SELF-HEALING (fully automated error recovery)
---------------------------------------------
  Every subcommand runs under a top-level supervisor that classifies failures and
  applies an automated remediation, then either retries in-process or RE-EXECs the
  process (which fully resets CUDA) and resumes from the last checkpoint. It is
  bounded (TMV_MAX_HEAL, default 12 restarts) so it can never loop forever, and
  journals every failure+fix to logs/heal.jsonl. Handled automatically:
    - missing python dep      -> pip-install the mapped package, restart
    - CUDA OOM / NVML(MIG)     -> set expandable_segments, climb the degradation
                                 ladder (seq-len -> grad-accum -> student size),
                                 restart with --resume
    - HF hub / network hiccup -> exponential backoff + retry (then restart)
    - OmniVoice TTS down       -> autostart the server (README cmd) + retry
    - corrupt/mismatched ckpt  -> restart from base adapters
    - low disk                 -> prune stale files, restart
  Disable with the global --no-heal flag. Best used via `auto` inside tmux.

--------------------------------------------------------------------------------
 TRANSFER TO SERVER + RUN (survives SSH drops via tmux)
--------------------------------------------------------------------------------
  # 1) copy just this file to the server (it needs nothing else):
  scp -P 30405 scripts/turkish_medvoice.py root@10.6.110.10:/root/ses_models/
  #    (or on the server:  git clone https://github.com/ArioMoniri/turso.git )

  # 2) open a tmux session so training keeps running if SSH drops:
  ssh -p 30405 root@10.6.110.10
  tmux new -s medvoice          # (re-attach later with:  tmux attach -t medvoice)

  # 3) one-time setup (creates venv, installs deps, prompts for HF token IF needed):
  cd /root/ses_models
  python3 turkish_medvoice.py setup            # will print the venv activate line
  #    activate the venv it created, then re-run subcommands inside it.

  # 4) EASIEST: one self-healing command runs the whole roadmap and recovers from
  #    errors on its own (data -> align -> s2s -> medical -> eval), resuming after
  #    any crash. Detach with Ctrl-b then d; it keeps going.
  python3 turkish_medvoice.py auto
  #    ...or drive it stage-by-stage (each is resumable and self-healing too):
  python3 turkish_medvoice.py doctor            # preflight + auto-fix
  python3 turkish_medvoice.py data
  python3 turkish_medvoice.py train --stage align
  python3 turkish_medvoice.py train --stage s2s
  python3 turkish_medvoice.py train --stage medical

  # 5) benchmark against baselines and write the report:
  python3 turkish_medvoice.py eval --suite all

  # 6) serve (native streaming) or the cascade fallback:
  python3 turkish_medvoice.py serve
  python3 turkish_medvoice.py serve --cascade

HF TOKEN: none of the *required* models are gated, so training needs NO token.
A token is only needed to PUSH private checkpoints. `setup` checks
huggingface_hub.whoami(); if you want to log in, it prompts once (getpass) and
caches to ~/.cache/huggingface/token. It also honors the HF_TOKEN env var. The
token is NEVER written into the repo.

NOTE: this is a research/education pipeline; the resulting model is NOT a
clinical decision tool. Medical outputs must be reviewed by a professional.
================================================================================
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Only stdlib is imported at module load. Heavy libs (torch, transformers,    #
#  datasets, ...) are imported lazily *inside* the functions that use them, so #
#  `setup` can install them before anything tries to import them.             #
# --------------------------------------------------------------------------- #

IS_TTY = sys.stdin.isatty()


# =========================================================================== #
#  CONFIG  — every model id, path and knob lives here. Override via env vars   #
#  (TMV_*) or a YAML passed with --config. Fallbacks flip a single flag.       #
# =========================================================================== #

def _env(key, default):
    return os.environ.get(key, default)


@dataclass
class Config:
    # ---- root / run layout -------------------------------------------------
    root:        str = _env("TMV_ROOT", "/root/ses_models")
    work:        str = _env("TMV_WORK", "/root/medvoice")          # our outputs live here
    venv:        str = _env("TMV_VENV", "/root/venv-medvoice")
    seed:        int = int(_env("TMV_SEED", "1234"))

    # ---- your existing on-server assets -----------------------------------
    whisper_ckpt:      str = _env("TMV_WHISPER_CKPT", "/root/ses_models/whisper-ft2")
    whisper_processor: str = _env("TMV_WHISPER_PROC", "openai/whisper-large-v3-turbo")
    omni_model:        str = _env("TMV_OMNI_MODEL", "/root/ses_models/omnivoice-ft1")
    omni_ref_wav:      str = _env("TMV_OMNI_REF", "/root/ses_models/ref/emin.wav")
    omni_ref_txt:      str = _env("TMV_OMNI_REFTXT", "/root/ses_models/ref/emin.txt")
    omni_lang:         str = _env("TMV_OMNI_LANG", "tr")
    # OpenAI-compatible endpoints you already run (used as robust TTS/STT fallbacks)
    omni_server_url:   str = _env("TMV_OMNI_URL", "http://127.0.0.1:8133/v1/audio/speech")
    stt_server_url:    str = _env("TMV_STT_URL",  "http://127.0.0.1:8135/v1/audio/transcriptions")

    # ---- student / teacher / eval models ----------------------------------
    student_llm: str = _env("TMV_STUDENT", "Qwen/Qwen2.5-7B-Instruct")   # Apache-2.0
    teacher_llm: str = _env("TMV_TEACHER", "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit")
    text_teacher_fallback: str = _env("TMV_TEACHER_TXT", "Qwen/Qwen2.5-7B-Instruct")
    eval_asr:    str = _env("TMV_EVAL_ASR", "openai/whisper-large-v3")    # neutral round-trip ASR
    eval_judge:  str = _env("TMV_EVAL_JUDGE", "Qwen/Qwen2.5-72B-Instruct-AWQ")  # optional local judge
    spk_sim:     str = _env("TMV_SPK_SIM", "speechbrain/spkrec-ecapa-voxceleb")

    # ---- projector / model dims -------------------------------------------
    whisper_dim:   int = 1280
    down_factor:   int = 2          # 50 Hz -> 25 Hz
    max_audio_sec: float = 30.0     # Whisper 30 s window
    max_seq_len:   int = int(_env("TMV_MAXSEQ", "3072"))

    # ---- QLoRA / training --------------------------------------------------
    load_in_4bit:  bool = _env("TMV_4BIT", "1") == "1"
    lora_r:        int = int(_env("TMV_LORA_R", "32"))
    lora_alpha:    int = int(_env("TMV_LORA_ALPHA", "32"))
    lora_dropout:  float = float(_env("TMV_LORA_DROPOUT", "0.05"))
    micro_batch:   int = int(_env("TMV_MBS", "1"))
    grad_accum:    int = int(_env("TMV_GA", "16"))
    grad_ckpt:     bool = _env("TMV_GCKPT", "1") == "1"
    attn_impl:     str = _env("TMV_ATTN", "flash_attention_2")   # falls back to sdpa
    save_steps:    int = int(_env("TMV_SAVE_STEPS", "500"))
    log_steps:     int = int(_env("TMV_LOG_STEPS", "10"))
    max_grad_norm: float = 1.0

    # per-stage LR / epochs (LoRA-tuned, not the paper's full-FT LRs)
    stage_lr:     dict = field(default_factory=lambda: {
        "align": 1.0e-4, "s2s": 2.0e-4, "medical": 1.0e-4, "distill": 1.0e-4})
    stage_epochs: dict = field(default_factory=lambda: {
        "align": 1, "s2s": 1, "medical": 3, "distill": 1})

    # ---- data --------------------------------------------------------------
    # (id, hf_config, split, audio_col, text_col, kind)   kind in {asr, instruct}
    asr_datasets: list = field(default_factory=lambda: [
        # column names verified on the 2026 Hub: both use 'transcription'
        ("ysdede/commonvoice_17_tr_fixed", None, "train", "audio", "transcription", "asr"),
        ("ysdede/khanacademy-turkish",     None, "train", "audio", "transcription", "asr"),
    ])
    # OPTIONAL heavy ASR set (218h). NOT auto-loaded (its parquet viewer is broken;
    # needs huggingface_hub.snapshot_download + custom parsing). To use it, download
    # manually and add a tuple to `asr_datasets` pointing at the local files.
    issai_repo: str = "issai/Turkish_Speech_Corpus"
    # Turkish instruction / dialogue text (has some medical)
    instruct_dataset: str = "turkish-nlp-suite/InstrucTurca"
    # real Turkish patient/doctor Q&A (gold medical answers -> best SFT targets)
    medqa_dataset: str = "kayrab/patient-doctor-qa-tr-167732"
    # eval sets (held OUT of training)
    fleurs_repo: str = "google/fleurs"
    mediaspeech_repo: str = "ymoslem/MediaSpeech"
    medturkquad_repo: str = "incidelen/MedTurkQuAD"   # EVAL ONLY (CC-BY-NC-ND)

    n_synth_medical: int = int(_env("TMV_N_MED", "20000"))
    n_general_s2s:   int = int(_env("TMV_N_S2S", "60000"))
    n_align:         int = int(_env("TMV_N_ALIGN", "60000"))

    # ---- streaming / serve -------------------------------------------------
    stream_chunk_tokens: int = int(_env("TMV_CHUNK", "12"))   # R:W ~ chunk cadence
    serve_host: str = _env("TMV_SERVE_HOST", "127.0.0.1")
    serve_port: int = int(_env("TMV_SERVE_PORT", "8140"))

    # ---- derived paths (filled in __post_init__) ---------------------------
    def __post_init__(self):
        self.work = str(Path(self.work))
        self.data_dir     = str(Path(self.work) / "data")
        self.synth_dir    = str(Path(self.work) / "data" / "synth")
        self.gazetteer    = str(Path(self.work) / "data" / "medical_gazetteer.jsonl")
        self.ckpt_dir     = str(Path(self.work) / "checkpoints")
        self.log_dir      = str(Path(self.work) / "logs")
        self.bench_dir    = str(Path(self.work) / "bench_results")
        self.hf_cache      = _env("HF_HOME", str(Path(self.work) / "hf_cache"))

    def stage_ckpt(self, stage):
        return str(Path(self.ckpt_dir) / stage)

    def ensure_dirs(self):
        for d in [self.work, self.data_dir, self.synth_dir, self.ckpt_dir,
                  self.log_dir, self.bench_dir, self.hf_cache]:
            Path(d).mkdir(parents=True, exist_ok=True)


CFG = Config()


# =========================================================================== #
#  Logging / small utilities                                                   #
# =========================================================================== #

def log(msg, *, err=False):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    stream = sys.stderr if err else sys.stdout
    print(line, file=stream, flush=True)      # flush -> tmux/nohup friendly
    try:
        Path(CFG.log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(CFG.log_dir) / "medvoice.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def die(msg, code=1):
    log("FATAL: " + msg, err=True)
    sys.exit(code)


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path):
    out = []
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def gpu_mem_gb():
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return round((total - free) / 1e9, 2), round(total / 1e9, 2)
    except Exception:
        pass
    return None, None


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# =========================================================================== #
#  SELF-HEALING ENGINE                                                          #
#  Classifies runtime failures and applies an automated remediation, then      #
#  either retries in-process or re-EXECs the process (with --resume) so long,   #
#  unattended runs recover on their own instead of dying. Every failure and    #
#  remediation is journaled to logs/heal.jsonl. Bounded so it can never loop    #
#  forever.                                                                     #
# =========================================================================== #

# Maximum number of self-heal RE-EXECs across the whole run (env-persisted).
MAX_HEAL_REEXEC = int(_env("TMV_MAX_HEAL", "12"))
# Degradation ladder level (env-persisted across re-execs).
HEAL_LEVEL = int(_env("TMV_HEAL_LEVEL", "0"))

# import-name -> pip spec, for auto-installing a missing dependency on the fly.
_IMPORT_TO_PKG = {
    "torch": "torch==2.8.0", "torchaudio": "torchaudio==2.8.0",
    "transformers": "transformers>=4.57.0", "peft": "peft>=0.14",
    "datasets": "datasets>=2.18", "bitsandbytes": "bitsandbytes>=0.45",
    "accelerate": "accelerate>=0.34", "huggingface_hub": "huggingface_hub>=0.25",
    "safetensors": "safetensors", "librosa": "librosa==0.10.2", "soundfile": "soundfile",
    "sentencepiece": "sentencepiece", "numpy": "numpy<2.3", "yaml": "pyyaml",
    "jiwer": "jiwer>=3.0.4", "speechbrain": "speechbrain", "resemblyzer": "resemblyzer",
    "torchmetrics": "torchmetrics[audio]", "fastapi": "fastapi", "uvicorn": "uvicorn[standard]",
    "requests": "requests", "multipart": "python-multipart",
    "omnivoice": "omnivoice", "vllm": "vllm>=0.17", "trnorm": "trnorm",
    "utmosv2": "git+https://github.com/sarulab-speech/UTMOSv2.git",
    "flash_attn": "flash-attn>=2.7.0", "torchcodec": "torchcodec",
    "phonemizer": "phonemizer", "torchmetrics.audio": "torchmetrics[audio]",
}

# Emergency-degradation ladder (applied CUMULATIVELY at startup from
# TMV_HEAL_LEVEL). It ONLY reduces the memory footprint of the SAME model
# (seq-len / grad-accum / micro-batch). It deliberately does NOT swap the student
# to a smaller model: that would change hidden_size and silently discard the
# projector + LoRA + optimizer state (a hidden restart-from-base), so if even the
# smallest footprint OOMs we surface the error instead.
MAX_HEAL_LEVEL = 4


def _apply_heal_level(cfg, level):
    if level <= 0:
        return
    log(f"[heal] applying degradation level {level} (seq/batch footprint only)", err=True)
    if level >= 1:
        cfg.max_seq_len = min(cfg.max_seq_len, 2048); cfg.grad_accum = max(cfg.grad_accum, 32)
    if level >= 2:
        cfg.max_seq_len = min(cfg.max_seq_len, 1536); cfg.micro_batch = 1
    if level >= 3:
        cfg.max_seq_len = min(cfg.max_seq_len, 1024); cfg.grad_accum = max(cfg.grad_accum, 64)
    if level >= 4:
        cfg.max_seq_len = min(cfg.max_seq_len, 768)


def _heal_journal(context, kind, exc, action, attempt):
    rec = {"time": time.time(), "context": context, "kind": kind,
           "error": f"{type(exc).__name__}: {str(exc)[:400]}",
           "action": action, "attempt": attempt, "heal_level": HEAL_LEVEL}
    try:
        append_jsonl(Path(CFG.log_dir) / "heal.jsonl", rec)
    except Exception:
        pass
    log(f"[heal] {context}: {kind} -> {action} (attempt {attempt})", err=True)


def _heal_tts_ports():
    import re
    ports = []
    for url in (CFG.omni_server_url, CFG.stt_server_url):
        m = re.search(r":(\d+)", url or "")
        if m:
            ports.append(m.group(1))
    return ports


# HF-hub errors that mean "the asset does not exist / you lack access" — these are
# PERMANENT (the user must fix), never retried.
_FATAL_ERR_NAMES = {
    "RepositoryNotFoundError", "GatedRepoError", "EntryNotFoundError",
    "RevisionNotFoundError", "FileNotFoundError",
    "NotADirectoryError", "PermissionError", "DatasetNotFoundError",
    "HFValidationError", "IsADirectoryError",
}
# transient network error TYPES (classified by type, not brittle substrings).
# LocalEntryNotFoundError is here (not fatal): hf_hub raises it when the Hub is
# unreachable AND the file isn't cached yet — i.e. a transient outage on a cold
# cache, which we want to retry, not surface.
_NET_ERR_NAMES = {
    "ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout", "SSLError",
    "ChunkedEncodingError", "IncompleteRead", "ProtocolError", "HfHubHTTPError",
    "LocalEntryNotFoundError",
}


def classify_error(exc):
    """Map an exception to a remediation class. Errs on the side of `fatal` (which
    is surfaced, not retried) so a permanent error can never drive an endless
    re-exec loop. Classes: missing_dep | oom | nvml | disk | tts | network | fatal."""
    name = type(exc).__name__
    mod = (type(exc).__module__ or "").split(".")[0]
    msg = (str(exc) or "").lower()

    if name in ("ModuleNotFoundError", "ImportError"):
        return "missing_dep"
    if name == "OutOfMemoryError" or "out of memory" in msg or "cuda oom" in msg:
        return "oom"
    if "nvml" in msg or "no cuda-capable device" in msg or "cuda unknown error" in msg:
        return "nvml"
    if name in _FATAL_ERR_NAMES:                 # missing asset / gated / 404 / perms
        return "fatal"
    if "no space left" in msg or "disk quota" in msg or "errno 28" in msg:
        return "disk"
    # TTS-server reachability — match the CONFIGURED ports so the supervisor can
    # auto-start the OmniVoice server (checked before generic network).
    if (any(p in msg for p in _heal_tts_ports()) or "omnivoice" in msg) and "cuda" not in msg:
        return "tts"
    # network transience by exception TYPE (robust) or explicit HTTP-5xx/429 text
    if mod in ("requests", "urllib", "urllib3", "http", "socket", "aiohttp") \
            or name in _NET_ERR_NAMES:
        return "network"
    if any(s in msg for s in ("temporarily unavailable", "max retries exceeded",
                              "connection reset", "connection aborted", "server error",
                              " 503", " 502", " 504", " 429")):
        return "network"
    return "fatal"                                # unknown -> surface, do not loop


def _pkg_for_import(exc):
    """Return (pip_spec_or_None, modname). Only WHITELISTED modules yield a spec;
    an unknown module returns (None, modname) so we never install an arbitrary
    package name parsed out of an (attacker-influenceable) error string."""
    import re
    msg = str(exc)
    modname = ""
    m = re.search(r"No module named ['\"]?([\w\-.]+)", msg)
    if m:
        modname = m.group(1).split(".")[0]
    else:
        # libs that raise a plain ImportError telling you to `install 'X'`
        # (e.g. datasets: "please install 'torchcodec'"). Only whitelisted names
        # below yield a spec, so this cannot install something arbitrary.
        m = re.search(r"install ['\"`]([\w\-.\[\]]+)['\"`]", msg)
        if m:
            modname = m.group(1).split("[")[0].split("==")[0]
    return _IMPORT_TO_PKG.get(modname), modname   # unmapped -> None (refuse to install)


def _auto_pip_install(spec, modname=None):
    """Install a WHITELISTED package into the current interpreter, at most once per
    module across the whole run (env-tracked) to prevent install/re-exec thrash on
    a transitively-broken import. Refuses anything not in the _IMPORT_TO_PKG values."""
    if not spec or spec not in set(_IMPORT_TO_PKG.values()):
        log(f"[heal] refusing to auto-install non-whitelisted dependency '{spec}'.", err=True)
        return False
    tried = set(filter(None, os.environ.get("TMV_HEAL_INSTALLED", "").split(",")))
    key = modname or spec
    if key in tried:
        log(f"[heal] '{key}' was already auto-installed once and still failing; "
            "surfacing instead of looping.", err=True)
        return False
    try:
        if spec.startswith("flash-attn"):
            _pip(sys.executable, [spec, "--no-build-isolation"])
        elif spec.startswith(("torch", "torchaudio")):
            _pip(sys.executable, [spec, "--index-url", TORCH_INDEX])
        else:
            _pip(sys.executable, [spec])
        os.environ["TMV_HEAL_INSTALLED"] = ",".join(sorted(tried | {key}))
        return True
    except Exception as e:
        log(f"[heal] auto-install of '{spec}' failed: {e}", err=True)
        return False


def _prune_checkpoints(cfg, keep=1):
    """Free disk by deleting all but the newest per-stage checkpoint dirs and old caches."""
    freed = []
    try:
        import glob
        for stage in ("align", "s2s", "medical", "distill"):
            # our layout keeps a single dir per stage, so only prune stray *.tmp / old opt
            d = Path(cfg.stage_ckpt(stage))
            for junk in glob.glob(str(d / "*.tmp")) + glob.glob(str(d / "optimizer.pt.bak")):
                try:
                    os.remove(junk); freed.append(junk)
                except Exception:
                    pass
    except Exception:
        pass
    log(f"[heal] pruned {len(freed)} stale files to free disk.", err=True)
    return len(freed) > 0


def _reexec(extra_env=None, ensure_args=()):
    """Fully restart this process (resets CUDA/context) with the SAME argv +
    optional extra env + ensured flags. Bounded by MAX_HEAL_REEXEC."""
    n = int(os.environ.get("TMV_HEAL_REEXEC_N", "0")) + 1
    if n > MAX_HEAL_REEXEC:
        log(f"[heal] giving up: exceeded {MAX_HEAL_REEXEC} self-heal restarts.", err=True)
        return False
    env = dict(os.environ)
    env["TMV_HEAL_REEXEC_N"] = str(n)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    argv = [sys.executable] + sys.argv
    for a in ensure_args:
        if a not in argv:
            argv.append(a)
    log(f"[heal] restarting process (#{n}/{MAX_HEAL_REEXEC}) ...", err=True)
    sys.stdout.flush(); sys.stderr.flush()
    os.execve(sys.executable, argv, env)   # never returns on success
    return False


def resilient(fn, *, retries=4, base_delay=3.0, context="op", on_give_up=None):
    """Run fn() with in-process automated remediation + capped exponential backoff.
    Handles the classes that CAN be fixed without a full restart (network, missing
    dep, transient load). Raises (or calls on_give_up) once retries are exhausted or
    the error is fatal. OOM/NVML are NOT retried here — they need a process restart
    and are handled by the top-level supervisor."""
    import random as _rnd
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:                     # noqa: BLE001 (intentional broad heal)
            attempt += 1
            kind = classify_error(e)
            # OOM/NVML need a process restart (handled by the top-level supervisor);
            # `fatal` is permanent -> surface at once so we never retry pointlessly.
            if kind in ("oom", "nvml", "fatal") or attempt > retries:
                _heal_journal(context, kind, e, "give-up", attempt)
                if on_give_up is not None:
                    return on_give_up(e)
                raise
            action = "backoff-retry"
            if kind == "missing_dep":
                spec, modname = _pkg_for_import(e)
                if not _auto_pip_install(spec, modname):
                    _heal_journal(context, kind, e, "give-up(non-whitelisted)", attempt)
                    raise                          # cannot safely install -> surface
                action = f"pip-install {spec}"
            elif kind == "disk":
                action = "prune-disk"; _prune_checkpoints(CFG)
            _heal_journal(context, kind, e, action, attempt)
            delay = min(60.0, base_delay * (2 ** (attempt - 1))) + _rnd.uniform(0, 1.5)
            time.sleep(delay)


def supervise(exc, args):
    """Top-level supervisor: last-resort automated recovery for an uncaught error
    from a subcommand. Applies a remediation and RE-EXECs the process (which resets
    CUDA and resumes from the last checkpoint). Returns only if it could not heal
    (caller should then re-raise)."""
    kind = classify_error(exc)
    # `auto` resumes via its own state file + cmd_train(resume=True), so only the
    # bare `train` subcommand needs the --resume flag re-appended.
    add_resume = ["--resume"] if getattr(args, "cmd", "") == "train" else []
    _heal_journal(getattr(args, "cmd", "?"), kind, exc, f"supervise/{kind}", 0)

    if kind == "missing_dep":
        spec, modname = _pkg_for_import(exc)
        if _auto_pip_install(spec, modname):       # whitelist + install-once guarded
            _reexec(ensure_args=add_resume)
    elif kind == "oom":
        if HEAL_LEVEL >= MAX_HEAL_LEVEL:
            log("[heal] OOM persists at the smallest footprint (level "
                f"{MAX_HEAL_LEVEL}); surfacing instead of looping.", err=True)
            return False
        env = {"TMV_HEAL_LEVEL": min(MAX_HEAL_LEVEL, HEAL_LEVEL + 1),
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        _reexec(extra_env=env, ensure_args=add_resume)   # climb the footprint ladder
    elif kind == "nvml":
        # MIG/driver hiccup: reset the allocator and restart WITHOUT shrinking
        # anything (the OOM ladder is the wrong remedy for a device fault).
        _reexec(extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                ensure_args=add_resume)
    elif kind == "network":
        time.sleep(10)
        _reexec(ensure_args=add_resume)
    elif kind == "disk":
        if _prune_checkpoints(CFG):
            _reexec(ensure_args=add_resume)
    elif kind == "tts":
        if _ensure_tts_server(CFG):
            _reexec(ensure_args=add_resume)
    # NOTE: corrupt/mismatched checkpoints are handled tolerantly inside
    # build_model (fresh projector + skipped LoRA on mismatch), so there is no
    # auto "drop adapters and restart" path here — that would silently throw away
    # resume progress on any error whose text merely mentions a shape/state_dict.
    return False   # could not heal -> caller re-raises


def _proactive_env():
    """Set environment that pre-empts the most common MIG/tokenizer/hang failures."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # MIG NVML
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # so a stalled HF download raises (and self-heals) instead of hanging forever
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


def _call_with_timeout(fn, timeout, *args, **kwargs):
    """Run fn(*args) but abort with TimeoutError after `timeout` seconds so a
    single hung call (e.g. a stuck TTS synthesis) can't freeze the whole run.
    Uses SIGALRM (Unix, main thread only)."""
    import signal
    if timeout <= 0 or not hasattr(signal, "SIGALRM"):
        return fn(*args, **kwargs)

    def _handler(signum, frame):
        raise TimeoutError(f"call exceeded {timeout}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _synth(tts, text, out_path=None):
    """TTS synth with a hard per-utterance timeout so one hung clip can't freeze
    the whole data build (tune with TMV_SYNTH_TIMEOUT seconds)."""
    to = int(os.environ.get("TMV_SYNTH_TIMEOUT", "90"))
    return _call_with_timeout(tts.synth, to, text, out_path=out_path)


# =========================================================================== #
#  Turkish text normalization  (decisive for fair WER — the i/İ/ı trap)        #
# =========================================================================== #

_TR_NUM = {
    "0": "sıfır", "1": "bir", "2": "iki", "3": "üç", "4": "dört", "5": "beş",
    "6": "altı", "7": "yedi", "8": "sekiz", "9": "dokuz",
}


def tr_lower(text):
    """Turkish-aware lowercasing: İ->i, I->ı (NOT the Python default)."""
    return (text.replace("İ", "i").replace("I", "ı")
                .replace("Ç", "ç").replace("Ğ", "ğ").replace("Ö", "ö")
                .replace("Ş", "ş").replace("Ü", "ü").lower())


def normalize_tr(text, expand_digits=True):
    """Best-effort Turkish normalizer for WER. Prefers `trnorm` if installed."""
    try:
        from trnorm import normalize as _trn      # optional, best-in-class
        return _trn(text)
    except Exception:
        pass
    import re
    t = tr_lower(str(text))
    if expand_digits:
        t = re.sub(r"\d", lambda m: " " + _TR_NUM[m.group()] + " ", t)
    t = re.sub(r"[^\wçğıöşü\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# =========================================================================== #
#  HF token handling — interactive prompt on first run (server-side)           #
# =========================================================================== #

def hf_login_if_needed(interactive=True, require=False):
    """Return the HF username if logged in. Prompt once (getpass) if a TTY and a
    token is wanted. Never writes a token into the repo. Reads/writes only
    ~/.cache/huggingface/token (the standard hub cache)."""
    try:
        from huggingface_hub import whoami, login
    except Exception:
        log("huggingface_hub not installed yet — run `setup` first.")
        return None

    # 1) already logged in (cached token or env)?
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        who = whoami(token=token)
        log(f"HF: authenticated as '{who.get('name', who)}'.")
        return who.get("name")
    except Exception:
        pass

    if token:
        try:
            login(token=token, add_to_git_credential=False)
            who = whoami()
            log(f"HF: logged in via HF_TOKEN as '{who.get('name')}'.")
            return who.get("name")
        except Exception as e:
            log(f"HF: HF_TOKEN present but invalid ({e}).", err=True)

    log("HF: not logged in. All REQUIRED models are ungated, so this is only "
        "needed to push private checkpoints.")
    if interactive and IS_TTY:
        import getpass
        try:
            tok = getpass.getpass("Paste an HF token to log in (or press Enter to skip): ").strip()
        except Exception:
            tok = ""
        if tok:
            try:
                login(token=tok, add_to_git_credential=False)
                who = whoami()
                log(f"HF: logged in as '{who.get('name')}' (cached to ~/.cache/huggingface/token).")
                return who.get("name")
            except Exception as e:
                log(f"HF: login failed: {e}", err=True)
    if require:
        die("An HF token is required for this operation but none was provided.")
    return None


# =========================================================================== #
#  SETUP  — venv + deps + smoke test                                           #
# =========================================================================== #

PIP_PACKAGES = [
    # core (pure-python / prebuilt wheels only — nothing here needs a C compiler)
    "transformers>=4.57.0", "accelerate>=0.34", "peft>=0.14", "datasets>=2.18",
    "bitsandbytes>=0.45", "huggingface_hub>=0.25", "safetensors",
    "librosa==0.10.2", "soundfile", "sentencepiece", "numpy<2.3", "pyyaml",
    # serving
    "fastapi", "uvicorn[standard]", "python-multipart", "requests",
    # eval
    "jiwer>=3.0.4", "torchmetrics[audio]", "speechbrain",
    # tts (your OmniVoice)
    "omnivoice",
]
PIP_OPTIONAL = [
    # resemblyzer pulls in webrtcvad, which COMPILES a C extension (needs
    # python3-dev). It is only a speaker-sim SANITY check in eval, so it is
    # optional and its failure is non-fatal.
    "resemblyzer",
    "trnorm", "whisper-normalizer",                 # Turkish normalization
    "vllm>=0.17", "qwen-omni-utils[decord]",        # teacher / judge inference
    "git+https://github.com/sarulab-speech/UTMOSv2.git",   # UTMOS MOS predictor
    "phonemizer",                                   # medical G2P (needs espeak-ng)
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
# system packages needed so compiled wheels (webrtcvad, etc.) can build, and for
# audio I/O + Turkish/medical G2P. Installed best-effort via apt.
APT_BUILD_DEPS = ["python3-dev", "build-essential", "ffmpeg", "libsndfile1", "espeak-ng"]


def cmd_setup(args):
    CFG.ensure_dirs()
    log("=== SETUP ===")
    log(f"python: {sys.version.split()[0]}  executable: {sys.executable}")

    # 1) create a dedicated venv with `uv` if available (fast), else stdlib venv
    venv_py = Path(CFG.venv) / "bin" / "python"
    if not venv_py.exists() and not args.no_venv:
        log(f"Creating venv at {CFG.venv} ...")
        if shutil.which("uv"):
            _run(["uv", "venv", "--python", "3.11", CFG.venv])
        else:
            _run([sys.executable, "-m", "venv", CFG.venv])
    if venv_py.exists():
        log(f"venv ready. ACTIVATE IT, then re-run subcommands:\n"
            f"    source {CFG.venv}/bin/activate")

    # 2) install deps into the venv (or current interpreter if --no-venv)
    py = str(venv_py) if venv_py.exists() and not args.no_venv else sys.executable
    if not args.skip_install:
        # 2a) system build deps first, so any C-extension wheel (e.g. webrtcvad)
        #     can compile instead of failing on a missing Python.h.
        _apt_build_deps()
        # 2b) modern build frontend (turns legacy setup.py installs into wheels)
        try:
            _pip(py, ["-U", "pip", "setuptools", "wheel"])
        except Exception as e:
            log(f"  pip/setuptools/wheel upgrade failed ({e}); continuing.", err=True)
        # 2c) PyTorch pinned to the driver-matched build (idempotent no-op if right)
        _ensure_correct_torch(py)
        # 2d) core + eval packages — FAULT-TOLERANT: if the batch fails, retry each
        #     package individually so one bad package can't abort the whole install.
        #     A pip CONSTRAINTS file keeps torch pinned so no dep can upgrade it to
        #     a cu13x build that won't run on a CUDA-12.x driver.
        cons = _write_torch_constraints()
        log("Installing core + eval packages ...")
        try:
            _pip(py, list(PIP_PACKAGES), constraints=cons)
        except Exception:
            log("  batch install failed -> installing package-by-package.", err=True)
            for pkg in PIP_PACKAGES:
                try:
                    _pip(py, [pkg], constraints=cons)
                except Exception as e:
                    log(f"  CORE '{pkg}' failed: {e}", err=True)
        # 2e) optional packages (all non-fatal)
        log("Installing optional packages (failures are non-fatal) ...")
        for pkg in PIP_OPTIONAL:
            try:
                _pip(py, [pkg], constraints=cons)
            except Exception as e:
                log(f"  optional '{pkg}' failed to install: {e}", err=True)
        # 2f) re-enforce torch in case a dependency still swapped it out
        _ensure_correct_torch(py)
        # flash-attn is an OPTIONAL speed-up: it needs the CUDA toolkit (nvcc) to
        # build, which containers often lack. Failure is fine — the model falls
        # back to attn_implementation='sdpa' automatically.
        try:
            _pip(py, ["flash-attn>=2.7.0", "--no-build-isolation"], constraints=cons)
        except Exception as e:
            log(f"  flash-attn not built ({str(e)[:80]}...) -> using 'sdpa' (fine).", err=True)

    # 3) HF auth check (only if requested / interactive)
    if py == sys.executable:
        hf_login_if_needed(interactive=True, require=False)
    else:
        log("Re-run `setup` INSIDE the venv to check HF auth, or just proceed to `data`.")

    # 4) smoke test (only meaningful inside the venv with torch present)
    if py == sys.executable and not args.skip_smoke:
        _smoke_test()

    log("=== SETUP DONE ===  Next:  python turkish_medvoice.py data")


def _run(cmd):
    log("$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def _pip(py, pkgs, constraints=None):
    extra = ["-c", constraints] if constraints and Path(constraints).exists() else []
    if shutil.which("uv"):
        _run(["uv", "pip", "install", "--python", py] + extra + pkgs)
    else:
        _run([py, "-m", "pip", "install"] + extra + pkgs)


# pinned so no dependency can upgrade torch to a CUDA-13 build that won't run on
# a CUDA-12.x driver. TORCH_PIN is the driver-matched version for this server.
TORCH_PIN = _env("TMV_TORCH", "2.8.0")
TORCH_CU = _env("TMV_TORCH_CU", "12")     # required torch.version.cuda major ("12")


def _write_torch_constraints():
    try:
        CFG.ensure_dirs()
        p = Path(CFG.work) / "pip-constraints.txt"
        p.write_text(f"torch=={TORCH_PIN}\ntorchaudio=={TORCH_PIN}\n")
        return str(p)
    except Exception:
        return None


def _torch_info(py):
    """Return (version, cuda_major) of the torch installed in `py`, or ('','')."""
    try:
        out = subprocess.check_output(
            [py, "-c", "import torch,json;print(json.dumps([torch.__version__, "
                       "(torch.version.cuda or '')]))"],
            stderr=subprocess.DEVNULL).decode()
        import json as _j
        ver, cu = _j.loads(out)
        return ver, (cu.split(".")[0] if cu else "")
    except Exception:
        return "", ""


def _ensure_correct_torch(py):
    """Install / force the driver-matched torch. Idempotent: a no-op if the right
    version+CUDA is already present, else (re)installs from the cu128 index. This
    is what prevents the '2.11.0+cu130 on a CUDA-12.8 driver' breakage."""
    ver, cu_major = _torch_info(py)
    if ver.startswith(TORCH_PIN) and cu_major == TORCH_CU:
        log(f"torch OK: {ver} (cuda {cu_major}.x) — matches the driver.")
        return
    if ver:
        log(f"[heal] torch is {ver} (cuda {cu_major}.x) — reinstalling the "
            f"driver-matched {TORCH_PIN}+cu128 build.", err=True)
    else:
        log(f"Installing PyTorch {TORCH_PIN} (cu128) ...")
    try:
        _pip(py, ["--force-reinstall", f"torch=={TORCH_PIN}", f"torchaudio=={TORCH_PIN}",
                  "--index-url", TORCH_INDEX])
    except Exception as e:
        log(f"[heal] torch (re)install failed: {e}", err=True)


def _has_module(py, mod):
    """True if `mod` imports in the target interpreter (avoids re-installing)."""
    try:
        subprocess.check_call([py, "-c", f"import {mod}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _apt_build_deps():
    """Best-effort install of the OS build/audio deps so C-extension wheels can
    compile (webrtcvad -> Python.h) and audio/G2P work. Skipped silently if apt
    is absent (non-Debian) or we lack privileges."""
    if not shutil.which("apt-get"):
        log("apt-get not found — skipping system build deps (install python3-dev "
            "+ build-essential manually if a wheel fails to compile).", err=True)
        return
    sudo = [] if os.geteuid() == 0 else (["sudo"] if shutil.which("sudo") else [])
    try:
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        log("Installing system build deps (python3-dev, build-essential, ffmpeg, ...) ...")
        subprocess.run(sudo + ["apt-get", "update", "-qq"], env=env, check=False)
        subprocess.run(sudo + ["apt-get", "install", "-y", "-qq"] + APT_BUILD_DEPS,
                       env=env, check=False)
    except Exception as e:
        log(f"  system dep install failed ({e}); continuing (a compiled wheel may "
            "fail — install python3-dev manually).", err=True)


def _smoke_test():
    log("--- smoke test ---")
    try:
        import torch
        used, total = gpu_mem_gb()
        log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
            f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}  "
            f"vram_used/total={used}/{total} GB")
    except Exception as e:
        log(f"torch import failed: {e}", err=True)
        return
    # whisper-ft2 encoder loads?
    try:
        from transformers import WhisperModel
        m = WhisperModel.from_pretrained(CFG.whisper_ckpt, torch_dtype="auto")
        log(f"whisper-ft2 loaded: encoder d_model={m.config.d_model}, "
            f"enc_layers={m.config.encoder_layers}. (encoder-only will be used)")
        del m
    except Exception as e:
        log(f"whisper-ft2 load failed: {e}", err=True)
    # OmniVoice reachable (python api or your running server)?
    tts = TTSBackend()
    log(f"TTS backend: {tts.describe()}")


# =========================================================================== #
#  Medical gazetteer  (ASR biasing + TTS pronunciation + MTER metric)          #
# =========================================================================== #

# INN -> Turkish orthographic/phonetic override so OmniVoice says Latin drug
# names the way a Turkish clinician does. Extend freely.
DRUG_PRONUNCIATION = {
    "ceftriaxone": "seftriakson", "ciprofloxacin": "siprofloksasin",
    "amoxicillin": "amoksisilin", "paracetamol": "parasetamol",
    "ibuprofen": "ibuprofen", "metformin": "metformin",
    "acetaminophen": "asetaminofen", "azithromycin": "azitromisin",
    "warfarin": "varfarin", "clopidogrel": "klopidogrel",
    "omeprazole": "omeprazol", "atorvastatin": "atorvastatin",
    "levothyroxine": "levotiroksin", "prednisolone": "prednizolon",
    "furosemide": "furosemid", "insulin": "insülin",
    "diazepam": "diazepam", "amlodipine": "amlodipin",
}

SEED_ICD10_TR = [
    ("E11", "Tip 2 diabetes mellitus"), ("I10", "Esansiyel hipertansiyon"),
    ("J45", "Astım"), ("J18", "Pnömoni"), ("K29", "Gastrit ve duodenit"),
    ("N39", "İdrar yolu enfeksiyonu"), ("M54", "Sırt ağrısı / bel ağrısı"),
    ("R51", "Baş ağrısı"), ("I21", "Akut miyokart enfarktüsü"),
    ("C34", "Bronş ve akciğer malign neoplazmı"), ("E78", "Lipoprotein metabolizması bozukluğu"),
    ("F41", "Anksiyete bozukluğu"), ("G43", "Migren"), ("K21", "Gastroözofageal reflü"),
]


def build_gazetteer(cfg):
    """Build ONE medical gazetteer used by ASR biasing, TTS pronunciation and MTER.
    Sources it can reach are best-effort; a solid seed is always written."""
    log("Building medical gazetteer ...")
    terms = {}  # term(lower) -> {"lang", "type", "pron"}

    def add(term, lang, typ, pron=None):
        t = term.strip()
        if not t:
            return
        terms[t.lower()] = {"term": t, "lang": lang, "type": typ,
                            "pron": pron or DRUG_PRONUNCIATION.get(t.lower())}

    # seed: ICD-10 TR diagnoses
    for code, desc in SEED_ICD10_TR:
        add(desc, "tr", "diagnosis")
        for w in desc.split():
            if len(w) > 4:
                add(w, "tr", "diagnosis_token")
    # seed: EN drug INNs (+ TR pronunciation)
    for inn, pron in DRUG_PRONUNCIATION.items():
        add(inn, "en", "drug", pron)
        add(pron, "tr", "drug")

    # optional: pull more approved drug names from ChEMBL if the MCP/HTTP is reachable
    try:
        import requests
        for q in ["metformin", "amoxicillin", "atorvastatin"]:
            r = requests.get("https://www.ebi.ac.uk/chembl/api/data/molecule/search",
                             params={"q": q, "format": "json"}, timeout=8)
            if r.ok:
                for m in r.json().get("molecules", [])[:20]:
                    nm = (m.get("pref_name") or "").strip()
                    if nm:
                        add(nm.lower(), "en", "drug")
    except Exception as e:
        log(f"  ChEMBL enrichment skipped: {e}")

    Path(cfg.gazetteer).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.gazetteer, "w", encoding="utf-8") as fh:
        for v in terms.values():
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    log(f"Gazetteer: {len(terms)} terms -> {cfg.gazetteer}")
    return terms


def load_gazetteer(cfg):
    rows = read_jsonl(cfg.gazetteer)
    if not rows:
        return build_gazetteer(cfg)
    return {r["term"].lower(): r for r in rows}


def apply_pronunciation(text, gaz):
    """Rewrite English drug names to Turkish phonetic spelling before TTS."""
    import re
    out = text
    for key, row in gaz.items():
        pron = row.get("pron")
        if pron and row.get("lang") == "en":
            out = re.sub(r"\b" + re.escape(row["term"]) + r"\b", pron, out, flags=re.IGNORECASE)
    return out


# =========================================================================== #
#  TTS backend (OmniVoice)  — python API -> HTTP server -> CLI, whichever works #
# =========================================================================== #

class TTSBackend:
    """Robust adapter around the user's Turkish OmniVoice FT. Tries, in order:
       (1) the `omnivoice` python API, (2) the OpenAI-compatible HTTP endpoint
       already running on the server, (3) None (caller must handle)."""

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.mode = None
        self._py = None
        self._init()

    def _init(self):
        # TMV_TTS_MODE = auto | http | python.
        #   http   -> use your running OmniVoice server (which loads your Turkish
        #             fine-tune omnivoice-ft1 via OMNI_MODEL) — RECOMMENDED so the
        #             fine-tune is used, not the base model the pip package pulls.
        #   python -> in-process `omnivoice` package (downloads/uses BASE OmniVoice).
        #   auto   -> python if it loads, else http.
        mode_pref = os.environ.get("TMV_TTS_MODE", "auto").lower()
        # (1) python API (base OmniVoice) — skipped when http is forced
        if mode_pref in ("auto", "python"):
            try:
                import omnivoice  # noqa
                self._py = _try_load_omnivoice_py(self.cfg)
                if self._py is not None:
                    self.mode = "python"
                    return
            except Exception:
                pass
            if mode_pref == "python":
                self.mode = None
                return
        # (2) HTTP endpoint (your omnivoice-ft1 server) — SELF-HEAL: autostart if down
        if _tts_reachable(self.cfg) or _ensure_tts_server(self.cfg):
            self.mode = "http"
            return
        self.mode = None

    def describe(self):
        if self.mode == "python":
            return "omnivoice python API (in-process)"
        if self.mode == "http":
            return f"OmniVoice HTTP server @ {self.cfg.omni_server_url}"
        return ("NONE reachable — start your TTS server (see README) or `pip install omnivoice`. "
                "Data-synth/serve TTS will be skipped until then.")

    def available(self):
        return self.mode is not None

    def synth(self, text, out_path=None, ref_wav=None):
        """Return 24 kHz float32 mono np.ndarray (and write WAV if out_path).
        SELF-HEAL: on an HTTP failure, try to bring the server back up and retry
        once before raising."""
        import numpy as np
        text = (text or "").strip()
        if not text:
            return np.zeros(1, dtype="float32"), 24000
        if self.mode == "python":
            wav, sr = self._py(text, ref_wav or self.cfg.omni_ref_wav)
        elif self.mode == "http":
            try:
                wav, sr = self._http(text)
            except Exception as e:
                log(f"[heal] TTS request failed ({e}); attempting recovery.", err=True)
                if _ensure_tts_server(self.cfg):
                    wav, sr = self._http(text)      # retry once after recovery
                else:
                    raise
        else:
            raise RuntimeError("No TTS backend available (see TTSBackend.describe()).")
        if out_path:
            import soundfile as sf
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, wav, sr)
        return wav, sr

    def _http(self, text):
        # Body matches the user's OmniVoice server contract (README):
        #   POST /v1/audio/speech  {"input": ..., "language": "tr"}  -> WAV 24kHz.
        # The clone reference voice (OMNI_REF) is configured server-side, so no
        # voice field is sent here. `voice` is added only if TMV sets it.
        import io
        import requests
        import soundfile as sf
        payload = {"input": text, "language": self.cfg.omni_lang}
        voice = os.environ.get("TMV_OMNI_VOICE")
        if voice:
            payload["voice"] = voice
        r = requests.post(self.cfg.omni_server_url, json=payload, timeout=120)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "json" in ctype:      # some servers return {"audio": <base64>} or a URL
            import base64
            j = r.json()
            b = j.get("audio") or j.get("data")
            if isinstance(b, str):
                raw = base64.b64decode(b)
                return sf.read(io.BytesIO(raw), dtype="float32")
            raise RuntimeError(f"TTS server returned JSON without audio: {list(j)[:5]}")
        wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        return wav, sr


def _try_load_omnivoice_py(cfg):
    """Best-effort in-process OmniVoice loader. Returns a callable(text, ref)->(wav,sr)
    or None. OmniVoice's exact python API varies by version; we probe common
    shapes and fall back gracefully to the HTTP path if none match."""
    try:
        import omnivoice
        # Common shape A: omnivoice.OmniVoice / .TTS with .generate/.synthesize
        for cls_name in ("OmniVoice", "TTS", "OmniVoiceTTS", "Synthesizer"):
            cls = getattr(omnivoice, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls(cfg.omni_model) if _accepts_one_arg(cls) else cls()
            except Exception:
                try:
                    model = cls.from_pretrained(cfg.omni_model)
                except Exception:
                    continue
            ref_txt = ""
            try:
                ref_txt = Path(cfg.omni_ref_txt).read_text(encoding="utf-8").strip()
            except Exception:
                pass

            def _call(text, ref_wav, _m=model, _rt=ref_txt):
                for meth in ("synthesize", "generate", "tts", "infer", "__call__"):
                    fn = getattr(_m, meth, None)
                    if fn is None:
                        continue
                    try:
                        out = fn(text=text, prompt_speech=ref_wav, prompt_text=_rt,
                                 language=cfg.omni_lang)
                    except TypeError:
                        try:
                            out = fn(text, ref_wav)
                        except Exception:
                            continue
                    return _coerce_wav(out)
                raise RuntimeError("omnivoice python API present but no callable matched")
            # sanity call deferred to first use
            return _call
    except Exception:
        return None
    return None


def _accepts_one_arg(cls):
    import inspect
    try:
        sig = inspect.signature(cls.__init__)
        return len([p for p in sig.parameters.values()
                    if p.name != "self" and p.default is inspect._empty]) >= 1
    except Exception:
        return False


def _coerce_wav(out):
    import numpy as np
    sr = 24000
    if isinstance(out, tuple) and len(out) == 2:
        wav, sr = out
    else:
        wav = out
    wav = np.asarray(wav, dtype="float32").reshape(-1)
    return wav, sr


def _tts_reachable(cfg):
    import requests
    base = cfg.omni_server_url.split("/v1/")[0]
    for url in (base + "/", cfg.omni_server_url):
        try:
            requests.get(url, timeout=4)
            return True
        except Exception:
            try:
                requests.options(cfg.omni_server_url, timeout=4)
                return True
            except Exception:
                continue
    return False


def _ensure_tts_server(cfg, timeout=90):
    """SELF-HEAL: make the OmniVoice TTS HTTP server reachable. If it is down and
    autostart is enabled (TMV_AUTOSTART_TTS=1, default), launch the documented
    uvicorn server in the OmniVoice venv, then poll for reachability. Returns True
    if reachable."""
    if _tts_reachable(cfg):
        return True
    if os.environ.get("TMV_AUTOSTART_TTS", "1") != "1":
        return False
    omni_py = os.environ.get("TMV_OMNI_VENV_PY", "/root/venv-omni/bin/python")
    app_dir = os.environ.get("TMV_OMNI_APP_DIR", str(Path(cfg.root) / "app"))
    if not (Path(omni_py).exists() and Path(app_dir).exists()):
        log(f"[heal] cannot autostart TTS (missing {omni_py} or {app_dir}); "
            "start it manually per README.", err=True)
        return False
    # Guard against spawning DUPLICATE GPU servers across re-execs / repeated
    # heals: if we already started one and it's still alive, just wait for it.
    pidfile = Path(cfg.log_dir) / ".tts_autostart.pid"
    try:
        if pidfile.exists():
            old = int(pidfile.read_text().strip() or "0")
            if old > 0:
                os.kill(old, 0)                    # raises if the process is gone
                log(f"[heal] a TTS autostart is already running (pid {old}); waiting.", err=True)
                t0 = time.time()
                while time.time() - t0 < timeout:
                    time.sleep(5)
                    if _tts_reachable(cfg):
                        return True
                return _tts_reachable(cfg)
    except Exception:
        pass   # stale/dead pid -> fall through and (re)spawn exactly one
    import re
    m = re.search(r":(\d+)", cfg.omni_server_url)
    port = m.group(1) if m else "8133"
    env = dict(os.environ)
    env.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "OMNI_MODEL": cfg.omni_model, "OMNI_REF": cfg.omni_ref_wav,
                "OMNI_REFTXT_FILE": cfg.omni_ref_txt, "OMNI_LANG": cfg.omni_lang})
    cmd = [omni_py, "-m", "uvicorn", "omnivoice_server:app",
           "--host", "127.0.0.1", "--port", port, "--app-dir", app_dir]
    log(f"[heal] autostarting OmniVoice TTS server: {' '.join(cmd)}", err=True)
    try:
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.log_dir) / "tts_server.log", "a") as logf:
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        pidfile.write_text(str(proc.pid))          # record so we don't double-spawn
    except Exception as e:
        log(f"[heal] TTS autostart failed: {e}", err=True)
        return False
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(5)
        if _tts_reachable(cfg):
            log("[heal] TTS server is up.")
            return True
    log("[heal] TTS server did not become reachable in time.", err=True)
    return False


# =========================================================================== #
#  STT helper (for cascade serve + round-trip eval)                            #
# =========================================================================== #

class ASRBackend:
    """Independent ASR for eval round-trip + cascade. Loads a local Whisper via
    transformers; `model_id` defaults to neutral whisper-large-v3."""

    def __init__(self, model_id=None, cfg=CFG):
        self.cfg = cfg
        self.model_id = model_id or cfg.eval_asr
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._pipe = pipeline("automatic-speech-recognition", model=self.model_id,
                              torch_dtype=dtype,
                              device=0 if torch.cuda.is_available() else -1,
                              chunk_length_s=30)

    def transcribe(self, wav, sr=16000, language="tr"):
        self._ensure()
        import numpy as np
        wav = np.asarray(wav, dtype="float32").reshape(-1)
        out = self._pipe({"array": wav, "sampling_rate": sr},
                         generate_kwargs={"language": language, "task": "transcribe"})
        return out["text"].strip()


# =========================================================================== #
#  DATA  — download, synthesize (offline, cached, resumable)                    #
# =========================================================================== #

def cmd_data(args):
    CFG.ensure_dirs()
    set_seed(CFG.seed)
    log("=== DATA ===")
    gaz = build_gazetteer(CFG)

    if args.only in (None, "align"):
        _build_align_manifest(CFG)
    if args.only in (None, "s2s"):
        _build_s2s_manifest(CFG)
    if args.only in (None, "medical"):
        _build_medical_triples(CFG, gaz, n=args.n_medical or CFG.n_synth_medical,
                               use_teacher=args.use_teacher)
    log("=== DATA DONE ===")


def _hf_load(repo, config=None, split=None, streaming=True, audio_col=None):
    def _do():
        from datasets import load_dataset
        # NOTE: no trust_remote_code — datasets>=4 removed it (all datasets we use
        # are standard Parquet, no loading script), and passing it now just warns.
        ds = load_dataset(repo, config, split=split, streaming=streaming,
                          cache_dir=CFG.hf_cache)
        if audio_col:
            # Decode audio OURSELVES (soundfile/librosa) rather than via datasets'
            # torchcodec backend: datasets>=4 dropped the soundfile decoder and
            # raises "please install 'torchcodec'". decode=False hands us raw bytes.
            try:
                from datasets import Audio
                ds = ds.cast_column(audio_col, Audio(decode=False))
            except Exception as e:
                log(f"  cast_column({audio_col}, decode=False) failed ({e}); "
                    "will try native decoding.", err=True)
        return ds
    # network transience (HF hub timeouts / 5xx / rate limits) auto-retries with backoff
    return resilient(_do, retries=5, base_delay=4.0, context=f"hf_load:{repo}")


def _decode_audio_field(field):
    """Decode a HF audio field to (float32 mono ndarray, sr) WITHOUT torchcodec.
    Handles decoded dicts ({'array','sampling_rate'}), decode=False dicts
    ({'bytes','path'}) and plain paths; mp3/opus fall back to librosa+ffmpeg."""
    import io
    import os as _os
    import tempfile
    import numpy as np
    import soundfile as sf
    import librosa

    def _mono(a):
        a = np.asarray(a, dtype="float32")
        return a.mean(axis=1) if a.ndim > 1 else a

    if isinstance(field, dict):
        if field.get("array") is not None:
            return _mono(field["array"]), int(field.get("sampling_rate", 16000))
        raw, path = field.get("bytes"), field.get("path")
        if raw:
            try:
                a, sr = sf.read(io.BytesIO(raw), dtype="float32")
                return _mono(a), sr
            except Exception:
                suf = _os.path.splitext(path or "")[1] or ".mp3"
                with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tf:
                    tf.write(raw); tmp = tf.name
                try:
                    a, sr = librosa.load(tmp, sr=None, mono=True)   # uses ffmpeg
                    return a.astype("float32"), sr
                finally:
                    try:
                        _os.remove(tmp)
                    except Exception:
                        pass
        if path:
            a, sr = librosa.load(path, sr=None, mono=True)
            return a.astype("float32"), sr
    elif isinstance(field, str):
        a, sr = librosa.load(field, sr=None, mono=True)
        return a.astype("float32"), sr
    raise ValueError("unrecognized audio field")


def _build_align_manifest(cfg):
    """Stage-A alignment data: (audio, transcript) -> the projector learns to map
    Whisper features into the LLM embedding space (ASR objective)."""
    man = Path(cfg.data_dir) / "align.jsonl"
    target = cfg.n_align
    if man.exists() and _manifest_count(man) >= target:
        log(f"align manifest already has {_manifest_count(man)} rows — done.")
        return
    log(f"Building alignment manifest -> {man}")
    audio_root = Path(cfg.data_dir) / "align_audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    already = _manifest_count(man)          # resume cursor: skip candidates already written
    written = already
    seen = 0
    for repo, conf, split, acol, tcol, _ in cfg.asr_datasets:
        if written >= target:
            break
        try:
            ds = _hf_load(repo, conf, split, streaming=True, audio_col=acol)
        except Exception as e:
            log(f"  {repo} load failed: {e}", err=True)
            continue
        for ex in ds:
            if written >= target:
                break
            txt = str(ex.get(tcol) or "").strip()
            if not txt:
                for _alt in ("transcription", "sentence", "text"):   # defensive
                    if ex.get(_alt):
                        txt = str(ex[_alt]).strip(); break
            if not txt:
                continue
            # Dump BEFORE the resume gate so `seen` counts only successfully-dumped
            # candidates, staying in lockstep with `written` (= manifest rows).
            # A dump that fails must NOT advance `seen`, else resume duplicates rows.
            wav_path = _dump_audio(ex.get(acol), audio_root, written)
            if wav_path is None:
                continue
            seen += 1
            if seen <= already:             # already written on a previous run
                continue
            append_jsonl(man, {"audio": wav_path, "prompt":
                               "Duyduğun Türkçe konuşmayı aynen yaz.", "target": txt,
                               "kind": "align"})
            written += 1
            if written % 2000 == 0:
                log(f"  align: {written} rows")
    log(f"align manifest: {written} rows")


def _build_s2s_manifest(cfg):
    """Stage-B general S2S: (instruction_speech, response_text). We synthesize the
    user's spoken instruction with OmniVoice from Turkish instruction text."""
    man = Path(cfg.data_dir) / "s2s.jsonl"
    target = cfg.n_general_s2s
    if man.exists() and _manifest_count(man) >= target:
        log(f"s2s manifest already has {_manifest_count(man)} rows — done.")
        return
    tts = TTSBackend()
    if not tts.available():
        log("  TTS unavailable -> writing TEXT-only s2s pairs (speech synth deferred). "
            "Start your OmniVoice server and re-run `data --only s2s` before training.", err=True)
    log(f"Building general S2S manifest -> {man}")
    audio_root = Path(cfg.synth_dir) / "s2s_audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    already = _manifest_count(man)
    written = already
    seen = 0
    try:
        ds = _hf_load(cfg.instruct_dataset, split="train", streaming=True)
    except Exception as e:
        log(f"  {cfg.instruct_dataset} load failed: {e}", err=True)
        return
    for ex in ds:
        if written >= target:
            break
        instr, resp = _extract_instruct(ex)
        if not instr or not resp:
            continue
        seen += 1
        if seen <= already:
            continue
        row = {"prompt": "", "target": resp, "kind": "s2s", "instruction_text": instr}
        if tts.available():
            try:
                wav_path = str(audio_root / f"s2s_{written:07d}.wav")
                _synth(tts, instr[:600], wav_path)
                row["audio"] = wav_path
            except Exception as e:
                log(f"  synth failed at {written}: {e}", err=True)
        append_jsonl(man, row)
        written += 1
        if written % 1000 == 0:
            log(f"  s2s: {written} rows")
    log(f"s2s manifest: {written} rows")


def _build_medical_triples(cfg, gaz, n, use_teacher=False):
    """Stage-C medical: real Turkish patient/doctor Q&A (gold answers) rendered to
    speech, plus gazetteer-seeded code-switched examples. Optionally augment with
    a teacher LLM (sequence-level KD)."""
    man = Path(cfg.data_dir) / "medical.jsonl"
    if man.exists() and _manifest_count(man) >= n:
        log(f"medical manifest already has {_manifest_count(man)} rows — done.")
        return
    tts = TTSBackend()
    if not tts.available():
        log("  TTS unavailable -> writing TEXT-only medical pairs (speech synth deferred). "
            "Start your OmniVoice server and re-run `data --only medical` before training.", err=True)
    log(f"Building medical triples -> {man} (target {n}, teacher={use_teacher})")
    audio_root = Path(cfg.synth_dir) / "medical_audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    already = _manifest_count(man)
    written = already
    seen = 0

    # (a) real gold Q&A -> best targets, avoids re-transcription error loops
    try:
        ds = _hf_load(cfg.medqa_dataset, split="train", streaming=True)
    except Exception as e:
        log(f"  {cfg.medqa_dataset} load failed: {e}", err=True)
        ds = []
    for ex in ds:
        if written >= n:
            break
        q, a = _extract_medqa(ex)
        if not q or not a:
            continue
        seen += 1
        if seen <= already:
            continue
        row = {"prompt": "", "target": a, "kind": "medical", "instruction_text": q}
        if tts.available():
            try:
                spoken = apply_pronunciation(q[:600], gaz)
                wav_path = str(audio_root / f"med_{written:07d}.wav")
                _synth(tts, spoken, wav_path)
                row["audio"] = wav_path
            except Exception as e:
                log(f"  synth failed at {written}: {e}", err=True)
        append_jsonl(man, row)
        written += 1
        if written % 500 == 0:
            log(f"  medical: {written} rows")

    # (b) optional teacher augmentation (code-switched, gazetteer-seeded)
    if use_teacher and written < n:
        _teacher_augment_medical(cfg, gaz, man, tts, audio_root, start=written, target=n)

    log(f"medical manifest: {_manifest_count(man)} rows")


def _teacher_augment_medical(cfg, gaz, man, tts, audio_root, start, target):
    """Generate extra Turkish medical dialogues with a teacher LLM, seeded with
    real terms from the gazetteer and code-switched TR/EN. Sequence-level KD."""
    gen = _load_text_generator(cfg)
    if gen is None:
        log("  teacher generator unavailable — skipping augmentation.", err=True)
        return
    import random
    terms = [r["term"] for r in gaz.values()]
    written = start
    while written < target:
        seed_terms = random.sample(terms, k=min(3, len(terms))) if terms else []
        prompt = (
            "Bir hasta ile doktor arasında kısa, gerçekçi bir Türkçe tıbbi diyalog yaz. "
            f"Şu terimleri doğal biçimde kullan (İngilizce ilaç/anatomi adlarını Türkçe cümle "
            f"içinde bırakabilirsin): {', '.join(seed_terms)}. "
            "Format:\nHASTA: <soru>\nDOKTOR: <bilgilendirici, güvenli yanıt>")
        try:
            text = gen(prompt)
            q, a = _split_dialogue(text)
            if not q or not a:
                continue
            row = {"prompt": "", "target": a, "kind": "medical",
                   "instruction_text": q, "source": "teacher"}
            if tts.available():
                spoken = apply_pronunciation(q[:600], gaz)
                wav_path = str(audio_root / f"medT_{written:07d}.wav")
                _synth(tts, spoken, wav_path)
                row["audio"] = wav_path
            append_jsonl(man, row)
            written += 1
            if written % 200 == 0:
                log(f"  medical(teacher): {written} rows")
        except Exception as e:
            log(f"  teacher gen failed: {e}", err=True)
            break


def _load_text_generator(cfg):
    """Return callable(prompt)->text. Tries the CONFIGURED teacher first
    (cfg.teacher_llm, e.g. Qwen3-Omni-30B AWQ — text stream only), via vLLM, then
    the lighter text fallback, then transformers; None if nothing loads. Whichever
    is used is logged so the run is honest about the actual teacher."""
    for tag, model_id in (("teacher", cfg.teacher_llm),
                          ("text-fallback", cfg.text_teacher_fallback)):
        try:
            from vllm import LLM, SamplingParams
            log(f"  loading {tag} teacher via vLLM: {model_id}")
            llm = LLM(model=model_id, gpu_memory_utilization=0.85,
                      max_model_len=4096, trust_remote_code=True)
            sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

            def _gen(prompt, _llm=llm, _sp=sp):
                out = _llm.generate([_chat_wrap(prompt)], _sp)
                return out[0].outputs[0].text.strip()
            log(f"  ACTIVE teacher = {model_id} (vLLM)")
            return _gen
        except Exception as e:
            log(f"  vLLM load of {model_id} failed ({e}).", err=True)
    # transformers fallback (text model only)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = cfg.text_teacher_fallback
        log(f"  loading text-fallback teacher via transformers: {mid}")
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

        def _gen(prompt, _t=tok, _m=mdl):
            msgs = [{"role": "user", "content": prompt}]
            ids = _t.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(_m.device)
            out = _m.generate(ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
            return _t.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        log(f"  ACTIVE teacher = {mid} (transformers)")
        return _gen
    except Exception as e:
        log(f"  transformers teacher unavailable ({e}).", err=True)
        return None


def _chat_wrap(prompt):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


# ---- dataset field extractors (robust to schema differences) -------------- #

def _extract_instruct(ex):
    # case-insensitive view so schemas like InstrucTurca's 'Input'/'Output' match
    low = {str(k).lower(): v for k, v in ex.items()}
    for a, b in [("input", "output"), ("instruction", "output"),
                 ("prompt", "response"), ("question", "answer"),
                 ("text", "target")]:
        if low.get(a) and low.get(b):
            return str(low[a]).strip(), str(low[b]).strip()
    # conversation format
    conv = low.get("conversations") or low.get("messages")
    if isinstance(conv, list) and len(conv) >= 2:
        return str(conv[0].get("value") or conv[0].get("content") or "").strip(), \
               str(conv[1].get("value") or conv[1].get("content") or "").strip()
    return None, None


def _extract_medqa(ex):
    # kayrab/patient-doctor-qa-tr-167732 columns: question_content / question_answer
    low = {str(k).lower(): v for k, v in ex.items()}
    for a, b in [("question_content", "question_answer"),
                 ("question", "answer"), ("input", "output"),
                 ("soru", "cevap"), ("patient", "doctor"), ("prompt", "response")]:
        if low.get(a) and low.get(b):
            return str(low[a]).strip(), str(low[b]).strip()
    return _extract_instruct(ex)


def _split_dialogue(text):
    q, a = "", ""
    for line in text.splitlines():
        u = line.strip()
        if u.upper().startswith(("HASTA", "PATIENT", "SORU")):
            q = u.split(":", 1)[-1].strip()
        elif u.upper().startswith(("DOKTOR", "DOCTOR", "CEVAP")):
            a = u.split(":", 1)[-1].strip()
    return q, a


def _dump_audio(audio_field, root, idx):
    """Persist a HF audio field to a 16 kHz mono wav; return path or None."""
    import soundfile as sf
    import librosa
    try:
        arr, sr = _decode_audio_field(audio_field)      # torchcodec-free decoder
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            sr = 16000
        if arr.size < 400:      # <25 ms -> junk
            return None
        p = Path(root) / f"a_{idx:07d}.wav"
        sf.write(p, arr, sr)
        return str(p)
    except Exception:
        return None


def _manifest_count(path):
    if not Path(path).exists():
        return 0
    return sum(1 for _ in open(path, "r", encoding="utf-8"))


# =========================================================================== #
#  MODEL  — native speech-LLM: frozen Whisper enc + projector + Qwen QLoRA     #
# =========================================================================== #

SPEECH_TOKEN = "<|speech_pad|>"   # placeholder we splice encoder features into


def build_model(cfg, for_training=True, adapter_dir=None):
    """Assemble the native model. Returns (model_wrapper, tokenizer, feat_extractor)."""
    import torch
    import torch.nn as nn
    from transformers import (AutoTokenizer, AutoModelForCausalLM, WhisperModel,
                              WhisperFeatureExtractor, BitsAndBytesConfig)

    # SELF-HEAL: if a prior load failed on a corrupt/mismatched checkpoint, the
    # supervisor re-execs with this flag so we start from base instead of looping.
    drop_adapters = os.environ.get("TMV_HEAL_DROP_ADAPTERS", "0") == "1"
    if drop_adapters and adapter_dir:
        log("[heal] TMV_HEAL_DROP_ADAPTERS=1 -> ignoring saved adapters, starting from base.",
            err=True)
        adapter_dir = None

    log("Loading tokenizer + Whisper encoder + Qwen student ...")
    # model/tokenizer downloads auto-retry on hub transience
    tok = resilient(lambda: AutoTokenizer.from_pretrained(cfg.student_llm),
                    context="load:tokenizer")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    feat = resilient(lambda: WhisperFeatureExtractor.from_pretrained(cfg.whisper_processor),
                     context="load:feat")

    # frozen Whisper encoder (encoder-only)
    wm = resilient(lambda: WhisperModel.from_pretrained(cfg.whisper_ckpt,
                                                        torch_dtype=torch.bfloat16),
                   context="load:whisper")
    encoder = wm.get_encoder()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    # student LLM (QLoRA NF4)
    attn = cfg.attn_impl
    try:
        import flash_attn  # noqa
    except Exception:
        if attn == "flash_attention_2":
            log("flash-attn not available -> attn_implementation='sdpa'.")
            attn = "sdpa"

    qcfg = None
    if cfg.load_in_4bit and for_training:
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16)
    llm = resilient(lambda: AutoModelForCausalLM.from_pretrained(
        cfg.student_llm, quantization_config=qcfg,
        torch_dtype=torch.bfloat16, attn_implementation=attn,
        device_map={"": 0} if torch.cuda.is_available() else None),
        context="load:student")
    if for_training:
        llm.config.use_cache = False          # required with gradient checkpointing

    d_llm = llm.config.hidden_size

    # ---- LoRA (training) or adapter application (inference) ----
    if for_training:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if qcfg is not None:
            llm = prepare_model_for_kbit_training(
                llm, use_gradient_checkpointing=cfg.grad_ckpt)
        elif cfg.grad_ckpt:
            llm.gradient_checkpointing_enable()
            llm.enable_input_require_grads()
        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                          lora_dropout=cfg.lora_dropout, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        llm = get_peft_model(llm, lcfg)
        # resume / curriculum init: load saved LoRA weights into this fresh adapter
        if adapter_dir and Path(adapter_dir, "lora").exists():
            _load_lora_weights(llm, str(Path(adapter_dir) / "lora"))
        llm.print_trainable_parameters()
    else:
        if adapter_dir and Path(adapter_dir, "lora").exists():
            from peft import PeftModel
            llm = PeftModel.from_pretrained(llm, str(Path(adapter_dir) / "lora"))
        llm.eval()

    dev = next(llm.parameters()).device

    # ---- projector: 50 Hz Whisper features -> 25 Hz LLM soft-tokens ----
    class SpeechProjector(nn.Module):
        def __init__(self, in_dim, out_dim, k):
            super().__init__()
            self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=k, stride=k)
            self.ln = nn.LayerNorm(out_dim)

        def forward(self, x):                 # x: [B, T, in_dim]
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            return self.ln(x)

    projector = SpeechProjector(cfg.whisper_dim, d_llm, cfg.down_factor).to(
        dtype=torch.bfloat16, device=dev)
    if adapter_dir and Path(adapter_dir, "projector.pt").exists():
        try:
            projector.load_state_dict(torch.load(
                str(Path(adapter_dir) / "projector.pt"), map_location=dev))
            log(f"Loaded projector from {adapter_dir}")
        except Exception as e:
            # e.g. dim mismatch after an emergency student-size fallback -> start fresh
            log(f"[heal] projector load skipped ({e}); using a fresh projector.", err=True)

    encoder = encoder.to(dev)
    model = NativeSpeechLLM(cfg, encoder, projector, llm, tok, feat)
    return model, tok, feat


def _load_lora_weights(peft_llm, lora_dir):
    """Load saved LoRA adapter weights into an EXISTING PeftModel (resume /
    curriculum init across stages)."""
    import torch
    try:
        from peft import set_peft_model_state_dict
    except Exception:
        from peft.utils import set_peft_model_state_dict
    sd = None
    st = Path(lora_dir) / "adapter_model.safetensors"
    bn = Path(lora_dir) / "adapter_model.bin"
    try:
        if st.exists():
            from safetensors.torch import load_file
            sd = load_file(str(st))
        elif bn.exists():
            sd = torch.load(str(bn), map_location="cpu")
    except Exception as e:
        log(f"  LoRA weight read failed ({e}).", err=True)
        return
    if sd is not None:
        try:
            res = set_peft_model_state_dict(peft_llm, sd)
            # verify weights actually landed (guards against silent adapter-name drift)
            unexpected = list(getattr(res, "unexpected_keys", []) or [])
            missing = list(getattr(res, "missing_keys", []) or [])
            if unexpected:
                log(f"  WARNING: {len(unexpected)} LoRA keys did not match the model "
                    f"(adapter naming drift?) — resume may be partial.", err=True)
            log(f"  Loaded {len(sd)} LoRA tensors from {lora_dir} "
                f"(missing={len(missing)}, unexpected={len(unexpected)}).")
        except Exception as e:
            log(f"  set_peft_model_state_dict failed ({e}).", err=True)


class NativeSpeechLLM:
    """Thin wrapper. Owns the frozen encoder, the trained projector, and the
    LoRA'd LLM; assembles speech+text embeddings and computes the LM loss."""

    def __init__(self, cfg, encoder, projector, llm, tok, feat):
        self.cfg, self.encoder, self.projector = cfg, encoder, projector
        self.llm, self.tok, self.feat = llm, tok, feat
        self.device = next(llm.parameters()).device

    # --- speech -> soft tokens ---
    def encode_speech(self, wav_16k):
        import torch
        import numpy as np
        arr = np.asarray(wav_16k, dtype="float32").reshape(-1)
        dur = len(arr) / 16000.0
        feats = self.feat(arr, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            enc = self.encoder(feats).last_hidden_state          # [1,1500,1280]
        valid = max(1, min(enc.shape[1], int(round(dur * 50))))  # trim padding
        enc = enc[:, :valid]
        soft = self.projector(enc)                               # [1, valid/2, d_llm]
        return soft.squeeze(0)                                   # [T, d_llm]

    # --- build one training/inference sequence ---
    def build_example_embeds(self, wav, prompt_text, target_text=None, system=None):
        import torch
        emb = self.llm.get_input_embeddings()
        system = system or ("Sen Türkçe konuşan, tıbbi terminolojiye hakim bir sesli "
                            "asistansın. Kısa, doğru ve güvenli yanıt ver.")
        pre = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt_text}"
        mid = "<|im_end|>\n<|im_start|>assistant\n"
        pre_ids = self.tok(pre, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        mid_ids = self.tok(mid, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        parts, labels = [], []
        pre_e = emb(pre_ids).squeeze(0)
        parts.append(pre_e);  labels += [-100] * pre_e.shape[0]
        if wav is not None:
            soft = self.encode_speech(wav).to(pre_e.dtype)
            parts.append(soft); labels += [-100] * soft.shape[0]
        mid_e = emb(mid_ids).squeeze(0)
        parts.append(mid_e);  labels += [-100] * mid_e.shape[0]
        # Everything so far (system+user prompt + SPEECH soft-tokens + assistant
        # header) is fixed and MUST be preserved. If we overflow, we trim the
        # TARGET, never the speech (dropping speech would train the model to
        # answer without listening).
        fixed_len = sum(p.shape[0] for p in parts)
        if target_text is not None:
            tgt_ids = self.tok(target_text + self.tok.eos_token, add_special_tokens=False,
                               return_tensors="pt").input_ids.to(self.device)
            budget = self.cfg.max_seq_len - fixed_len
            if tgt_ids.shape[1] > max(1, budget):
                tgt_ids = tgt_ids[:, :max(1, budget)].clone()
                tgt_ids[0, -1] = self.tok.eos_token_id          # keep an EOS to learn stopping
            tgt_e = emb(tgt_ids).squeeze(0)
            parts.append(tgt_e); labels += tgt_ids.squeeze(0).tolist()
        seq = torch.cat(parts, 0)                                # [L, d_llm]
        lab = torch.tensor(labels, device=self.device)
        if seq.shape[0] > self.cfg.max_seq_len:                  # only if audio alone is huge
            over = seq.shape[0] - self.cfg.max_seq_len           # trim front of prompt text
            seq, lab = seq[over:], lab[over:]
        return seq, lab

    def forward_batch(self, batch):
        """batch: list of (wav, prompt, target). Returns loss."""
        import torch
        seqs, labs = [], []
        for wav, prompt, target in batch:
            s, l = self.build_example_embeds(wav, prompt, target)
            seqs.append(s); labs.append(l)
        L = max(s.shape[0] for s in seqs)
        H = seqs[0].shape[1]
        B = len(seqs)
        inp = torch.zeros(B, L, H, device=self.device, dtype=seqs[0].dtype)
        att = torch.zeros(B, L, device=self.device, dtype=torch.long)
        lab = torch.full((B, L), -100, device=self.device, dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            n = s.shape[0]
            inp[i, :n] = s; att[i, :n] = 1; lab[i, :n] = l
        # With gradient checkpointing + inputs_embeds, the checkpointed graph
        # needs an input that requires grad. For text-only rows (no projector
        # output in the graph) force it on the leaf.
        if self.llm.training and not inp.requires_grad:
            inp.requires_grad_(True)
        out = self.llm(inputs_embeds=inp, attention_mask=att, labels=lab)
        return out.loss

    # --- generation (native): speech-in -> text tokens (streamable) ---
    def generate_ids(self, wav, prompt_text="", max_new_tokens=256, streamer=None):
        import torch
        seq, _ = self.build_example_embeds(wav, prompt_text, target_text=None)
        inp = seq.unsqueeze(0)
        att = torch.ones(1, seq.shape[0], device=self.device, dtype=torch.long)
        gen = self.llm.generate(inputs_embeds=inp, attention_mask=att,
                                max_new_tokens=max_new_tokens, do_sample=False,
                                streamer=streamer, pad_token_id=self.tok.pad_token_id)
        return gen

    # --- save/load (only projector + LoRA adapters — small & resumable) ---
    def save_adapters(self, out_dir):
        import torch
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(str(Path(out_dir) / "lora"))
        torch.save(self.projector.state_dict(), str(Path(out_dir) / "projector.pt"))

    def trainable_parameters(self):
        ps = [p for p in self.projector.parameters() if p.requires_grad]
        ps += [p for p in self.llm.parameters() if p.requires_grad]
        return ps


# =========================================================================== #
#  TRAIN  — QLoRA, gradient accumulation, checkpoint/resume, VRAM fallback      #
# =========================================================================== #

STAGE_MANIFEST = {"align": "align.jsonl", "s2s": "s2s.jsonl",
                  "medical": "medical.jsonl", "distill": "medical.jsonl"}
STAGE_ORDER = ["align", "s2s", "medical"]


def cmd_train(args):
    CFG.ensure_dirs()
    set_seed(CFG.seed)
    stage = args.stage
    log(f"=== TRAIN stage={stage} ===")

    man = Path(CFG.data_dir) / STAGE_MANIFEST[stage]
    rows = read_jsonl(man)
    rows = [r for r in rows if r.get("target")]
    if stage == "distill":
        # distill == SEQUENCE-LEVEL knowledge distillation: train only on rows whose
        # targets were produced by the teacher LLM (`data --use-teacher`). Without
        # teacher rows there is nothing to distill, so fail loudly instead of
        # silently re-running the medical SFT.
        rows = [r for r in rows if r.get("source") == "teacher"]
        if not rows:
            die("distill stage found no teacher-generated rows. Run "
                "`data --only medical --use-teacher` first (sequence-level KD).")
    if not rows:
        die(f"No training rows in {man}. Run `data` first (or `data --only {stage}`).")
    log(f"{len(rows)} rows loaded from {man} (stage={stage}).")

    # Speech-requirement guard: these stages must train a SPEECH-in model. If the
    # manifest has little/no audio (e.g. OmniVoice was down during `data`), we
    # would silently train a TEXT-only model and the benchmark would measure the
    # wrong thing. Refuse unless explicitly overridden.
    audio_rows = sum(1 for r in rows if r.get("audio"))
    frac = audio_rows / max(1, len(rows))
    if stage in ("align", "s2s", "medical") and frac < 0.5:
        log(f"WARNING: only {audio_rows}/{len(rows)} ({frac:.0%}) rows carry speech audio.",
            err=True)
        if not args.allow_textonly:
            die("Refusing to train a speech-less model. Start your OmniVoice TTS server "
                f"(README) and rebuild data (`data --only {stage}`), or pass "
                "--allow-textonly to intentionally train text-only.")

    # init from the previous stage's adapters (curriculum), unless --from-scratch
    init_dir = None
    if not args.from_scratch:
        prev = None
        if stage in STAGE_ORDER and STAGE_ORDER.index(stage) > 0:
            prev = CFG.stage_ckpt(STAGE_ORDER[STAGE_ORDER.index(stage) - 1])
        elif stage == "distill":
            prev = CFG.stage_ckpt("medical")     # distill builds on the medical model
        if prev and Path(prev, "projector.pt").exists():
            init_dir = prev
            log(f"Initializing from previous stage adapters: {prev}")

    # OOM / NVML / transient crashes here propagate to the top-level supervisor
    # (main), which classifies them, escalates the degradation ladder, and
    # re-EXECs with --resume so training continues from the last checkpoint.
    _train_loop(CFG, stage, rows, init_dir=init_dir, resume=args.resume)
    log(f"=== TRAIN stage={stage} DONE ===  ckpt: {CFG.stage_ckpt(stage)}")


def _train_loop(cfg, stage, rows, init_dir=None, resume=False):
    import random
    import torch
    from bitsandbytes.optim import PagedAdamW8bit

    ckpt_dir = cfg.stage_ckpt(stage)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    state_path = Path(ckpt_dir) / "trainer_state.json"

    # Choose where adapters come from: a real checkpoint for this stage takes
    # priority on --resume; otherwise fall back to the curriculum init_dir. This
    # matters for the OOM re-exec, which passes --resume even before any
    # checkpoint has been written — without this it would lose the previous stage.
    have_ckpt = Path(ckpt_dir, "projector.pt").exists()
    adapter_dir = ckpt_dir if (resume and have_ckpt) else init_dir
    model, tok, feat = build_model(cfg, for_training=True, adapter_dir=adapter_dir)
    params = model.trainable_parameters()
    lr = cfg.stage_lr.get(stage, 1e-4)
    opt = PagedAdamW8bit(params, lr=lr, weight_decay=0.0)

    start_step, start_epoch = 0, 0
    if resume and have_ckpt and state_path.exists():
        st = read_json(state_path, {})
        start_step = st.get("global_step", 0)
        start_epoch = st.get("epoch", 0)
        op = Path(ckpt_dir) / "optimizer.pt"
        if op.exists():
            try:
                # weights_only=False: bnb 8-bit optimizer state holds non-tensor objects
                opt.load_state_dict(torch.load(op, map_location=model.device,
                                               weights_only=False))
                log(f"Resumed optimizer + step {start_step}.")
            except Exception as e:
                log(f"optimizer resume failed ({e}); continuing with fresh optimizer.", err=True)

    epochs = cfg.stage_epochs.get(stage, 1)
    micro = cfg.micro_batch
    ga = cfg.grad_accum
    global_step = start_step
    micro_count = 0                       # counts ACTUAL backward() calls (not loop index)
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        random.Random(cfg.seed + epoch).shuffle(rows)
        opt.zero_grad(set_to_none=True)
        micro_count = 0                   # reset per epoch (partial window flushed below)
        pending = False
        for bi in range(0, len(rows), micro):
            batch_rows = rows[bi:bi + micro]
            batch = [(_load_wav(r.get("audio")),
                      r.get("prompt") or r.get("instruction_text") or "",
                      r["target"]) for r in batch_rows]
            batch = [b for b in batch if b[2]]
            if not batch:
                continue
            loss = model.forward_batch(batch) / ga
            loss.backward()
            pending = True
            micro_count += 1
            if micro_count % ga == 0:                       # true accumulation boundary
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending = False
                global_step += 1
                if global_step % cfg.log_steps == 0:
                    used, total = gpu_mem_gb()
                    sps = global_step / max(1e-6, time.time() - t0)
                    log(f"stage={stage} ep={epoch} step={global_step} "
                        f"loss={loss.item() * ga:.4f} lr={lr:g} "
                        f"vram={used}/{total}GB {sps:.2f} step/s")
                if global_step % cfg.save_steps == 0:
                    _save_ckpt(model, opt, ckpt_dir, global_step, epoch)
        if pending:                                         # flush the final partial window
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
        _save_ckpt(model, opt, ckpt_dir, global_step, epoch + 1)
    _save_ckpt(model, opt, ckpt_dir, global_step, epochs)
    log(f"Training finished: {global_step} optimizer steps in {time.time() - t0:.0f}s.")


def _save_ckpt(model, opt, ckpt_dir, step, epoch):
    import torch
    model.save_adapters(ckpt_dir)
    torch.save(opt.state_dict(), str(Path(ckpt_dir) / "optimizer.pt"))
    write_json(Path(ckpt_dir) / "trainer_state.json",
               {"global_step": step, "epoch": epoch, "time": time.time()})
    log(f"  checkpoint saved @ step {step} -> {ckpt_dir}")


def _load_wav(path):
    if not path:
        return None
    try:
        import librosa
        arr, sr = librosa.load(path, sr=16000, mono=True)
        return arr
    except Exception:
        return None


# =========================================================================== #
#  EVAL  — the "best-in-era" Turkish native voice benchmark                    #
# =========================================================================== #

def cmd_eval(args):
    CFG.ensure_dirs()
    set_seed(CFG.seed)
    suite = args.suite
    log(f"=== EVAL suite={suite} ===")
    results = read_json(Path(CFG.bench_dir) / "results.json", {}) or {}

    if suite in ("asr", "all"):
        results["asr"] = eval_asr(CFG, limit=args.limit); _free_gpu()
    if suite in ("tts", "all"):
        results["tts"] = eval_tts(CFG, limit=args.limit); _free_gpu()
    if suite in ("s2s", "all"):
        results["s2s"] = eval_s2s(CFG, limit=args.limit); _free_gpu()
    if suite in ("medical", "all"):
        results["medical"] = eval_medical(CFG, limit=args.limit); _free_gpu()
    if suite in ("latency", "all"):
        results["latency"] = eval_latency(CFG); _free_gpu()

    results["_meta"] = {"time": time.time(), "student": CFG.student_llm,
                        "ckpt": CFG.stage_ckpt("medical")}
    write_json(Path(CFG.bench_dir) / "results.json", results)
    _print_report(results)
    log(f"=== EVAL DONE ===  full JSON: {Path(CFG.bench_dir) / 'results.json'}")


def _free_gpu():
    """Release GPU memory between heavy eval models so they don't co-reside."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _wer_cer(refs, hyps):
    import jiwer
    refs_n = [normalize_tr(r) for r in refs]
    hyps_n = [normalize_tr(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(refs_n, hyps_n) if r.strip()]
    if not pairs:
        return {"wer": None, "cer": None, "n": 0}
    R, H = zip(*pairs)
    return {"wer": round(jiwer.wer(list(R), list(H)), 4),
            "cer": round(jiwer.cer(list(R), list(H)), 4), "n": len(pairs)}


def eval_asr(cfg, limit=200):
    """ASR-understanding: WER/CER on FLEURS-tr (+ MediaSpeech) for our whisper-ft2
    AND baselines. Uses the mandatory Turkish normalizer."""
    log("[eval] ASR (FLEURS-tr / MediaSpeech) ...")
    out = {}
    try:
        refs, wavs = _load_fleurs_tr(cfg, limit)
    except Exception as e:
        log(f"  FLEURS load failed: {e}", err=True)
        return {"error": str(e)}
    models = {"whisper-ft2 (ours)": cfg.whisper_ckpt,
              "whisper-large-v3 (base)": "openai/whisper-large-v3",
              "whisper-large-v3-turbo (base)": cfg.whisper_processor}
    if args_has_baseline("mms"):
        models["mms-1b-all"] = "facebook/mms-1b-all"
    for name, mid in models.items():
        asr = None
        try:
            asr = ASRBackend(mid, cfg)
            hyps = [asr.transcribe(w, 16000, "tr") for w in wavs]
            out[name] = _wer_cer(refs, hyps)
            log(f"    {name}: WER={out[name]['wer']} CER={out[name]['cer']} (n={out[name]['n']})")
        except Exception as e:
            out[name] = {"error": str(e)}
            log(f"    {name}: FAILED {e}", err=True)
        finally:
            del asr           # free each Whisper pipeline before loading the next
            _free_gpu()
    return out


def eval_tts(cfg, limit=100):
    """TTS-generation: round-trip intelligibility WER (neutral whisper-v3), MOS
    (UTMOSv2 if present), and speaker similarity vs the reference voice."""
    log("[eval] TTS (round-trip WER + MOS + speaker sim) ...")
    tts = TTSBackend(cfg)
    if not tts.available():
        return {"error": "no TTS backend available"}
    sentences = _tts_eval_sentences(cfg, limit)
    asr = ASRBackend("openai/whisper-large-v3", cfg)   # INDEPENDENT judge
    hyps, refs, wavs = [], [], []
    for s in sentences:
        try:
            wav, sr = _synth(tts, s)
            wav16 = _to16k(wav, sr)
            hyps.append(asr.transcribe(wav16, 16000, "tr"))
            refs.append(s); wavs.append((wav, sr))
        except Exception as e:
            log(f"    synth/asr failed: {e}", err=True)
    res = {"round_trip": _wer_cer(refs, hyps)}
    res["mos_utmos"] = _mos_score([w for w, _ in wavs], [sr for _, sr in wavs])
    res["speaker_sim"] = _speaker_sim(cfg, [w for w, _ in wavs], [sr for _, sr in wavs])
    log(f"    round-trip WER={res['round_trip']['wer']}  MOS={res['mos_utmos']}  "
        f"spk_sim={res['speaker_sim']}")
    return res


def eval_s2s(cfg, limit=100):
    """Full speech-to-speech: spoken-QA accuracy (exact/normalized match on the
    text transcript of the model's answer) for our native model vs cascade."""
    log("[eval] S2S spoken-QA (native vs cascade) ...")
    items = _spoken_qa_set(cfg, limit)
    if not items:
        return {"error": "no spoken-QA eval items (need TTS to render questions)"}
    out = {}
    # native model
    try:
        model, tok, feat = build_model(cfg, for_training=False,
                                       adapter_dir=cfg.stage_ckpt("medical"))
        correct = 0
        for it in items:
            gen = model.generate_ids(it["wav"], it.get("prompt", ""), max_new_tokens=64)
            ans = tok.decode(gen[0], skip_special_tokens=True)
            if _match(ans, it["answer"]):
                correct += 1
        out["native (ours)"] = {"acc": round(correct / len(items), 4), "n": len(items)}
        del model, tok, feat        # free the native 7B before loading the cascade 7B
    except Exception as e:
        out["native (ours)"] = {"error": str(e)}
        log(f"    native S2S failed: {e}", err=True)
    _free_gpu()
    # cascade baseline (whisper-ft2 ASR -> student text -> match)
    try:
        out["cascade (whisper-ft2 + student)"] = _eval_cascade_qa(cfg, items)
    except Exception as e:
        out["cascade (whisper-ft2 + student)"] = {"error": str(e)}
    log(f"    S2S: {out}")
    return out


def eval_medical(cfg, limit=200):
    """Medical-Term Error Rate (MTER): entity error over gazetteer spans, split by
    Turkish vs English-code-switch, on the model's spoken/text answers."""
    log("[eval] Medical-Term Error Rate ...")
    gaz = load_gazetteer(cfg)
    items = _medical_eval_set(cfg, limit)
    if not items:
        return {"error": "no medical eval items"}
    try:
        model, tok, _ = build_model(cfg, for_training=False,
                                    adapter_dir=cfg.stage_ckpt("medical"))
    except Exception as e:
        return {"error": f"model load failed: {e}"}
    tr_hit = tr_tot = en_hit = en_tot = 0
    for it in items:
        gen = model.generate_ids(it["wav"], it.get("prompt", ""), max_new_tokens=128)
        ans = tr_lower(tok.decode(gen[0], skip_special_tokens=True))
        for term in it["expected_terms"]:
            row = gaz.get(term.lower(), {})
            is_en = row.get("lang") == "en"
            present = term.lower() in ans or (row.get("pron", "") and row["pron"] in ans)
            if is_en:
                en_tot += 1; en_hit += int(bool(present))
            else:
                tr_tot += 1; tr_hit += int(bool(present))
    mter = {"mter_tr": round(1 - tr_hit / tr_tot, 4) if tr_tot else None,
            "mter_en_codeswitch": round(1 - en_hit / en_tot, 4) if en_tot else None,
            "n_tr": tr_tot, "n_en": en_tot}
    log(f"    MTER: {mter}")
    return mter


def eval_latency(cfg):
    """Time-to-first-audio + RTF + end-to-end turn latency, native vs cascade."""
    log("[eval] Latency (TTFB / RTF / e2e) ...")
    import numpy as np
    probe = np.zeros(16000 * 3, dtype="float32")   # 3 s silence probe
    res = {}
    tts = TTSBackend(cfg)
    # native TTFB
    try:
        model, tok, _ = build_model(cfg, for_training=False,
                                    adapter_dir=cfg.stage_ckpt("medical"))
        res["native"] = _measure_native_latency(model, tok, tts, probe)
    except Exception as e:
        res["native"] = {"error": str(e)}
    # cascade TTFB (full ASR + full gen + full TTS)
    try:
        res["cascade"] = _measure_cascade_latency(cfg, tts, probe)
    except Exception as e:
        res["cascade"] = {"error": str(e)}
    log(f"    latency: {res}")
    return res


# ---- eval helpers --------------------------------------------------------- #

def args_has_baseline(name):
    return name in (globals().get("_ACTIVE_BASELINES") or [])


def _load_fleurs_tr(cfg, limit):
    ds = _hf_load(cfg.fleurs_repo, "tr_tr", "test", streaming=True, audio_col="audio")
    refs, wavs = [], []
    for ex in ds:
        refs.append(str(ex.get("transcription") or ex.get("raw_transcription") or ""))
        arr, sr = _decode_audio_field(ex["audio"])          # torchcodec-free
        wavs.append(_to16k(arr, sr))
        if len(refs) >= limit:
            break
    return refs, wavs


def _tts_eval_sentences(cfg, limit):
    """UNIQUE sentences (no replication): FLEURS-tr real refs + curated medical
    pronunciation-stress sentences. Returns at most `limit` distinct sentences."""
    out = []
    try:
        refs, _ = _load_fleurs_tr(cfg, max(1, limit - 10))
        out += refs
    except Exception:
        pass
    med = ["Hastaya günde iki kez beş yüz miligram amoksisilin reçete edildi.",
           "Kan basıncı yüksekti, amlodipin dozu artırıldı.",
           "Siprofloksasin tedavisine rağmen ateş devam ediyor.",
           "Metformin ve insülin birlikte kullanılıyor.",
           "Migren atakları için profilaktik tedavi başlandı.",
           "Warfarin dozu INR değerine göre ayarlandı.",
           "Klopidogrel ve asetilsalisilik asit birlikte verildi.",
           "Levotiroksin sabah aç karnına alınmalıdır.",
           "Furosemid ile ödem kontrol altına alındı.",
           "Atorvastatin ile kolesterol düzeyi düştü."]
    out += med
    uniq = list(dict.fromkeys(out))          # de-duplicate, preserve order
    if len(uniq) < limit:
        log(f"    [note] only {len(uniq)} UNIQUE TTS eval sentences available "
            f"(requested {limit}); reporting the true count.", err=True)
    return uniq[:limit]


def _load_medturkquad(cfg, limit):
    """Best-effort real Turkish MEDICAL questions from MedTurkQuAD (EVAL-ONLY,
    CC-BY-NC-ND). Returns list of (question, short_answer). Empty on failure."""
    out = []
    try:
        ds = _hf_load(cfg.medturkquad_repo, split="validation", streaming=True)
    except Exception:
        try:
            ds = _hf_load(cfg.medturkquad_repo, split="train", streaming=True)
        except Exception as e:
            log(f"    MedTurkQuAD unavailable ({e}); using curated medical probes only.", err=True)
            return out
    for ex in ds:
        low = {str(k).lower(): v for k, v in ex.items()}
        q = str(low.get("question") or "").strip()
        ans = low.get("answers") or low.get("answer") or ""
        if isinstance(ans, dict):
            texts = ans.get("text") or []
            ans = texts[0] if texts else ""
        elif isinstance(ans, list):
            ans = ans[0] if ans else ""
        a = str(ans).strip()
        if q:
            out.append((q, a))
        if len(out) >= limit:
            break
    return out


def _spoken_qa_set(cfg, limit):
    """Turkish spoken-QA eval items (UNIQUE, no replication). Curated short-answer
    factual set + real MedTurkQuAD questions, each rendered to speech."""
    tts = TTSBackend(cfg)
    if not tts.available():
        return []
    qa = [("Vücudun en büyük organı nedir?", "deri"),
          ("Kalp kaç odacıktan oluşur?", "dört"),
          ("İnsülin hangi organda üretilir?", "pankreas"),
          ("Tansiyon ölçen alete ne denir?", "tansiyon aleti"),
          ("Kırmızı kan hücrelerini üreten doku nedir?", "kemik iliği"),
          ("Kanın pıhtılaşmasını sağlayan hücreler nelerdir?", "trombosit"),
          ("Böbrek taşı hangi organda oluşur?", "böbrek"),
          ("Vücudun ana solunum kası hangisidir?", "diyafram")]
    pairs = list(qa) + _load_medturkquad(cfg, max(0, limit - len(qa)))
    pairs = pairs[:limit]
    items = []
    for q, a in pairs:
        wav, sr = _synth(tts, q)
        items.append({"wav": _to16k(wav, sr), "answer": a, "prompt": ""})
    log(f"    spoken-QA: {len(items)} UNIQUE items "
        f"({'curated+MedTurkQuAD' if len(items) > len(qa) else 'curated only'}).")
    return items


def _medical_eval_set(cfg, limit):
    """Medical-term probes (UNIQUE). Curated code-switch probes + real MedTurkQuAD
    questions whose expected terms are mined from the gazetteer."""
    tts = TTSBackend(cfg)
    if not tts.available():
        return []
    gaz = load_gazetteer(cfg)
    probes = [("Amoksisilin ne için kullanılır?", ["amoksisilin"]),
              ("Ciprofloxacin yan etkileri nelerdir?", ["siprofloksasin"]),
              ("Diyabet tedavisinde metformin nasıl etki eder?", ["metformin"]),
              ("Hipertansiyon nedir?", ["hipertansiyon"]),
              ("Astım krizinde ne yapılmalı?", ["astım"]),
              ("Warfarin kullanan hastada nelere dikkat edilmeli?", ["varfarin"]),
              ("Atorvastatin kolesterolü nasıl düşürür?", ["atorvastatin"])]
    items = [{"q": q, "terms": t} for q, t in probes]
    # augment with real MedTurkQuAD questions; expected terms = gazetteer hits
    for q, _a in _load_medturkquad(cfg, max(0, limit - len(items))):
        ql = tr_lower(q)
        hits = [row["term"] for key, row in gaz.items() if key in ql]
        if hits:
            items.append({"q": q, "terms": hits[:3]})
        if len(items) >= limit:
            break
    items = items[:limit]
    out = []
    for it in items:
        wav, sr = _synth(tts, it["q"])
        out.append({"wav": _to16k(wav, sr), "expected_terms": it["terms"], "prompt": ""})
    log(f"    medical-term eval: {len(out)} UNIQUE probes.")
    return out


def _eval_cascade_qa(cfg, items):
    asr = ASRBackend(cfg.whisper_ckpt, cfg)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.student_llm)
    mdl = AutoModelForCausalLM.from_pretrained(cfg.student_llm, torch_dtype=torch.bfloat16,
                                               device_map={"": 0})
    correct = 0
    for it in items:
        q = asr.transcribe(it["wav"], 16000, "tr")
        msgs = [{"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
        gen = mdl.generate(ids, max_new_tokens=64, do_sample=False)
        ans = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        correct += int(_match(ans, it["answer"]))
    res = {"acc": round(correct / len(items), 4), "n": len(items)}
    del asr, mdl, tok
    _free_gpu()
    return res


def _match(hyp, ref):
    return normalize_tr(ref) in normalize_tr(hyp)


def _mos_score(wavs, srs):
    try:
        import tempfile
        import soundfile as sf
        import utmosv2
        model = utmosv2.create_model(pretrained=True)
        scores = []
        for w, sr in zip(wavs, srs):
            # UTMOSv2.predict is keyword-only: predict(data=..., sr=...) or input_path=...
            try:
                val = model.predict(data=w, sr=sr)
            except TypeError:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    sf.write(tf.name, w, sr)
                    val = model.predict(input_path=tf.name)
            scores.append(float(val["predicted_mos"] if isinstance(val, dict) else val))
        return round(sum(scores) / len(scores), 3)
    except Exception as e:
        log(f"    UTMOSv2 unavailable ({e}); trying torchmetrics DNSMOS.", err=True)
    try:
        import torch
        from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
        dns = DeepNoiseSuppressionMeanOpinionScore(16000, False)
        vals = []
        for w, sr in zip(wavs, srs):
            vals.append(float(dns(torch.tensor(_to16k(w, sr)))[-1]))
        return {"dnsmos_ovrl": round(sum(vals) / len(vals), 3), "note": "not naturalness"}
    except Exception as e:
        return {"error": f"no MOS predictor ({e})"}


def _speaker_sim(cfg, wavs, srs):
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        enc = EncoderClassifier.from_hparams(source=cfg.spk_sim,
                                             savedir=str(Path(cfg.hf_cache) / "ecapa"))
        import soundfile as sf
        import numpy as np
        ref = enc.encode_batch(torch.tensor(_load_wav(cfg.omni_ref_wav)).unsqueeze(0)).squeeze()
        sims = []
        for w, sr in zip(wavs, srs):
            e = enc.encode_batch(torch.tensor(_to16k(w, sr)).unsqueeze(0)).squeeze()
            sims.append(float(torch.nn.functional.cosine_similarity(ref, e, dim=0)))
        return round(sum(sims) / len(sims), 4)
    except Exception as e:
        return {"error": f"speaker-sim unavailable ({e})"}


def _measure_native_latency(model, tok, tts, probe):
    import time as _t
    from transformers import TextIteratorStreamer
    import threading
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    t0 = _t.perf_counter()
    th = threading.Thread(target=model.generate_ids,
                          kwargs={"wav": probe, "prompt_text": "Merhaba, nasılsın?",
                                  "max_new_tokens": 64, "streamer": streamer})
    th.start()
    first_text_t = None
    buf = ""
    for tokt in streamer:
        if first_text_t is None:
            first_text_t = _t.perf_counter() - t0
        buf += tokt
        if len(buf.split()) >= model.cfg.stream_chunk_tokens:
            break
    ttfa = None
    if tts.available() and buf.strip():
        _ = tts.synth(buf)
        ttfa = _t.perf_counter() - t0
    th.join(timeout=30)
    return {"ttf_text_s": round(first_text_t or -1, 3),
            "ttf_audio_s": round(ttfa or -1, 3)}


def _measure_cascade_latency(cfg, tts, probe):
    import time as _t
    asr = ASRBackend(cfg.whisper_ckpt, cfg)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.student_llm)
    mdl = AutoModelForCausalLM.from_pretrained(cfg.student_llm, torch_dtype=torch.bfloat16,
                                               device_map={"": 0})
    t0 = _t.perf_counter()
    q = asr.transcribe(probe, 16000, "tr")
    ids = tok.apply_chat_template([{"role": "user", "content": q or "Merhaba"}],
                                  add_generation_prompt=True, return_tensors="pt").to(mdl.device)
    gen = mdl.generate(ids, max_new_tokens=64, do_sample=False)
    text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
    ttfa = None
    if tts.available() and text.strip():
        _ = tts.synth(text)
        ttfa = _t.perf_counter() - t0
    return {"ttf_audio_s": round(ttfa or -1, 3),
            "note": "cascade waits for full ASR + full gen + full TTS"}


def _to16k(wav, sr):
    import numpy as np
    import librosa
    wav = np.asarray(wav, dtype="float32").reshape(-1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


def _print_report(results):
    log("\n================ TURKISH NATIVE VOICE BENCHMARK ================")
    for suite, data in results.items():
        if suite.startswith("_"):
            continue
        log(f"\n### {suite.upper()}")
        log(json.dumps(data, ensure_ascii=False, indent=2))
    log("===============================================================\n")


# =========================================================================== #
#  SERVE  — streaming native S2S (and cascade fallback)                        #
# =========================================================================== #

def cmd_serve(args):
    CFG.ensure_dirs()
    log(f"=== SERVE ({'cascade' if args.cascade else 'native'}) on "
        f"{CFG.serve_host}:{CFG.serve_port} ===")
    import io
    import numpy as np
    import soundfile as sf
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import Response
    import uvicorn

    app = FastAPI(title="turkish-medvoice")
    tts = TTSBackend(CFG)
    state = {}

    def _lazy():
        if "ready" in state:
            return
        if args.cascade:
            state["asr"] = ASRBackend(CFG.whisper_ckpt, CFG)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            state["tok"] = AutoTokenizer.from_pretrained(CFG.student_llm)
            state["llm"] = AutoModelForCausalLM.from_pretrained(
                CFG.student_llm, torch_dtype=torch.bfloat16, device_map={"": 0})
        else:
            model, tok, _ = build_model(CFG, for_training=False,
                                        adapter_dir=CFG.stage_ckpt("medical"))
            state["model"], state["tok"] = model, tok
        state["ready"] = True

    @app.post("/v1/audio/voice-chat")
    async def voice_chat(file: UploadFile = File(...)):
        _lazy()
        raw = await file.read()
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
        wav16 = _to16k(wav, sr)
        if args.cascade:
            q = state["asr"].transcribe(wav16, 16000, "tr")
            import torch
            tok, llm = state["tok"], state["llm"]
            ids = tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True, return_tensors="pt").to(llm.device)
            gen = llm.generate(ids, max_new_tokens=256, do_sample=False)
            text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        else:
            model, tok = state["model"], state["tok"]
            gen = model.generate_ids(wav16, "", max_new_tokens=256)
            text = tok.decode(gen[0], skip_special_tokens=True)
        gaz = load_gazetteer(CFG)
        speech, srr = tts.synth(apply_pronunciation(text, gaz)) if tts.available() else (np.zeros(1), 24000)
        buf = io.BytesIO(); sf.write(buf, speech, srr, format="WAV")
        # HTTP headers are latin-1 only; Turkish (ş/ı/ğ/İ...) must be URL-encoded.
        from urllib.parse import quote
        return Response(content=buf.getvalue(), media_type="audio/wav",
                        headers={"X-Response-Text": quote(text[:2000])})

    @app.get("/health")
    def health():
        return {"ok": True, "mode": "cascade" if args.cascade else "native",
                "tts": tts.describe()}

    uvicorn.run(app, host=CFG.serve_host, port=CFG.serve_port)


# =========================================================================== #
#  DOCTOR (preflight + auto-fix)  &  AUTO (unattended self-healing roadmap)     #
# =========================================================================== #

def cmd_doctor(args):
    """Preflight the environment and auto-fix what it can: CUDA/MIG env, missing
    deps, low disk, asset paths, TTS reachability (with autostart), HF auth."""
    CFG.ensure_dirs()
    log("=== DOCTOR (preflight + auto-fix) ===")
    healthy = True
    # CUDA / MIG
    try:
        import torch
        used, total = gpu_mem_gb()
        log(f"  torch {torch.__version__} cuda={torch.cuda.is_available()} "
            f"vram={used}/{total}GB alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
        if not torch.cuda.is_available():
            log("  WARNING: CUDA not available.", err=True); healthy = False
    except Exception as e:
        log(f"  torch missing ({e}) -> auto-installing (a fix, not a hard failure).", err=True)
        _auto_pip_install(_IMPORT_TO_PKG["torch"], "torch")
    # core deps — installing a missing one is a FIX, so it does not mark unhealthy
    for mod in ("transformers", "peft", "datasets", "bitsandbytes", "librosa",
                "soundfile", "jiwer", "huggingface_hub", "fastapi", "requests"):
        try:
            __import__(mod)
        except Exception:
            log(f"  missing '{mod}' -> installing {_IMPORT_TO_PKG.get(mod)}", err=True)
            _auto_pip_install(_IMPORT_TO_PKG.get(mod), mod)
    # disk
    try:
        du = shutil.disk_usage(CFG.work)
        log(f"  disk free at {CFG.work}: {du.free / 1e9:.1f} GB")
        if du.free / 1e9 < 20:
            log("  low disk -> pruning stale files.", err=True); _prune_checkpoints(CFG)
    except Exception:
        pass
    # asset paths
    for name, pth in (("whisper-ft2", CFG.whisper_ckpt), ("omnivoice-ft1", CFG.omni_model),
                      ("ref wav", CFG.omni_ref_wav)):
        exists = Path(pth).exists()
        log(f"  asset {name}: {pth} -> {'OK' if exists else 'MISSING'}")
        if not exists:
            healthy = False
    # TTS reachability (+ autostart) and HF auth
    log(f"  TTS reachable/started: {_ensure_tts_server(CFG)}")
    hf_login_if_needed(interactive=False, require=False)
    log(f"=== DOCTOR done (healthy={healthy}) ===")
    return healthy


def cmd_auto(args):
    """Run the WHOLE roadmap unattended: doctor -> data -> train(align,s2s,medical)
    -> eval. Each step is idempotent and resumable; progress is journaled to
    roadmap_state.json so that after any crash the top-level supervisor re-EXECs
    `auto` and it continues from the exact step it left off (train uses --resume)."""
    import argparse as _ap
    CFG.ensure_dirs()
    state_path = Path(CFG.work) / "roadmap_state.json"
    state = read_json(state_path, {}) or {}

    def _done(step):
        return state.get(step, {}).get("done")

    def _mark(step):
        state[step] = {"done": True, "time": time.time()}
        write_json(state_path, state)
        # Credit forward progress: a completed step resets the self-heal restart
        # budget so a long, genuinely-advancing roadmap is not aborted by failures
        # accumulated across earlier (already-recovered) stages. Repeated failures
        # WITHIN one stuck step stay bounded (the budget only resets on success).
        os.environ["TMV_HEAL_REEXEC_N"] = "0"

    def _run_step(step):
        """Run a step; return True only on REAL success. A raised exception
        propagates to the top-level supervisor (OOM ladder / restart)."""
        if step == "doctor":
            return bool(cmd_doctor(_ap.Namespace()))
        if step == "data":
            cmd_data(_ap.Namespace(only=None, n_medical=None, use_teacher=args.use_teacher))
            counts = {m: _manifest_count(Path(CFG.data_dir) / f"{m}.jsonl")
                      for m in ("align", "s2s", "medical")}
            log(f"[auto] data manifests: {counts}")
            ok = counts["align"] >= 100 and counts["s2s"] >= 1 and counts["medical"] >= 1
            if not ok:
                log("[auto] data produced empty/undersized manifests — likely a "
                    "network or OmniVoice-TTS outage. NOT marking data done.", err=True)
            return ok
        if step.startswith("train:"):
            st = step.split(":", 1)[1]
            cmd_train(_ap.Namespace(stage=st, resume=True, from_scratch=False,
                                    allow_textonly=args.allow_textonly))
            return Path(CFG.stage_ckpt(st), "projector.pt").exists()   # a saved ckpt proves it ran
        if step == "eval:all":
            cmd_eval(_ap.Namespace(suite="all", limit=args.limit, baselines=[]))
            res = read_json(Path(CFG.bench_dir) / "results.json", {}) or {}

            def _errored(v):
                if not isinstance(v, dict):
                    return False
                if v.get("error"):
                    return True
                # a suite that is a dict-of-subresults (e.g. latency's native/
                # cascade, or asr's per-model rows) is errored iff EVERY sub-dict
                # errored — so it can't silently count as a passing suite.
                subs = [sv for sv in v.values() if isinstance(sv, dict)]
                return bool(subs) and all(sv.get("error") for sv in subs)

            suites = [k for k in res if not k.startswith("_")]
            ok = any(not _errored(res[k]) for k in suites)
            if not ok:
                log("[auto] every eval suite errored — NOT marking eval done.", err=True)
            return ok
        return True

    plan = ["doctor", "data", "train:align", "train:s2s", "train:medical", "eval:all"]
    log(f"=== AUTO roadmap (self-healing, resumable): {plan} ===")
    for step in plan:
        if _done(step):
            log(f"[auto] {step} already complete — skipping.")
            continue
        log(f"[auto] >>>>>> {step}")
        if _run_step(step):
            _mark(step)
        else:
            # a soft failure that healing can't fix (empty corpus, unhealthy env):
            # surface it (die() is not auto-healed) so it isn't hidden behind a
            # green checkmark and the user can fix the root cause.
            die(f"[auto] step '{step}' did not succeed — see the log above. "
                "Fix the cause (e.g. start the TTS server / check the network / "
                f"provide missing assets), then re-run: turkish_medvoice.py auto")
    log("=== AUTO roadmap complete. Benchmark: "
        f"{Path(CFG.bench_dir) / 'results.json'} ===")


# =========================================================================== #
#  main / argparse                                                             #
# =========================================================================== #

def _apply_yaml(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in (cfg or {}).items():
        if hasattr(CFG, k):
            setattr(CFG, k, v)
    CFG.__post_init__()


def main():
    _proactive_env()          # MIG/tokenizer env before anything imports torch
    p = argparse.ArgumentParser(
        prog="turkish_medvoice.py",
        description="Native Turkish medical speech-to-speech: setup/data/train/eval/serve/doctor/auto")
    p.add_argument("--config", help="optional YAML overriding CONFIG fields")
    p.add_argument("--no-heal", action="store_true",
                   help="disable the top-level self-healing supervisor (raise on error)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="venv + deps + HF login + smoke test")
    s.add_argument("--no-venv", action="store_true")
    s.add_argument("--skip-install", action="store_true")
    s.add_argument("--skip-smoke", action="store_true")
    s.set_defaults(func=cmd_setup)

    d = sub.add_parser("data", help="download + synthesize training data")
    d.add_argument("--only", choices=["align", "s2s", "medical"], default=None)
    d.add_argument("--n-medical", type=int, default=None)
    d.add_argument("--use-teacher", action="store_true",
                   help="augment medical data with a teacher LLM (seq-level KD)")
    d.set_defaults(func=cmd_data)

    t = sub.add_parser("train", help="QLoRA train a stage (resumable)")
    t.add_argument("--stage", choices=["align", "s2s", "medical", "distill"], required=True)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--from-scratch", action="store_true",
                   help="do NOT init from the previous stage's adapters")
    t.add_argument("--allow-textonly", action="store_true",
                   help="train even if the manifest has little/no speech audio")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="run the Turkish voice benchmark")
    e.add_argument("--suite", choices=["asr", "tts", "s2s", "medical", "latency", "all"],
                   default="all")
    e.add_argument("--limit", type=int, default=200)
    e.add_argument("--baselines", nargs="*", default=[],
                   help="extra baselines to include, e.g. mms xtts qwen-omni")
    e.set_defaults(func=cmd_eval)

    sv = sub.add_parser("serve", help="streaming voice-chat endpoint")
    sv.add_argument("--cascade", action="store_true", help="use the safe cascade path")
    sv.set_defaults(func=cmd_serve)

    doc = sub.add_parser("doctor", help="preflight checks + auto-fix (deps, disk, assets, TTS)")
    doc.set_defaults(func=cmd_doctor)

    au = sub.add_parser("auto", help="run the whole roadmap unattended (self-healing, resumable)")
    au.add_argument("--use-teacher", action="store_true")
    au.add_argument("--allow-textonly", action="store_true")
    au.add_argument("--limit", type=int, default=200)
    au.set_defaults(func=cmd_auto)

    args = p.parse_args()
    if args.config:
        _apply_yaml(args.config)
    _apply_heal_level(CFG, HEAL_LEVEL)          # emergency-degradation ladder (env-persisted)
    if getattr(args, "baselines", None):
        globals()["_ACTIVE_BASELINES"] = args.baselines

    # ---- top-level SELF-HEALING supervisor ----
    try:
        args.func(args)
    except (KeyboardInterrupt, SystemExit):
        raise                                   # user abort / intentional die() -> do not heal
    except Exception as e:                       # noqa: BLE001
        import traceback
        log(f"UNHANDLED {type(e).__name__}: {e}", err=True)
        log(traceback.format_exc(), err=True)
        if getattr(args, "no_heal", False):
            raise
        supervise(e, args)                       # remediates + re-EXECs, or returns if unhealable
        raise                                    # not healed -> surface the error


if __name__ == "__main__":
    main()
