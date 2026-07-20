#!/usr/bin/env bash
# =============================================================================
#  turkish-medvoice RUN WIZARD
#  One command sets up and runs the whole native Turkish voice-to-voice pipeline:
#    clean previous run -> fetch latest trainer -> detect assets -> size to disk
#    -> start OmniVoice TTS -> setup -> doctor -> data -> train -> eval
#
#  It NEVER touches: other projects under /data, your venv, tmux sessions, or the
#  model assets in ses_models. It only removes artifacts THIS project generated.
#
#  A background watchdog stops the run BEFORE the box maxes out on disk or RAM,
#  and the trainer shrinks its sequence length before it can OOM the GPU.
#
#  Run it inside tmux:
#    tmux new -s medvoice
#    bash run_medvoice.sh                  # soft clean (keeps built data + hf cache)
#    bash run_medvoice.sh --clean full     # also delete built training data
#    bash run_medvoice.sh --clean all      # also delete the HF model cache
#    bash run_medvoice.sh --preset hardcore --yes
#    bash run_medvoice.sh --no-run         # set everything up, don't start training
# =============================================================================
set -Eeuo pipefail

# ---------------------------------------------------------------- 0. settings
SES=${SES:-/data/ses_models}                 # your model assets
WORK=${WORK:-/data/medvoice}                 # our outputs
VENV=${VENV:-/data/venv-medvoice}            # python venv
PORT_TTS=${PORT_TTS:-8133}
PRESET=${PRESET:-standard}
RAW_URL=${RAW_URL:-https://raw.githubusercontent.com/ArioMoniri/turso/main/scripts/turkish_medvoice.py}

# resource ceilings, percent of capacity. WARN = prune, STOP = halt the run.
DISK_WARN=${DISK_WARN:-85};  DISK_STOP=${DISK_STOP:-93}
RAM_WARN=${RAM_WARN:-88};    RAM_STOP=${RAM_STOP:-95}
VRAM_WARN=${VRAM_WARN:-92}
RESERVE_GB=${RESERVE_GB:-30}                 # keep free for models/cache/checkpoints

CLEAN=soft; ASSUME_YES=0; DO_RUN=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)  CLEAN="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --no-run) DO_RUN=0; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m[wizard]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[wizard]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[wizard] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

# SAFETY: this script rm -rf's paths under $WORK, and $WORK sits on a mount shared
# with unrelated projects. Refuse any WORK that is empty, relative, a system dir, a
# bare mount root, or the assets dir — a typo there must never eat someone's work.
validate_paths() {
  [[ -n "${WORK:-}" ]] || die "WORK is empty."
  [[ "$WORK" = /* ]]   || die "WORK must be an absolute path (got '$WORK')."
  case "${WORK%/}" in
    ""|"/"|"/data"|"/root"|"/home"|"/usr"|"/etc"|"/var"|"/opt"|"/srv"|"/mnt"|"/tmp")
      die "refusing to use WORK='$WORK' — that is a system/mount root, not a workspace." ;;
  esac
  [[ $(awk -F/ '{c=0; for(i=1;i<=NF;i++) if($i!="") c++; print c}' <<<"${WORK%/}") -ge 2 ]] \
    || die "WORK='$WORK' is too shallow; use something like /data/medvoice."
  if [[ -n "${SES:-}" && "${WORK%/}" == "${SES%/}" ]]; then
    die "WORK and SES are the same path ('$WORK') — that would delete your model assets."
  fi
  case "${WORK%/}" in
    "${SES%/}"/*) die "WORK ('$WORK') is inside SES ('$SES') — cleaning would hit your assets." ;;
  esac
}

# ------------------------------------------------------- 1. resource helpers
disk_pct()     { df -P  "$1" 2>/dev/null | awk 'NR==2{gsub(/%/,"",$5); print $5+0}'; }
disk_free_gb() { df -PBG "$1" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4+0}'; }
ram_pct()      { free 2>/dev/null | awk '/^Mem:/{printf "%d", ($2-$7)*100/$2}'; }
vram_pct()     {
  if ! command -v nvidia-smi >/dev/null 2>&1; then echo 0; return; fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' 'NR==1{printf "%d", $1*100/$2}' || echo 0
}

# Free space WITHOUT destroying anything the run still needs.
prune_safe() {
  find "$WORK" \( -name '*.tmp' -o -name '*.jsonl.tmp' \) -delete 2>/dev/null || true
  find "${HF_HOME:-$WORK/hf_cache}" ~/.cache/huggingface -name '*.incomplete' -delete 2>/dev/null || true
  rm -rf "${WORK:?}/bench_results"/* 2>/dev/null || true
  find "$WORK/logs" -name '*.log' -size +512M -exec truncate -s 64M {} \; 2>/dev/null || true
  command -v pip     >/dev/null 2>&1 && pip cache purge >/dev/null 2>&1 || true
  command -v apt-get >/dev/null 2>&1 && apt-get clean   >/dev/null 2>&1 || true
  sync || true
}

# ------------------------------------------------- 2. background resource cop
start_watchdog() {
  local logf="$WORK/logs/watchdog.log"
  mkdir -p "$WORK/logs"
  (
    set +e                                  # a probe failure must never kill the cop
    gone=0
    while true; do
      sleep 60
      d=$(disk_pct "$WORK"); r=$(ram_pct); v=$(vram_pct)
      d=${d:-0}; r=${r:-0}; v=${v:-0}
      printf '%s disk=%s%% ram=%s%% vram=%s%%\n' "$(date +%H:%M:%S)" "$d" "$r" "$v" >>"$logf"

      if [ "$d" -ge "$DISK_WARN" ]; then
        echo "$(date +%H:%M:%S) disk ${d}% >= ${DISK_WARN}% -> pruning" >>"$logf"
        prune_safe >>"$logf" 2>&1
        d=$(disk_pct "$WORK"); d=${d:-0}
      fi
      if [ "$r" -ge "$RAM_WARN" ]; then
        echo "$(date +%H:%M:%S) ram ${r}% -> dropping page cache" >>"$logf"
        sync; echo 3 >/proc/sys/vm/drop_caches 2>/dev/null
        r=$(ram_pct); r=${r:-0}
      fi
      if [ "$v" -ge "$VRAM_WARN" ]; then
        echo "$(date +%H:%M:%S) vram ${v}% (trainer shrinks max_seq_len itself)" >>"$logf"
      fi

      # about to max out -> stop the run cleanly rather than die from ENOSPC/OOM
      if [ "$d" -ge "$DISK_STOP" ] || [ "$r" -ge "$RAM_STOP" ]; then
        echo "$(date +%H:%M:%S) CRITICAL disk=${d}% ram=${r}% -> STOPPING the run" >>"$logf"
        pkill -INT -f turkish_medvoice.py 2>/dev/null    # SIGINT: lets it checkpoint
        sleep 30
        pkill -9   -f turkish_medvoice.py 2>/dev/null
        exit 0
      fi

      # exit once the trainer has finished (5 consecutive misses)
      if pgrep -f turkish_medvoice.py >/dev/null 2>&1; then gone=0; else gone=$((gone+1)); fi
      [ "$gone" -ge 5 ] && { echo "$(date +%H:%M:%S) trainer gone -> watchdog exit" >>"$logf"; exit 0; }
    done
  ) &
  WATCHDOG_PID=$!
  say "resource watchdog running (pid $WATCHDOG_PID) -> $logf"
  trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT
}

# ---------------------------------------------------------- 2b. venv bootstrap
# The venv dir may exist but be empty/broken (no bin/python). Falling back to the
# system interpreter would scatter multi-GB wheels into /usr and break the TTS
# server, so build a real venv instead.
ensure_venv() {
  PY="$VENV/bin/python"
  if [[ -x "$PY" ]]; then say "venv: $VENV"; return 0; fi
  if [[ -e "$VENV" && ! -d "$VENV" ]]; then die "$VENV exists but is not a directory."; fi

  local base; base=$(command -v python3) || die "no python3 on PATH to build a venv with."
  if [[ -d "$VENV" ]]; then
    local n; n=$(ls -A "$VENV" 2>/dev/null | wc -l | tr -d ' ')
    warn "no interpreter at $PY — $VENV exists but holds $n entries; repairing it."
  fi
  say "creating venv at $VENV (base: $base) ..."
  if ! "$base" -m venv "$VENV" >/dev/null 2>&1; then
    warn "python3 -m venv failed — installing python3-venv and retrying ..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq >/dev/null 2>&1 || true
      apt-get install -y python3-venv python3-dev >/dev/null 2>&1 || true
    fi
    "$base" -m venv "$VENV" >/dev/null 2>&1 || {
      warn "still failing — recreating from scratch (--clear) ..."
      "$base" -m venv --clear "$VENV" || die "could not create a venv at $VENV"
    }
  fi
  PY="$VENV/bin/python"
  [[ -x "$PY" ]] || die "venv created but $PY is missing."
  "$PY" -m pip install -q --upgrade pip setuptools wheel 2>/dev/null || warn "pip self-upgrade failed (continuing)."
  say "venv ready: $PY ($("$PY" -V 2>&1))"
}

# ------------------------------------------------------------ 3. asset detect
has_hf_weights() {
  compgen -G "$1/model*.safetensors" >/dev/null 2>&1 && return 0
  compgen -G "$1/pytorch_model*.bin" >/dev/null 2>&1 && return 0
  return 1
}

# A usable training encoder must be an HF transformers Whisper dir. A CTranslate2 /
# faster-whisper export (model.bin + vocabulary.*) looks similar but cannot be loaded.
is_hf_whisper() {
  [[ -f "$1/config.json" ]] || return 1
  has_hf_weights "$1" || return 1
  grep -q '"model_type"[[:space:]]*:[[:space:]]*"whisper"' "$1/config.json" 2>/dev/null
}

detect_assets() {
  [[ -d "$SES" ]] || die "assets dir not found: $SES   (export SES=/path/to/ses_models)"

  # OmniVoice TTS: whatever omnivoice-* dir exists (ft1, v3, ...) — newest wins
  OMNI_DIR=$(find "$SES" -maxdepth 1 -type d -name 'omnivoice*' | sort -V | tail -1)
  [[ -n "$OMNI_DIR" ]] || die "no omnivoice* model dir under $SES"

  # Whisper encoder MUST be HF-format; a *-ct2 (CTranslate2) export cannot be loaded
  # by transformers, so fall back to the base model rather than crash mid-training.
  # An explicit WHISPER=/path (or a pre-set TMV_WHISPER_CKPT) always wins.
  WHISPER_DIR="${WHISPER:-${TMV_WHISPER_CKPT:-}}"
  if [[ -n "$WHISPER_DIR" ]]; then
    say "whisper pinned by env -> $WHISPER_DIR"
    if [[ -d "$WHISPER_DIR" ]] && ! is_hf_whisper "$WHISPER_DIR"; then
      warn "pinned $WHISPER_DIR is NOT an HF transformers Whisper dir — the trainer will"
      warn "  fall back to the base model. Unset WHISPER/TMV_WHISPER_CKPT to auto-detect."
    fi
  else
    # search *hf* names first, then anything else; format check is the real gate
    while read -r d; do
      [[ -z "$d" ]] && continue
      if is_hf_whisper "$d"; then WHISPER_DIR="$d"; break; fi
    done < <( { find "$SES" -maxdepth 1 -type d -name 'whisper*hf*';
                find "$SES" -maxdepth 1 -type d -name 'whisper*' ! -name '*hf*'; } 2>/dev/null )
  fi

  if [[ -z "$WHISPER_DIR" ]]; then
    WHISPER_DIR="openai/whisper-large-v3-turbo"
    local found; found=$(find "$SES" -maxdepth 1 -type d -name 'whisper*' -printf '%f ' 2>/dev/null || true)
    warn "no HF-format Whisper under $SES (found: ${found:-none})."
    warn "  A CTranslate2/faster-whisper export CANNOT be used as a training encoder."
    warn "  -> using BASE $WHISPER_DIR. Training works, but you lose your Turkish"
    warn "     fine-tune's encoder quality. To keep it, place its HF export"
    warn "     (config.json with model_type=whisper + model.safetensors) under $SES."
  fi

  REF_WAV="$SES/ref/emin.wav"; REF_TXT="$SES/ref/emin.txt"; APP_DIR="$SES/app"
  say "assets detected:"
  say "  whisper encoder : $WHISPER_DIR"
  say "  omnivoice TTS   : $OMNI_DIR"
  say "  ref voice       : $REF_WAV"
  say "  tts app dir     : $APP_DIR"
}

# ------------------------------------------------- 4. size the run to the disk
# ~0.25 MB per synthesized clip; scale the targets so we cannot fill the SSD.
autoscale_to_disk() {
  local free usable max_clips want
  free=$(disk_free_gb "$WORK"); free=${free:-0}
  usable=$(( free - RESERVE_GB ))
  say "free disk at $WORK: ${free}GB (reserve ${RESERVE_GB}GB) -> ${usable}GB for speech synthesis"
  if [ "$usable" -lt 3 ]; then
    die "only ${free}GB free — free space, lower RESERVE_GB, or use --clean full."
  fi
  max_clips=$(( usable * 1024 * 4 ))                 # ~4 clips per MB
  case "$PRESET" in
    smoke)    want_s2s=800;    want_med=400   ;;
    hardcore) want_s2s=120000; want_med=60000 ;;
    *)        want_s2s=40000;  want_med=20000 ;;
  esac
  want=$(( want_s2s + want_med ))
  if [ "$want" -gt "$max_clips" ]; then
    want_s2s=$(( want_s2s * max_clips / want ))
    want_med=$(( want_med * max_clips / want ))
    warn "disk fits ~${max_clips} clips but preset '$PRESET' wants ${want}."
    warn "  scaling down -> TMV_N_S2S=$want_s2s TMV_N_MED=$want_med (prevents an SSD blowout)"
  fi
  export TMV_N_S2S=$want_s2s TMV_N_MED=$want_med
}

# ------------------------------------------------------------- 5. clean phase
clean_previous() {
  local -a targets=(
    "$WORK/checkpoints" "$WORK/bench_results" "$WORK/logs"
    "$WORK/roadmap_state.json" "$WORK/vocab_ext.json"
  )
  if [[ "$CLEAN" == "full" || "$CLEAN" == "all" ]]; then targets+=("$WORK/data"); fi
  if [[ "$CLEAN" == "all" ]]; then targets+=("$WORK/hf_cache"); fi

  say "clean mode '$CLEAN' — the following are from previous runs and all regenerable:"
  local found=0 t
  for t in "${targets[@]}"; do
    if [[ -e "$t" ]]; then
      printf '    %-44s %s\n' "$t" "$(du -sh "$t" 2>/dev/null | cut -f1)"
      found=1
    fi
  done
  if [ "$found" -eq 0 ]; then say "    (nothing to clean)"; fi
  say "NOT touched: $SES, $VENV, other projects under /data, tmux sessions."

  if [ "$found" -eq 1 ] && [ "$ASSUME_YES" -eq 0 ]; then
    printf '\033[1;33m[wizard]\033[0m delete the above? [y/N] '
    read -r ans
    [[ "$ans" =~ ^[Yy]$ ]] || die "aborted by user."
  fi
  pkill -9 -f turkish_medvoice.py 2>/dev/null || true
  for t in "${targets[@]}"; do rm -rf "$t"; done
  find "$WORK" \( -name '*.tmp' -o -name '*.jsonl.tmp' \) -delete 2>/dev/null || true
  find ~/.cache/huggingface -name '*.incomplete' -delete 2>/dev/null || true
  mkdir -p "$WORK" "$WORK/logs"
  say "clean done."
}

# ------------------------------------------------------------ 6. TTS server
tts_up() { curl -sS -o /dev/null --max-time 4 "http://127.0.0.1:$PORT_TTS/" >/dev/null 2>&1; }

start_tts() {
  if tts_up; then say "OmniVoice TTS already up on :$PORT_TTS"; return 0; fi
  if [[ ! -f "$APP_DIR/omnivoice_server.py" ]]; then
    warn "no $APP_DIR/omnivoice_server.py — cannot start TTS automatically."
    return 0
  fi
  say "starting OmniVoice TTS ($(basename "$OMNI_DIR")) on :$PORT_TTS ..."
  mkdir -p "$WORK/logs"
  OMNI_MODEL="$OMNI_DIR" OMNI_REF="$REF_WAV" OMNI_REFTXT_FILE="$REF_TXT" OMNI_LANG=tr \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "$PY" -m uvicorn omnivoice_server:app --app-dir "$APP_DIR" \
        --host 127.0.0.1 --port "$PORT_TTS" >>"$WORK/logs/tts_server.log" 2>&1 &
  echo $! >"$WORK/logs/.tts.pid"
  local i
  for i in $(seq 1 40); do
    sleep 3
    if tts_up; then say "TTS is up (pid $(cat "$WORK/logs/.tts.pid"))."; return 0; fi
  done
  warn "TTS did not come up in 120s — see $WORK/logs/tts_server.log"
  warn "  data would be built TEXT-ONLY and the run will stop at the speech gate."
  return 0
}

# ===================================================================== MAIN ==
say "=== turkish-medvoice run wizard ==="
[[ $EUID -eq 0 ]] || warn "not root: page-cache drop / apt clean will be skipped."

validate_paths
ensure_venv

mkdir -p "$WORK" "$WORK/logs"
clean_previous
detect_assets

say "fetching the latest trainer from GitHub ..."
curl -fsSL "$RAW_URL" -o "$WORK/turkish_medvoice.py" || die "download failed: $RAW_URL"
say "trainer: $WORK/turkish_medvoice.py ($(wc -l <"$WORK/turkish_medvoice.py") lines)"

# ---- the full environment the trainer needs --------------------------------
export TMV_ROOT="$SES" TMV_WORK="$WORK" TMV_VENV="$VENV"
export TMV_WHISPER_CKPT="$WHISPER_DIR"
export TMV_OMNI_MODEL="$OMNI_DIR" TMV_OMNI_REF="$REF_WAV" TMV_OMNI_REFTXT="$REF_TXT"
export TMV_TTS_MODE=http TMV_OMNI_URL="http://127.0.0.1:$PORT_TTS/v1/audio/speech"
export TMV_OMNI_VENV_PY="$PY" TMV_OMNI_APP_DIR="$APP_DIR"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# VRAM ceilings — the trainer shrinks max_seq_len before it can OOM
export TMV_MAXSEQ=${TMV_MAXSEQ:-2048} TMV_MAX_AUDIO_SEC=${TMV_MAX_AUDIO_SEC:-20}
export TMV_VRAM_GUARD=${TMV_VRAM_GUARD:-0.88} TMV_VRAM_SHRINK=${TMV_VRAM_SHRINK:-0.92}
export TMV_SYNTH_WORKERS=${TMV_SYNTH_WORKERS:-2}

autoscale_to_disk
start_watchdog
start_tts

say "environment:"
env | grep -E '^(TMV_|HF_HOME|PYTORCH_CUDA)' | sort | sed 's/^/    /'

if [ "$DO_RUN" -eq 0 ]; then
  say "--no-run: setup complete. Start it with:"
  say "  cd $WORK && $PY turkish_medvoice.py --preset $PRESET auto"
  exit 0
fi

cd "$WORK"
say "installing / verifying deps (setup) ..."
"$PY" turkish_medvoice.py setup --skip-smoke
say "preflight (doctor) ..."
"$PY" turkish_medvoice.py doctor || warn "doctor reported issues — see above."
say "roadmap: data -> train(align,s2s,medical) -> eval   [preset=$PRESET]"
"$PY" turkish_medvoice.py --preset "$PRESET" auto
say "=== wizard finished ==="
