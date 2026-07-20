#!/usr/bin/env bash
# =============================================================================
#  turkish-medvoice RUN WIZARD
#  One command sets up and runs the whole native Turkish voice-to-voice pipeline:
#    validate -> venv -> clean previous -> fetch trainer -> detect assets
#    -> size to disk -> start OmniVoice TTS -> setup -> doctor -> data
#    -> train(align,s2s,medical) -> eval
#
#  SAFETY MODEL: this script deletes files under $WORK, which lives on a mount
#  shared with unrelated projects. It will ONLY clean a directory that carries its
#  own marker (.medvoice-workspace) or is recognisably a medvoice workspace. A
#  mistyped WORK aborts instead of deleting someone else's project.
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
VENV=${VENV:-/data/venv-medvoice}            # training venv (torch 2.8.0+cu128)
OMNI_VENV=${OMNI_VENV:-/data/venv-omni}      # TTS venv — MUST stay separate (see below)
PORT_TTS=${PORT_TTS:-8133}
TTS_WAIT=${TTS_WAIT:-300}                     # seconds to wait for the TTS server
PRESET=${PRESET:-standard}
RAW_URL=${RAW_URL:-https://raw.githubusercontent.com/ArioMoniri/turso/main/scripts/turkish_medvoice.py}
MARKER_NAME=.medvoice-workspace

# resource ceilings, percent of capacity. WARN = prune, STOP = halt the run.
DISK_WARN=${DISK_WARN:-85};  DISK_STOP=${DISK_STOP:-93}
RAM_STOP=${RAM_STOP:-95}
VRAM_WARN=${VRAM_WARN:-92}
RESERVE_GB=${RESERVE_GB:-30}                 # keep free for models/cache/checkpoints

CLEAN=soft; ASSUME_YES=0; DO_RUN=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      [[ $# -ge 2 && "$2" != --* ]] || { echo "--clean needs a value: soft|full|all" >&2; exit 2; }
      CLEAN="$2"; shift 2 ;;
    --preset)
      [[ $# -ge 2 && "$2" != --* ]] || { echo "--preset needs a value: smoke|standard|hardcore" >&2; exit 2; }
      PRESET="$2"; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --no-run) DO_RUN=0; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$CLEAN" in soft|full|all) ;; *) echo "invalid --clean '$CLEAN' (soft|full|all)" >&2; exit 2 ;; esac
case "$PRESET" in smoke|standard|hardcore) ;; *) echo "invalid --preset '$PRESET' (smoke|standard|hardcore)" >&2; exit 2 ;; esac

say()  { printf '\033[1;36m[wizard]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[wizard]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[wizard] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------- 1. path safety (identity)
# A depth/blacklist check cannot tell /data/medvoice from /data/briefer, so the
# real gate is a marker file this wizard owns. Existing medvoice workspaces are
# adopted automatically; anything else non-empty aborts.
validate_paths() {
  local p
  for p in WORK VENV OMNI_VENV; do
    [[ -n "${!p:-}" ]]  || die "$p is empty."
    [[ "${!p}" = /* ]]  || die "$p must be an absolute path (got '${!p}')."
  done
  # resolve symlinks so a symlinked WORK cannot redirect rm -rf into another project
  WORK=$(readlink -f "$WORK" 2>/dev/null || echo "$WORK")
  VENV=$(readlink -f "$VENV" 2>/dev/null || echo "$VENV")
  OMNI_VENV=$(readlink -f "$OMNI_VENV" 2>/dev/null || echo "$OMNI_VENV")
  SES=$(readlink -f "$SES"  2>/dev/null || echo "$SES")

  for p in "$WORK" "$VENV" "$OMNI_VENV"; do
    case "${p%/}" in
      ""|"/"|"/data"|"/root"|"/home"|"/usr"|"/etc"|"/var"|"/opt"|"/srv"|"/mnt"|"/tmp"|"/boot")
        die "refusing to use '$p' — that is a system/mount root, not a workspace." ;;
    esac
    [[ $(awk -F/ '{c=0; for(i=1;i<=NF;i++) if($i!="") c++; print c}' <<<"${p%/}") -ge 2 ]] \
      || die "'$p' is too shallow; use something like /data/medvoice."
  done
  [[ "${WORK%/}" != "${SES%/}" ]] || die "WORK == SES ('$WORK') — that would delete your model assets."
  case "${WORK%/}" in "${SES%/}"/*) die "WORK ('$WORK') is inside SES — cleaning would hit your assets." ;; esac
  case "${VENV%/}" in "${WORK%/}"/*) die "VENV ('$VENV') is inside WORK — cleaning would delete your venv." ;; esac
  case "${OMNI_VENV%/}" in "${WORK%/}"/*) die "OMNI_VENV ('$OMNI_VENV') is inside WORK — cleaning would delete it." ;; esac

  # IDENTITY GATE: only ever clean a directory this wizard created or adopted.
  local marker="${WORK%/}/$MARKER_NAME"
  if [[ -d "$WORK" && ! -e "$marker" ]]; then
    if [[ -f "${WORK%/}/turkish_medvoice.py" || -f "${WORK%/}/roadmap_state.json" \
          || -f "${WORK%/}/data/align.jsonl" ]]; then
      say "adopting existing medvoice workspace at $WORK"
    elif [[ -n "$(ls -A "$WORK" 2>/dev/null)" ]]; then
      die "WORK='$WORK' is non-empty and carries no $MARKER_NAME marker — refusing to
  clean a directory this wizard did not create. If this really is your medvoice
  workspace, adopt it deliberately:   touch '$marker'"
    fi
  fi
  mkdir -p "$WORK" || die "cannot create WORK='$WORK'."
  : >"$marker"     || die "cannot write workspace marker '$marker'."
  say "workspace: $WORK (marker ok)"
}

# ------------------------------------------------------- 2. resource helpers
disk_pct()     { df -P  "$1" 2>/dev/null | awk 'NR==2{gsub(/%/,"",$5); print $5+0}'; }
disk_free_gb() { df -PBG "$1" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4+0}'; }
# MemAvailable already discounts reclaimable page cache, so this is true pressure.
ram_pct()      { free 2>/dev/null | awk '/^Mem:/{printf "%d", ($2-$7)*100/$2}'; }
vram_pct()     {
  if ! command -v nvidia-smi >/dev/null 2>&1; then echo 0; return; fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' 'NR==1{printf "%d", $1*100/$2}' || echo 0
}

# Free space WITHOUT destroying anything the run still needs. Deliberately does
# NOT touch *.jsonl.tmp (the trainer's in-flight atomic manifest writes) and only
# sweeps OUR hf cache, never the shared ~/.cache/huggingface used by other projects.
prune_safe() {
  find "${HF_HOME:-$WORK/hf_cache}" -name '*.incomplete' -delete 2>/dev/null || true
  rm -rf "${WORK:?}/bench_results"/* 2>/dev/null || true
  find "$WORK/logs" -maxdepth 2 -name '*.log' -size +512M -exec truncate -s 64M {} \; 2>/dev/null || true
  command -v pip >/dev/null 2>&1 && "$PY" -m pip cache purge >/dev/null 2>&1 || true
  sync || true
}

# ------------------------------------------------- 3. background resource cop
start_watchdog() {
  local logf="$WORK/logs/watchdog.log"
  mkdir -p "$WORK/logs"
  (
    set +e                                  # a probe failure must never kill the cop
    seen=0
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
      [ "$r" -ge 85 ] && echo "$(date +%H:%M:%S) ram ${r}% (MemAvailable-based; page cache already discounted)" >>"$logf"
      [ "$v" -ge "$VRAM_WARN" ] && echo "$(date +%H:%M:%S) vram ${v}% (trainer shrinks max_seq_len itself)" >>"$logf"

      # about to max out -> stop the run cleanly rather than die from ENOSPC/OOM
      if [ "$d" -ge "$DISK_STOP" ] || [ "$r" -ge "$RAM_STOP" ]; then
        echo "$(date +%H:%M:%S) CRITICAL disk=${d}% ram=${r}% -> STOPPING the run" >>"$logf"
        pkill -INT -f "$WORK/turkish_medvoice.py" 2>/dev/null   # SIGINT: lets it checkpoint
        sleep 30
        pkill -9   -f "$WORK/turkish_medvoice.py" 2>/dev/null
        exit 0
      fi

      # Exit only AFTER we have seen the trainer and it then disappears, so the
      # cop can never disarm itself during the long setup/download phase.
      if pgrep -f "$WORK/turkish_medvoice.py" >/dev/null 2>&1; then
        seen=1; gone=0
      elif [ "$seen" -eq 1 ]; then
        gone=$((gone+1))
        [ "$gone" -ge 3 ] && { echo "$(date +%H:%M:%S) trainer finished -> watchdog exit" >>"$logf"; exit 0; }
      fi
    done
  ) &
  WATCHDOG_PID=$!
  say "resource watchdog running (pid $WATCHDOG_PID) -> $logf"
  trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT
}

# ---------------------------------------------------------- 4. venv bootstrap
ensure_venv() {
  PY="$VENV/bin/python"
  if [[ -x "$PY" ]] && "$PY" -c 'import sys' >/dev/null 2>&1; then say "venv: $VENV"; return 0; fi
  [[ -e "$VENV" && ! -d "$VENV" ]] && die "$VENV exists but is not a directory."

  local base; base=$(command -v python3) || die "no python3 on PATH to build a venv with."
  if [[ -d "$VENV" ]]; then
    warn "no working interpreter at $PY ($(ls -A "$VENV" 2>/dev/null | wc -l | tr -d ' ') entries) — repairing."
  fi
  say "creating venv at $VENV (base: $base) ..."
  if ! "$base" -m venv "$VENV" >/dev/null 2>&1; then
    warn "python3 -m venv failed — installing python3-venv and retrying ..."
    if command -v apt-get >/dev/null 2>&1; then
      DEBIAN_FRONTEND=noninteractive apt-get update -qq </dev/null >/dev/null 2>&1 || true
      DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-dev </dev/null >/dev/null 2>&1 || true
    fi
    if ! "$base" -m venv "$VENV" >/dev/null 2>&1; then
      warn "recreating the venv from scratch (--clear wipes $VENV) ..."
      "$base" -m venv --clear "$VENV" || die "could not create a venv at $VENV"
    fi
  fi
  PY="$VENV/bin/python"
  [[ -x "$PY" ]] || die "venv created but $PY is missing."
  "$PY" -m pip install -q --upgrade pip setuptools wheel 2>/dev/null || warn "pip self-upgrade failed (continuing)."
  say "venv ready: $PY ($("$PY" -V 2>&1))"
}

# ------------------------------------------------------- 4b. TTS venv (separate)
# The `omnivoice` package pulls torch 2.13+cu130, which is UNUSABLE on this box's
# CUDA-12.8 driver and would destroy the training venv's torch 2.8.0+cu128. The
# asset README is explicit about this ("yeni omnivoice torch 2.13-cu130 ceker,
# GERI ZORLA!"), which is why TTS gets its own venv and we talk to it over HTTP.
ensure_omni_venv() {
  OMNI_PY="$OMNI_VENV/bin/python"
  if [[ -x "$OMNI_PY" ]] && "$OMNI_PY" -c 'import omnivoice' >/dev/null 2>&1; then
    say "TTS venv: $OMNI_VENV (omnivoice present)"; return 0
  fi
  say "building the separate OmniVoice TTS venv at $OMNI_VENV ..."
  say "  (kept apart from $VENV because omnivoice would drag torch to cu130)"
  local base
  base=$(command -v python3.11 || command -v python3) || die "no python3 to build the TTS venv."
  if [[ ! -x "$OMNI_PY" ]]; then
    "$base" -m venv "$OMNI_VENV" >/dev/null 2>&1 || die "cannot create TTS venv at $OMNI_VENV"
  fi
  OMNI_PY="$OMNI_VENV/bin/python"
  "$OMNI_PY" -m pip install -q --upgrade pip wheel >/dev/null 2>&1 || true
  say "  installing omnivoice + fastapi + uvicorn ..."
  "$OMNI_PY" -m pip install -q omnivoice fastapi "uvicorn[standard]" \
    || die "omnivoice install failed in $OMNI_VENV"
  # README step 3: force torch BACK to cu128 after omnivoice drags in cu130
  say "  re-pinning torch 2.8.0+cu128 in the TTS venv (omnivoice pulls cu130) ..."
  "$OMNI_PY" -m pip install -q --force-reinstall torch==2.8.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128 \
    || warn "torch re-pin failed in $OMNI_VENV — TTS may not see the GPU."
  "$OMNI_PY" -c 'import omnivoice, torch; print("  omnivoice ok | torch", torch.__version__, "| cuda", torch.cuda.is_available())' \
    || die "TTS venv still cannot import omnivoice."
  say "TTS venv ready: $OMNI_PY"
}

# ------------------------------------------------------------ 5. asset detect
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

  # OmniVoice TTS: newest-by-name that actually looks like a model dir
  OMNI_DIR=""
  while read -r d; do
    [[ -z "$d" ]] && continue
    if [[ -f "$d/config.json" ]] && has_hf_weights "$d"; then OMNI_DIR="$d"; break; fi
  done < <(find "$SES" -maxdepth 1 -type d -name 'omnivoice*' | sort -Vr)
  [[ -n "$OMNI_DIR" ]] || die "no usable omnivoice* model dir under $SES (need config.json + weights)."

  # explicit WHISPER=/path (or a pre-set TMV_WHISPER_CKPT) always wins
  WHISPER_DIR="${WHISPER:-${TMV_WHISPER_CKPT:-}}"
  if [[ -n "$WHISPER_DIR" ]]; then
    say "whisper pinned by env -> $WHISPER_DIR"
    if [[ -d "$WHISPER_DIR" ]] && ! is_hf_whisper "$WHISPER_DIR"; then
      warn "pinned $WHISPER_DIR is NOT an HF transformers Whisper dir — the trainer will"
      warn "  fall back to the base model. Unset WHISPER/TMV_WHISPER_CKPT to auto-detect."
    fi
  else
    while read -r d; do
      [[ -z "$d" ]] && continue
      if is_hf_whisper "$d"; then WHISPER_DIR="$d"; break; fi
    done < <( { find "$SES" -maxdepth 1 -type d -name 'whisper*hf*';
                find "$SES" -maxdepth 1 -type d -name 'whisper*' ! -name '*hf*'; } 2>/dev/null )
  fi
  if [[ -z "$WHISPER_DIR" ]]; then
    WHISPER_DIR="openai/whisper-large-v3-turbo"
    warn "no HF-format Whisper under $SES — a CTranslate2 export CANNOT be a training"
    warn "  encoder, so using BASE $WHISPER_DIR. You lose your Turkish fine-tune's encoder."
  fi

  REF_WAV="$SES/ref/emin.wav"; REF_TXT="$SES/ref/emin.txt"; APP_DIR="$SES/app"
  say "assets detected:"
  say "  whisper encoder : $WHISPER_DIR"
  say "  omnivoice TTS   : $OMNI_DIR"
  say "  ref voice       : $REF_WAV"
  say "  tts app dir     : $APP_DIR"
}

# ------------------------------------------------- 6. size the run to the disk
autoscale_to_disk() {
  local free usable max_clips want
  free=$(disk_free_gb "$WORK"); free=${free:-0}
  usable=$(( free - RESERVE_GB ))
  say "free disk at $WORK: ${free}GB (reserve ${RESERVE_GB}GB) -> ${usable}GB for speech synthesis"
  [ "$usable" -lt 3 ] && die "only ${free}GB free — free space, lower RESERVE_GB, or use --clean full."
  max_clips=$(( usable * 1024 * 4 ))                 # ~0.25MB per clip
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

# ------------------------------------------------------------- 7. clean phase
clean_previous() {
  local -a targets=(
    "$WORK/checkpoints" "$WORK/bench_results" "$WORK/logs"
    "$WORK/roadmap_state.json" "$WORK/vocab_ext.json"
  )
  [[ "$CLEAN" == "full" || "$CLEAN" == "all" ]] && targets+=("$WORK/data")
  [[ "$CLEAN" == "all" ]] && targets+=("$WORK/hf_cache")

  local found=0 t
  for t in "${targets[@]}"; do [[ -e "$t" ]] && found=1; done
  if [ "$found" -eq 0 ]; then say "clean ($CLEAN): nothing from a previous run."; return 0; fi

  say "clean mode '$CLEAN' — these are from previous runs and all regenerable:"
  for t in "${targets[@]}"; do
    [[ -e "$t" ]] && printf '    %-44s %s\n' "$t" "$(du -sh "$t" 2>/dev/null | cut -f1)"
  done
  say "NOT touched: $SES, $VENV, and everything outside $WORK."

  if [ "$ASSUME_YES" -eq 0 ]; then
    if [ ! -t 0 ]; then
      die "stdin is not a terminal, so the delete confirmation cannot be shown.
  Re-run with --yes to confirm non-interactively, or run it in a tmux/ssh shell."
    fi
    printf '\033[1;33m[wizard]\033[0m delete the above? [y/N] '
    read -r ans || die "no answer read; aborting without deleting."
    [[ "$ans" =~ ^[Yy]$ ]] || die "aborted by user (nothing deleted)."
  fi

  # stop only OUR trainer, and give it a chance to checkpoint first
  if pgrep -f "$WORK/turkish_medvoice.py" >/dev/null 2>&1; then
    say "stopping a running trainer from this workspace (SIGINT, 15s grace) ..."
    pkill -INT -f "$WORK/turkish_medvoice.py" 2>/dev/null || true
    sleep 15
    pkill -9   -f "$WORK/turkish_medvoice.py" 2>/dev/null || true
  fi
  for t in "${targets[@]}"; do rm -rf "$t"; done
  find "$WORK" -maxdepth 4 \( -name '*.tmp' -o -name '*.jsonl.tmp' \) -delete 2>/dev/null || true
  say "clean done."
}

# ------------------------------------------------------------ 8. TTS server
tts_up() { curl -sS -o /dev/null --max-time 4 "http://127.0.0.1:$PORT_TTS/" >/dev/null 2>&1; }

start_tts() {
  TTS_PID_FILE="$WORK/$MARKER_NAME.tts.pid"      # outside logs/, which clean wipes
  if tts_up; then say "OmniVoice TTS already up on :$PORT_TTS (reusing it)"; return 0; fi
  # reap an orphan from an aborted run that is holding VRAM but not serving
  if [[ -f "$TTS_PID_FILE" ]]; then
    local old; old=$(cat "$TTS_PID_FILE" 2>/dev/null || echo "")
    if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
      warn "orphan TTS pid $old is alive but not serving — terminating it."
      kill "$old" 2>/dev/null || true; sleep 5; kill -9 "$old" 2>/dev/null || true
    fi
    rm -f "$TTS_PID_FILE"
  fi
  if [[ ! -f "$APP_DIR/omnivoice_server.py" ]]; then
    warn "no $APP_DIR/omnivoice_server.py — cannot start TTS automatically."
    return 0
  fi
  say "starting OmniVoice TTS ($(basename "$OMNI_DIR")) on :$PORT_TTS ..."
  mkdir -p "$WORK/logs"
  OMNI_MODEL="$OMNI_DIR" OMNI_REF="$REF_WAV" OMNI_REFTXT_FILE="$REF_TXT" OMNI_LANG=tr \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "$OMNI_PY" -m uvicorn omnivoice_server:app --app-dir "$APP_DIR" \
        --host 127.0.0.1 --port "$PORT_TTS" >>"$WORK/logs/tts_server.log" 2>&1 &
  echo $! >"$TTS_PID_FILE"
  # Watch the PROCESS as well as the port: a server that crashed on import should
  # report its traceback at once, not after the full timeout with no explanation.
  local waited=0 pid
  pid=$(cat "$TTS_PID_FILE" 2>/dev/null || echo "")
  while [ "$waited" -lt "$TTS_WAIT" ]; do
    sleep 3; waited=$((waited + 3))
    if tts_up; then say "TTS is up after ${waited}s (pid $pid)."; return 0; fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      warn "TTS process died after ${waited}s. Tail of $WORK/logs/tts_server.log:"
      tail -15 "$WORK/logs/tts_server.log" >&2 2>/dev/null || true
      warn "  data would be built TEXT-ONLY and the run will stop at the speech gate."
      return 0
    fi
  done
  warn "TTS did not come up in ${TTS_WAIT}s — see $WORK/logs/tts_server.log"
  warn "  data would be built TEXT-ONLY and the run will stop at the speech gate."
  return 0
}

# ===================================================================== MAIN ==
say "=== turkish-medvoice run wizard ==="
[[ $EUID -eq 0 ]] || warn "not root: apt-get fallbacks in venv setup will be skipped."

validate_paths
ensure_venv
ensure_omni_venv
clean_previous
mkdir -p "$WORK/logs"
detect_assets

say "fetching the latest trainer from GitHub ..."
# atomic + bounded: never clobber a good copy with a truncated/stalled download
curl -fsSL --max-time 300 --retry 3 --retry-delay 5 "$RAW_URL" -o "$WORK/turkish_medvoice.py.part" \
  || die "download failed: $RAW_URL"
head -1 "$WORK/turkish_medvoice.py.part" | grep -q '^#!' \
  || die "downloaded trainer does not look like a script — refusing to run it."
"$PY" -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$WORK/turkish_medvoice.py.part" \
  || die "downloaded trainer failed to parse — refusing to run a truncated file."
mv -f "$WORK/turkish_medvoice.py.part" "$WORK/turkish_medvoice.py"
say "trainer: $WORK/turkish_medvoice.py ($(wc -l <"$WORK/turkish_medvoice.py") lines)"

# ---- the full environment the trainer needs --------------------------------
export TMV_ROOT="$SES" TMV_WORK="$WORK" TMV_VENV="$VENV"
export TMV_WHISPER_CKPT="$WHISPER_DIR"
export TMV_OMNI_MODEL="$OMNI_DIR" TMV_OMNI_REF="$REF_WAV" TMV_OMNI_REFTXT="$REF_TXT"
export TMV_TTS_MODE=http TMV_OMNI_URL="http://127.0.0.1:$PORT_TTS/v1/audio/speech"
export TMV_OMNI_VENV_PY="$OMNI_PY" TMV_OMNI_APP_DIR="$APP_DIR"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# VRAM ceilings. These OVERRIDE the preset (env wins in the trainer) — say so out
# loud rather than silently downgrading a hardcore run.
if [[ -z "${TMV_MAXSEQ:-}" && "$PRESET" == "hardcore" ]]; then
  say "note: capping TMV_MAXSEQ=2048 for VRAM safety (hardcore's default is 4096)."
  say "      export TMV_MAXSEQ=4096 before running to keep the preset value."
fi
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
  say "  (the TTS server on :$PORT_TTS is left running for that run)"
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
