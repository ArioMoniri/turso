#!/usr/bin/env bash
# =============================================================================
#  run_cwm.sh — tmux launcher / wizard for the CWM-mini from-scratch trainer.
#
#  Self-sufficient: picks the biggest-free-disk data root, builds a venv,
#  fetches cwm.py, and runs `python cwm.py auto` inside a resurrect-able tmux
#  session. The Python side is self-healing/resumable/resource-guarded; this
#  script only sets up the box and supervises the tmux session.
#
#  QUICKSTART (server):
#     curl -fsSL https://raw.githubusercontent.com/ArioMoniri/turso/main/scripts/run_cwm.sh -o run_cwm.sh
#     bash run_cwm.sh start          # setup + launch `auto` in tmux 'cwm'
#     bash run_cwm.sh attach         # watch it
#     bash run_cwm.sh status         # resource + progress snapshot
#     bash run_cwm.sh stop           # stop the run (checkpoint is safe to resume)
#     bash run_cwm.sh start          # resume where it left off
#
#  SUBCOMMANDS: start | attach | stop | status | logs | clean [soft|all] |
#               smoke | bench | doctor | eval <stage> | shell
# =============================================================================
set -Eeuo pipefail

SESSION="${CWM_TMUX:-cwm}"
RAW_BASE="${CWM_RAW_BASE:-https://raw.githubusercontent.com/ArioMoniri/turso/main/scripts}"
PY_MIN="3.10"

log()  { printf '\033[0;36m[cwm]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[cwm][warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[0;31m[cwm][err]\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
#  Pick the data root: the writable candidate mount with the MOST free space.  #
# --------------------------------------------------------------------------- #
pick_root() {
  if [[ -n "${CWM_ROOT:-}" ]]; then echo "$CWM_ROOT"; return; fi
  local best="" best_free=0 c free
  for c in /data /workspace /mnt/data "$HOME"; do
    [[ -d "$c" ]] || continue
    if mkdir -p "$c/.cwm_wtest" 2>/dev/null; then rmdir "$c/.cwm_wtest" 2>/dev/null || true; else continue; fi
    free=$(df -Pk "$c" 2>/dev/null | awk 'NR==2{print $4+0}')
    if (( free > best_free )); then best_free=$free; best="$c"; fi
  done
  [[ -n "$best" ]] || best="$HOME"
  echo "$best/cwm"
}

ROOT="$(pick_root)"
export CWM_ROOT="$ROOT"
WORK="$(dirname "$ROOT")/cwm_work"        # venv + code live beside the data root
VENV="$WORK/venv"
CODE="$WORK/cwm.py"
MARKER="$WORK/.cwm-workspace"
mkdir -p "$WORK" "$ROOT"

# --------------------------------------------------------------------------- #
ensure_tmux() { command -v tmux >/dev/null 2>&1 || die "tmux not installed (apt-get install -y tmux)"; }

ensure_python() {
  command -v python3 >/dev/null 2>&1 || die "python3 not found"
  local v; v=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
  awk -v a="$v" -v b="$PY_MIN" 'BEGIN{split(a,x,".");split(b,y,".");exit !(x[1]>y[1]||(x[1]==y[1]&&x[2]>=y[2]))}' \
    || die "python >= $PY_MIN required (have $v)"
}

ensure_venv() {
  if [[ ! -f "$VENV/bin/activate" ]]; then
    log "creating venv at $VENV"
    python3 -m venv "$VENV" || die "venv creation failed"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -c 'import torch' 2>/dev/null || {
    log "installing deps (first run; this is the slow part) ..."
    pip install -q --upgrade pip
    # torch first (respect a preinstalled CUDA build if the box already has one)
    python -c 'import torch' 2>/dev/null || pip install -q torch
    pip install -q numpy sentencepiece soundfile librosa "datasets>=2.18" huggingface_hub moshi einops
  }
  touch "$MARKER"
}

fetch_code() {
  # A local cwm.py next to this script wins (drop a commit-pinned copy here to
  # bypass the network entirely). Otherwise fetch FRESH — the raw.githubusercontent
  # "main" URL is CDN-cached ~5 min, so we hit the GitHub API (not cached) first
  # and fall back to a cache-busted raw URL. Pin a ref/commit with CWM_REF.
  local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "$here/cwm.py" ]]; then
    cp -f "$here/cwm.py" "$CODE"; log "using local cwm.py at $here/cwm.py"
    return
  fi
  local ref="${CWM_REF:-main}" cb="cb=$(date +%s)$$"
  if curl -fsSL -H "Accept: application/vnd.github.raw" \
       "https://api.github.com/repos/ArioMoniri/turso/contents/scripts/cwm.py?ref=${ref}&${cb}" \
       -o "$CODE.tmp" 2>/dev/null && [[ -s "$CODE.tmp" ]]; then
    mv -f "$CODE.tmp" "$CODE"; log "fetched cwm.py via GitHub API (ref=${ref}, fresh)"
    return
  fi
  log "API fetch failed; trying cache-busted raw ($RAW_BASE)"
  curl -fsSL "$RAW_BASE/cwm.py?${cb}" -o "$CODE.tmp" && [[ -s "$CODE.tmp" ]] && mv -f "$CODE.tmp" "$CODE" \
    || die "download failed. Set CWM_REF=<commit>, or drop cwm.py next to run_cwm.sh ($here/cwm.py)."
}

hf_login_hint() {
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    warn "No HF token in env. Gated/large datasets may fail. Export HF_TOKEN=... if needed."
  fi
}

preflight() {
  ensure_tmux; ensure_python; ensure_venv; fetch_code; hf_login_hint
  export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  log "root=$ROOT  work=$WORK"
  local free; free=$(df -Ph "$ROOT" | awk 'NR==2{print $4}')
  log "free disk @ root: $free"
}

# --------------------------------------------------------------------------- #
cmd_in_tmux() {   # cmd_in_tmux "<python subcommand args>"
  ensure_tmux
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    warn "tmux session '$SESSION' already exists. attach: bash run_cwm.sh attach"
    return 0
  fi
  local inner="source '$VENV/bin/activate'; export CWM_ROOT='$ROOT' HF_HOME='${HF_HOME:-$ROOT/hf_cache}' \
PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}'; \
cd '$WORK'; echo '[cwm] launching: python cwm.py $*'; \
python cwm.py $*; ec=\$?; echo \"[cwm] exited code \$ec\"; echo 'window stays open; detach with Ctrl-b d'; exec bash"
  tmux new-session -d -s "$SESSION" "$inner"
  log "started tmux '$SESSION' running: python cwm.py $*"
  log "attach:  bash run_cwm.sh attach     |  status: bash run_cwm.sh status"
}

action="${1:-start}"; shift || true
case "$action" in
  start)   preflight; cmd_in_tmux "auto" ;;
  smoke)   preflight; cmd_in_tmux "smoke" ;;
  bench)   preflight; cmd_in_tmux "bench" ;;
  eval)    preflight; cmd_in_tmux "eval --stage ${1:-b3}" ;;
  doctor)  preflight; source "$VENV/bin/activate"; cd "$WORK"; python cwm.py doctor ;;
  attach)  ensure_tmux; tmux attach -t "$SESSION" || die "no session '$SESSION' (start it first)" ;;
  stop)
    ensure_tmux
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      # SIGINT the python so it checkpoints, then kill the session
      tmux send-keys -t "$SESSION" C-c; sleep 3; tmux kill-session -t "$SESSION" || true
      log "stopped '$SESSION'. Checkpoint is safe — 'start' resumes."
    else warn "no session '$SESSION'"; fi ;;
  status)
    echo "== disk =="; df -Ph "$ROOT" | awk 'NR==1||NR==2'
    echo "== ram ==";  free -h 2>/dev/null | awk 'NR==1||NR==2' || vm_stat | head -5
    echo "== gpu =="; command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "no nvidia-smi"
    echo "== tmux =="; tmux has-session -t "$SESSION" 2>/dev/null && echo "session '$SESSION' RUNNING" || echo "session '$SESSION' not running"
    echo "== roadmap =="; [[ -f "$ROOT/roadmap_state.json" ]] && cat "$ROOT/roadmap_state.json" || echo "(no roadmap_state.json yet)"
    echo "== last metrics =="; ls -1t "$ROOT/logs"/metrics_*.jsonl.last 2>/dev/null | head -1 | xargs -r cat || echo "(none)"
    echo "== reports =="; ls -1 "$ROOT/logs"/REPORT_*.md 2>/dev/null || echo "(none)"
    echo "== alerts =="; tail -n 5 "$ROOT/logs/ALERT.md" 2>/dev/null || echo "(none)" ;;
  logs)    tail -n "${1:-120}" -f "$ROOT/logs/cwm.log" ;;
  clean)
    mode="${1:-soft}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    case "$mode" in
      soft) log "clean soft: tmp shards + logs only (keeps data/ckpt/tokenizer)"
            find "$ROOT" -name '*.tmp' -o -name '*.part' | xargs -r rm -f
            rm -f "$ROOT/logs/"*.log 2>/dev/null || true ;;
      all)  read -r -p "DELETE ALL of $ROOT (data+checkpoints+logs)? type 'yes': " a
            [[ "$a" == "yes" ]] && { rm -rf "$ROOT"; log "wiped $ROOT"; } || log "aborted" ;;
      *)    die "clean mode: soft|all" ;;
    esac ;;
  shell)   preflight; source "$VENV/bin/activate"; cd "$WORK"; exec bash ;;
  *) cat <<EOF
run_cwm.sh — CWM-mini launcher   (root: $ROOT)
  start           setup + run the full roadmap (auto) in tmux '$SESSION' [resumable]
  attach          attach to the tmux session
  status          disk/ram/gpu/roadmap/metrics/alerts snapshot
  stop            SIGINT (checkpoint) then kill the session
  logs [N]        tail -f the run log
  smoke | bench   run just the smoke / throughput gate
  doctor          preflight resource + dependency report
  eval <stage>    run eval for a trained stage (default b3)
  clean soft|all  soft = tmp+logs;  all = wipe the whole root (asks first)
  shell           drop into the venv shell at the work dir
EOF
    ;;
esac
