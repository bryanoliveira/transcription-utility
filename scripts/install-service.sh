#!/usr/bin/env bash
# Install `transcribe serve` as a systemd service.
#
#   ./scripts/install-service.sh --port 8000 --model qwen3-1.7b --keep-alive 5
#   ./scripts/install-service.sh --system --run-as asr --host 0.0.0.0
#   ./scripts/install-service.sh --uninstall
#
# Defaults to a --user service (no root). Use --system for a machine-wide one.
set -euo pipefail

NAME=transcribe
MODEL=qwen3-1.7b
LANGUAGE=pt
HOST=127.0.0.1
PORT=8000
KEEP_ALIVE=5
EXTRA=""
SCOPE=user
RUN_AS=""
UNINSTALL=0
PRINT_ONLY=0
EXEC=""

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)       NAME=$2; shift 2 ;;
    --model)      MODEL=$2; shift 2 ;;
    --language)   LANGUAGE=$2; shift 2 ;;
    --host)       HOST=$2; shift 2 ;;
    --port)       PORT=$2; shift 2 ;;
    --keep-alive) KEEP_ALIVE=$2; shift 2 ;;
    --exec)       EXEC=$2; shift 2 ;;
    --extra)      EXTRA=$2; shift 2 ;;
    --system)     SCOPE=system; shift ;;
    --run-as)     RUN_AS=$2; shift 2 ;;
    --uninstall)  UNINSTALL=1; shift ;;
    --print)      PRINT_ONLY=1; shift ;;
    -h|--help)    sed -n '2,8p' "$0"; exit 0 ;;
    *)            die "unknown option $1" ;;
  esac
done

if [[ $SCOPE == user ]]; then
  SYSTEMCTL=(systemctl --user)
  JOURNAL=(journalctl --user)
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  WANTED_BY=default.target
else
  SYSTEMCTL=(sudo systemctl)
  JOURNAL=(sudo journalctl)
  UNIT_DIR=/etc/systemd/system
  WANTED_BY=multi-user.target
fi
UNIT="$UNIT_DIR/$NAME.service"

if [[ $UNINSTALL == 1 ]]; then
  "${SYSTEMCTL[@]}" disable --now "$NAME.service" 2>/dev/null || true
  if [[ $SCOPE == user ]]; then rm -f "$UNIT"; else sudo rm -f "$UNIT"; fi
  "${SYSTEMCTL[@]}" daemon-reload
  echo "removed $UNIT"
  exit 0
fi

# Find the transcribe entry point. A venv install is the normal case, and the
# unit needs an absolute path because systemd starts with a bare environment.
if [[ -z $EXEC ]]; then
  if [[ -n ${VIRTUAL_ENV:-} && -x "$VIRTUAL_ENV/bin/transcribe" ]]; then
    EXEC="$VIRTUAL_ENV/bin/transcribe"
  else
    EXEC=$(command -v transcribe || true)
  fi
fi
[[ -n $EXEC && -x $EXEC ]] || die "cannot find 'transcribe'. Activate your venv, or pass --exec /path/to/transcribe"

command -v ffmpeg >/dev/null || die "ffmpeg not on PATH — the service needs it to chunk audio"

# ffmpeg/ffprobe are shelled out to, so the unit must carry a usable PATH.
BIN_PATH="$(dirname "$EXEC"):$(dirname "$(command -v ffmpeg)"):/usr/local/bin:/usr/bin:/bin"

unit_text() {
  cat <<UNIT
[Unit]
Description=transcription-utility ASR API ($MODEL)
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
ExecStart=$EXEC serve --model $MODEL --language $LANGUAGE --host $HOST --port $PORT --keep-alive $KEEP_ALIVE $EXTRA
Environment=PATH=$BIN_PATH
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=5
# Model loads can be slow; don't let systemd shoot it during startup.
TimeoutStartSec=600
$( [[ -n $RUN_AS ]] && echo "User=$RUN_AS" )

[Install]
WantedBy=$WANTED_BY
UNIT
}

if [[ $PRINT_ONLY == 1 ]]; then
  unit_text
  exit 0
fi

if [[ $SCOPE == user ]]; then
  mkdir -p "$UNIT_DIR"
  unit_text > "$UNIT"
else
  sudo mkdir -p "$UNIT_DIR"
  unit_text | sudo tee "$UNIT" >/dev/null
fi

"${SYSTEMCTL[@]}" daemon-reload
"${SYSTEMCTL[@]}" enable --now "$NAME.service"

if [[ $SCOPE == user ]]; then
  # Without lingering, a --user service dies when the last session closes.
  loginctl enable-linger "$USER" 2>/dev/null \
    || echo "note: could not enable lingering; service stops when you log out"
fi

echo "installed $UNIT"
echo
"${SYSTEMCTL[@]}" --no-pager status "$NAME.service" | head -5 || true
cat <<EOF

  health:  curl http://$HOST:$PORT/health
  logs:    ${JOURNAL[*]} -u $NAME -f
  stop:    ${SYSTEMCTL[*]} stop $NAME
  remove:  $0 --uninstall --name $NAME$( [[ $SCOPE == system ]] && echo " --system" )
EOF
