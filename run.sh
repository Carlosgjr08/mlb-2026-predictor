#!/usr/bin/env bash
# Daily run: catch up on results, retrain, and show the picks.
#   bash run.sh              -> all upcoming games
#   bash run.sh today        -> just today's slate
#   bash run.sh tomorrow     -> just tomorrow's slate
#   bash run.sh 2026-07-25   -> a specific date
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate || { echo "No .venv here — set it up first."; exit 1; }

echo "==> Fetching latest results (this takes a minute)..."
python3 -m mlb_predictor fetch
echo "==> Retraining the model on the new data..."
python3 -m mlb_predictor train
echo

case "${1:-}" in
  today)    day=$(date +%Y-%m-%d) ;;
  tomorrow) day=$(date -v+1d +%Y-%m-%d 2>/dev/null || date -d tomorrow +%Y-%m-%d) ;;
  "")       day="" ;;
  *)        day="$1" ;;
esac

if [ -n "$day" ]; then
  echo "==> Picks for $day:"
  python3 -m mlb_predictor predict --upcoming --date "$day"
else
  echo "==> Next upcoming picks:"
  python3 -m mlb_predictor predict --upcoming --limit 30
fi
