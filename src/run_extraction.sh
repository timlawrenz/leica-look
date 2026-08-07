#!/bin/bash
# run_extraction.sh — GPU-scheduled embedding extraction runner
# Launches with: setsid nohup bash src/run_extraction.sh > logs/extraction_runner.log 2>&1 &
set -euo pipefail

SCHEDULER="/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py"
GPU="4090"
JOB_ID="leica-embeddings-4"
PROJECT_DIR="/home/tim/source/activity/leica-look"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
RUNNER_LOG="${LOG_DIR}/extraction_runner_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$RUNNER_LOG") 2>&1

echo "=== GPU Extraction Runner ==="
echo "Started: $(date)"
echo "Job ID: $JOB_ID"
echo "GPU: $GPU"

# Step 1: Request
echo ""
echo "--- Step 1: Request GPU ---"
python3 "$SCHEDULER" request \
    --gpu "$GPU" \
    --project leica-look \
    --vram 20 \
    --duration 6h \
    --job-id "$JOB_ID" 2>&1
echo "Request submitted."

# Step 2: Poll until claimed (up to 30 min)
echo ""
echo "--- Step 2: Poll for GPU ---"
MAX_POLLS=36  # 36 × 30s = 18 min
POLL_COUNT=0
while [ $POLL_COUNT -lt $MAX_POLLS ]; do
    RESULT=$(python3 "$SCHEDULER" poll --gpu "$GPU" --job-id "$JOB_ID" 2>&1)
    echo "  [$POLL_COUNT] $RESULT"
    case "$RESULT" in
        *claimed*)
            echo "  ✅ GPU claimed!"
            break
            ;;
        *not_my_turn*|*gpu_busy*)
            echo "  Waiting... (30s)"
            sleep 30
            POLL_COUNT=$((POLL_COUNT + 1))
            ;;
        *gpu_unhealthy*|*thermal_cooldown*|*vram_insufficient*|*not_queued*|*scheduler_unavailable*)
            echo "  ❌ Fatal: $RESULT"
            exit 1
            ;;
        *)
            echo "  Unknown response, waiting..."
            sleep 30
            POLL_COUNT=$((POLL_COUNT + 1))
            ;;
    esac
done

if [ $POLL_COUNT -ge $MAX_POLLS ]; then
    echo "❌ Timed out waiting for GPU."
    exit 1
fi

# Step 3: Activate
echo ""
echo "--- Step 3: Activate ---"
python3 "$SCHEDULER" activate --gpu "$GPU" --job-id "$JOB_ID" --progress-unit model 2>&1
echo "Activated."

# Step 4: Start heartbeat loop in background
echo ""
echo "--- Step 4: Start heartbeat loop ---"
HEARTBEAT_LOG="${LOG_DIR}/heartbeat_${JOB_ID}.log"
(
    MODEL_COUNT=0
    while true; do
        sleep 240  # every 4 minutes
        python3 "$SCHEDULER" heartbeat \
            --gpu "$GPU" \
            --job-id "$JOB_ID" \
            --progress "$MODEL_COUNT" \
            --vram-used 20 2>&1 >> "$HEARTBEAT_LOG"
        echo "  [heartbeat] model_count=$MODEL_COUNT at $(date +%H:%M:%S)" >> "$HEARTBEAT_LOG"
    done
) &
HEARTBEAT_PID=$!
echo "Heartbeat PID: $HEARTBEAT_PID"

# Step 5: Run extraction
echo ""
echo "--- Step 5: Run extraction ---"
cd "$PROJECT_DIR"
EXTRACTION_LOG="${LOG_DIR}/extraction_$(date +%Y%m%d_%H%M).log"

export HF_HOME=/mnt/nas-ai-models/huggingface-cache
.venv/bin/python src/extract_embeddings.py 2>&1 | tee "$EXTRACTION_LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Extraction exited with code: $EXIT_CODE"

# Step 6: Stop heartbeats
kill $HEARTBEAT_PID 2>/dev/null || true
wait $HEARTBEAT_PID 2>/dev/null || true

# Step 7: Release
echo ""
echo "--- Step 7: Release GPU ---"
if [ $EXIT_CODE -eq 0 ]; then
    STATUS="completed"
else
    STATUS="failed"
fi
python3 "$SCHEDULER" release --gpu "$GPU" --job-id "$JOB_ID" --status "$STATUS" 2>&1
echo "Released ($STATUS)."

echo ""
echo "=== Runner finished: $(date) ==="
exit $EXIT_CODE
