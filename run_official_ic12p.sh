#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TIMEOUT="${TIMEOUT:-7200}"
RUN_ID="${RUN_ID:-official-$(date -u +%Y%m%dT%H%M%SZ)}"

UWRMAXSAT_BIN="${UWRMAXSAT_BIN:-$HOME/solver-build/uwrmaxsat/build/release/bin/uwrmaxsat}"

DATA_DIRS=(
  data_table03_origin
  data_table06_forb
  data_table07_fixed
  data_table08_prec
)

# Kích hoạt môi trường Python.
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
elif [[ -f "$ROOT/.venv-ubuntu/bin/activate" ]]; then
    source "$ROOT/.venv-ubuntu/bin/activate"
else
    echo "ERROR: không tìm thấy .venv hoặc .venv-ubuntu."
    exit 1
fi

# Kiểm tra source và dữ liệu.
if [[ ! -f "$ROOT/src/Main.py" || ! -f "$ROOT/src/ORG_new.py" ]]; then
    echo "ERROR: script phải được đặt tại thư mục gốc repository."
    exit 1
fi

for data_dir in "${DATA_DIRS[@]}"; do
    if [[ ! -d "$ROOT/$data_dir" ]]; then
        echo "ERROR: không tồn tại folder $data_dir"
        exit 1
    fi
done

# Kiểm tra UWrMaxSAT.
if [[ ! -x "$UWRMAXSAT_BIN" ]]; then
    echo "ERROR: không tìm thấy executable:"
    echo "  $UWRMAXSAT_BIN"
    exit 1
fi

export UWRMAXSAT_BIN
UWRMAXSAT_SHA256="$(sha256sum "$UWRMAXSAT_BIN" | awk '{print $1}')"
export UWRMAXSAT_SHA256

OUT="$ROOT/output/ic12p-$RUN_ID"

mkdir -p \
    "$OUT/logs" \
    "$OUT/main" \
    "$OUT/org"

# Lưu thông tin lần chạy.
{
    echo "run_id=$RUN_ID"
    echo "started_utc=$(date -u --iso-8601=seconds)"
    echo "root=$ROOT"
    echo "timeout_per_configuration=$TIMEOUT"
    echo "main_encoding_variant=imp12+"
    echo "main_solver=all"
    echo "main_domain_mode=both"
    echo "main_sat_backend=cadical"
    echo "main_maxsat_backend=uwrmaxsat"
    echo "uwrmaxsat_bin=$UWRMAXSAT_BIN"
    echo "uwrmaxsat_sha256=$UWRMAXSAT_SHA256"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo
    echo "=== GIT STATUS ==="
    git status --short 2>/dev/null || true
    echo
    echo "=== CPU ==="
    lscpu
    echo
    echo "=== MEMORY ==="
    free -h
    echo
    echo "=== DISK ==="
    df -h "$ROOT"
} > "$OUT/environment.txt"

printf "stage\tfolder\texit_code\tfinished_utc\n" > "$OUT/status.tsv"

run_stage() {
    local stage="$1"
    local folder="$2"
    shift 2

    local log_file="$OUT/logs/${stage}_${folder}.log"

    echo
    echo "============================================================"
    echo "START:  $stage / $folder"
    echo "UTC:    $(date -u --iso-8601=seconds)"
    echo "LOG:    $log_file"
    echo "============================================================"

    "$@" 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}

    printf "%s\t%s\t%s\t%s\n" \
        "$stage" \
        "$folder" \
        "$rc" \
        "$(date -u --iso-8601=seconds)" \
        >> "$OUT/status.tsv"

    if (( rc != 0 )); then
        echo
        echo "ERROR: $stage / $folder kết thúc với exit code $rc"
        echo "Xem log:"
        echo "  $log_file"
        return "$rc"
    fi

    echo "COMPLETED: $stage / $folder"
    return 0
}

echo "Output directory: $OUT"
echo "Timeout: $TIMEOUT giây cho mỗi configuration"
echo "UWrMaxSAT: $UWRMAXSAT_BIN"

# ============================================================
# Giai đoạn 1: ORG_new baseline
# Một run cho mỗi canonical instance.
# ============================================================

for data_dir in "${DATA_DIRS[@]}"; do
    run_stage "org" "$data_dir" \
        python -u src/ORG_new.py \
            --data-dir "$data_dir" \
            --timeout "$TIMEOUT" \
            --uwrmaxsat-bin "$UWRMAXSAT_BIN" \
            --uwrmaxsat-sha256 "$UWRMAXSAT_SHA256" \
            --csv "$OUT/org/${data_dir}_org_new.csv" \
            --excel-dir "$OUT/org/excel_${data_dir}"

    rc=$?
    if (( rc != 0 )); then
        exit "$rc"
    fi
done

# ============================================================
# Giai đoạn 2: Main — chỉ IC12+
#
# Không truyền precedence flags:
# - instance không precedence: pairwise + direct
# - instance có precedence: đủ bốn P×G
# ============================================================

for data_dir in "${DATA_DIRS[@]}"; do
    run_stage "main" "$data_dir" \
        python -u src/Main.py \
            --data-dir "$data_dir" \
            --solver all \
            --maxsat-backend uwrmaxsat \
            --uwrmaxsat-bin "$UWRMAXSAT_BIN" \
            --uwrmaxsat-sha256 "$UWRMAXSAT_SHA256" \
            --sat-backend cadical \
            --domain-mode both \
            --encoding-variant 'imp12+' \
            --timeout "$TIMEOUT" \
            --csv "$OUT/main/${data_dir}_aggregate.csv" \
            --long-csv "$OUT/main/${data_dir}_detailed.csv" \
            --excel-dir "$OUT/main/excel_${data_dir}" \
            --verbose

    rc=$?
    if (( rc != 0 )); then
        exit "$rc"
    fi
done

echo "finished_utc=$(date -u --iso-8601=seconds)" \
    >> "$OUT/environment.txt"

tar \
    -C "$ROOT/output" \
    -czf "$ROOT/output/ic12p-$RUN_ID.tar.gz" \
    "ic12p-$RUN_ID"

echo
echo "============================================================"
echo "BENCHMARK HOÀN THÀNH"
echo "Results: $OUT"
echo "Archive: $ROOT/output/ic12p-$RUN_ID.tar.gz"
echo "============================================================"
