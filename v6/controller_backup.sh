#!/usr/bin/env bash
# Stage snapshot plus all prerequisites needed to restore it. Run between stages.
set -euo pipefail
run_dir="$(realpath "${1:?Usage: bash v6/controller_backup.sh RUN_DIR STAGE REMOTE}")"
stage="${2:?Specify prepared, labels, training, evaluation, or comparison}"
remote="${3:?Specify gdrive:PSSN2/v6/controller-text-100x3}"
case "$stage" in
  prepared) members=(prepared) ;;
  labels) members=(prepared labels) ;;
  training) members=(prepared labels training) ;;
  evaluation) members=(prepared labels training evaluation) ;;
  comparison) members=(prepared labels training evaluation comparison) ;;
  *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
for member in "${members[@]}"; do test -d "$run_dir/$member"; done
for extra in logs source-v6; do
  if test -d "$run_dir/$extra"; then members+=("$extra"); fi
done
archive="$run_dir/$stage.tar.gz"
tar -C "$run_dir" -czf "$archive.tmp" "${members[@]}"
mv "$archive.tmp" "$archive"
rclone copyto "$archive" "${remote%/}/$stage.tar.gz" \
  --progress --checksum --contimeout 20s --timeout 2m --retries 3
rclone check "$run_dir" "$remote" --one-way --include "/$stage.tar.gz"
printf 'Verified Drive snapshot: %s/%s.tar.gz\n' "${remote%/}" "$stage"
