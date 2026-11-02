#!/usr/bin/env bash
# Run after labeling, training, evaluation, or unseen-test stage returns.
set -euo pipefail
run_dir="$(realpath "${1:?Usage: bash v5/controller_backup.sh RUN_DIR STAGE REMOTE}")"
stage="${2:?Specify labels, training, evaluation, or unseen-test}"
remote="${3:?Specify gdrive:PSSN2/v5/controller-100x3}"
case "$stage" in labels|training|evaluation|unseen-test) ;; *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;; esac
test -d "$run_dir/$stage"
archive="$run_dir/$stage.tar.gz"
members=("$stage")
for extra in logs source-v5; do
  if test -d "$run_dir/$extra"; then members+=("$extra"); fi
done
tar -C "$run_dir" -czf "$archive.tmp" "${members[@]}"
mv "$archive.tmp" "$archive"
rclone copyto "$archive" "${remote%/}/$stage.tar.gz" \
  --progress --checksum --contimeout 20s --timeout 2m --retries 3
rclone check "$run_dir" "$remote" --one-way --include "/$stage.tar.gz"
printf 'Verified Drive backup: %s/%s.tar.gz\n' "${remote%/}" "$stage"
