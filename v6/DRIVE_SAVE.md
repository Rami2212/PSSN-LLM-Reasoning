# Save and restore V6 with Google Drive

Drive destination: **My Drive / PSSN2 / v6 / controller-text-100x3**.
Use an rclone Google Drive remote called `gdrive`. This is a server backup
workflow; it does not require mounting Drive or copying model weights.

## 1. Configure the remote on the headless server

If this command succeeds, reuse the existing remote:

```bash
rclone lsd gdrive:PSSN2
```

Otherwise:

```bash
rclone config
```

Choose new remote `n`, name `gdrive`, storage `drive`, blank client ID and secret,
scope `1`, blank service-account file, advanced config `n`. Answer `n` when
asked to use a browser on the headless server. Run the exact `rclone authorize
"drive" ...` command it prints on your Windows PC with rclone installed.
Complete browser login and paste the returned token into the server's
`config_token>` prompt. Finish and save the remote.

Use the exact authorization command shown by your installed version. Official
instructions: [rclone headless setup](https://rclone.org/remote_setup/).

```bash
export V6_RUN=/root/pssn-work/PSSN2/v6/controller-text-100x3
export V6_REMOTE=gdrive:PSSN2/v6/controller-text-100x3
mkdir -p "$V6_RUN"
rclone mkdir "$V6_REMOTE"
rclone lsd gdrive:PSSN2/v6
```

## 2. Snapshots after each stage

The automatic [pipeline](run_pipeline.sh) calls these between completed stages.
When running stages manually, call the matching command after each completes:

```bash
bash v6/controller_backup.sh "$V6_RUN" prepared "$V6_REMOTE"
bash v6/controller_backup.sh "$V6_RUN" labels "$V6_REMOTE"
bash v6/controller_backup.sh "$V6_RUN" training "$V6_REMOTE"
bash v6/controller_backup.sh "$V6_RUN" evaluation "$V6_REMOTE"
bash v6/controller_backup.sh "$V6_RUN" comparison "$V6_REMOTE"
```

Run the lines as their stages become available, not all before training. The
script archives the requested stage **and all preceding prerequisite folders**,
plus `logs` and `source-v6` when present. `comparison.tar.gz` therefore contains
the full experiment evidence. It creates the archive locally, uploads with
`rclone copyto --checksum` and checks the remote snapshot. A failed upload
leaves local data intact; retry the same command. Completed snapshots with the
same name are updated by the next successful copy.

Do not snapshot a stage while it is actively writing. To back up an unfinished
long stage, stop it with Ctrl+C, wait for the process to exit and the lock to
release, then snapshot the available stage directory. It may contain partial
evidence; inspect `run_status.json` after restore before calling it completed.
Resume on the same recorded environment with matching source and options.

Snapshots include prompts, reference summaries, generation token IDs, saved
outputs, labels, the fitted controller, validation selection, evaluation,
comparison results, environment records and source. They exclude model caches,
the virtual environment and Hugging Face/rclone credentials. Treat the remote
as your private research backup and respect original data/model terms.

## 3. Restore on another server

Recreate the Python/ROCm environment from [SERVER_SETUP.md](SERVER_SETUP.md).
Use a new empty experiment directory, then download **one latest available
snapshot**, such as the complete comparison:

```bash
export V6_RUN=/root/pssn-work/PSSN2/v6/controller-text-100x3-restored
export V6_REMOTE=gdrive:PSSN2/v6/controller-text-100x3
mkdir -p "$V6_RUN"
rclone copyto "$V6_REMOTE/comparison.tar.gz" "$V6_RUN/comparison.tar.gz" --progress --checksum
tar -tzf "$V6_RUN/comparison.tar.gz" | head -n 20
tar -xzf "$V6_RUN/comparison.tar.gz" -C "$V6_RUN"
```

For an earlier checkpoint replace `comparison.tar.gz` with `labels.tar.gz`,
`training.tar.gz` or another available snapshot. Each contains its prerequisites;
you do not need to extract every older archive over it.

Inspect results without generation:

```bash
cat "$V6_RUN/comparison/comparison.md"
cat "$V6_RUN/training/training_summary.json"
```

To resume, the active repository `v6` source must match the archived `source-v6`
and its manifest hashes. Preserve the complete sibling directory structure
(`prepared`, `labels`, `training`, etc.). A different GPU or dependency version
must use a new comparison directory for timings; existing JSON/CSV reports
remain readable everywhere. A copied controller joblib must be trusted and
loaded with compatible scikit-learn/joblib versions.

## 4. Download the final archive to Windows

In Windows PowerShell, after the final snapshot exists on the server:

```powershell
$serverHost = 'YOUR_SERVER_IP'
scp "root@${serverHost}:/root/pssn-work/PSSN2/v6/controller-text-100x3/comparison.tar.gz" "$env:USERPROFILE\Downloads\v6-controller-text-100x3.tar.gz"
```

Alternatively use your Windows rclone remote to download directly from Drive:

```powershell
rclone copyto gdrive:PSSN2/v6/controller-text-100x3/comparison.tar.gz "$env:USERPROFILE\Downloads\v6-controller-text-100x3.tar.gz" --progress --checksum
```

The experiment files remain on the server and Drive after download.
