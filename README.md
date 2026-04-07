# UFC-PoseEstimation

Fine-tuning Ultralytics YOLO pose models for MMA / UFC fighter keypoint estimation. Includes dataset preparation from bbox-only labels and training scripts.

## Google Colab

Upload [colab/MMA_Pose_YOLO_Training.ipynb](colab/MMA_Pose_YOLO_Training.ipynb) to Colab (GPU runtime). **Mount Google Drive**, set `DRIVE_DATASET_DIR` to the folder that contains `data.yaml` and your YOLO pose splits, and `DRIVE_OUTPUT_DIR` where runs should be saved. Training uses fast `/content` scratch; `best.pt`, metrics JSON, and plots are written to Drive.

## Local validation insights

After training, you can summarize **official val metrics** (mAP) and **pose-only** punch/kick heuristics on the val split, plus a Plotly **3D skeleton gallery** (2D keypoints in a 3D viewer — no metric depth):

```bash
python val_pose_insights.py --weights runs/pose/train/weights/best.pt --data mma_pose_dataset/data.yaml --out-dir insights_out
```

Outputs: `insights_out/val_insights_summary.json`, `insights_out/val_3d_gallery.html`. Strike/tactic fields are **heuristics** on model predictions, not labeled fight stats. Use `--skip-val` to skip `model.val()` if it is too slow.

Frame order uses the **first digit run** after `_mp4-` (e.g. `..._mp4-0162_jpg.rf.<hash>` → frame **162**). If no `_mp4-<digits>` appears in a stem, the script falls back to a **single flat sequence** for predict + heuristics. Install **plotly** for the HTML gallery; if no punch/kick peaks fire, the gallery still shows **neutral** poses unless you pass `--no-neutral-gallery`.

## Action recognition (LSTM + MMAction2)

The Colab notebook **§10–§12** builds **labeled clip manifests** and trains a small **LSTM** on sequences shaped `(T, 17, 2|3)` per `.npy` file. **You must assign class labels** (e.g. `jab`, `hook`) in the CSV; `unlabeled` rows are skipped by the LSTM cell.

**Manifest format** (header required):

| Column   | Description |
|----------|-------------|
| `clip_id` | Unique id (MMAction2 `frame_dir`) |
| `label`   | Class name or integer id |
| `npy_path`| Absolute path to `(T,17,2)` or `(T,17,3)` array |
| `split`   | `train` or `val` |
| `img_h`, `img_w` | Optional; default 720×1280 in MMAction2 pickle |

Example: [schemas/action_clip_manifest.example.csv](schemas/action_clip_manifest.example.csv).

**Local tools** (run from repo root; requires `pyyaml` for the GT builder):

```bash
# Val GT → sequences/ + manifest_template.csv (edit labels before training)
python tools/build_manifest_from_val_gt.py --data-yaml mma_pose_dataset/data.yaml --out-dir ./action_clips

# Labeled manifest → MMAction2 pickle (see MMAction2 docs for layout)
python tools/export_mmaction2_skeleton.py --manifest ./action_clips/manifest.csv --out-dir ./mmaction_data --normalize
```

**ST-GCN starter config:** copy [configs/mmaction2/stgcn_coco17_2d.py](configs/mmaction2/stgcn_coco17_2d.py) into a cloned [MMAction2](https://github.com/open-mmlab/mmaction2) `configs/skeleton/stgcn/` directory, set `custom_num_classes`, place `mmaction_custom.pkl` under `mmaction2/data/skeleton/`, then train from the **mmaction2 repo root**. **Colab:** PyTorch / CUDA / MMCV / MMAction2 versions must match; follow the [official install guide](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) — wheels break when mixed blindly.
