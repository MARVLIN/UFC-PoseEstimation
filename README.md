# UFC-PoseEstimation

Fine-tuning Ultralytics YOLO pose models for MMA / UFC fighter keypoint estimation. Includes dataset preparation from bbox-only labels and training scripts.

## Google Colab

Upload [colab/MMA_Pose_YOLO_Training.ipynb](colab/MMA_Pose_YOLO_Training.ipynb) to Colab (GPU runtime). By default it **downloads the Mendeley zip via `wget`**, unzips under `/content`, auto-finds `data.yaml`, trains, and writes `best.pt`, metrics, and plots to `/content/mma_pose_work/exports` (or set `SAVE_OUTPUT_TO_DRIVE = True` and use Drive). Alternatively set `DATA_SOURCE = "drive"` and point `DRIVE_DATASET_DIR` at your dataset folder.
