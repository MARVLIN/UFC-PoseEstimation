# UFC-PoseEstimation

Fine-tuning Ultralytics YOLO pose models for MMA / UFC fighter keypoint estimation. Includes dataset preparation from bbox-only labels and training scripts.

## Google Colab

Upload [colab/MMA_Pose_YOLO_Training.ipynb](colab/MMA_Pose_YOLO_Training.ipynb) to Colab (GPU runtime). **Mount Google Drive**, set `DRIVE_DATASET_DIR` to the folder that contains `data.yaml` and your YOLO pose splits, and `DRIVE_OUTPUT_DIR` where runs should be saved. Training uses fast `/content` scratch; `best.pt`, metrics JSON, and plots are written to Drive.
