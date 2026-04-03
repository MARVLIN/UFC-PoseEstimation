# UFC-PoseEstimation

Fine-tuning Ultralytics YOLO pose models for MMA / UFC fighter keypoint estimation. Includes dataset preparation from bbox-only labels and training scripts.

## Google Colab

Upload [colab/MMA_Pose_YOLO_Training.ipynb](colab/MMA_Pose_YOLO_Training.ipynb) to Colab (GPU runtime). Mount Drive, set `DRIVE_DATASET_DIR` to the folder that contains `data.yaml` and the `train` / `valid` / `test` trees. The notebook trains YOLO pose, saves `best.pt` and metrics to Drive, and exports MAE/RMSE plus multiple training plots.
