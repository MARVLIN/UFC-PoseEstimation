# Custom ST-GCN for COCO-17 2D clips exported by yoloPose/tools/export_mmaction2_skeleton.py
#
# INSTALL (Colab / local): clone MMAction2 and copy this file into:
#   mmaction2/configs/skeleton/stgcn/stgcn_coco17_2d.py
# so `_base_` resolves next to the upstream ST-GCN config.
#
# Place the pickle at: mmaction2/data/skeleton/mmaction_custom.pkl
# (or set custom_ann below to an absolute path).
#
# Train (from mmaction2 repo root, GPU):
#   mim train mmaction2 stgcn_coco17_2d.py
#
# Edit custom_num_classes to match rows in label_map.txt from export.

_base_ = "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"

custom_num_classes = 3
custom_ann = "data/skeleton/mmaction_custom.pkl"

model = dict(cls_head=dict(num_classes=custom_num_classes))

dataset_type = "PoseDataset"
ann_file = custom_ann

train_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSampleFrames", clip_len=64),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSampleFrames", clip_len=64, num_clips=1, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSampleFrames", clip_len=64, num_clips=10, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split="train",
        ),
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split="val",
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_begin=1, val_interval=1)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        eta_min=0,
        T_max=24,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]

auto_scale_lr = dict(enable=False, base_batch_size=128)
