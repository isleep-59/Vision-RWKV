# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='OrderFinetuneVRWKV',
        finetune_cfg=dict(
            in_channels=3,
            img_size=224,
            patch_size=16,
            stride=16,
            embed_dims=192,
            num_patches=196,),
        backbone_cfg=dict(
            type='VRWKV',),
        backbone_ckpt='/workspace/Order-Matters/Vision-RWKV/classification/checkpoint/vrwkv_t_in1k_224.pth',),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=192,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict()
)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True)
evaluation = dict(interval=1)
# 8 gpus
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=24,
)