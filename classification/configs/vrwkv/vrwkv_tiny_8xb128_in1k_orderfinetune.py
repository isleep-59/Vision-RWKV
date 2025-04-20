_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

reorder_module_cfg = dict(
    type='Reorder',
    finetune_cfg=dict(
        img_size=224, 
        patch_size=16,
        stride=16,    
        in_channels=3,
        embed_dims=192,     
        num_patches=196,    
    ),
)

# model settings
vision_rwkv_classifier_cfg  = dict(
    type='ImageClassifier', 
    backbone=dict(
        type='VRWKV',
        img_size=224,
        patch_size=16,
        embed_dims=192,),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=192,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    train_cfg=dict()
)

model = dict(
    type='ReorderImageClassifier', # New top-level type
    reorder_module_cfg=reorder_module_cfg,
    image_classifier_module_cfg=vision_rwkv_classifier_cfg,
    pretrained_checkpoint_path='/workspace/Order-Matters/Vision-RWKV/classification/checkpoint/vrwkv_t_in1k_224.pth',
    reg_weight=300000,
)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True)
evaluation = dict(interval=1)
# 8 gpus
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=24,
)