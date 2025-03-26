import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, get_root_logger,
                         setup_multi_processes, wrap_distributed_model,
                         wrap_non_distributed_model)

import urllib3
urllib3.disable_warnings()

from geneticalgorithm.geneticalgorithm import geneticalgorithm as ga
import mmcls_custom

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib

    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

def split_image(image: torch.Tensor, patch_size = 16):
    """
    将 (C, H, W) 形状的图像分割成 14x14=196 个 16x16 小块，并编号
    - image: (C, 224, 224) 的张量
    - patch_size: 每个小块的大小，默认 16
    """
    C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, "图像尺寸必须是 patch_size 的整数倍"
    
    num_patches = (H // patch_size) * (W // patch_size)
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(num_patches, C, patch_size, patch_size)
    indices = np.arange(num_patches).reshape(H // patch_size, W // patch_size).flatten()
    return indices, patches

def rearrange_patches(patches, indices, new_order):
    """
    重新排列小块顺序
    - patches: (196, 3, 16, 16) 的张量
    - indices: (196,) 编号数组
    - new_order: (196,) 重新排列的索引顺序
    """
    assert len(new_order) == len(indices), "新顺序的大小必须等于196"
    reordered_patches = patches[new_order]
    return reordered_patches


def merge_patches(patches, indices, patch_size=16):
    """
    将 196 个小块合并成一张完整的图像
    - patches: (196, 3, 16, 16) 的张量
    - indices: (196,) 编号数组
    - patch_size: 每个小块的大小，默认 16
    """
    num_patches = len(indices)
    H, W = patch_size * (int(np.sqrt(num_patches))), patch_size * (int(np.sqrt(num_patches)))
    patches = patches.reshape(int(np.sqrt(num_patches)), int(np.sqrt(num_patches)), 3, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).reshape(3, H, W)
    return patches


@torch.no_grad()
def evaluate(model, image, target):
    """
    计算图像的损失
    - image: (C, H, W) 的图像张量
    - model: 模型
    """
    image = image.unsqueeze(0)
    target = torch.from_numpy(target).unsqueeze(0)

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    output = model(image, target)
    output = torch.from_numpy(np.array(output))
    loss = criterion(output, target)

    return loss.item()


def f(new_order, kwargs):
    """
    优化目标函数
    - new_order: 重新排列的索引顺序
    """

    model = kwargs["model"]
    image = kwargs["image"]
    target = kwargs["target"]

    indices, patches = split_image(image, 16)
    patches = rearrange_patches(patches, indices, new_order)
    rearranged_image = merge_patches(patches, indices, 16)
    loss = evaluate(model, rearranged_image, target)
    
    return loss


def solve(model, image, target):
    """
    优化问题求解
    - model: 模型
    - image: (C, H, W) 的图像张量
    - target: 目标标签
    """
    algorithm_param = {'max_num_iteration': 10,\
                    'population_size':100,\
                    'mutation_probability':0.05,\
                    'elit_ratio': 0.05,\
                    'crossover_probability': 0.8,\
                    'parents_portion': 0.5,\
                    'crossover_type':'pmx',\
                    'mutation_type':'swap',\
                    'max_iteration_without_improv':None}

    m=ga(function=f,\
            dimension=196,\
            function_timeout=10000,\
            algorithm_parameters=algorithm_param,\
            model=model,\
            image=image,\
            target=target)

    m.run()

def main(args):
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = args.device or auto_select_device()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    # loader_cfg = dict(
    #     # cfg.gpus will be ignored if distributed
    #     num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
    #     dist=distributed,
    #     round_up=True,
    # )
    # # The overall dataloader settings
    # loader_cfg.update({
    #     k: v
    #     for k, v in cfg.data.items() if k not in [
    #         'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
    #         'test_dataloader'
    #     ]
    # })
    # test_loader_cfg = {
    #     **loader_cfg,
    #     'shuffle': False,  # Not shuffle by default
    #     'sampler_cfg': None,  # Not use sampler by default
    #     **cfg.data.get('test_dataloader', {}),
    # }
    # # the extra round_up data will be removed during gpu/cpu collect
    # data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    if not distributed:
        model = wrap_non_distributed_model(
            model, device=cfg.device, device_ids=cfg.gpu_ids)
        if cfg.device == 'ipu':
            from mmcv.device.ipu import cfg2options, ipu_model_wrapper
            opts = cfg2options(cfg.runner.get('options_cfg', {}))
            if fp16_cfg is not None:
                model.half()
            model = ipu_model_wrapper(model, opts, fp16_cfg=fp16_cfg)
        model.CLASSES = CLASSES
        show_kwargs = args.show_options or {}
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
        #                           **show_kwargs)
    else:
        model = wrap_distributed_model(
            model, device=cfg.device, broadcast_buffers=False)
        # outputs = multi_gpu_test(model, data_loader, args.tmpdir,
        #                          args.gpu_collect)
    
    image = dataset[0]['img']
    target = dataset.data_infos[0]['gt_label']

    solve(model, image, target)
    # show_kwargs = args.show_options or {}
    # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
    #                             **show_kwargs)    

    # rank, _ = get_dist_info()
    # if rank == 0:
    #     results = {}
    #     logger = get_root_logger()
    #     if args.metrics:
    #         eval_results = dataset.evaluate(
    #             results=outputs,
    #             metric=args.metrics,
    #             metric_options=args.metric_options,
    #             logger=logger)
    #         results.update(eval_results)
    #         for k, v in eval_results.items():
    #             if isinstance(v, np.ndarray):
    #                 v = [round(out, 2) for out in v.tolist()]
    #             elif isinstance(v, Number):
    #                 v = round(v, 2)
    #             else:
    #                 raise ValueError(f'Unsupport metric type: {type(v)}')
    #             print(f'\n{k} : {v}')
    #     if args.out:
    #         if 'none' not in args.out_items:
    #             scores = np.vstack(outputs)
    #             pred_score = np.max(scores, axis=1)
    #             pred_label = np.argmax(scores, axis=1)
    #             pred_class = [CLASSES[lb] for lb in pred_label]
    #             res_items = {
    #                 'class_scores': scores,
    #                 'pred_score': pred_score,
    #                 'pred_label': pred_label,
    #                 'pred_class': pred_class
    #             }
    #             if 'all' in args.out_items:
    #                 results.update(res_items)
    #             else:
    #                 for key in args.out_items:
    #                     results[key] = res_items[key]
    #         print(f'\ndumping results to {args.out}')
    #         mmcv.dump(results, args.out)