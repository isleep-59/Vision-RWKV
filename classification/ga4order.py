import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
import torchvision.utils as vutils
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
import datetime

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
    indices = np.arange(num_patches)
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
    patches = patches.permute(2, 0, 3, 1, 4).reshape(3, H, W)
    return patches

def reorder(image: torch.Tensor, new_order: np.array):
    """
    将图像重新排列
    - image: (C, H, W) 的张量
    - indices: (196,) 编号数组
    - new_order: (196,) 重新排列的索引顺序
    """
    indices, patches = split_image(image)
    patches = rearrange_patches(patches, indices, new_order)
    rearranged_image = merge_patches(patches, indices)
    return rearranged_image


def evaluate(output: torch.Tensor, target: torch.Tensor):
    """
    计算图像的损失
    - image: (C, H, W) 的图像张量
    - model: 模型
    """
    loss = -torch.log(output[0][target[0]])

    return loss


def test_inf(model, image: torch.Tensor, target: torch.Tensor):
    """
    图像推理
    - model: 模型
    - image: (C, H, W) 的图像张量
    - target: 目标标签
    """
    image = image.unsqueeze(0)
    model.eval()
    output = model(image, return_loss=False)
    output = torch.from_numpy(np.array(output))
    return output


def f(new_order: np.array, kwargs):
    """
    优化目标函数
    - new_order: 重新排列的索引顺序
    """

    model, image, target = kwargs["model"], kwargs["image"], kwargs["target"]
    reordered_image = reorder(image, new_order)
    output = test_inf(model, reordered_image, target)
    loss = evaluate(output, target)

    var = np.arange(196)
    mismatch_num = np.sum(new_order != var)
    
    return loss + 0.1 * mismatch_num


def save_images(new_order, image, original_image_path, reordered_image_path):
    """
    保存排列前后的图像
    - new_order: 重新排列的索引顺序
    - image: 原始图像
    - original_image_path: 原始图像保存路径
    - reordered_image_path: 重新排列图像保存路径
    """
    vutils.save_image(image, original_image_path)
    reordered_image = reorder(image, new_order)
    vutils.save_image(reordered_image, reordered_image_path)


def solve(model, image: torch.Tensor, target: torch.Tensor, index: int, output: torch.Tensor, log_directory: str):
    """
    优化问题求解
    - model: 模型
    - image: (C, H, W) 的图像张量
    - target: 目标标签
    - index: 图像索引
    """
    algorithm_param = {'max_num_iteration': 100,\
                    'population_size':25,\
                    'mutation_probability':0.8,\
                    'elit_ratio': 0.05,\
                    'crossover_probability': 0.001,\
                    'parents_portion': 0.5,\
                    'crossover_type':'pmx',\
                    'mutation_type':'swap',\
                    'init_type':'sequential',\
                    'max_iteration_without_improv':None}

    m=ga(function=f,\
            dimension=196,\
            function_timeout=10000,\
            algorithm_parameters=algorithm_param,\
            log_limit=100,\
            log_directory=f'{log_directory}/{index}',\
            model=model,\
            image=image,\
            target=target)

    m.run()

    # dir_name = f'{m.init_type}_{m.current_time}'
    # dir_path = f'/workspace/Order-Matters/Vision-RWKV/classification/ga4order_logs/{dir_name}/'

    # 保存算法参数
    if not os.path.exists(f'{log_directory}/algorithm_parameter'):
        with open(f'{log_directory}/algorithm_parameter', 'w') as file:
            for key, value in algorithm_param.items():
                file.write(f'{key}: {value}\n')

    # 保存每次搜索结果
    dir_path = f'{log_directory}/{index}'
    os.makedirs(dir_path, exist_ok=True)
    np.savetxt(f'{dir_path}/logits.npy', output.cpu().numpy().T)
    var = np.arange(196)
    mismatch_num = np.sum(m.best_variable.astype(int) != var)
    with open(f'{dir_path}/label_change.txt', 'w') as file:
        file.write(f'{index}: \n\tgt_label: {target[0]} \n\tinf_label: {torch.argmax(output)} \n\tbeforeloss: {evaluate(output, target)} \n\tafter_loss: {m.best_function - 0.1 * mismatch_num} \n\tmismatch_num: {mismatch_num}\n\n')
    original_image_path = f'{dir_path}/original_image.png'
    reordered_image_path = f'{dir_path}/rearranged_image.png'
    save_images(m.best_variable, image, original_image_path, reordered_image_path)


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
    else:
        model = wrap_distributed_model(
            model, device=cfg.device, broadcast_buffers=False)
    
    # image = dataset[10]['img']
    # target = dataset.data_infos[10]['gt_label']

    # solve(model, image, target)

    # new_order = np.loadtxt('/workspace/Order-Matters/Vision-RWKV/classification/ga4order_logs/random_2025-03-30_14-23-20/best_solution.npy')
    # reordered_image = reorder(image, new_order)
    # test_inf(model, reordered_image, target)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_directory = f'/workspace/Order-Matters/Vision-RWKV/classification/wrong_classification/{current_time}'
    for index in range(0, 100):
        image = dataset[index]['img']
        target = dataset.data_infos[index]['gt_label']
        target = torch.from_numpy(target)
        target = target.unsqueeze(0)
        output = test_inf(model, image, target)
        if torch.argmax(output) != target:
            solve(model, image, target, index, output, log_directory)   
              

    # index = 35        
    # image = dataset[index]['img']
    # target = dataset.data_infos[index]['gt_label']
    # target = torch.from_numpy(target)
    # target = target.unsqueeze(0)
    # output = test_inf(model, image, target)
    # if torch.argmax(output) != target:
    #     solve(model, image, target, index, output) 
