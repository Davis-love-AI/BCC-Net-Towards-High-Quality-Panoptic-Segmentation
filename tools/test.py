import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from pycocotools.cocoeval import COCOeval

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from tools.panoptic_evaluate import combine_predictions, logger_init_pq


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--submit', action='store_true', help='Prepare submission for coco19')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    if args.submit:
        dataset = build_dataset(cfg.data.submit)
    else:
        dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        result_root = osp.dirname(args.out)
        if not osp.exists(result_root):
            os.makedirs(result_root)
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)

        # eval_types = args.eval
        # if eval_types:
        #     print('Starting evaluate {}'.format(' and '.join(eval_types)))
        #     if eval_types == ['proposal_fast']:
        #         result_file = args.out
        #         coco_eval(result_file, eval_types, dataset.coco)
        #     else:
        #         if not isinstance(outputs[0], dict):
        #             result_files = results2json(dataset, outputs, args.out)
        #             coco_eval(result_files, eval_types, dataset.coco)
        #         else:
        #             for name in outputs[0]:
        #                 print('\nEvaluating {}'.format(name))
        #                 outputs_ = [out[name] for out in outputs]
        #                 result_file = args.out + '.{}'.format(name)
        #                 result_files = results2json(dataset, outputs_,
        #                                             result_file)
        #                 coco_eval(result_files, eval_types, dataset.coco)

        print('\ncreating result_file_instance and result_file_semantic...')
        result_file_instance = args.out + '.instance.json'  # will be created as intermediate file by result2json()
        result_file_semantic = args.out + '.semantic.json'  # will be created as intermediate file by result2json()
        result_file_aux = args.out + '.aux.json'  # may be created
        results2json(dataset, outputs, result_file_instance, result_file_semantic, result_file_aux)
        logger = logger_init_pq(result_root)

        # evaluate mAP
        res_types = ['bbox', 'segm']
        cocoGt = dataset.coco
        cocoDt = cocoGt.loadRes(result_file_instance)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                logger.info('{:20s}|{:10.3f}'.format(key, val))

        # Evaluate PQ
        print('\ncombining instance and semantic predictions...')
        if not args.submit:
            combine_predictions(
                result_file_semantic,
                result_file_instance,
                result_file_aux,
                cfg.data.test.ann_file.replace('instances', 'panoptic'),  # data/coco/annotations/panoptic_val2017.json
                cfg.data.test.cat_pan_file_discrete,
                "data/coco/annotations/panoptic_val2017",
                work_dir=result_root,
                mode='test',
                logger=logger)
        else:
            combine_predictions(
                result_file_semantic,
                result_file_instance,
                result_file_aux,
                cfg.data.submit.ann_file,  # data/coco/annotations/image_info_test-dev2017.json
                cfg.data.submit.cat_pan_file_discrete,
                None,  # no gt_folder
                work_dir=result_root,
                mode='submit',
                logger=logger)


if __name__ == '__main__':
    main()
