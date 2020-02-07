from collections import OrderedDict
import torch
import torch.distributed as dist
from torch._utils import (_flatten_dense_tensors, _unflatten_dense_tensors,
                          _take_tensors)
from mmcv.runner import OptimizerHook


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


fisher_dict = dict()
count_dict = dict()

def compute_fisher_info(named_parameters):
    gradient_dict = dict()
    gradient_dict.update(
        {name: parameter.grad for name, parameter in named_parameters if parameter.grad is not None})
    for k in gradient_dict:
        gradient_dict[k] = torch.mean(gradient_dict[k] ** 2)
    global fisher_dict
    global count_dict
    for k in gradient_dict:
        module = k.split('.')[1] + '_' +  k.split('.')[2] + '_' + k.split('.')[3]
        count_dict[module] = count_dict.get(module, 0)
        cnt = count_dict[module]
        fisher_dict[module] = fisher_dict.get(module, 0)*cnt/(cnt+1) + gradient_dict[k]/(cnt+1)
        count_dict[module] += 1
    return fisher_dict


class DistOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        # print(compute_fisher_info(runner.model.named_parameters()))
        allreduce_grads(runner.model.parameters(), self.coalesce,
                        self.bucket_size_mb)
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()
