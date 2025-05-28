import torch.utils.checkpoint as cp


# 备份原来的两个函数
_orig_checkpoint = cp.checkpoint
_orig_checkpoint_sequential = cp.checkpoint_sequential


# 用 non-reentrant（只会一次 forward + 一次 backward）
def _no_reentrant_checkpoint(fn, *args, **kwargs):
    return _orig_checkpoint(fn, *args, use_reentrant=False, **kwargs)


def _no_reentrant_checkpoint_seq(functions, segments, *inputs, **kwargs):
    return _orig_checkpoint_sequential(functions, segments, *inputs, use_reentrant=False, **kwargs)


# 覆盖全局
cp.checkpoint = _no_reentrant_checkpoint
cp.checkpoint_sequential = _no_reentrant_checkpoint_seq

from .model import LlavaLlamaForCausalLM
