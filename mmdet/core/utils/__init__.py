<<<<<<< HEAD
from .dist_utils import DistOptimizerHook, allreduce_grads, reduce_mean
from .misc import flip_tensor, mask2ndarray, multi_apply, unmap, vectorize_labels

__all__ = [
    'allreduce_grads', 'DistOptimizerHook', 'reduce_mean', 'multi_apply',
    'unmap', 'mask2ndarray', 'flip_tensor', 'vectorize_labels'
=======
from .dist_utils import (DistOptimizerHook, all_reduce_dict, allreduce_grads,
                         reduce_mean)
from .misc import flip_tensor, mask2ndarray, multi_apply, unmap

__all__ = [
    'allreduce_grads', 'DistOptimizerHook', 'reduce_mean', 'multi_apply',
    'unmap', 'mask2ndarray', 'flip_tensor', 'all_reduce_dict'
>>>>>>> b4a84cc9cf38ee4c0fa6f6264e9916c8ba7cd3b3
]
