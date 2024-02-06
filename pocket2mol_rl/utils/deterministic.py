from torch_scatter import (
    scatter_add as _scatter_add,
    scatter_sum as _scatter_sum,
    scatter_softmax as _scatter_softmax,
)
import torch


# TODO: Test this
def scatter_add(src, index, dim=0, dim_size=None, **kwargs):
    if torch.are_deterministic_algorithms_enabled():
        assert len(kwargs) == 0
        assert dim == 0
        # src: (n,), index: (2, n), dim_size: n

        if index.numel() == 0:
            _shape = tuple([dim_size] + list(src.shape[1:]))
            return torch.zeros(_shape, dtype=src.dtype, device=src.device)

        assert index.ndim == 1
        assert 0 <= index.min() and index.max() < dim_size
        assert len(src) == len(index)

        if src.ndim == 1:
            val = torch.zeros_like(src, shape=(dim_size,))
            val = torch.scatter_add(val, 0, index, src)
            return val
        elif src.ndim == 2:
            src_flattened = src.flatten()
            index_flattened = (
                src.shape[1] * index[:, None]
                + torch.arange(src.shape[1], device=index.device, dtype=index.dtype)[
                    None, :
                ]
            ).flatten()
            val = torch.zeros(
                dim_size * src.shape[1], dtype=src.dtype, device=src.device
            )
            val = torch.scatter_add(val, 0, index_flattened, src_flattened)
            val = val.view(dim_size, src.shape[1])
            return val
        elif src.ndim == 3:
            src_flattened = src.flatten()
            index_flattened = (
                src.shape[1] * src.shape[2] * index[:, None, None]
                + torch.arange(
                    src.shape[1] * src.shape[2], device=index.device, dtype=index.dtype
                ).view(1, src.shape[1], src.shape[2])
            ).flatten()
            val = torch.zeros(
                dim_size * src.shape[1] * src.shape[2],
                dtype=src.dtype,
                device=src.device,
            )
            val = torch.scatter_add(val, 0, index_flattened, src_flattened)
            val = val.view(dim_size, src.shape[1], src.shape[2])
            return val
        else:
            raise ValueError(f"src.ndim must be 1 or 2, got {src.ndim}", src.shape)

        """ 
        device = src.device
        result = _scatter_add(src.cpu(), index=index.cpu(), dim=dim, dim_size=dim_size)
        return result.to(device)
        """

    else:
        return _scatter_add(src, index=index, dim=dim, dim_size=dim_size, **kwargs)


def scatter_sum(*args, **kwargs):
    return scatter_add(*args, **kwargs)
    if torch.are_deterministic_algorithms_enabled():
        # Use this only for inference
        device = src.device
        result = _scatter_sum(src.cpu(), index=index.cpu(), **kwargs)
        return result.to(device)
    else:
        return _scatter_sum(src, index, **kwargs)


def scatter_softmax(src, index, dim=0, **kwargs):
    if torch.are_deterministic_algorithms_enabled():
        # Explot the fact that Tensor.scatter_add_() has deterministic cuda implementation for 1D tensors

        assert dim == 0
        assert src.ndim == 2
        assert index.ndim == 1
        assert src.shape[0] == len(index)
        assert len(kwargs) == 0

        src_flattened = src.flatten()
        index_flattened = (
            src.shape[1] * index[:, None]
            + torch.arange(src.shape[1], device=index.device, dtype=index.dtype)[
                None, :
            ]
        ).flatten()

        result_flattened = _scatter_softmax(src_flattened, index_flattened, dim=0)
        result = result_flattened.view_as(src)

        """
        device = src.device
        result = _scatter_softmax(src.cpu(), index=index.cpu(), **kwargs)
        return result.to(device)
        """

        return result

    else:
        return _scatter_softmax(src, index, dim=dim, **kwargs)


def assign_val_at_idx(tensor, idx, val):
    prev_deterministic_setting = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    tensor[idx] = val
    torch.use_deterministic_algorithms(prev_deterministic_setting)
    return tensor
