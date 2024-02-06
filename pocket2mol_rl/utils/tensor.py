import torch
import torch.nn.functional as F
from typing import List, Optional


def concat_with_padding(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """Concatenate two tensors with padding
    tensor has shape (N, D), where N is the number of data and D is the feature dimension
    for example, tensor1 has shape (5, 10) and tensor2 has shape (3, 5)
    then the output will be (8, 10) and tensor2 will be padded with zeros

    if one of the tensor is empty and have more features than the other tensor,
    the other tensor will be padded with zeros.

    Args:
        tensor1 (torch.Tensor): tensor 1
        tensor2 (torch.Tensor): tensor 2
        dim (int, optional): dimension to concatenate. Defaults to 1.
    Returns:
        torch.Tensor: concatenated tensor with larger tensor dimension"""
    if tensor1.shape[0] == 0:
        return pad_tensor(tensor2, tensor1.shape[1])
    elif tensor2.shape[0] == 0:
        return pad_tensor(tensor1, tensor2.shape[1])

    max_columns = max(tensor1.shape[1], tensor2.shape[1])

    tensor1 = pad_tensor(tensor1, max_columns)
    tensor2 = pad_tensor(tensor2, max_columns)

    return torch.cat([tensor1, tensor2], dim=0)


def pad_tensor(tensor, target_columns):
    padding_columns = target_columns - tensor.shape[1]
    if padding_columns < 0:
        return tensor

    padding = (0, padding_columns, 0, 0)
    return F.pad(tensor, padding, value=0)


def decompose_tensors(x: torch.Tensor, nums, dim=0):
    assert 0 <= dim < x.ndim
    assert sum(nums) == x.shape[dim]
    l = []
    start = 0
    for num in nums:
        l.append(x.narrow(dim, start, num))
        start += num
    return l


def decompose_tensors_and_getindex(x: torch.Tensor, nums, i, dim=0):
    assert 0 <= dim < x.ndim
    assert sum(nums) == x.shape[dim]
    assert 0 <= i < len(nums)
    start = sum(nums[:i])
    length = nums[i]
    return x.narrow(dim, start, length)


def added_concat(l: List[torch.Tensor], nums: List[int], dim=0):
    assert 0 <= dim < l[0].ndim
    xs = []
    to_add = 0
    for x, num in zip(l, nums):
        xs.append(x + to_add)
        to_add += num
    return torch.cat(xs, dim=dim)


def nonnan_running_average(
    x: torch.Tensor, alpha: float, acc: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """dimension-wise Running average of non-nan values.
    (1-alpha)^n*x0 + (1-alpha)^{n-1}*alpha*x1+...+alpha*xn
    where x0, x1, ..., xn are non-nan values

    Args:
        x (torch.Tensor): (N, d), stacked data to take running average
        alpha (float): 1 - (the decay rate)
        acc (torch.Tensor): (1, d), previous running average, if any. Defaults to None.
    Returns:
        z (torch.Tensor): (1, d), the running average
    """
    assert 0 < alpha < 1
    assert len(x.shape) == 2
    n, d = x.shape
    if acc is None:
        acc = torch.nan * torch.ones_like(x[0]).unsqueeze(0)
    assert acc.shape == (1, d)
    device = acc.device
    assert x.device == device

    nonnan = ~x.isnan()
    cumsum_nonnan = torch.cumsum(nonnan, dim=0)
    num_nonnan = cumsum_nonnan[-1].unsqueeze(0)

    y_coeff = (1 - alpha) ** (num_nonnan - cumsum_nonnan) * alpha
    zerod_x = torch.where(nonnan, x, torch.tensor(0.0, device=device))

    z = (1 - alpha) ** num_nonnan * acc + torch.sum(
        y_coeff * zerod_x, dim=0, keepdim=True
    )

    if torch.any(acc.isnan()):
        y_coeff_altered = torch.where(
            cumsum_nonnan == 1, (1 - alpha) ** (num_nonnan - cumsum_nonnan), y_coeff
        )
        z = torch.where(
            acc.isnan(),
            torch.where(
                x.isnan().all(dim=0, keepdim=True),
                torch.tensor(torch.nan, device=device),
                torch.sum(y_coeff_altered * zerod_x, dim=0, keepdim=True),
            ),
            z,
        )

    return z
