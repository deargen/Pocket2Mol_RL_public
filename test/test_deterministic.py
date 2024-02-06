from torch_scatter import (
    scatter_add as _scatter_add,
    scatter_sum as _scatter_sum,
    scatter_softmax as _scatter_softmax,
)
from pocket2mol_rl.utils.deterministic import scatter_add, scatter_sum, scatter_softmax
import torch
import unittest


class TestDeterministic(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0"

    def test_scatter_sum(self):
        l = []
        src = torch.randn(4, 16, device=self.device)
        index = torch.tensor([0, 1, 0, 1], device=self.device)
        dim_size = 3
        l.append((src, index, dim_size))

        src = torch.randn(6, 8, 16, device=self.device)
        index = torch.tensor([2, 2, 2, 1, 1, 0], device=self.device)
        dim_size = 3
        l.append((src, index, dim_size))

        for src, index, dim_size in l:
            torch.use_deterministic_algorithms(True)
            result1 = scatter_sum(src, index, dim=0, dim_size=dim_size)
            result2 = scatter_add(src, index, dim=0, dim_size=dim_size)
            torch.use_deterministic_algorithms(False)
            result3 = _scatter_sum(src, index, dim=0, dim_size=dim_size)
            result4 = _scatter_add(src, index, dim=0, dim_size=dim_size)

            torch.testing.assert_allclose(result1, result2)
            torch.testing.assert_allclose(result2, result3)
            torch.testing.assert_allclose(result3, result4)

    def test_scatter_softmax(self):
        l = []
        src = torch.randn(8, 16, device=self.device)
        index = torch.tensor([1, 1, 1, 0, 0, 2, 3, 3], device=self.device)
        l.append((src, index))

        index = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], device=self.device)
        l.append((src, index))

        for src, index in l:
            torch.use_deterministic_algorithms(True)
            result1 = scatter_softmax(src, index, dim=0)
            torch.use_deterministic_algorithms(False)
            result2 = _scatter_softmax(src, index, dim=0)
            torch.testing.assert_allclose(result1, result2)
