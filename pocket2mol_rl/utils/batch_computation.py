from functools import partial
from typing import List, Any, TypeVar, Callable, Iterable
from math import ceil

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def iter_batches(*data_iters, batch_size: int = None, n=None) -> Iterable[List[T1]]:
    return BatchIteration(*data_iters, batch_size=batch_size, n=n)


class BatchIteration:
    def __init__(self, *data_iters, batch_size: int = None, n=None):
        """Iterate over batches of data from multiple iterables.

        Args:
            *data_iter: iterables of data
            batch_size (int): the batch size
            n (int, optional): the number of data to iterate over. Defaults to None. (Used in case data_iter does not have __len__ attribute)
        """
        assert len(data_iters) >= 1, len(data_iters)
        assert batch_size is not None
        assert batch_size >= 1, batch_size
        if n is None:
            assert any(hasattr(data_iter, "__len__") for data_iter in data_iters)
            n = min(
                len(data_iter)
                for data_iter in data_iters
                if hasattr(data_iter, "__len__")
            )
        assert n >= 1, n
        self.n = n

        self.data_iters = data_iters
        self.batch_size = batch_size
        self.n = n

    def _iter_single(self):
        batch = []
        for i, x in enumerate(self.data_iters[0]):
            if i == self.n:
                break
            batch.append(x)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __iter__(self):
        if len(self.data_iters) == 1:
            yield from self._iter_single()
            return

        batches = tuple([] for _ in range(len(self.data_iters)))
        for i, t in enumerate(zip(*self.data_iters)):
            if i == self.n:
                break
            for j, x in enumerate(t):
                batches[j].append(x)
            if len(batches[0]) == self.batch_size:
                yield batches
                batches = tuple([] for _ in range(len(self.data_iters)))
        if len(batches[0]) > 0:
            yield batches

    def __len__(self):
        return ceil(self.n / self.batch_size)


class OnDemandBatchComputation:
    def __init__(
        self, batches: List[List[T1]], f: Callable[[List[T1]], List[T2]], **kwargs
    ):
        """Suppose we have
        (1) a list of data, partitioned into batches.
        (2) and a function f, that is of the form f([x1,...,xn]) == [g(x1), ..., g(xn)] for some function g.
        We want to sequentially get the value of g
        (1) on a random (independent of the batches) number of data at each time,
        (2) but internally compute in batches using f (e.g. for efficiency).
        This class provides such a functionality.

        Args:
            batches (List[List[T1]]): the batches of data
            f (Callable[[List[T1]], List[T2]]): the function to compute on the batches
            **kwargs: the keyward arguments to pass to f
        """
        self.batches = batches
        self.func = partial(f, **kwargs)
        self.batch_idx = 0
        self.data_idx = 0
        self.fetched_vals = []

    @property
    def num_total_data(self):
        return sum(len(batch) for batch in self.batches)

    @property
    def num_remaining_data(self):
        return (
            sum(len(batch) for batch in self.batches[self.batch_idx :]) - self.data_idx
        )

    def get(self, n: int, return_data=False) -> List[T2]:
        """
        Get the next `n` values from the iterator.

        Args:
            n (int): The number of values to get.
            return_data (bool, optional): Whether to return the corresponding data for each value.
                Defaults to False.

        Returns:
            List[T2] or Tuple[List[T2], List[T1]]: The next `n` values from the iterator.
                If `return_data` is True, a tuple of two lists is returned, where the first list
                contains the values and the second list contains the corresponding data.
                Otherwise, only the list of values is returned.
        """
        assert n >= 1, n
        if n > self.num_remaining_data:
            raise ValueError(f"Not enough data to get: {n} > {self.num_remaining_data}")

        vals = []
        data = [] if return_data else None

        while len(vals) < n:
            # fetch more data if needed
            if self.data_idx == 0:
                self.fetched_vals = self.func(self.batches[self.batch_idx])
                assert len(self.fetched_vals) == len(self.batches[self.batch_idx]), (
                    len(self.fetched_vals),
                    len(self.batches[self.batch_idx]),
                )

            # get the next `n` values
            assert self.data_idx < len(self.batches[self.batch_idx])
            num_remaining_to_get = n - len(vals)
            end_idx = self.data_idx + num_remaining_to_get
            if end_idx < len(self.batches[self.batch_idx]):
                vals.extend(self.fetched_vals[self.data_idx : end_idx])
                if return_data:
                    data.extend(self.batches[self.batch_idx][self.data_idx : end_idx])
                self.data_idx = end_idx
                break
            else:
                vals.extend(self.fetched_vals[self.data_idx :])
                if return_data:
                    data.extend(self.batches[self.batch_idx][self.data_idx :])
                self.batch_idx += 1
                self.data_idx = 0

        return (vals, data) if return_data else vals


class Null:
    pass


def group_compute_merge(
    group_fn,
    compute_fn,
    *data_lists,
):
    """Group the data by group_fn, compute compute_fn on each group, and merge the results while retaining the original order. Multiples arguments and multiple outputs are supported.

    Args:
        group_fn (Callable[[A1,...,An], Any]): the function to group the data
        compute_fn (Union[
            Callable[[List[A1],...,List[An]], List[B]],
            Callable[[List[A1],...,List[An]], Tuple[List[B1],...,List[Bm]]]]): the function to compute on each group
        data_lists (Tuple[List[A1],...,List[An]]): the data
    Returns:
        Union[List[B], Tuple[List[B1],...,List[Bm]]]: the merged results
    """
    assert len(data_lists) >= 1
    num_data = len(data_lists[0])
    assert all(len(l) == num_data for l in data_lists)

    if num_data == 0:
        return []

    group_to_idxs = {}
    group_to_data_lists = {}
    for i, data_tuple in enumerate(zip(*data_lists)):
        label = group_fn(*data_tuple)
        group_to_idxs.setdefault(label, []).append(i)
        if not label in group_to_data_lists:
            group_to_data_lists[label] = tuple([] for _ in range(len(data_lists)))
        for j, data in enumerate(data_tuple):
            group_to_data_lists[label][j].append(data)

    assert len(group_to_idxs) == len(group_to_data_lists) >= 1, (
        group_to_idxs,
        group_to_data_lists,
    )

    result = [Null() for _ in range(num_data)]
    for i, group in enumerate(group_to_idxs):
        idxs = group_to_idxs[group]
        data_lists = group_to_data_lists[group]
        vals = compute_fn(*data_lists)
        if i == 0:
            if isinstance(vals, tuple):
                result = tuple([Null() for _ in range(num_data)] for _ in vals)
                result_is_tuple = True
            else:
                result = [Null() for _ in range(num_data)]
                result_is_tuple = False
        if result_is_tuple:
            assert all(len(idxs) == len(val) for val in vals)
            for j in range(len(vals)):
                for idx, val in zip(idxs, vals[j]):
                    result[j][idx] = val
        else:
            assert len(idxs) == len(vals)
            for idx, val in zip(idxs, vals):
                result[idx] = val
    if result_is_tuple:
        assert all(not isinstance(y, Null) for x in result for y in x)
    else:
        assert all(not isinstance(x, Null) for x in result)
    return result
