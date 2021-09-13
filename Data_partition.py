import math
import functools
import numpy as np
import torchvision.datasets as dataset

# Highlight: n_auxi_workers should be no more than n_workers

def build_non_iid_by_dirichlet(random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]

    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(indices2targets[from_index : (num_indices if idx == num_splits - 1 else to_index)])
        from_index = to_index


    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):

            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets == _class)[0]
                # idx_class = _targets[idx_class]

                # sampling.
                try:
                    proportions = random_state.dirichlet(np.repeat(non_iid_alpha, _n_workers))
                    proportions = np.array(
                        [p * (len(idx_j) < _targets_size / _n_workers) for p, idx_j in zip(proportions, _idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1]
                    _idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(_idx_batch, np.split(idx_class, proportions))]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch

    return idx_batch


def create_indices(data, partition_type, partition_count, non_iid_alpha, indices):
    if partition_type == "origin":
        pass
    elif partition_type == "non_iid_dirichlet":
        num_classes = len(np.unique(data.targets))
        num_indices = len(indices)
        n_workers = len(partition_count)

        list_of_indices = build_non_iid_by_dirichlet(
            random_state=np.random.RandomState(7),
            indices2targets=np.array([(idx, target) for idx, target in enumerate(data.targets) if idx in indices]),
            # indices2targets=np.array(functools.reduce(lambda a, b: a + b,[[idx] * int(num_indices / num_classes) for idx in range(num_classes)],)),
            # indices2targets=np.array([(idx, target) for idx, target in enumerate(data.targets)]),
            non_iid_alpha=non_iid_alpha,
            num_classes=num_classes,
            num_indices=num_indices,
            n_workers=n_workers,
        )
        # print(list_of_indices)
        indices = functools.reduce(lambda a, b: a + b, list_of_indices)
    else:
        raise NotImplementedError(
            f"The partition scheme={partition_type} is not implemented yet"
        )
    return indices


def partition_indices(data, partition_type, partition_count, non_iid_alpha):
    data_size = len(data)
    pre_indices = np.array([x for x in range(0, data_size)])
    indices = create_indices(data=data, partition_type=partition_type, partition_count=partition_count, non_iid_alpha=non_iid_alpha, indices=pre_indices)

    # partition indices.
    partitions = []
    from_index = 0
    data_size = len(data)
    for partition_size in partition_count:
        to_index = from_index + int(partition_size * data_size)
        partitions.append(indices[from_index:to_index])
        from_index = to_index
    # record_class_distribution(partitions=partitions, targets=data.targets, print_fn=print())
    return partitions


def record_class_distribution(partitions, targets, print_fn):
    targets_of_partitions = {}
    targets_np = np.array(targets)
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(targets_np[partition], return_counts=True)
        targets_of_partitions[idx] = list(zip(unique_elements, counts_elements))
    print_fn(f"the histogram of the targets in the partitions: {targets_of_partitions.items()}")
    return targets_of_partitions


def demo():
    world_size = 10
    n_workers = [1.0 / world_size for _ in range(world_size)]
    train_data = dataset.CIFAR10(root='../dataset/cifar10', train=True, download=True)
    test_data = dataset.CIFAR10(root='../dataset/cifar10', train=False, download=False)

    partitioned_indices = partition_indices(data=train_data, partition_type='non_iid_dirichlet', partition_count=n_workers, non_iid_alpha=100)
    print(f'the number of training data in single client = {len(partitioned_indices[0])}, by total {len(train_data)}')


# demo()