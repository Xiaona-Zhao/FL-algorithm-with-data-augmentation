import math
import functools
import numpy as np
import torchvision.datasets as dataset


def Dirichlet_distribution_partition_function(random_state, index_target, non_iid_alpha, num_class, num_indices,
                                              num_workers):
    global _index_batch
    num_auxi_workers = num_workers

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(num_workers / num_auxi_workers)
    split_n_workers = [
        num_auxi_workers
        if idx < num_splits - 1
        else num_workers - num_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / num_workers for _n_workers in split_n_workers]

    for index, ratio in enumerate(split_ratios):
        to_index = from_index + int(num_auxi_workers / num_workers * num_indices)
        splitted_targets.append(index_target[from_index: (num_indices if index == num_splits - 1 else to_index)])
        from_index = to_index

    index_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _num_workers = min(num_auxi_workers, num_workers)
        num_workers = num_workers - num_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _num_workers):

            _index_batch = [[] for _ in range(_num_workers)]
            for _class in range(num_class):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets == _class)[0]
                # sampling.
                try:
                    proportions = random_state.dirichlet(np.repeat(non_iid_alpha, _num_workers))
                    proportions = np.array(
                        [p * (len(idx_j) < _targets_size / _num_workers) for p, idx_j in
                         zip(proportions, _index_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1]
                    _index_batch = [idx_j + idx.tolist() for idx_j, idx in
                                    zip(_index_batch, np.split(idx_class, proportions))]
                    sizes = [len(idx_j) for idx_j in _index_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        index_batch += _index_batch

    return index_batch


def create_index(data, partition_count, non_iid_alpha, indices):
    num_class = len(np.unique(data.targets))
    num_indices = len(indices)
    num_workers = len(partition_count)

    list_of_indices = Dirichlet_distribution_partition_function(
        random_state=np.random.RandomState(1),
        index_target=np.array([(idx, target) for idx, target in enumerate(data.targets) if idx in indices]),
        non_iid_alpha=non_iid_alpha,
        num_class=num_class,
        num_indices=num_indices,
        num_workers=num_workers,
    )
    indices = functools.reduce(lambda a, b: a + b, list_of_indices)
    return indices


def recording_function(partitions, targets, print_function):
    partitioned_targets = {}
    targets_np = np.array(targets)
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(targets_np[partition], return_counts=True)
        partitioned_targets[idx] = list(zip(unique_elements, counts_elements))
    print_function(f"Partitioned targets: {partitioned_targets.items()}")
    return partitioned_targets


def partition_indices(data, partition_count, non_iid_alpha):
    data_size = len(data)
    predefined_indices = np.array([x for x in range(0, data_size)])

    indices = create_index(data=data, partition_count=partition_count, non_iid_alpha=non_iid_alpha,
                           indices=predefined_indices)

    # partition indices.
    partitioned_indices = []
    from_index = 0
    data_size = len(data)
    for partition_size in partition_count:
        to_index = from_index + int(partition_size * data_size)
        partitioned_indices.append(indices[from_index:to_index])
        from_index = to_index
    # recording_function(partitions=partitions, targets=data.targets, print_fn=print())
    return partitioned_indices


def demo():
    world_size = 10
    n_workers = [1.0 / world_size for _ in range(world_size)]
    train_data = dataset.CIFAR10(root='../dataset/cifar10', train=True, download=True)
    test_data = dataset.CIFAR10(root='../dataset/cifar10', train=False, download=False)

    partitioned_indices = partition_indices(data=train_data, partition_count=n_workers, non_iid_alpha=100)
    print(f'the number of training data in single client = {len(partitioned_indices[0])}, with total number {len(train_data)}')


# demo()
