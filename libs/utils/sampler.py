import torch

from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled.
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def make_batch_sampler(
    dataset, images_per_batch, num_iters=None, start_iter=0, shuffle=True
):
    sampler = make_sampler(dataset, shuffle)

    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler
