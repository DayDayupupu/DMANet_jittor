import numpy as np
import numba as nb
import jittor as jt


class RandomContinuousSampler:
    """随机连续采样器，返回指定数量的连续索引。"""
    def __init__(self, data_len, num, data_index):
        self.dataset = data_len
        self.num = num
        self.data_index = data_index
        self.indices = self._get_indices()

    def _get_indices(self):
        return random_batch_indice(self.dataset, self.num, self.data_index)

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


# get random continuous random numbers
@nb.jit()
def random_batch_indice(data_len, num, index_list):
    """
    :param data_len: length of dataset
    :param num: continuous random numbers, e.g. num=2
    """
    data_list = list(range(data_len))
    split_list = []
    for idx in range(data_len//num):
        batch = data_list[idx*num:(idx+1)*num]
        split_list.append(batch)
    # 移除包含 index_list 中元素的 batch
    split_list = [batch for batch in split_list if not any(item in index_list for item in batch)]
    split_list = np.array(split_list)
    np.random.shuffle(split_list)
    return split_list.reshape(-1)
