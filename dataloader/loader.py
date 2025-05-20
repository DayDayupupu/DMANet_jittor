import numpy as np


class Loader:
    def __init__(self, dataset, mode, batch_size, num_workers, drop_last):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.mode = mode
        self.collate_fn = collate_events if mode == "training" else collate_events_test


        # 计算总批次数
        self.total_batches = len(dataset) // batch_size
        if not drop_last and len(dataset) % batch_size != 0:
            self.total_batches += 1

    def __iter__(self):
        """返回数据迭代器"""
        batch_data = []
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            batch_data.append(data)
            
            if len(batch_data) == self.batch_size:
                # 使用collate_fn处理批次数据
                # print(batch_data)
                yield self.collate_fn(batch_data)
                batch_data = []
        
        # 处理最后一个不完整的批次
        if not self.drop_last and len(batch_data) > 0:
            yield self.collate_fn(batch_data)

    def __len__(self):
        """返回数据加载器长度"""
        return self.total_batches


def collate_events(data):
    batch_labels = []
    batch_pos_events = []
    batch_neg_events = []
    idx_batch = 0

    for d in data:  # different batch
        for idx in range(len(d[0])):
            label = d[0][idx]
            lb = np.concatenate([label, idx_batch*np.ones((len(label), 1), dtype=np.float32)], 1)

            batch_labels.append(lb)
            idx_batch += 1
        batch_pos_events.append(d[1])
        batch_neg_events.append(d[2])
    labels = np.concatenate(batch_labels, 0)
    return labels, batch_pos_events, batch_neg_events


def collate_events_test(data):
    labels = []
    pos_events = []
    neg_events = []
    idx_batch = 0

    for d in data:
        for idx in range(len(d[0])):
            label = d[0][idx]
            lb = np.concatenate([label, idx_batch * np.ones((len(label), 1), dtype=np.float32)], 1)
            labels.append(lb)
            idx_batch += 1
        pos_events.append(d[1])
        neg_events.append(d[2])

    labels = np.concatenate(labels, 0)

    return labels, pos_events, neg_events
