"""
Function for loading data
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class MalwareDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, data, labels, families):
        self.drebin_inputs = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.families = families

    def __len__(self):
        # 返回数据集大小
        return len(self.drebin_inputs)

    def __getitem__(self, idx):
        # 获取单个样本，返回特征、标签、家族信息
        drebin_input = self.drebin_inputs[idx]
        label = self.labels[idx]
        family = self.families[idx]
        return drebin_input, label, family


def load_range_dataset(start_month, end_month, dataset_name="gen_androzoo_drebin"):
    if start_month != end_month:
        file_name = f'{start_month}to{end_month}_selected.npz'
    else:
        file_name = f'{start_month}_selected.npz'

    root_dir = os.getcwd()
    abs_path = os.path.join(root_dir, "data", dataset_name, file_name)

    print(abs_path)

    data = np.load(abs_path, allow_pickle=True)
    drebin_inputs, labels, families = data['X_train'], data['y_train'], data['y_mal_family']

    # convert to one-hot vector
    binary_labels = labels[labels != 0] = 1
    num_classes = 2
    one_hot_labels = np.eye(num_classes)[binary_labels]

    # all_train_family has 'benign'
    # 将 label 为 0 的软件的家族设置为 benign
    benign_len = drebin_inputs.shape[0] - families.shape[0]
    benign_family = np.full(benign_len, 'benign')
    all_families = np.concatenate((families, benign_family), axis=0)

    return drebin_inputs, binary_labels, all_families


if __name__ == "__main__":
    drebin_inputs, labels, families = load_range_dataset("2019-01", "2019-12")
    trainDataset = MalwareDataset(drebin_inputs, labels, families)
    print(len(trainDataset))
    print(trainDataset[0: 10])
