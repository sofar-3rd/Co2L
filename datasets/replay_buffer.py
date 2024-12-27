import numpy as np


class ReplayBuffer:
    def __init__(self, n_sample_per_class=5):
        self.n_per_class = n_sample_per_class
        # 字典['类别':[sample1, sample2]...]
        self.dict = {}

    def update(self, feature, label, family):
        if family not in self.dict:
            self.dict[family] = {'feature': [], 'label': [], 'family': []}
        else:
            n = len(self.dict[family]['feature'])
            if n == self.n_per_class:
                index = np.random.randint(low=0, high=self.n_per_class)
                self.dict[family]['feature'][index] = feature
                self.dict[family]['label'][index] = label
                self.dict[family]['family'][index] = family

        self.dict[family]['feature'].append(feature)
        self.dict[family]['label'].append(label)
        self.dict[family]['family'].append(family)

    def save(self, file_path):
        # 将dict保存至file_path中
        np.save(file_path, self.dict, allow_pickle=True)

    def load(self, file_path):
        # 从file_path读取categories和buffer
        self.dict = np.load(file_path, allow_pickle=True).item()

    def get_sample(self):
        features = []
        labels = []
        families = []
        for _, v in self.dict.items():
            features += v['feature']
            labels += v['label']
            families += v['family']

        return features, labels, families
