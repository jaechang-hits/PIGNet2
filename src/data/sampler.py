import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import time


class PDBBatchSampler(Sampler):
    def __init__(self, keys, batch_size):
        """
        Args:
            dataset_size (int): 전체 데이터셋 크기
            group_ids (list): 각 데이터의 그룹 ID를 담은 리스트
        """
        super().__init__(None)  # None 대신 데이터셋을 넣어도 됩니다
        self.dataset_size = len(keys)
        self.group_ids = np.array(self._make_group(keys))

        # 유니크한 그룹 ID들을 가져옴
        self.unique_groups = np.unique(self.group_ids)

        # 각 그룹별 인덱스 저장
        self.group_indices = {
            group: np.where(self.group_ids == group)[0] for group in self.unique_groups
        }
        self.batch_size = batch_size

    def _make_group(self, keys):
        pdb_ids = sorted(list(set([k.split("_")[0] for k in keys])))
        pdb_id_to_group_id = {pdb: i for i, pdb in enumerate(pdb_ids)}
        group_ids = []
        for i, key in enumerate(keys):
            pdb_id = key.split("_")[0]
            group_ids.append(pdb_id_to_group_id[pdb_id])
        return group_ids

    def __iter__(self):
        seed = int(time.time() * 1000)
        random.seed(seed)
        # 그룹별 인덱스를 섞음
        indices = []
        group_indices_copy = dict(self.group_indices)
        for group in group_indices_copy:
            random.shuffle(group_indices_copy[group])
        # 각 그룹에서 랜덤하게 샘플링
        while len(indices) < self.dataset_size:
            # group = list(group_indices_copy.keys())[0]
            group_list = list(group_indices_copy.keys())
            random.shuffle(group_list)
            for group in group_list:
                if len(group_indices_copy[group]) > 0:
                    # 현재 그룹에서 가능한 만큼 인덱스 추가
                    available = min(self.batch_size, len(group_indices_copy[group]))
                    batch_indices = group_indices_copy[group][:available]
                    indices.extend(batch_indices)
                    group_indices_copy[group] = group_indices_copy[group][available:]
                    if len(group_indices_copy[group]) == 0:
                        del group_indices_copy[group]
        return iter(indices)
