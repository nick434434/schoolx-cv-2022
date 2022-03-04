from typing import Union, Iterable, Tuple
import os
from pickle import dump, load

import faiss
import numpy as np
from faiss import Index, IndexFlatL2, read_index, IDSelector, IDSelectorRange


class FeatureStorage:
    def __init__(self, filename_index: str, filename_users: Union[str, None] = None, num_features: int = 128,
                 users: Union[list, None] = None):
        self.num_features = num_features

        assert filename_users is not None or users is not None, ""
        self.index_filename = filename_index
        self.users_helper = filename_users if filename_users is not None else users

        self.index = self.init_index()
        self.users = self.init_users()

    def init_index(self) -> Index:
        if os.path.isfile(self.index_filename):
            index: IndexFlatL2 = read_index(self.index_filename)
        else:
            index = IndexFlatL2(self.num_features)
        return index

    def init_users(self) -> dict:
        if os.path.isfile(self.users_helper):
            with open(self.users_helper, "rb") as f:
                users = load(f)
        else:
            users = dict()
            for i in range(len(self)):
                if self.users_helper[i] not in users:
                    users[self.users_helper[i]] = [i]
                else:
                    users[self.users_helper[i]].append(i)
        return users

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, features: np.ndarray) -> None:
        self.index.add(features)
        faiss.write_index(self.index_filename)

    def delete(self, start: int, end: int = -1) -> None:
        selector = IDSelector(start) if end == -1 else IDSelectorRange(start, end)
        self.index.remove_ids(selector)
        faiss.write_index(self.index_filename)

    def search(self, features: np.ndarray, neighbours: int) -> Iterable[Tuple[float, int]]:
        dist, ind = self.index.search(features, neighbours)
        return zip(dist[0], ind[0])
