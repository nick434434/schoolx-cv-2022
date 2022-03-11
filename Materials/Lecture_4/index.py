from typing import Union, Iterable, Tuple
import os
from pickle import dump, load

import faiss
import numpy as np
from cv2 import imread, waitKey
from faiss import Index, IndexFlatL2, read_index, IDSelectorRange

from face_worker import FaceWorker


class FeatureStorage:
    def __init__(self, filename_index: str, filename_users: Union[str, None] = None, num_features: int = 128,
                 users: Union[list, None] = None, empty_index: bool = False):
        self.num_features = num_features

        assert filename_users is not None or users is not None, ""
        self.index_filename = filename_index
        self.users_helper = filename_users if filename_users is not None else users

        self.index = self.init_index(empty_index)
        self.users = self.init_users(empty_index)

    def init_index(self, empty: bool = False) -> Index:
        if not empty and os.path.isfile(self.index_filename):
            index: IndexFlatL2 = read_index(self.index_filename)
        else:
            index = IndexFlatL2(self.num_features)
            if empty:
                # TODO: save to file
                pass
        return index

    def init_users(self, empty: bool = False) -> dict:
        if not empty and os.path.isfile(self.users_helper):
            with open(self.users_helper, "rb") as f:
                users = load(f)
        else:
            users = dict()
            for i in range(len(self)):
                if self.users_helper[i] not in users:
                    users[self.users_helper[i]] = [i]
                else:
                    users[self.users_helper[i]].append(i)
            if empty:
                # TODO: save to file
                pass
        return users

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, features: np.ndarray) -> None:
        if len(features.shape) == 1:
            features = features.reshape((1, -1))
        self.index.add(features.astype(np.float32))
        faiss.write_index(self.index, self.index_filename)

    def delete(self, start: int, end: int = -1) -> None:
        selector = IDSelectorRange(start, start + 1) if end == -1 else IDSelectorRange(start, end)
        self.index.remove_ids(selector)
        faiss.write_index(self.index, self.index_filename)

    def search(self, features: np.ndarray, neighbours: int) -> Iterable[Tuple[float, int]]:
        if len(features.shape) == 1:
            features = features.reshape((1, -1))
        dist, ind = self.index.search(features.astype(np.float32), neighbours)
        return zip(dist[0], ind[0])


if __name__ == "__main__":
    fs = FeatureStorage(filename_index="./data/features.index", filename_users="./data/users.pkl", num_features=128,
                        empty_index=True)

    img = imread("./data/face_image.png")
    fw = FaceWorker(img)
    fw()
    fw.show("Keanu based")
    assert len(fw.features) == 1
    fs.add(fw.features[0])

    img = imread("./data/keanu.jpg")
    fw(img)
    fw.show("Keanu diff")
    assert len(fw.features) == 1
    results_similar = fs.search(fw.features[0], 1)

    img = imread("./data/1.jpg")
    fw(img)
    fw.show("Random guy")
    assert len(fw.features) == 1
    results_different = fs.search(fw.features[0], 1)

    img = imread("./data/gyllenhaal.jpg")
    fw(img)
    fw.show("Bearded guy")
    assert len(fw.features) == 1
    results_different_2 = fs.search(fw.features[0], 1)

    print([res[0] for res in results_similar], [res[0] for res in results_different], [res[0] for res in results_different_2])
    waitKey(0)
