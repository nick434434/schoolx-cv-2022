from typing import Union, Iterable, Tuple
import os
from pickle import dump, load
from multiprocessing import Lock

import faiss
import numpy as np
from cv2 import imread, waitKey
from faiss import Index, IndexFlatL2, read_index, IDSelectorRange, IDSelectorBatch

from Materials.Lecture_4.face_worker import FaceWorker
from Materials.config import CONFIG


class FeatureStorage:
    write_op_lock = Lock()

    def __init__(self, filename_index: str, filename_users: Union[str, None] = None,
                 num_features: int = 128, users: Union[list, None] = None, empty_index: bool = False):
        self.num_features = num_features

        assert filename_users is not None or users is not None, ""
        self.index_filename = filename_index
        self.users_filename = filename_users
        self.users_helper = users

        self.index = self.init_index(empty_index)
        self.users, self.users_faiss_list = self.init_users(empty_index)

    def init_index(self, empty: bool = False) -> Index:
        if not empty and os.path.isfile(self.index_filename):
            index: IndexFlatL2 = read_index(self.index_filename)
        else:
            index = IndexFlatL2(self.num_features)
            if empty:
                faiss.write_index(index, self.index_filename)
        return index

    def init_users(self, empty: bool = False) -> Tuple[dict, list]:
        if not empty and os.path.isfile(self.users_filename):
            with open(self.users_filename, "rb") as f:
                users, users_list = load(f)
        else:
            if self.users_helper is None and len(self) > 0:
                raise ValueError("users argument must contain a list of usernames behind each vector inside Faiss index"
                                 " file")
            users = dict()
            for i in range(len(self)):
                user = self.users_helper[i]
                if user not in users:
                    users[user] = {i}
                else:
                    users[user].add(i)

            users_list = self.users_helper if self.users_helper is not None else []
            if empty:
                with open(self.users_filename, "wb") as f:
                    dump((users, users_list), f)
        return users, users_list

    def update_with_files(self):
        # Ensuring all processes are up-to-date before making any operation
        self.index = self.init_index(empty=False)
        self.users, self.users_faiss_list = self.init_users(empty=False)

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, features: np.ndarray, username: str) -> None:
        self.update_with_files()

        if len(features.shape) == 1:
            features = features.reshape((1, -1))

        # Lock!
        FeatureStorage.write_op_lock.acquire(True, 1.)

        self.index.add(features.astype(np.float32))

        if username in self.users:
            self.users[username].add(len(self) - 1)
        else:
            self.users[username] = {len(self) - 1}
        self.users_faiss_list.append(username)

        faiss.write_index(self.index, self.index_filename)
        with open(self.users_filename, "wb") as f:
            dump((self.users, self.users_faiss_list), f)

        # Unlock!
        FeatureStorage.write_op_lock.release()

    def delete(self, start: Union[int, str], end: int = -1) -> None:
        self.update_with_files()

        if isinstance(start, int):
            selector = IDSelectorRange(start, start + 1) if end <= start else IDSelectorRange(start, end)

            # Lock!
            FeatureStorage.write_op_lock.acquire(True, 1.)

            self.index.remove_ids(selector)

            elems_to_remove = set()
            for elem in reversed(range(start, end)):
                elems_to_remove.add(elem)
                del self.users_faiss_list[elem]
            for key in self.users:
                self.users[key].difference_update(elems_to_remove)

            faiss.write_index(self.index, self.index_filename)
            with open(self.users_filename, "wb") as f:
                dump((self.users, self.users_faiss_list), f)

            # Unlock!
            FeatureStorage.write_op_lock.release()
        else:
            # Start is username in this case
            ids_to_be_deleted = self.users[start]
            selector = IDSelectorBatch(len(ids_to_be_deleted), list(ids_to_be_deleted))

            # Lock!
            FeatureStorage.write_op_lock.acquire(True, 1.)

            self.index.remove_ids(selector)
            self.users.pop(start)
            for id_remove in reversed(list(ids_to_be_deleted)):
                del self.users_faiss_list[id_remove]

            faiss.write_index(self.index, self.index_filename)
            with open(self.users_filename, "wb") as f:
                dump((self.users, self.users_faiss_list), f)

            # Unlock!
            FeatureStorage.write_op_lock.release()

    def search(self, features: np.ndarray, neighbours: int) -> Iterable[Tuple[float, int, str]]:
        self.update_with_files()

        if len(features.shape) == 1:
            features = features.reshape((1, -1))
        dist, ind = self.index.search(features.astype(np.float32), neighbours)
        usernames = [self.users_faiss_list[index] for index in ind[0]]
        return list(zip(dist[0], ind[0], usernames))


if __name__ == "__main__":
    fs = FeatureStorage(filename_index="data/test.index", filename_users="data/test_users.pkl",
                        num_features=128, empty_index=True)

    img = imread("./data/face_image.png")
    fw = FaceWorker(img)
    fw()
    fw.show("Keanu based")
    assert len(fw.features) == 1
    fs.add(fw.features[0], "kreeves")

    img = imread("./data/keanu.jpg")
    fw(img)
    fw.show("Keanu diff")
    assert len(fw.features) == 1
    keanu_features = fw.features[0]
    keanu = fs.search(keanu_features, 1)

    img = imread("./data/1.jpg")
    fw(img)
    fw.show("Random guy")
    assert len(fw.features) == 1
    random_guy = fs.search(fw.features[0], 1)

    img = imread("./data/gyllenhaal.jpg")
    fw(img)
    fw.show("Gyllenhaal 1")
    assert len(fw.features) == 1
    gyl_not_added = fs.search(fw.features[0], 1)

    # Add Gyllenhaal
    fs.add(fw.features[0], "jgyllenhaal")

    img = imread("./data/gyllenhaal_no_beard.jpg")
    fw(img)
    fw.show("Gyllenhaal 2")
    assert len(fw.features) == 1
    gyl_added = fs.search(fw.features[0], 1)

    keanu_after_gyl = fs.search(keanu_features, 1)

    print(keanu, random_guy, gyl_not_added, gyl_added, keanu_after_gyl)
    waitKey(0)
