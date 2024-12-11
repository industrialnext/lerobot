import logging
from pathlib import Path
import multiprocessing as mp
from typing import Callable
import ctypes
import gc

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION

from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import get_cameras

from datasets import Dataset, Features, Image, Sequence, Value


import code

class InextLeRobotDataset(LeRobotDataset):

    FILE_PATTERN = "episode_*.hdf5"

    def __init__(
        self,
        root: Path,
        fps: float,
        video: bool = False,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
    ):
        self._compressed_images = "sim" not in root.name
        # Data directory
        self._root_dir = mp.RawArray(ctypes.c_char, str(root.absolute()).encode())
        self._image_transforms = image_transforms
        self._delta_timestamps = delta_timestamps
        self._num_samples = 0
        self._num_episodes = 0
        self._features = None

        episode_filenames = []
        sample_index_ep_map = []
        for file_path in root.rglob(self.FILE_PATTERN):
            with h5py.File(file_path, "r") as data:
                logging.info(f"Checking {file_path}")
                check_aloha_data_format(data, self._compressed_images)
                # Get sample count for each episode
                num_sample = data["/action"].shape[0]
                if self._features is None:
                    self._features = create_hf_features(data, video)

            for _ in range(num_sample):
                sample_index_ep_map.append(self._num_samples)
                sample_index_ep_map.append(self._num_episodes)

            # Add file name
            filename = file_path.relative_to(root)
            shared_str_filename = mp.RawArray(ctypes.c_char, str(filename).encode())
            episode_filenames.append(ctypes.cast(shared_str_filename, ctypes.c_char_p))

            # Update count
            self._num_samples += num_sample
            self._num_episodes += 1

        gc.collect()

        assert self._num_episodes > 0, "No episodes found!"

        # Create index and episode file map
        shared_idx_map = mp.RawArray(ctypes.c_uint32, sample_index_ep_map)
        self._sample_index_map = np.frombuffer(shared_idx_map, dtype=np.uint32)
        # Reshape to 2d
        self._sample_index_map = self._sample_index_map.reshape((-1, 2))

        self._episode_filenames = mp.RawArray(ctypes.c_char_p, episode_filenames) 

        self.info = {
            "codebase_version": CODEBASE_VERSION,
            "fps": fps,
            "video": video,
        }

        logging.info(f"Found total {self._num_episodes} episodes with {self._num_samples} samples")

    @property
    def features(self) -> datasets.Features:
        pass

    @property
    def num_samples(self) -> int:
        """Number of samples/frames."""
        return self._num_samples

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return self._num_episodes

    def __getitem__(self, idx):
        pass

    def __repr__(self):
        pass


def check_aloha_data_format(data: h5py._hl.files.File, compressed_images: bool):
    assert "/action" in data
    assert "/observations/qpos" in data

    assert data["/action"].ndim == 2
    assert data["/observations/qpos"].ndim == 2

    num_frames = data["/action"].shape[0]
    assert num_frames == data["/observations/qpos"].shape[0]

    for camera in get_cameras(data):
        assert num_frames == data[f"/observations/images/{camera}"].shape[0]

        if compressed_images:
            assert data[f"/observations/images/{camera}"].ndim == 2
        else:
            assert data[f"/observations/images/{camera}"].ndim == 4
            b, h, w, c = data[f"/observations/images/{camera}"].shape
            assert c < h and c < w, f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided."


def create_hf_features(data: h5py._hl.files.File, video: bool) -> Dataset:
    features = {}

    for camera in get_cameras(data):
        if video:
            features["observation.images.{camera}"] = VideoFrame()
        else:
            features["observation.images.{camera}"] = Image()

    features["observation.state"] = Sequence(
        length=data["/observations/qpos"].shape[1], feature=Value(dtype="float32", id=None)
    )
    if "/observations/qvel" in data:
        features["observation.velocity"] = Sequence(
            length=data["/observations/qvel"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "/observations/effort" in data:
        features["observation.effort"] = Sequence(
            length=data["/observations/effort"].shape[1], feature=Value(dtype="float32", id=None)
        )
    features["action"] = Sequence(
        length=data["/action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    return features

def load_hdf5_to_dict(
    episode_path: Path,
    episode_index, int,
    total_index: int,
    index: int,
    fps: int,
    delta_timestamps: dict[list[float]] | None = None,
):
    data_dict = {}
    # Read an episode
    with h5py.File(episode_path, "r") as ep:
        num_frames = ep["/action"].shape[0]
        assert num_frames > index

        data_dict["observation.state"] = torch.from_numpy(ep["/observations/qpos"][index:index+1])
        data_dict["action"] = torch.from_numpy(ep["/action"][index:index+1])
        data_dict["episode_index"] = torch.tensor([episode_index])
        data_dict["frame_index"] = torch.tensor([index])
        data_dict["timestamp"] = torch.tensor([index]) / fps
        if delta_timestamps is not None:

        # last step of demonstration is considered done
        data_dict["next.done"] = torch.tensor([index == (num_frames - 1)], dtype=torch.bool)

        if "/observations/qvel" in ep:
            data_dict["observation.velocity"] = torch.from_numpy(ep["/observations/qvel"][index:index+1])
        if "/observations/effort" in ep:
            data_dict["observation.effort"] = torch.from_numpy(ep["/observations/effort"][index:index+1])

        for camera in get_cameras(ep):
            img_key = f"observation.images.{camera}"

            if compressed_images:
                import cv2

                # load one compressed image after the other in RAM and uncompress
                imgs_array = np.array([cv2.imdecode(ep[f"/observations/images/{camera}"][index], 1)])
            else:
                # load all images in RAM
                imgs_array = ep[f"/observations/images/{camera}"][index:index+1]

            data_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

    data_dict["index"] = torch.tensor([total_index])
    return data_dict

import time

def test_write_func(d: InextLeRobotDataset):
    print(id(d._root_dir))
    print(id(d._episode_filenames))
    #c = 33
    #for i in range(len(d.l)):
    #    #print("write:", id(p), p[:])
    #    p = d.l[i]
    #    p[-1] = chr(c).encode()
    #    f = d._episode_filenames[i]
    #    print("write:", p[:], f)
    #    time.sleep(1)
    #    c+=1

    #for i in range(len(d._sample_index_map)):
    #    print(d._sample_index_map[i])
    #    d._sample_index_map[i][0] = 10100 + i
    #    print("write:", d._sample_index_map[i])
    #    time.sleep(1)
    print("write: ", d._num_samples)
    d._num_samples = -1010101
    print("write: ", d._num_samples)

def test_read_func(d: InextLeRobotDataset):
    print(id(d._root_dir))
    print(id(d._episode_filenames))
    time.sleep(5)
    #for p in d._episode_filenames:
    #    #print("read: ", id(p), p)
    #    print("read: ", p)
    #    time.sleep(1)

    #for i in range(len(d._sample_index_map)):
    #    print("read: ", d._sample_index_map[i])
    #    time.sleep(1)

    print("read: ", d._num_samples)

if __name__ == "__main__":
    i = InextLeRobotDataset(Path("/home/elton/data/inext/nov_11_to_dec_3_cam123/sim/dec_3/sim"))
    print(i._sample_index_map)
    print(i._sample_index_map.shape)

    p1 = mp.Process(target=test_write_func, args=(i,))
    p2 = mp.Process(target=test_read_func, args=(i,))
    p1.start()
    p2.start()

    #code.interact(local=locals())
