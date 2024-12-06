import logging
from pathlib import Path
import multiprocessing as mp
from typing import Callable
import ctypes

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


import code

class InextLeRobotDataset(LeRobotDataset):

    FILE_PATTERN = "episode_*.hdf5"

    def __init__(
        self,
        root: Path,
        image_transforms: Callable | None = None,
    ):
        self._compressed_images = "sim" not in root.name
        # Data directory
        self._root_dir = mp.RawArray(ctypes.c_char, str(root.absolute()).encode())
        self._image_transforms = image_transforms
        self._sample_count = 0
        self._episodes_count = 0

        episode_filenames = []
        for file_path in root.rglob(self.FILE_PATTERN):
            with h5py.File(file_path, "r") as data:
                logging.info(f"Checking {file_path}")
                self._check_data_format(data)
                # Get sample count for each episode
                self._sample_count += data["/action"].shape[0]

            # Add file name
            filename = file_path.relative_to(root)
            shared_str_filename = mp.RawArray(ctypes.c_char, str(filename).encode())

            episode_filenames.append(ctypes.cast(shared_str_filename, ctypes.c_char_p))
            self._episodes_count += 1

        assert self._episodes_count > 0, "No episodes found!"

        self._episode_filenames = mp.RawArray(ctypes.c_char_p, episode_filenames) 

        logging.info(f"Found total {self._episodes_count} episodes with {self._sample_count} samples")


    def _check_data_format(self, data: h5py._hl.files.File):
        assert "/action" in data
        assert "/observations/qpos" in data

        assert data["/action"].ndim == 2
        assert data["/observations/qpos"].ndim == 2

        num_frames = data["/action"].shape[0]
        assert num_frames == data["/observations/qpos"].shape[0]

        for camera in self._get_cameras(data):
            assert num_frames == data[f"/observations/images/{camera}"].shape[0]

            if self._compressed_images:
                assert data[f"/observations/images/{camera}"].ndim == 2
            else:
                assert data[f"/observations/images/{camera}"].ndim == 4
                b, h, w, c = data[f"/observations/images/{camera}"].shape
                assert c < h and c < w, f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided."

    def _get_cameras(self, hdf5_data):
        # TODO(rcadene): add depth
        return [key for key in hdf5_data["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


import time

def test_write_func(d: InextLeRobotDataset):
    print(id(d._root_dir))
    print(id(d._episode_filenames))
    c = 33
    for i in range(len(d.l)):
        #print("write:", id(p), p[:])
        p = d.l[i]
        p[-1] = chr(c).encode()
        f = d._episode_filenames[i]
        print("write:", p[:], f)
        time.sleep(1)
        c+=1

def test_read_func(d: InextLeRobotDataset):
    print(id(d._root_dir))
    print(id(d._episode_filenames))
    time.sleep(0.5)
    for p in d._episode_filenames:
        #print("read: ", id(p), p)
        print("read: ", p)
        time.sleep(1)

if __name__ == "__main__":

    i = InextLeRobotDataset(Path("/home/elton/data/inext/nov_11_to_dec_3_cam123/sim/dec_3/sim"))
    p1 = mp.Process(target=test_write_func, args=(i,))
    p2 = mp.Process(target=test_read_func, args=(i,))
    p1.start()
    p2.start()

    #code.interact(local=locals())
