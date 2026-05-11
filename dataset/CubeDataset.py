from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import os
import torch


class CubeDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        path,
        blocks_per_direction,
        block_sample_method,
        mode,
        snr_threshold=None,
        cube_size=(300, 300, 300),
        block_size=(128, 128, 128),
        swc_scale=(1.0, 1.0, 1.0),
        transforms=None,
        skip_ids=[],
    ):
        assert mode in ["cube", "block"]
        assert block_sample_method in ["order", "random"]
        self.blocks_per_direction = blocks_per_direction
        self.num_blocks_per_file = blocks_per_direction**3
        self.block_sample_method = block_sample_method
        self.cube_size = np.array(cube_size)
        self.block_size = np.array(block_size)
        self.transforms = transforms
        self.mode = mode
        self.swc_scale = swc_scale
        self.snr_threshold = snr_threshold
        ds = np.genfromtxt(path, delimiter=",", dtype=str)
        annos = []
        for d in ds:
            if os.path.basename(d[0]).replace(".tif", "") in skip_ids:
                continue
            annos.append(
                {
                    "img_path": os.path.join(dataset_dir, d[0]),
                    "swc_path": os.path.join(dataset_dir, d[1]),
                    "mask_path": os.path.join(dataset_dir, d[2]) if len(d) > 2 else None,
                }
            )
        self.annos = annos

    def __len__(self):
        if self.mode == "cube":
            return len(self.annos)
        elif self.mode == "block":
            return len(self.annos) * self.num_blocks_per_file

    def __getitem__(self, idx):
        if self.mode == "cube":
            anno = self.annos[idx]
            image = tiff.imread(anno["img_path"]).astype(np.float32)
            image = np.sqrt(image) / 255.0
            label = tiff.imread(anno["mask_path"]).astype(np.float32)
            cube_name = os.path.basename(anno["img_path"]).split(".")[0]
            if label is None:
                data = {"image": image, "cube_name": cube_name}
            else:
                data = {"image": image, "mask": label, "cube_name": cube_name}
        elif self.mode == "block":
            file_idx = idx // self.num_blocks_per_file
            anno = self.annos[file_idx]
            image = tiff.imread(anno["img_path"]).astype(np.float32)
            image = np.sqrt(image) / 255.0
            label = tiff.imread(anno["mask_path"]).astype(np.float32)
            if self.block_sample_method == "order":
                block_idx = idx % self.num_blocks_per_file
                step_size = (
                    self.cube_size - self.block_size
                ) // self.blocks_per_direction

                x_coord = block_idx // (self.blocks_per_direction**2)
                y_coord = (block_idx % (self.blocks_per_direction**2)) // (
                    self.blocks_per_direction
                )
                z_coord = block_idx % self.blocks_per_direction

                start_z = step_size[0] * z_coord
                start_y = step_size[1] * y_coord
                start_x = step_size[2] * x_coord
                end_z = start_z + self.block_size[0]
                end_y = start_y + self.block_size[1]
                end_x = start_x + self.block_size[2]
            elif self.block_sample_method == "random":
                start_ends = self.cube_size - self.block_size
                mask = None
                assert (
                    np.sum(label) >= 20
                ), f"Too few foreground pixels in the whole cube: {np.sum(label)}"
                max_attempts = 10
                attempts = 0
                while (mask is None or np.sum(mask) < 50) and attempts < max_attempts:
                    attempts += 1
                    start_z = np.random.randint(0, start_ends[0])
                    start_y = np.random.randint(0, start_ends[1])
                    start_x = np.random.randint(0, start_ends[2])
                    end_z = start_z + self.block_size[0]
                    end_y = start_y + self.block_size[1]
                    end_x = start_x + self.block_size[2]
                    mask = label[start_z:end_z, start_y:end_y, start_x:end_x]
            image = image[start_z:end_z, start_y:end_y, start_x:end_x]
            mask = label[start_z:end_z, start_y:end_y, start_x:end_x]
            data = {"image": image, "mask": mask}

        if self.snr_threshold is not None:
            from utils.image import compute_snr_proxy

            snr = compute_snr_proxy(data["image"])
            data["snr"] = torch.tensor([snr], dtype=torch.float32)
            data["snr_label"] = torch.tensor(
                [float(snr >= self.snr_threshold)], dtype=torch.float32
            )

        if self.transforms:
            data = self.transforms(data)
        return data
