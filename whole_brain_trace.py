from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple
from collections import deque
import torch
import os
import re
import time
import ast
import numpy as np
import json
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import click
import tifffile as tiff
from monai.transforms import ScaleIntensityRangePercentiles, ScaleIntensity

from swclib.data.swc import Swc
from swclib.data.swc_node import SwcNode
from swclib.data.swc_forest import SwcForest
from swclib.whole_brain.tifreader import WBTReader
from swclib.image.segment_soma import segment_soma_from_seed

from skeleton import run_trace

Vec3 = Tuple[int, int, int]


@dataclass(frozen=True)
class Cube:
    iz: int
    iy: int
    ix: int
    start_coord: Vec3
    end_coord: Vec3
    cube_size: Vec3


class _AxisGrid:
    """
    1D grid with stride.
    """

    def __init__(self, axis_len: int, stride: int, tail_mode: str = "padding"):
        """
        tail_mode: "padding" (default) means the last window is [last, last+win) but only part of it is valid;
        """
        assert axis_len > 0
        assert stride > 0
        self.axis_len = axis_len
        self.stride = stride
        self.tail_mode = tail_mode

        if axis_len % stride == 0:
            self.max_n = axis_len // stride
        else:
            self.max_n = axis_len // stride + 1  # include tail start

    def start_at(self, i: int) -> int:
        if not (0 <= i < self.max_n):
            raise IndexError(f"axis index out of range: {i} (max_n={self.max_n})")
        return i * self.stride

    def is_valid_index(self, i: int) -> bool:
        return 0 <= i < self.max_n

    def index_of_floor(self, x: int) -> int:
        """
        Given x in [0, axis_len), return an index i such that start_at(i) <= x
        (closest start not greater than x, with tail handled).
        """
        if x <= 0:
            return 0
        i = x // self.stride
        return int(i)


class Grid:
    """
    Implicit grid for brain ROI:
    - roi_bbox: ((x0,y0,z0), (x1,y1,z1)) in global voxel coordinates
    - cube_size: (cx, cy, cz) in voxels, e.g. (300,300,300)
    - step_size: (sx, sy, sz) in voxels, e.g. (200,200,200), the stride of the grid, can be smaller than cube_size for better coverage and more candidates, but also more redundancy and computation.
    """

    def __init__(
        self,
        roi_bbox: Tuple[Vec3, Vec3] = ((17416, 0, 0), (33509, 25698, 11600)),
        cube_size: Vec3 = (300, 300, 300),
        step_size: Vec3 = (200, 200, 200),
    ):
        self.roi_min = roi_bbox[0]
        self.roi_max = roi_bbox[1]
        self.cube_size = cube_size
        self.step_size = step_size

        x0, y0, z0 = roi_bbox[0]
        x1, y1, z1 = roi_bbox[1]
        self.depth = z1 - z0
        self.height = y1 - y0
        self.width = x1 - x0

        self.ax = _AxisGrid(self.width, step_size[0])
        self.ay = _AxisGrid(self.height, step_size[1])
        self.az = _AxisGrid(self.depth, step_size[2])

    # ---------------------------
    # Basic indexing (cube index space)
    # ---------------------------
    def cube_origin_from_index(self, idx_xyz: Vec3) -> Vec3:
        ix, iy, iz = idx_xyz
        x0, y0, z0 = self.roi_min
        ox = x0 + self.ax.start_at(ix)
        oy = y0 + self.ay.start_at(iy)
        oz = z0 + self.az.start_at(iz)
        return (ox, oy, oz)

    def get_cube_by_index(self, idx_xyz: Vec3):
        """
        Return Cube if this 100^3 cell is a center cell of some cube; otherwise MarginCell.
        Args:
            - idx_xyz: cell index (ix, iy, iz) in cell grid
        """
        origin = self.cube_origin_from_index(idx_xyz)

        return Cube(
            ix=idx_xyz[0],
            iy=idx_xyz[1],
            iz=idx_xyz[2],
            start_coord=origin,
            end_coord=(
                origin[0] + self.cube_size[0],
                origin[1] + self.cube_size[1],
                origin[2] + self.cube_size[2],
            ),
            cube_size=self.cube_size,
        )

    # ---------------------------
    # Query by voxel coordinate
    # ---------------------------
    def find_best_cube_for_point(
        self, coord: Tuple[float, float, float], topk: int = 3
    ) -> List[Tuple[Cube, float]]:
        """
        Given a voxel coordinate (x, y, z), find up to topk cubes (300^3) that contain
        this point, rank them by distance from cube center to the point, and return:

            [(Cube, distance_to_cube_center), ...]

        sorted by ascending distance.

        If the point is outside ROI, return an empty list.
        """
        x, y, z = coord
        x0, y0, z0 = self.roi_min
        x1, y1, z1 = self.roi_max

        if not (x0 <= x < x1 and y0 <= y < y1 and z0 <= z < z1):
            return []

        # Relative coordinate inside ROI
        rx, ry, rz = (x - x0, y - y0, z - z0)
        ix, iy, iz = (
            int(rx // self.step_size[0]),
            int(ry // self.step_size[1]),
            int(rz // self.step_size[2]),
        )

        # Candidate cube starts along each axis
        ix_starts = [ix - 1, ix, ix + 1]
        iy_starts = [iy - 1, iy, iy + 1]
        iz_starts = [iz - 1, iz, iz + 1]

        ix_starts = [s for s in ix_starts if 0 <= s < self.ax.max_n]
        iy_starts = [s for s in iy_starts if 0 <= s < self.ay.max_n]
        iz_starts = [s for s in iz_starts if 0 <= s < self.az.max_n]

        if not ix_starts or not iy_starts or not iz_starts:
            return []

        candidates = []  # [(dist2, ix, iy, iz, sx, sy, sz), ...]

        for ix in ix_starts:
            for iy in iy_starts:
                for iz in iz_starts:
                    sx, sy, sz = self.cube_origin_from_index((ix, iy, iz))

                    if not (sx <= x < sx + self.cube_size[0]):
                        continue
                    if not (sy <= y < sy + self.cube_size[1]):
                        continue
                    if not (sz <= z < sz + self.cube_size[2]):
                        continue

                    # Cube center
                    ccx = sx + self.cube_size[0] / 2.0
                    ccy = sy + self.cube_size[1] / 2.0
                    ccz = sz + self.cube_size[2] / 2.0

                    dx = x - ccx
                    dy = y - ccy
                    dz = z - ccz
                    dist2 = dx * dx + dy * dy + dz * dz

                    candidates.append((dist2, ix, iy, iz, sx, sy, sz))

        if not candidates:
            return []

        candidates.sort(key=lambda t: t[0])

        results = []
        for dist2, ix, iy, iz, sx, sy, sz in candidates[:topk]:
            cube = Cube(
                ix=ix,
                iy=iy,
                iz=iz,
                start_coord=(sx, sy, sz),
                end_coord=(
                    sx + self.cube_size[0],
                    sy + self.cube_size[1],
                    sz + self.cube_size[2],
                ),
                cube_size=self.cube_size,
            )
            results.append(cube)

        return results


class LocalSwcIno:
    """
    {
        "nid":{"merged_nid": (nid_old, nid_new), ...},
         ...
    }
    """
    def __init__(self, json_path):
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                self.info = json.load(f)
        elif os.path.exists(json_path.replace(".json", ".swc")):
            self.info = self.create_local_swc_info(Swc(json_path.replace(".json", ".swc")))
            self.save_to_json(json_path)
        else:
            raise FileNotFoundError(f"Neither {json_path} nor {json_path.replace('.json', '.swc')} exists.")
        self._rebuild_global_index()
    
    def create_local_swc_info(self, swc):
        info = {}
        for node in swc.nodes:
            info[str(node)] = {"merged_nid": (None, None), "global_nid":None}
        return info

    def _rebuild_global_index(self):
        self.global_to_local_index = {}
        for local_nid, value in self.info.items():
            global_nid = value.get("global_nid", None)
            if global_nid is not None:
                self.global_to_local_index.setdefault(global_nid, set()).add(local_nid)

    def _remove_from_global_index(self, local_nid, global_nid):
        if global_nid is None:
            return
        local_nids = self.global_to_local_index.get(global_nid)
        if local_nids is None:
            return
        local_nids.discard(local_nid)
        if len(local_nids) == 0:
            del self.global_to_local_index[global_nid]

    def _add_to_global_index(self, local_nid, global_nid):
        if global_nid is None:
            return
        self.global_to_local_index.setdefault(global_nid, set()).add(local_nid)

    def get_merged_node(self, nid_old):
        nid_old = str(nid_old)
        assert nid_old in self.info
        return self.info[nid_old].get("merged_nid", (None, None))

    def get_global_nid(self, nid_old):
        nid_old = str(nid_old)
        assert nid_old in self.info, f"nid {nid_old} not in local swc info"
        return self.info[nid_old].get("global_nid", None)
    
    def global_to_local(self, global_nid):
        local_nids = self.global_to_local_index.get(global_nid)
        if not local_nids:
            return None
        return next(iter(local_nids))
    
    def set_info(self, local_nid, global_nid, merged_nid):
        local_nid = str(local_nid)
        assert local_nid in self.info
        old_global_nid = self.info[local_nid].get("global_nid", None)
        self._remove_from_global_index(local_nid, old_global_nid)
        self.info[local_nid]["global_nid"] = global_nid
        self.info[local_nid]["merged_nid"] = merged_nid
        self._add_to_global_index(local_nid, global_nid)

    def set_null_by_global_nid(self, global_nids):
        for global_nid in global_nids:
            local_nids = list(self.global_to_local_index.get(global_nid, ()))
            for local_nid in local_nids:
                self.info[local_nid]["global_nid"] = None
                self.info[local_nid]["merged_nid"] = (None, None)
                self._remove_from_global_index(local_nid, global_nid)
    
    def save_to_json(self, json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.info, f, ensure_ascii=False, indent='\t')


class WholeBrainTrace:
    """
    Main class for whole-brain tracing.
    """

    def __init__(
        self,
        slice_dir: str = "/data2/CH1/slices",
        slice_name_pattern: str = "CH1_{z:05d}.tif",
        work_dir: str = "outputs",
        trace_model_cfg: dict = {
            "method": "seg+trace",
            "seg_model_cfg_path": "models/dynunet.py",
            "seg_model_ckpt_path": "outputs/neuron-seg/CH1-iter10000/dynunet-dice/2026-01-26-14-06-56/ckpts/dynunet-dice_step-8500_valmetric_NSD@1-0.9128.ckpt",
            "trace_model_name": "Kimimaro",
        },
        grid_params: dict = {
            "roi_bbox": None,
            "cube_size": (300, 300, 300),
            "step_size": (200, 200, 200),
        },
        merge_params: dict = {
            "merge_iou_threshold": 0.9,
            "merge_match_max_dist": 1,
            "min_newtree_length": 3,
        },
        search_params: dict = {
            "mode": "bfs",  # "dfs" or "bfs",
            "candidate_node_dist_to_margin": 3,
            "max_search_iter": 2000,
        },
        post_process_params: dict = {
            "resample_distance": 2.0,
            "remove_duplicate_nodes": True,
        },
        save_intermediate_tree: bool = True,
        save_intermediate_every: int = 50,
        verbose: bool = True,
    ):
        self.slice_dir = slice_dir
        self.work_dir = work_dir
        self.trace_model_cfg = trace_model_cfg
        self.grid_params = grid_params
        self.merge_params = merge_params
        self.search_params = search_params
        self.post_process_params = post_process_params
        self.save_intermediate_tree = save_intermediate_tree
        self.save_intermediate_every = save_intermediate_every
        self.verbose = verbose

        # tracing model
        if self.trace_model_cfg["method"] == "seg+trace":
            from omegaconf import OmegaConf
            import hydra
            from monai.inferers.inferer import SlidingWindowInfererAdapt

            cfg = OmegaConf.load(self.trace_model_cfg["seg_model_cfg_path"])
            self.seg_model = hydra.utils.instantiate(cfg).to("cuda:0")
            ckpt = torch.load(
                self.trace_model_cfg["seg_model_ckpt_path"], map_location=f"cuda:0"
            )
            model_chkpt = {
                k.replace("model.", ""): e
                for k, e in ckpt["state_dict"].items()
                if "model" in k
            }
            self.seg_model.load_state_dict(model_chkpt)
            self.seg_model.eval()
            self.sliding_window_inferer = SlidingWindowInfererAdapt(
                roi_size=(128, 128, 128), sw_batch_size=4, overlap=0.5
            )
        # sub cube reader
        self.cube_reader = WBTReader(slice_dir, slice_name_pattern)
        if self.grid_params["roi_bbox"] is None:
            self.grid_params["roi_bbox"] = self.cube_reader.get_bbox()

        # cache swc results for each cube by "cube_{ix}_{iy}_{iz}.swc" under grid_cache_dir
        self.grid_cache_dir = f"{work_dir}/grid_cache"
        self.cube_mask_cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(work_dir)),
            ".cube_mask_cache",
            self._seg_model_name(),
        )
        self.cube_cache_dir = f"{os.path.dirname(os.path.dirname(work_dir))}/.cube_cache"
        self.grid_fmt = "cube_{ix}_{iy}_{iz}.swc"
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.grid_cache_dir, exist_ok=True)
        if self.trace_model_cfg["method"] == "seg+trace":
            os.makedirs(self.cube_cache_dir, exist_ok=True)
            os.makedirs(self.cube_mask_cache_dir, exist_ok=True)
        self.grid = Grid(**self.grid_params)

        # log
        if self.save_intermediate_tree:
            self.imtermediate_tree_dir = f"{work_dir}/intermediate_trees"
            os.makedirs(self.imtermediate_tree_dir, exist_ok=True)
        self.log_path = os.path.join(work_dir, "trace_log.jsonl")

    # ----------------------------
    # Logging utilities
    # ----------------------------
    def _now_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def _tic(self):
        return time.perf_counter()

    def _toc(self, tic):
        return round(time.perf_counter() - tic, 6)

    def _write_log(self, **kwargs):
        record = {
            "time": self._now_str(),
            **kwargs,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _wait_for_stable_file(self, path: str, timeout: float = 30.0, interval: float = 0.2) -> bool:
        deadline = time.time() + timeout
        last_size = -1
        while time.time() < deadline:
            try:
                stat = os.stat(path)
            except OSError:
                time.sleep(interval)
                continue
            size = stat.st_size
            if size > 0 and time.time() - stat.st_mtime > 2.0:
                return True
            if size > 0 and size == last_size:
                return True
            last_size = size
            time.sleep(interval)
        return False

    def _read_tiff_cache(
        self,
        path: str,
        use_memmap: bool = True,
        expected_shape: Optional[Tuple[int, ...]] = None,
        wait_timeout: float = 30.0,
    ):
        if not os.path.exists(path):
            return None
        if not self._wait_for_stable_file(path, timeout=wait_timeout):
            print(f"Warning: cache file is not stable yet, regenerating it: {path}")
            return None
        try:
            if use_memmap:
                try:
                    data = tiff.memmap(path)
                except Exception:
                    data = tiff.imread(path)
            else:
                data = tiff.imread(path)
            if expected_shape is not None and tuple(data.shape) != tuple(expected_shape):
                print(
                    f"Warning: cache shape mismatch for {path}: "
                    f"got {tuple(data.shape)}, expected {tuple(expected_shape)}"
                )
                return None
            return data
        except Exception as e:
            print(f"Warning: failed to read cache {path}, regenerating it: {e}")
            return None

    def _is_tiff_cache_ready(
        self, path: str, expected_shape: Optional[Tuple[int, ...]] = None
    ) -> bool:
        return (
            self._read_tiff_cache(
                path,
                use_memmap=False,
                expected_shape=expected_shape,
            )
            is not None
        )

    def _write_tiff_cache_atomic(self, path: str, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}.tif"
        try:
            tiff.imwrite(
                tmp_path,
                data,
                compression=None,
                metadata=None,
            )
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _seg_model_name(self) -> str:
        seg_model_name = self.trace_model_cfg.get("seg_model_name")
        if seg_model_name is None:
            seg_model_cfg_path = self.trace_model_cfg.get("seg_model_cfg_path", "")
            seg_model_name = os.path.splitext(os.path.basename(seg_model_cfg_path))[0]
        seg_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(seg_model_name)).strip("_")
        return seg_model_name or "unknown_seg_model"

    # ----------------------------
    # Core functions
    # ----------------------------
    def trace_cube(self, cube: Cube):
        swc_path = os.path.join(
            self.grid_cache_dir,
            self.grid_fmt.format(ix=cube.ix, iy=cube.iy, iz=cube.iz),
        )
        if os.path.exists(swc_path):
            return swc_path
        start = cube.start_coord
        end = cube.end_coord
        cube_mask_save_path = os.path.join(
            self.cube_mask_cache_dir,
            f"cube_{cube.ix}_{cube.iy}_{cube.iz}.tif",
        )
        if self.trace_model_cfg["method"] == "seg+trace":
            trace_needs_mask_path = self.trace_model_cfg["trace_model_name"] in [
                "APP2",
                "neuTube",
                "smartTrace",
            ]
            expected_shape = cube.cube_size[::-1]
            seg = None
            if trace_needs_mask_path and self._is_tiff_cache_ready(
                cube_mask_save_path, expected_shape=expected_shape
            ):
                seg = cube_mask_save_path
            elif os.path.exists(cube_mask_save_path):
                seg = self._read_tiff_cache(
                    cube_mask_save_path, expected_shape=expected_shape
                )

            if seg is None:
                cube_path = os.path.join(
                    self.cube_cache_dir,
                    f"cube_{cube.ix}_{cube.iy}_{cube.iz}.tif",
                )
                vol = self._read_tiff_cache(cube_path, expected_shape=expected_shape)
                if vol is None:
                    vol = self.cube_reader.read_region(start[::-1], end[::-1])
                    if tuple(vol.shape) != tuple(expected_shape):
                        raise ValueError(
                            f"Unexpected cube shape for cube "
                            f"({cube.ix},{cube.iy},{cube.iz}): "
                            f"got {tuple(vol.shape)}, expected {tuple(expected_shape)}"
                        )
                    self._write_tiff_cache_atomic(cube_path, vol)
                vol_tensor = (
                    np.sqrt(vol) / 255.0
                )  # simple normalization for now, can be improved
                vol_tensor = np.clip(vol_tensor, 0, 1)
                vol_tensor = torch.from_numpy(vol_tensor).float().unsqueeze(0).unsqueeze(0)
                with torch.inference_mode():
                    seg = (
                        self.sliding_window_inferer(
                            vol_tensor.to("cuda:0"),
                            self.seg_model,
                        )
                        .squeeze()
                        .cpu()
                    )
                    seg = (seg.sigmoid() * 255).numpy().astype(np.uint8)
                self._write_tiff_cache_atomic(cube_mask_save_path, seg)
                if trace_needs_mask_path:
                    seg = cube_mask_save_path
            ok, _ = run_trace(seg, swc_path, self.trace_model_cfg["trace_model_name"], max_try=3)
            if not ok:
                print(f"Error: trace failed for cube ({cube.ix},{cube.iy},{cube.iz}), mask saved at {cube_mask_save_path}")
                swc = Swc()
                swc.save_to_swc(swc_path, reindex=True, sort_by_id=True, radius=1)
                return swc_path
                # raise RuntimeError(
                #     f"trace failed: method={self.trace_model_cfg['trace_model_name']}, "
                #     f"cube=({cube.ix},{cube.iy},{cube.iz}), mask={seg}, swc={swc_path}"
                # )
        swc = Swc(swc_path)
        swc.add_offset(start)  # convert to global coord
        if self.post_process_params["resample_distance"] is not None:
            swc.resample(min_distance=self.post_process_params["resample_distance"])
        swc.save_to_swc(swc_path, reindex=True, sort_by_id=True, radius=1)
        return swc_path

    def get_node_distance_to_cube_marigin(self, node: SwcNode, cube: Cube):
        cx0, cy0, cz0 = cube.start_coord
        cx1, cy1, cz1 = cube.end_coord
        x, y, z = node.coord
        dz = min(abs(z - cz0), abs(z - cz1))
        dy = min(abs(y - cy0), abs(y - cy1))
        dx = min(abs(x - cx0), abs(x - cx1))
        return min(dx, dy, dz)

    def merge_tree(
        self,
        T,
        swc_path,
        cube: Cube,
        leaf: SwcNode,
        merge_iou_threshold=0.9,
        merge_match_max_dist=1,
        min_newtree_length=3,
        is_loose=False,
        iter=None,
    ):
        """Merge the traced sub-tree from swc_path into the global tree T, using edge matching with from_fibers."""
        roi = (cube.start_coord, cube.end_coord)
        Tnodes = T.get_node_list(roi=roi)
        if len(Tnodes) == 0 or leaf not in Tnodes:
            return T, []
        Tnodes_kdtree = cKDTree([n[:] for n in Tnodes])
        swc = SwcForest(swc_path)
        if swc.size() == 0:
            if self.verbose:
                print(f"Warning: traced SWC has no nodes, skip merge: {swc_path}")
            return T, []
        local_swc_info = LocalSwcIno(swc_path.replace(".swc", ".json"))
        from_fibers = leaf.get_rerooted_subtree_fibers(roi)
        if len(from_fibers) == 0:
            return T, []

        # find junction node
        candidates = swc.get_nearest_node(
            leaf.coord, topk=max(5, int(merge_match_max_dist * 5))
        )
        candidates_new = []
        for candidate in candidates:
            if not isinstance(candidate, (tuple, list)) or len(candidate) < 2:
                if self.verbose:
                    print(f"Warning: invalid nearest-node candidate: {candidate}")
                continue
            node, dist = candidate[0], candidate[1]
            if dist > merge_match_max_dist:
                break
            f = True
            for n in candidates_new:
                if n[0].root == node.root:
                    f = False
                    break
            if f:
                candidates_new.append((node, dist))
        candidates = candidates_new

        # merge junction node
        candiate_t_junction_nodes = []
        for candidate in candidates:
            if not isinstance(candidate, (tuple, list)) or len(candidate) < 2:
                if self.verbose:
                    print(f"Warning: invalid nearest-node candidate: {candidate}")
                continue
            node, dist = candidate[0], candidate[1]
            junction_node_t, old2new = node.get_rerooted_tree(
                nid_start=T.next_id(),
                ntype=0 if leaf.ntype == 1 else leaf.ntype,
                return_old2new=True,
            )
            new2old = {v: k for k, v in old2new.items()}
            fibers = junction_node_t.get_subtree_fibers(roi=roi, with_root=True)
            is_junction = False
            for fiber in fibers:
                l1 = fiber.length
                for from_fiber in from_fibers:
                    l2 = from_fiber.length
                    overlap = fiber.get_overlap_length_with(
                        from_fiber, dist_threshold=merge_match_max_dist
                    )
                    if (
                        overlap / (l1 + 1e-8) > merge_iou_threshold
                        or overlap / (l2 + 1e-8) > merge_iou_threshold
                        or (is_loose and overlap >= 5)
                    ):
                        is_junction = True
                        if fiber[1].parent is not None and overlap > merge_match_max_dist * 1.05:
                            fiber[1].parent = None
                        break
            if is_junction and junction_node_t.get_subtree_length() >= min_newtree_length:
                # check if subtree has been merged to global tree, if so break the global tree at the first branch point to avoid cycle
                # local_new<-->local<-->global
                junction_node_t_local = new2old[junction_node_t]
                global_nid = local_swc_info.get_global_nid(junction_node_t_local.nid)
                if global_nid is not None:
                    merged_local_nid, merged_global_nid = local_swc_info.get_merged_node(junction_node_t_local.nid)
                    merged_local = swc.get_node_by_nid(merged_local_nid)
                    if merged_local in old2new:
                        # has been merged to another node
                        # break in local new
                        break_node_local_new = junction_node_t
                        merged_local_new = old2new[merged_local]
                        temp_local_new = merged_local_new
                        while temp_local_new.parent is not None and temp_local_new != junction_node_t:
                            if len(temp_local_new.parent.children) > 1:
                                break_node_local_new = temp_local_new
                            temp_local_new = temp_local_new.parent
                        for child in break_node_local_new.children:
                            if child is not None:
                                child.parent = None
                        # break in global tree
                        break_node_global_nid = local_swc_info.get_global_nid(new2old[break_node_local_new].nid)
                        break_node_global = T.get_node_by_nid(break_node_global_nid)
                        if break_node_global is not None:
                            break_node_global.parent = None
                            break_node_global_subtree_nodes = break_node_global.get_subtree_node_list()
                            local_swc_info.set_null_by_global_nid([n.nid for n in break_node_global_subtree_nodes])
                        else:
                            # from cubes' overlap area
                            print(f"Warning: break node global nid {break_node_global_nid} not found in T")
                # mask overlap nodes
                nodes = junction_node_t.get_subtree_node_list()
                if len(nodes) > 0:
                    old_nodes = [new2old[node] for node in nodes]
                    old_coords = np.array(
                        [[node[0], node[1], node[2]] for node in old_nodes],
                        dtype=float,
                    )
                    leaf_coord = np.array([leaf[0], leaf[1], leaf[2]], dtype=float)
                    keep_mask = np.linalg.norm(old_coords - leaf_coord, axis=1) >= 5
                    if np.any(keep_mask):
                        dists = Tnodes_kdtree.query(old_coords[keep_mask])[0]
                        nodes_to_mask = np.array(nodes, dtype=object)[keep_mask][dists <= 2]
                        old_nodes_to_mask = np.array(old_nodes, dtype=object)[keep_mask][dists <= 2]
                        for node, node_old in zip(nodes_to_mask, old_nodes_to_mask):
                            node.parent = None
                            node_old.parent = None
                swc.save_to_file(swc_path)
                if junction_node_t.get_subtree_length() < min_newtree_length:
                    continue
                T.link_child(leaf, junction_node_t)
                candiate_t_junction_nodes.append(junction_node_t)
                merged_local_nid, merged_global_nid = new2old[junction_node_t].nid, junction_node_t.nid
                for node in junction_node_t.get_subtree_node_list():
                    local_nid = new2old[node].nid
                    local_swc_info.set_info(local_nid, node.nid, (merged_local_nid, merged_global_nid))
                local_swc_info.save_to_json(swc_path.replace(".swc", ".json"))
        return T, candiate_t_junction_nodes

    def run(self, T: SwcForest, Q=None, final_save_path=None, iter=0):
        """Run the whole brain tracing process, starting from initial tree T and trace queue Q."""
        if final_save_path is None:
            final_save_path = os.path.join(self.work_dir, f"tree.swc")
        if Q is None:
            Q = T.get_leaf_nodes()
        loose_iters = max(3, len(Q))
        if self.search_params["mode"] == "dfs":
            container = list(Q)
            pop_node = container.pop
            push_node = container.append
        elif self.search_params["mode"] == "bfs":
            container = deque(Q)
            pop_node = container.popleft
            push_node = container.append
        else:
            raise ValueError(f"Invalid search mode: {self.search_params['mode']}")
        visited = set()
        queued = {tuple(q[:]) for q in container}
        start_time = time.time()
        while container:
            iter += 1
            if iter > self.search_params["max_search_iter"]:
                print(
                    f"Reached max search iterations ({self.search_params['max_search_iter']}), stop."
                )
                break
            if self.verbose:
                print(f"Iteration {iter}, queue size: {len(container)}")

            # trace new cube from a endpoint node
            node = pop_node()
            queued.discard(tuple(node[:]))
            if node not in T.get_node_list():
                continue
            cubes = self.grid.find_best_cube_for_point(node, topk=3)
            visited.add(tuple(node[:]))
            trace_time, merge_time = 0, 0
            for cube in cubes:
                trace_tic = self._tic()
                swc_path = self.trace_cube(cube)
                trace_toc = self._toc(trace_tic)
                trace_time += trace_toc

                # merge traced tree into global tree
                merge_tic = self._tic()
                T, candiate_t_junction_nodes = self.merge_tree(
                    T,
                    swc_path,
                    cube,
                    node,
                    **self.merge_params,
                    iter=iter,
                )
                merge_toc = self._toc(merge_tic)
                merge_time += merge_toc
                if len(candiate_t_junction_nodes) > 0:
                    break
            if (
                self.save_intermediate_tree
                and self.save_intermediate_every > 0
                and iter % self.save_intermediate_every == 0
            ):  # save growth process for debugging
                T.save_to_file(
                    os.path.join(self.imtermediate_tree_dir, f"tree_iter{iter:02d}.swc")
                )

            # search new nodes for tracing
            added_length = 0
            search_tic = self._tic()
            for junction_node_t in candiate_t_junction_nodes:
                length = junction_node_t.get_subtree_length()
                if length < 1:
                    continue
                added_length += length
                # add leafs
                leafs = junction_node_t.get_subtree_leafs()
                for leaf in leafs:
                    node_key = tuple(leaf[:])
                    if node_key not in visited and node_key not in queued:
                        push_node(leaf)
                        queued.add(node_key)
                        visited.add(node_key)
                # add margin nodes
                nodes = junction_node_t.get_subtree_node_list()
                for node in nodes:
                    if (
                        node not in leafs
                        and self.get_node_distance_to_cube_marigin(node, cube)
                        <= self.search_params["candidate_node_dist_to_margin"]
                    ):
                        node_key = tuple(node[:])
                        if node_key not in visited and node_key not in queued:
                            push_node(node)
                            queued.add(node_key)
                            visited.add(node_key)
            search_toc = self._toc(search_tic)

            if self.verbose:
                print(
                    f"Added length: {added_length:.2f}, time elapsed: {time.time() - start_time:.2f}s, "
                )
            self._write_log(
                iter=iter,
                cube_xyz=([cube.ix, cube.iy, cube.iz] if cube is not None else None),
                trace_time=trace_time,
                merge_time=merge_time,
                search_time=search_toc,
                added_length=added_length,
            )
        T.save_to_file(os.path.join(self.work_dir, f"tree-raw.swc"))

        # post-process the final tree
        swc = Swc(os.path.join(self.work_dir, f"tree-raw.swc"))
        if self.post_process_params["remove_duplicate_nodes"]:
            swc.remove_duplicate_nodes()
        if self.post_process_params["resample_distance"] is not None:
            swc.resample(min_distance=self.post_process_params["resample_distance"])
        swc.save_to_swc(final_save_path, reindex=True, sort_by_id=True, radius=1)

        return final_save_path

    def trace_from_init_tree(self, tree, save_path=None, iter=0):
        if isinstance(tree, str):
            tree = SwcForest(tree)
        assert isinstance(tree, SwcForest)
        return self.run(tree, final_save_path=save_path, iter=iter)

    def trace_from_soma_center(
        self, soma_center_coord, soma_roi_radius=(30, 60, 60), fiber_start_tolerance=18
    ):
        soma_cube = self.grid.find_best_cube_for_point(soma_center_coord)[0]

        # get soma mask and distance to soma
        soma_img = self.cube_reader.read_region(
            soma_cube.start_coord[::-1], soma_cube.end_coord[::-1]
        )
        import tifffile as tiff

        tiff.imwrite("z_soma_cube.tif", soma_img)
        soma_center_coord_rel_zyx = (
            int(soma_center_coord[2] - soma_cube.start_coord[2]),
            int(soma_center_coord[1] - soma_cube.start_coord[1]),
            int(soma_center_coord[0] - soma_cube.start_coord[0]),
        )
        soma_mask = segment_soma_from_seed(
            soma_img, soma_center_coord_rel_zyx, roi_radius=soma_roi_radius
        )
        soma_mask = soma_mask.astype(bool)
        dist2soma = distance_transform_edt(~soma_mask)

        # get initial tree from soma mask
        T = SwcForest()
        soma_node = SwcNode(
            nid=T.next_id(),
            ntype=1,
            coord=soma_center_coord,
        )
        T.add_tree(soma_node)
        soma_cube_swc_path = self.trace_cube(soma_cube)
        breakpoint()
        swc = Swc(soma_cube_swc_path)
        swc.add_offset(
            [
                -soma_cube.start_coord[0],
                -soma_cube.start_coord[1],
                -soma_cube.start_coord[2],
            ]
        )
        swc.save_to_swc("z_soma_cube_traced.swc")
        tree = SwcForest(soma_cube_swc_path)
        for root in tree.roots:
            node = tree.get_nearest_node(soma_center_coord, root, topk=1)[0]
            rz, ry, rx = (
                int(node.coord[2] - soma_cube.start_coord[2]),
                int(node.coord[1] - soma_cube.start_coord[1]),
                int(node.coord[0] - soma_cube.start_coord[0]),
            )
            if dist2soma[rz, ry, rx] <= fiber_start_tolerance:
                subtree_root = node.get_rerooted_tree(nid_start=T.next_id())
                for child in subtree_root.children:
                    length = child.get_subtree_length()
                    if length < 5:
                        child.parent = None
                length = subtree_root.get_subtree_length()
                if length >= 5:
                    T.link_child(soma_node, subtree_root)
        if self.save_intermediate_tree:
            T.save_to_file(
                os.path.join(self.imtermediate_tree_dir, f"tree_init_soma.swc")
            )
        return self.run(T)


@click.group()
def cli():
    pass


def _parse_neuron_ids(neuron_id_values):
    if neuron_id_values is None:
        return []

    if isinstance(neuron_id_values, str):
        raw_values = [neuron_id_values]
    else:
        raw_values = list(neuron_id_values)

    parsed_ids = []
    for raw in raw_values:
        value = str(raw).strip()
        if not value:
            continue

        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise click.BadParameter(
                    "Invalid list format for --neuron-id. Example: \"['neuron-08','neuron-12']\""
                ) from exc
            if not isinstance(parsed, (list, tuple)):
                raise click.BadParameter(
                    "--neuron-id list format must be a list or tuple."
                )
            parsed_ids.extend([str(x).strip() for x in parsed if str(x).strip()])
            continue

        if "," in value:
            parsed_ids.extend([x.strip() for x in value.split(",") if x.strip()])
            continue

        parsed_ids.append(value)

    clean_ids = []
    seen = set()
    for nid in parsed_ids:
        nid = nid.replace(".swc", "")
        if nid and nid not in seen:
            clean_ids.append(nid)
            seen.add(nid)
    return clean_ids


@cli.command()
@click.option("--slice-dir", default="/data2/CH1/slices")
@click.option("--slice-name-pattern", default="CH1_{z:05d}.tif")
@click.option("--hemisphere-anno-path", default="/data2/CH1/swcs/hemispheres_anno.txt")
@click.option(
    "--soma-initial-tree-dir",
    default="/data2/CH1/swcs/single-neurons-step2-somatree100",
)
@click.option("--seg-model-cfg-path", default="configs/model/dynunet.yaml")
@click.option("--seg-model-ckpt-path", required=True)
@click.option("--trace-model-name", default="Kimimaro")
@click.option("--search-method", default="bfs")
@click.option("--save-dir", default="outputs/whole_brain_trace")
@click.option(
    "--neuron-id",
    multiple=True,
    help=(
        "Only trace specified neuron id(s). Supports repeated option, comma-separated values, "
        "or Python-list string. Examples: --neuron-id neuron-08 --neuron-id neuron-12; "
        "--neuron-id neuron-08,neuron-12; --neuron-id \"['neuron-08','neuron-12']\"."
    ),
)
@click.option(
    "--save-intermediate-every",
    default=10,
    type=int,
    show_default=True,
    help="Save one intermediate tree every N iterations. Use 0 to disable.",
)
def trace_from_init_tree(
    slice_dir,
    slice_name_pattern,
    hemisphere_anno_path,
    soma_initial_tree_dir,
    seg_model_cfg_path,
    seg_model_ckpt_path,
    trace_model_name,
    search_method,
    save_dir,
    neuron_id,
    save_intermediate_every,
):
    save_dir = os.path.join(
        save_dir,
        f"{os.path.basename(seg_model_cfg_path).split('.')[0]}_{trace_model_name}_{search_method}",
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(hemisphere_anno_path, "r") as f:
        ds = json.load(f)
        nids = [
            d["swc_name"].replace(".swc", "")
            for d in ds["neurons"]
            if d["hemisphere"] == "right"
        ]

    selected_neuron_ids = _parse_neuron_ids(neuron_id)
    if len(neuron_id) > 0:
        if len(selected_neuron_ids) == 0:
            raise click.BadParameter("--neuron-id cannot be empty after parsing.")
        missing_ids = [nid for nid in selected_neuron_ids if nid not in nids]
        if missing_ids:
            raise click.BadParameter(
                f"Neuron id(s) not found in hemisphere annotation: {missing_ids}"
            )
        nids = selected_neuron_ids
    for nid in nids:
        work_dir = os.path.join(save_dir, nid)
        os.makedirs(work_dir, exist_ok=True)
        save_path = os.path.join(work_dir, f"{nid}.swc")
        if os.path.exists(save_path):
            print(f"Neuron {nid} already traced, skip.")
            continue
        print(f"Tracing neuron {nid}...")
        tracer = WholeBrainTrace(
            slice_dir=slice_dir,
            slice_name_pattern=slice_name_pattern,
            work_dir=work_dir,
            trace_model_cfg={
                "method": "seg+trace",
                "seg_model_cfg_path": seg_model_cfg_path,
                "seg_model_ckpt_path": seg_model_ckpt_path,
                "trace_model_name": trace_model_name,
            },
            merge_params={
                "merge_iou_threshold": 0.7,
                "merge_match_max_dist": 5,
                "min_newtree_length": 3,
            },
            search_params={
                "mode": search_method,
                "candidate_node_dist_to_margin": 3,
                "max_search_iter": 3000,
            },
            save_intermediate_tree=save_intermediate_every > 0,
            save_intermediate_every=save_intermediate_every,
        )
        iter = 0
        init_tree_swc_path = os.path.join(soma_initial_tree_dir, f"{nid}.swc")
        if os.path.exists(tracer.log_path):
            print("Start from existing log...")
            with open(tracer.log_path, "r") as f:
                ls = f.readlines()
                for l in ls[::-1]:
                    record = json.loads(l)
                    iter = record['iter']
                    if os.path.exists(os.path.join(tracer.imtermediate_tree_dir, f"tree_iter{iter:02d}.swc")):
                        print(f"Found intermediate tree for iter {iter}, start from it.")
                        iter = int(iter)
                        init_tree_swc_path = os.path.join(tracer.imtermediate_tree_dir, f"tree_iter{iter:02d}.swc")
                        break
        tracer.trace_from_init_tree(init_tree_swc_path, save_path, iter=iter)


if __name__ == "__main__":
    cli()
