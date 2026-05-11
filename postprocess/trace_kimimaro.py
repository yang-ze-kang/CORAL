import kimimaro
import tifffile
from pathlib import Path
from tqdm import tqdm
import numpy as np

def trace_kimimaro(tif, out_path, foreground_thres=0.5, kimimaro_anisotropy=(0.35, 0.35, 1.0), mode='xyz'):
    if isinstance(tif, str) or isinstance(tif, Path):
        labels = tifffile.imread(tif)
    else:
        labels = tif
    maxi = np.max(labels)
    labels = (labels>=foreground_thres*maxi).astype(np.uint8)
    skels = kimimaro.skeletonize(
        labels,
        teasar_params={
            "scale": 1.5,
            "const": 10,  # physical units
            "pdrf_scale": 100000,
            "pdrf_exponent": 2,
            "soma_acceptance_threshold": 3500,  # physical units
            "soma_detection_threshold": 750,  # physical units
            "soma_invalidation_const": 300,  # physical units
            "soma_invalidation_scale": 2,
            "max_paths": 10000,  # default None
        },
        # object_ids=[ ... ], # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        dust_threshold=100,  # skip connected components with fewer than this many voxels
        anisotropy=kimimaro_anisotropy[::-1] if mode=='xyz' else kimimaro_anisotropy,  # default True
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        fill_holes=False,  # default False
        fix_avocados=False,  # default False
        progress=False,  # default False, show progress bar
        parallel=1,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
    )
    if len(skels)>0:
        swc = skels[1].to_swc()
    else:
        swc = "# none"
    with open(out_path, "w") as f:
        rows = swc.split("\n")
        for row in rows:
            if row.startswith("#"):
                f.write(row + "\n")
            else:
                ds = row.split(" ")
                if len(ds) != 7:
                    f.write(row + "\n")
                else:
                    id, t, z, y, x, r, pid = ds
                    f.write(f"{id} {t} {float(x)/kimimaro_anisotropy[0]} {float(y)/kimimaro_anisotropy[1]} {float(z)/kimimaro_anisotropy[2]} {r} {pid}\n")
    return swc

if __name__=='__main__':
    # mask Kimimacro
    # mask_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/dataset/guolab-etv133/mask_radius1")
    # swc_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/dataset/guolab-etv133/mask_radius1_swcs_kimimaro")

    # swc_dir.mkdir(parents=True, exist_ok=True)
    # files = mask_dir.glob("*.tif")
    # files = sorted(files)
    # for path in tqdm(files):
    #     id = path.stem.replace('_mask', '')
    #     swc_path = swc_dir / f"{id}.swc"
    #     trace_kimimaro(path, swc_path)
    #     # break

    tif_path = '/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/outputs/whole_brain_trace/dynunet_APP2_bfs/neuron-19/cube_cache/cube_99_55_16.tif'
    save_swc_path = 'z.swc'
    trace_kimimaro(tif_path, save_swc_path)