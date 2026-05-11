from pathlib import Path
from tifffile import tifffile
import numpy as np
from tqdm import tqdm

from swclib.image.mask2swc import Mask2Swc

def trace_skel(tif, out_path, soma_path=None):
    if isinstance(tif, str) or isinstance(tif, Path):
        labels = tifffile.imread(tif)
    else:
        labels = tif
    maxi = np.max(labels)
    labels = (labels>0.5*maxi).astype(np.uint8)
    converter = Mask2Swc()
    converter.run(labels, out_path, soma_path=soma_path, verbos=False)


if __name__ == "__main__":
    # etv
    soma_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/data/guolab/etv133_block_swc_yzk_refine_soma/somas")
    # mask_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-seg/outputs/neuron-seg/seg-3dunet/preds/train_seg-3dunet_step-2370_val_clDiceMetric-0/guolab-etv133")
    # swc_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-seg/outputs/neuron-seg/seg-3dunet/preds/train_seg-3dunet_step-2370_val_clDiceMetric-0/guolab-etv133/swcs")

    # mask
    # mask_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/dataset/guolab-etv133/mask_radius1")
    # swc_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/dataset/guolab-etv133/mask_radius1_swcs_skel")

    # etv DeepBranchTracer
    # mask_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/baselines/DeepBranchTracer-main/data/etv-133/results/pre_centerline_test")
    # swc_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/baselines/DeepBranchTracer-main/data/etv-133/results/pre_centerline_test/swcs")

    # etv NETracer
    # mask_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/baselines/NETracer/data/etv-133/results/centerline")
    # swc_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/baselines/NETracer/data/etv-133/results/centerline/swcs")

    # etv133 DynUNet skeleton
    # mask_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/outputs/neuron-seg/dynunet/2026-01-07-16-42-22/preds/dynunet_step-2370_val_clDiceMetric-0.7413/guolab-etv133")
    # swc_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/outputs/neuron-seg/dynunet/2026-01-07-16-42-22/preds/dynunet_step-2370_val_clDiceMetric-0.7413/guolab-etv133/swcs")

    # etv133 VNet skeleton
    # mask_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/outputs/neuron-seg/vnet/2026-01-07-19-46-43/preds/vnet_step-2570_val_clDiceMetric-0.7313/guolab-etv133")
    # swc_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/outputs/neuron-seg/vnet/2026-01-07-19-46-43/preds/vnet_step-2570_val_clDiceMetric-0.7313/guolab-etv133/swcs")

    # etv133 ImprovedVNet skeleton
    mask_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/outputs/neuron-seg/ivnet/2026-01-07-20-45-10/preds/ivnet_step-3755_val_clDiceMetric-0.7343/guolab-etv133")
    swc_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/outputs/neuron-seg/ivnet/2026-01-07-20-45-10/preds/ivnet_step-3755_val_clDiceMetric-0.7343/guolab-etv133/swcs")

    
    # MADM-TBR2-I3
    # mask_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-seg/outputs/neuron-seg/dynunet_cldice0.3_iter10000-trail2/preds/train_dynunet_cldice0/guolab-MADM-TBR2-I3")
    # swc_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/neuron-seg/outputs/neuron-seg/dynunet_cldice0.3_iter10000-trail2/preds/train_dynunet_cldice0/guolab-MADM-TBR2-I3/swcs")
    # soma_dir = Path("/gpfs-flash/hulab/yangzekang/neuron/data/guolab/MADM-TBR2-I3_block/swc")
    
    swc_dir.mkdir(parents=True, exist_ok=True)
    converter = Mask2Swc()
    files = mask_dir.glob("*.tif")
    files = sorted(files)
    for path in tqdm(files):
    # for path in mask_dir.glob("*.pro.skl.tif"):
        id = path.stem.replace('.pro.skl', '').replace('_mask', '')
        swc_path = swc_dir / f"{id}.swc"
        soma_path = soma_dir / f"{id}_somas.swc"
        assert soma_path.exists()
        mask = tifffile.imread(path)
        # breakpoint()
        mask = (mask > 0.5*255).astype(np.uint8)
        # breakpoint()
        # mask = (mask > 0.5).astype(np.uint8)
        converter.run(mask, swc_path, soma_path=soma_path, verbos=False)