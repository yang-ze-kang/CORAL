import glob
import argparse
from tqdm import tqdm
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from swclib.data.swc_soma import refine_with_soma

URL = "http://127.0.0.1:8000/trace_vaa3d_app2"


def trace_app2(tif_file, swc_file):
    assert isinstance(tif_file, str) and isinstance(swc_file, str)
    try:
        with open(tif_file, "rb") as f:
            files = {"file": f}
            r = requests.post(URL, files=files, timeout=1200)

        if r.status_code != 200:
            print(f"[FAIL] {tif_file} -> status {r.status_code}")
            return False
        else:
            with open(swc_file, "wb") as out:
                out.write(r.content)
            return True
    except Exception as e:
        print(f"[ERROR] {tif_file}: {e}")
        return False


if __name__ == "__main__":
    # tif_file = '/data1/yangzekang/neuron/CH1-cubes1723/masks/cube300_x20700_y12800_z1400_mask.tif'
    tif_file = '/data1/yangzekang/neuron/neuron-trace/outputs/neuron-seg/CH1-iter10000/dynunet-dice/2026-01-26-14-06-56/preds/dynunet-dice_step-8500_valmetric_NSD@1-0.9128/CH1-cubes1723/cube300_x17700_y5900_z5000.tif'
    swc_file = 'temp3.swc'
    trace_app2(tif_file, swc_file)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mask_dir", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--image_dir", type=str, default=None)
    # parser.add_argument("--image_weight", type=float, default=0.0)
    # parser.add_argument("--refined_soma", action="store_true")
    # parser.add_argument("--soma_dir", type=str, default="/gpfs-flash/hulab/yangzekang/neuron/data/guolab/etv133_block_swc_yzk_refine_soma/somas")

    # args = parser.parse_args()
    # os.makedirs(args.output_dir, exist_ok=True)

    # files = glob.glob(os.path.join(args.mask_dir, "*.tif"))

    # test_anno_path = "/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/data_split/guolab-etv133-radius1/test.txt"
    # ds = np.genfromtxt(test_anno_path, delimiter=',', dtype=str)[:,0]
    # ids = [os.path.basename(d).split('.')[0] for d in ds]

    # # filter
    # files = [file for file in files if os.path.basename(file).split('.')[0].replace('_mask',"") in ids]

    # assert args.image_dir is None

    # tasks = []
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     for file in files:
    #         out_path = os.path.join(
    #             args.output_dir,
    #             os.path.basename(file).replace(".tif", ".swc"),
    #         )
    #         if os.path.isfile(out_path):
    #             continue
    #         tasks.append(executor.submit(trace_neutube, file, out_path))

    #     for _ in tqdm(as_completed(tasks), total=len(tasks)):
    #         pass
    
    # save_swc_dir = args.output_dir + '-refinedsoma'
    # os.makedirs(save_swc_dir, exist_ok=True)
    # for swc_path in os.listdir(args.output_dir):
    #     swc_path = os.path.join(args.output_dir, swc_path)
    #     if str(swc_path).endswith('_somas.swc') or not str(swc_path).endswith('.swc'):
    #         continue
    #     soma_path = args.soma_dir + "/" + os.path.basename(swc_path).replace('.swc', '_somas.swc').replace('_mask', '')
    #     out_path = save_swc_dir + "/" + os.path.basename(swc_path)
    #     refine_with_soma(swc_path, soma_path, out_path, scale=(1.0, 1.0, 1/0.35))