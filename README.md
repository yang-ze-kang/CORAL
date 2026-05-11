# CORAL

<p align="center">
  <img src="docs/coral_logo.png" alt="CORAL logo" width="420">
</p>

Code for **CORAL: A Benchmark for Structure-aware and Brain-wide Neuron Reconstruction in Light Microscopy**.

CORAL provides training and inference code for neuron segmentation, block-level reconstruction, and brain-wide reconstruction.

## Dataset

The CORAL dataset is available on Hugging Face:

<https://huggingface.co/datasets/yangzekang2000/CORAL>

Example download command:

```bash
huggingface-cli download yangzekang2000/CORAL \
  --repo-type dataset \
  --local-dir /path/to/CORAL-data
```

After downloading, update the dataset paths in `configs/data/C2-cubes1937.yaml` if your local directory differs from the default paths.

## Installation
```bash
conda create -n coral python=3.9 -y
conda activate coral
pip install -r requirements.txt

git clone git@github.com:yang-ze-kang/swclib.git
cd swclib
pip install .
```

`requirements.txt` is intentionally minimal and is derived from the packages directly imported by this repository. It also installs the local CUDA extensions under `models/causal-conv1d` and `models/mamba`.

## Repository Layout

```text
configs/                  Hydra configs for data, models, and training
dataset/                  Dataset and dataloader utilities
models/                   Segmentation networks and local CUDA extensions
modules/                  Lightning training modules
loss/                     Loss functions
utils/                    Metrics, image, SWC, GPU, and miscellaneous utilities
postprocess/              Block-level tracing backends
TraceAPI/                 FastAPI wrapper for Vaa3D-based tracing methods
train.py                  Segmentation training entry point
test.py                   Segmentation inference/testing entry point
skeleton.py               Block-level reconstruction from predicted masks
whole_brain_trace.py      Brain-wide tracing from image slices and initial trees
calculate_metrics.py      Block-level SWC metric calculation
calculate_metrics_wb.py   Brain-wide SWC metric calculation
scripts/                  Aggregation and helper scripts
```

## Segmentation Training

Training uses Hydra configs in `configs/`. Because `train.py` expects a config name, pass one explicitly with `--config-name`.

Available C2 segmentation configs include:

```text
C2_seg_dynunet
C2_seg_dynunet_cldice
C2_seg_vnet
C2_seg_ivnet
C2_seg_unetr
C2_seg_swinunetr
C2_seg_dscnet
C2_seg_mednext
C2_seg_segmamba
C2_seg_adtlnet
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name C2_seg_dynunet
```

Training outputs are written under:

```text
outputs/<project_name>/<run_name>/<timestamp>/
```

## Segmentation Inference

Use `test.py` with the same config and a checkpoint path:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config-name C2_seg_dynunet \
  path_to_chkpt=/path/to/checkpoint.ckpt
```

Predicted masks are saved to the corresponding `preds/` directory. Offline segmentation metrics are computed after inference by `utils.offline_metrics`.

## Block-Level Reconstruction

Use `skeleton.py` to convert predicted segmentation masks into SWC reconstructions.

Supported tracing methods include:

```text
Kimimaro
neuTube
APP2
smartTrace
SPE-DNR
NETracer
```

Example:

```bash
python skeleton.py \
  --pred_dir /path/to/predicted_masks \
  --out_swc_dir /path/to/output_swcs \
  --method Kimimaro \
  --workers 16 \
  --skip_existing
```

Some methods use the raw image as additional input:

```bash
python skeleton.py \
  --raw_dir /path/to/raw_cubes \
  --pred_dir /path/to/predicted_masks \
  --out_swc_dir /path/to/output_swcs \
  --method NETracer \
  --workers 8 \
  --skip_existing
```

`neuTube`, `APP2` and `smartTrace` call the local Vaa3D service in `TraceAPI`. Start the API before using those methods:

```bash
cd TraceAPI
uvicorn app:app --host 127.0.0.1 --port 8000
```

For detailed TraceAPI installation, endpoints, request examples, and troubleshooting, see `TraceAPI/README.md`.

## Block-Level Metrics

Compute SWC reconstruction metrics with:

```bash
python calculate_metrics.py \
  --gt_swc_dir /path/to/ground_truth_swcs \
  --pred_swc_dir /path/to/predicted_swcs \
  --results_dir /path/to/metric_results \
  --check-total-num 1022 \
  --workers 16 \
  --skip_existing
```

Aggregate multiple runs:

```bash
python scripts/aggregate_trace_metrics.py \
  --run-dirs /path/to/run1 /path/to/run2 \
  --methods APP2,Kimimaro,neuTube,smartTrace,SPE-DNR \
  --output outputs/summary_metrics.json

python scripts/print_summary_table.py \
  --summary-json outputs/summary_metrics.json \
  --metrics ssd.sd,ssd.ssd,point.micro_f1,length.micro_f1,keypoints.micro_f1,fiber.micro_f1 \
  --sort-by length.micro_f1
```

## Brain-Wide Tracing

`whole_brain_trace.py` traces neurons across whole-brain image slices. The main command is `trace-from-init-tree`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python whole_brain_trace.py trace-from-init-tree \
  --slice-dir /path/to/slices \
  --slice-name-pattern '^CXU_slice_(\d{5})_CH1\.tif' \
  --hemisphere-anno-path /path/to/brain_hemisphere.json \
  --soma-initial-tree-dir /path/to/soma_initial_trees \
  --seg-model-cfg-path configs/model/dynunet.yaml \
  --seg-model-ckpt-path /path/to/checkpoint.ckpt \
  --trace-model-name Kimimaro \
  --search-method bfs \
  --save-dir outputs/whole_brain_trace
```

Trace only selected neurons:

```bash
CUDA_VISIBLE_DEVICES=0 python whole_brain_trace.py trace-from-init-tree \
  --slice-dir /path/to/slices \
  --hemisphere-anno-path /path/to/brain_hemisphere.json \
  --soma-initial-tree-dir /path/to/soma_initial_trees \
  --seg-model-cfg-path configs/model/dynunet.yaml \
  --seg-model-ckpt-path /path/to/checkpoint.ckpt \
  --trace-model-name Kimimaro \
  --neuron-id neuron-08,neuron-12
```

Compute brain-wide metrics:

```bash
python calculate_metrics_wb.py \
  --gt_swc_dir /path/to/ground_truth_swcs \
  --pred_swc_dir /path/to/predicted_whole_brain_swcs \
  --results_dir /path/to/whole_brain_metric_results \
  --workers 16 \
  --skip_existing
```
