import torch.nn as nn
import os
import sys
import torch
from thop import profile
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from omegaconf import OmegaConf
from hydra.utils import instantiate
from copy import deepcopy


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def format_num(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    elif n >= 1e9:
        return f"{n / 1e9:.2f}G"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    else:
        return str(int(n))

def measure_peak_memory(model, input_tensor, do_backward=True):
    device = input_tensor.device
    assert device.type == "cuda"

    torch.cuda.reset_peak_memory_stats(device)

    model.train()
    output = model(input_tensor)
    if isinstance(output, dict):
        output = output['mask']
    loss = output.sum()

    if do_backward:
        loss.backward()

    peak_mem = torch.cuda.max_memory_allocated(device)
    return peak_mem  # bytes

def format_mem(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} G sB"

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@dataclass
class ThroughputResult:
    volumes_per_sec: float
    ms_per_volume: float
    iters: int
    total_volumes: int
    total_time_s: float


@torch.no_grad()
def measure_volume_throughput(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int, int],  # (B, C, D, H, W)
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,           # AMP dtype; use float32 if you want
    use_amp: bool = True,
    warmup_iters: int = 20,
    measure_iters: int = 100,
    volumes_per_iter: Optional[int] = None,       # If 1 iter processes >B "effective volumes", set it
    channels_last_3d: bool = False,               # set True if you use channels_last_3d
) -> ThroughputResult:
    model = model.to(device).eval()

    x = torch.randn(*input_shape, device=device, dtype=torch.float32)
    if channels_last_3d:
        x = x.contiguous(memory_format=torch.channels_last_3d)

    B = input_shape[0]
    if volumes_per_iter is None:
        volumes_per_iter = B  # default: each iter processes B volumes

    # Warmup (to stabilize kernels, caches, autotune)
    for _ in range(warmup_iters):
        if use_amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = model(x)
        else:
            _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Measure
    t0 = time.perf_counter()
    for _ in range(measure_iters):
        if use_amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = model(x)
        else:
            _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_volumes = measure_iters * volumes_per_iter
    vps = total_volumes / total_time
    ms_per_vol = (total_time / total_volumes) * 1000.0

    return ThroughputResult(
        volumes_per_sec=vps,
        ms_per_volume=ms_per_vol,
        iters=measure_iters,
        total_volumes=total_volumes,
        total_time_s=total_time,
    )


if __name__ == "__main__":
    # model_name = 'dynunet'
    # model_name = 'vnet'
    # model_name = 'unetr'
    # model_name = 'swin_unetr'
    # model_name = 'mednext'
    # model_name = 'segmamba'
    # model_name = 'ivnet'
    # model_name = 'dscnet'
    # model_name = 'adtlnet'
    model_name = 'gbpnet'
    yaml_model_dir = '/data1/yangzekang/neuron/neuron-trace/configs/model'
    # yaml_model_dir = '/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/configs/model'
    cfg = OmegaConf.load(os.path.join(yaml_model_dir, model_name + '.yaml'))
    model = instantiate(cfg).to('cuda')

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(format_num(total), format_num(trainable))
        
    dummy_input = torch.randn(1, 1, 128, 128, 128).to('cuda')
    dummy_input1 = torch.randn(1, 1, 47, 78, 78).to('cuda')
    dummy_input2 = torch.randn(1, 1, 141, 234, 234).to('cuda')

    model_profile = deepcopy(model).eval()
    flops, params = profile(model_profile, inputs=(dummy_input1,dummy_input2,), verbose=False)

    print("Params:", format_num(params))
    print(f"FLOPs:{flops / 1e9:.2f}G",)

    res = measure_volume_throughput(
        model=model,
        input_shape=(1, 1, 128, 128, 128),  # one 3D volume
        use_amp=True,
        dtype=torch.float16,
        warmup_iters=30,
        measure_iters=200,
    )
    print(f"Throughput: {res.volumes_per_sec:.3f} volumes/s")
    print(f"Latency:    {res.ms_per_volume:.3f} ms/volume")
    print(f"Total:      {res.total_volumes} volumes in {res.total_time_s:.3f}s")


    peak_bytes = measure_peak_memory(model, dummy_input, do_backward=True)
    print("Peak GPU memory:", format_mem(peak_bytes))
