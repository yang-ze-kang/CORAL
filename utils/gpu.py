from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetName
)


def get_gpu_with_max_free_vram():
    nvmlInit()
    try:
        n = nvmlDeviceGetCount()
        best_i, best_free = None, -1.0
        info_list = []
        for i in range(n):
            h = nvmlDeviceGetHandleByIndex(i)
            mem = nvmlDeviceGetMemoryInfo(h)
            free_mb = mem.free / (1024 ** 2)

            name_raw = nvmlDeviceGetName(h)
            name = name_raw.decode("utf-8", errors="ignore") if isinstance(name_raw, (bytes, bytearray)) else str(name_raw)

            info_list.append((i, name, free_mb))
            if free_mb > best_free:
                best_i, best_free = i, free_mb

        return best_i, best_free, info_list
    finally:
        nvmlShutdown()