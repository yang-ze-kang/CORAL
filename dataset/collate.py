import torch

def make_collate_fn(non_stack_keys=None):
    non_stack_keys = set(non_stack_keys or [])

    def collate_fn(batch):
        out = {}
        keys = batch[0].keys()

        for k in keys:
            values = [b[k] for b in batch]
            if k in non_stack_keys:
                out[k] = values
            else:
                if isinstance(values[0], torch.Tensor):
                    out[k] = torch.stack(values, dim=0)
                else:
                    out[k] = values
        return out

    return collate_fn