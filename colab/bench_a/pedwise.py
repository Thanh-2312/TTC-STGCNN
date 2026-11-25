"""B-retrain variant F: pedestrian-wise kernels. Every Conv2d kernel of width > 1 on the last axis (the pedestrian axis
in Social-STGCNN, DMRGCN and Graph-TERN) gets width 1 and no padding on that axis. apply() must run before the
repository code builds its model; the repositories call nn.Conv2d at construction time."""
import torch.nn as nn

Conv2d = nn.Conv2d
CHANGED = []


def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


class PedWiseConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *args, **kw):
        assert not isinstance(padding, str), 'string padding is not used by these models'
        (kh, kw_), (ph, pw) = _pair(kernel_size), _pair(padding)
        if kw_ > 1:
            CHANGED.append(((kh, kw_), (kh, 1)))
            kw_, pw = 1, 0
        super().__init__(in_channels, out_channels, (kh, kw_), stride, (ph, pw), *args, **kw)


def apply():
    nn.Conv2d = PedWiseConv2d
