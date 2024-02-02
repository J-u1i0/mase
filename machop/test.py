from math import ceil, log2

from numpy import ndarray
from torch import Tensor
import torch


def _integer_quantize(
    x: Tensor | ndarray, width: int, frac_width: int = None, is_signed: bool = True
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    if frac_width is None:
        frac_width = width // 2

    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    # thresh = 2 ** (width - 1)
    scale = 2**frac_width

    if isinstance(x, (Tensor, ndarray)):
        return torch.clamp(torch.round(x.mul(scale)), int_min, int_max).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return torch.clamp(torch.round(x * scale), int_min, int_max) / scale

x = torch.tensor([-0.0503])
print(_integer_quantize(x, 8, 4))
