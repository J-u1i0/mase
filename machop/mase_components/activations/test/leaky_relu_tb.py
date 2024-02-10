import torch

from torch import nn
from torch.autograd.function import InplaceFunction

import cocotb

from cocotb.triggers import Timer

import pytest
import cocotb

from mase_cocotb.runner import mase_runner

pytestmark = pytest.mark.simulator_required

# snippets
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# wrap through module to make it a function
my_clamp = MyClamp.apply
my_round = MyRound.apply

def bin_to_dec(arr: list):
    string = ''.join('1' if x else '0' for x in arr)
    return int(string, 2)

# fixed-point quantization with a bias
def quantize(x, bits, bias):  # bits = 32
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = 2 ** (bits - 1)
    scale = 2**bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)

class VerificationCase:

    # Parameters
    PARAMETERS = {
            "DATA_IN_0_PRECISION_0" : 8,
            "DATA_IN_0_PRECISION_1" : 1,
            "ALPHA_INT" : 3,
            "ALPHA_FRAC" : 5,
            "DATA_IN_0_TENSOR_SIZE_DIM_0" : 2,
            "DATA_IN_0_TENSOR_SIZE_DIM_1" : 1,
            "DATA_IN_0_PARALLELISM_DIM_0" : 2,
            "DATA_IN_0_PARALLELISM_DIM_1" : 1,
            "DATA_OUT_0_PRECISION_0" : 8,
            "DATA_OUT_0_PRECISION_1" : 1,
            "DATA_OUT_0_TENSOR_SIZE_DIM_0" : 2,
            "DATA_OUT_0_TENSOR_SIZE_DIM_1" : 1,
            "DATA_OUT_0_PARALLELISM_DIM_0" : 2,
            "DATA_OUT_0_PARALLELISM_DIM_1" : 1
    }

    # Used for calculating the scale in the quantisation of the inputs.
    bias = 4

    def __init__(self, samples=2):
        self.samples = samples
        alpha = self.PARAMETERS["ALPHA_INT"] / 2 ** self.PARAMETERS["ALPHA_FRAC"]
        self.m = nn.LeakyReLU(negative_slope=alpha)
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)

    def single_run(self):
        xs = torch.rand(self.PARAMETERS["DATA_IN_0_TENSOR_SIZE_DIM_0"])
        r1, r2 = 4, -4
        xs = (r1 - r2) * xs + r2
        # 8-bit, (5, 3)
        xs = quantize(xs, self.PARAMETERS["DATA_IN_0_PRECISION_0"], self.bias)
        return xs, self.m(xs)

    def get_dut_parameters(self):
        return self.PARAMETERS

    def get_dut_input(self, i):
        inputs = self.inputs[i]
        shifted_integers = (inputs * (2**self.bias)).int()
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        outputs = self.outputs[i]
        shifted_integers = (outputs * (2**self.bias)).floor().int()
        return shifted_integers.numpy().tolist()


@cocotb.test()
async def test_leaky_relu(dut):
    """Test integer based Relu"""
    test_case = VerificationCase(samples=100)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)

        dut.data_in_0.value = x
        await Timer(2, units="ns")
        for i, val in enumerate(dut.data_out_0.value):
            assert val.signed_integer == y[i], f"output q was incorrect on the {i}th cycle"


if __name__ == "__main__":
    print("Executing testbench ...")
    tb = VerificationCase()
    mase_runner(
            module_param_list=[tb.get_dut_parameters()]
    )
