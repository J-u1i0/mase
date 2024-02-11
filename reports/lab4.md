# Lab 4

## Task 1: Read this page for more information on the hardware metadata pass. Why we have such a metadata setup? How is this different from the software metadata?
`add_hardware_metadata_analysis_pass`
Adds the matadata for the hardware.
* Describes constraints of the node operation for static analysis or possible transformation.

### Hardware metadata structure
* `is_implicit` -> whether the node is mapped to hardware or software only.
* `verilog_param` -> parameters for verilog modules
* `toolchain`
* `module` -> name of hardware module.
* `device_id`
* `interface` -> for the parameters {weight, biases} decide on storage type and transposing data before emitting.
* `dependence_files` -> file dependences to instantiate the module in the top level module.

### Example: 
```
{
    "common": {...},
    "software": {},
    "hardware": {
        "is_implicit": False,
        "interface": {
            "weight": {"storage": "BRAM", "transpose": False},
            "bias": {"storage": "BRAM", "transpose": False},
        },
        "toolchain": "INTERNAL",
        "module": "fixed_linear",
        "device_id": -1,
        "dependence_files": [
            "cast/fixed_cast.sv",
            "fixed_arith/fixed_dot_product.sv",
            ...
            "common/join2.sv",
            "linear/fixed_linear.sv",
        ],
        "verilog_param": {
            "DATA_IN_0_PRECISION_0": 8,
            "DATA_IN_0_PRECISION_1": 3,
            ...
            "DATA_OUT_0_PARALLELISM_1_DIM_2": 1,
        },
    },
}
```

### Why we have such a metadata?
The main reason for having hardware metadata is for customisation of the instatiation of the modules and how they communicate with each other.
The metadata abstracts the design of the hardware to a list of parameters that can be tuned to generate models that fit the required specifications.

The parameters from the metadata are essential in instantiating the correct precision and range of
* input ports
* output ports
* internal wires
* internal registers
These parameters allow the instantiation of modules with different amounts of memory requirements and computational precision.
For exampe if the Neural Network were to be Binarised then the weights and biases of the linear layers would be set to a width of 1.

Theh dependence files are key in allowing Verilator to generate the RTL from a hierarchical design where modules are stored in different files.

The interface allows to set the storage type and transposing for learnable parameters of a Neural Network layer. For example, the weights and biases of a Linear Layer.

### How is this different from software metadata?
The Hardware metadata focuses on the instantiation of hardware modules. This means that it focuses on what the parameters are and what the file dependences are for the modules.
<br>
On the other hand the Software metadata focuses on the arguments (inputs and internal variables) and the outputs for the nodes in the graph and displays the requested statistics on these I/O.

Hardware metadata
```
INFO     Inspecting graph [add_common_meta_param_analysis_pass]
INFO:chop.passes.graph.analysis.report.report_node:Inspecting graph [add_common_meta_param_analysis_pass]
INFO     
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| Node name   | Fx Node op    | Mase type           | Mase op     | Hardware Param                                                     |
+=============+===============+=====================+=============+====================================================================+
| x           | placeholder   | placeholder         | placeholder | {'device_id': 0, 'is_implicit': True}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| flatten     | call_function | implicit_func       | flatten     | {'device_id': 0, 'is_implicit': True}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| fc1         | call_module   | module_related_func | linear      | {'dependence_files': ['cast/fixed_cast.sv',                        |
|             |               |                     |             |                       'fixed_arith/fixed_dot_product.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_vector_mult.sv',          |
|             |               |                     |             |                       'fixed_arith/register_slice.sv',             |
|             |               |                     |             |                       'fixed_arith/fixed_accumulator.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree.sv',           |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree_layer.sv',     |
|             |               |                     |             |                       'fixed_arith/fixed_mult.sv',                 |
|             |               |                     |             |                       'common/join2.sv',                           |
|             |               |                     |             |                       'linear/fixed_linear.sv'],                   |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'bias': {'storage': 'BRAM', 'transpose': False},    |
|             |               |                     |             |                'weight': {'storage': 'BRAM', 'transpose': False}}, |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_linear',                                         |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'BIAS_PARALLELISM_DIM_0': 784,                  |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_2': 1,                    |
|             |               |                     |             |                    'BIAS_PRECISION_0': 32,                         |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_0': 784,                  |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_2': 1,                    |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 784,          |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 784,          |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_0': 784,                |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_1': 784,                |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_2': 1,                  |
|             |               |                     |             |                    'WEIGHT_PRECISION_0': 32,                       |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_0': 784,                |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_1': 784,                |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_2': 1}}                 |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| relu        | call_function | module_related_func | relu        | {'dependence_files': ['activations/fixed_relu.sv'],                |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'inplace': {}},                                     |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_relu',                                           |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 784,          |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 784,          |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'INPLACE': False}}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| fc2         | call_module   | module_related_func | linear      | {'dependence_files': ['cast/fixed_cast.sv',                        |
|             |               |                     |             |                       'fixed_arith/fixed_dot_product.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_vector_mult.sv',          |
|             |               |                     |             |                       'fixed_arith/register_slice.sv',             |
|             |               |                     |             |                       'fixed_arith/fixed_accumulator.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree.sv',           |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree_layer.sv',     |
|             |               |                     |             |                       'fixed_arith/fixed_mult.sv',                 |
|             |               |                     |             |                       'common/join2.sv',                           |
|             |               |                     |             |                       'linear/fixed_linear.sv'],                   |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'bias': {'storage': 'BRAM', 'transpose': False},    |
|             |               |                     |             |                'weight': {'storage': 'BRAM', 'transpose': False}}, |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_linear',                                         |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'BIAS_PARALLELISM_DIM_0': 3136,                 |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_2': 1,                    |
|             |               |                     |             |                    'BIAS_PRECISION_0': 32,                         |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_0': 3136,                 |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_2': 1,                    |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 3136,         |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 3136,         |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_0': 3136,               |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_1': 784,                |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_2': 1,                  |
|             |               |                     |             |                    'WEIGHT_PRECISION_0': 32,                       |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_0': 3136,               |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_1': 784,                |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_2': 1}}                 |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| relu_1      | call_function | module_related_func | relu        | {'dependence_files': ['activations/fixed_relu.sv'],                |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'inplace': {}},                                     |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_relu',                                           |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 3136,            |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 3136,            |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 3136,         |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 3136,         |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'INPLACE': False}}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| fc3         | call_module   | module_related_func | linear      | {'dependence_files': ['cast/fixed_cast.sv',                        |
|             |               |                     |             |                       'fixed_arith/fixed_dot_product.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_vector_mult.sv',          |
|             |               |                     |             |                       'fixed_arith/register_slice.sv',             |
|             |               |                     |             |                       'fixed_arith/fixed_accumulator.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree.sv',           |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree_layer.sv',     |
|             |               |                     |             |                       'fixed_arith/fixed_mult.sv',                 |
|             |               |                     |             |                       'common/join2.sv',                           |
|             |               |                     |             |                       'linear/fixed_linear.sv'],                   |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'bias': {'storage': 'BRAM', 'transpose': False},    |
|             |               |                     |             |                'weight': {'storage': 'BRAM', 'transpose': False}}, |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_linear',                                         |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'BIAS_PARALLELISM_DIM_0': 10,                   |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_2': 1,                    |
|             |               |                     |             |                    'BIAS_PRECISION_0': 32,                         |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_0': 10,                   |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_2': 1,                    |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 3136,            |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 3136,            |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 10,           |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 10,           |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_0': 10,                 |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_1': 3136,               |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_2': 1,                  |
|             |               |                     |             |                    'WEIGHT_PRECISION_0': 32,                       |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_0': 10,                 |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_1': 3136,               |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_2': 1}}                 |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| output      | output        | output              | output      | {'device_id': 0, 'is_implicit': True}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
INFO:chop.passes.graph.analysis.report.report_node:
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| Node name   | Fx Node op    | Mase type           | Mase op     | Hardware Param                                                     |
+=============+===============+=====================+=============+====================================================================+
| x           | placeholder   | placeholder         | placeholder | {'device_id': 0, 'is_implicit': True}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| flatten     | call_function | implicit_func       | flatten     | {'device_id': 0, 'is_implicit': True}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| fc1         | call_module   | module_related_func | linear      | {'dependence_files': ['cast/fixed_cast.sv',                        |
|             |               |                     |             |                       'fixed_arith/fixed_dot_product.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_vector_mult.sv',          |
|             |               |                     |             |                       'fixed_arith/register_slice.sv',             |
|             |               |                     |             |                       'fixed_arith/fixed_accumulator.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree.sv',           |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree_layer.sv',     |
|             |               |                     |             |                       'fixed_arith/fixed_mult.sv',                 |
|             |               |                     |             |                       'common/join2.sv',                           |
|             |               |                     |             |                       'linear/fixed_linear.sv'],                   |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'bias': {'storage': 'BRAM', 'transpose': False},    |
|             |               |                     |             |                'weight': {'storage': 'BRAM', 'transpose': False}}, |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_linear',                                         |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'BIAS_PARALLELISM_DIM_0': 784,                  |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_2': 1,                    |
|             |               |                     |             |                    'BIAS_PRECISION_0': 32,                         |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_0': 784,                  |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_2': 1,                    |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 784,          |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 784,          |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_0': 784,                |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_1': 784,                |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_2': 1,                  |
|             |               |                     |             |                    'WEIGHT_PRECISION_0': 32,                       |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_0': 784,                |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_1': 784,                |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_2': 1}}                 |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| relu        | call_function | module_related_func | relu        | {'dependence_files': ['activations/fixed_relu.sv'],                |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'inplace': {}},                                     |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_relu',                                           |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 784,          |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 784,          |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'INPLACE': False}}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| fc2         | call_module   | module_related_func | linear      | {'dependence_files': ['cast/fixed_cast.sv',                        |
|             |               |                     |             |                       'fixed_arith/fixed_dot_product.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_vector_mult.sv',          |
|             |               |                     |             |                       'fixed_arith/register_slice.sv',             |
|             |               |                     |             |                       'fixed_arith/fixed_accumulator.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree.sv',           |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree_layer.sv',     |
|             |               |                     |             |                       'fixed_arith/fixed_mult.sv',                 |
|             |               |                     |             |                       'common/join2.sv',                           |
|             |               |                     |             |                       'linear/fixed_linear.sv'],                   |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'bias': {'storage': 'BRAM', 'transpose': False},    |
|             |               |                     |             |                'weight': {'storage': 'BRAM', 'transpose': False}}, |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_linear',                                         |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'BIAS_PARALLELISM_DIM_0': 3136,                 |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_2': 1,                    |
|             |               |                     |             |                    'BIAS_PRECISION_0': 32,                         |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_0': 3136,                 |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_2': 1,                    |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 784,             |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 3136,         |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 3136,         |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_0': 3136,               |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_1': 784,                |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_2': 1,                  |
|             |               |                     |             |                    'WEIGHT_PRECISION_0': 32,                       |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_0': 3136,               |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_1': 784,                |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_2': 1}}                 |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| relu_1      | call_function | module_related_func | relu        | {'dependence_files': ['activations/fixed_relu.sv'],                |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'inplace': {}},                                     |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_relu',                                           |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 3136,            |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 3136,            |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 3136,         |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 3136,         |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'INPLACE': False}}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| fc3         | call_module   | module_related_func | linear      | {'dependence_files': ['cast/fixed_cast.sv',                        |
|             |               |                     |             |                       'fixed_arith/fixed_dot_product.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_vector_mult.sv',          |
|             |               |                     |             |                       'fixed_arith/register_slice.sv',             |
|             |               |                     |             |                       'fixed_arith/fixed_accumulator.sv',          |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree.sv',           |
|             |               |                     |             |                       'fixed_arith/fixed_adder_tree_layer.sv',     |
|             |               |                     |             |                       'fixed_arith/fixed_mult.sv',                 |
|             |               |                     |             |                       'common/join2.sv',                           |
|             |               |                     |             |                       'linear/fixed_linear.sv'],                   |
|             |               |                     |             |  'device_id': -1,                                                  |
|             |               |                     |             |  'interface': {'bias': {'storage': 'BRAM', 'transpose': False},    |
|             |               |                     |             |                'weight': {'storage': 'BRAM', 'transpose': False}}, |
|             |               |                     |             |  'is_implicit': False,                                             |
|             |               |                     |             |  'module': 'fixed_linear',                                         |
|             |               |                     |             |  'toolchain': 'INTERNAL',                                          |
|             |               |                     |             |  'verilog_param': {'BIAS_PARALLELISM_DIM_0': 10,                   |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_PARALLELISM_DIM_2': 1,                    |
|             |               |                     |             |                    'BIAS_PRECISION_0': 32,                         |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_0': 10,                   |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_1': 1,                    |
|             |               |                     |             |                    'BIAS_TENSOR_SIZE_DIM_2': 1,                    |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_1': 3136,            |
|             |               |                     |             |                    'DATA_IN_0_PARALLELISM_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_IN_0_PRECISION_0': 32,                    |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_0': 1,               |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_1': 3136,            |
|             |               |                     |             |                    'DATA_IN_0_TENSOR_SIZE_DIM_2': 1,               |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_1': 10,           |
|             |               |                     |             |                    'DATA_OUT_0_PARALLELISM_0_DIM_2': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_PRECISION_0': 32,                   |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_0': 1,            |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_1': 10,           |
|             |               |                     |             |                    'DATA_OUT_0_TENSOR_SIZE_0_DIM_2': 1,            |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_0': 10,                 |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_1': 3136,               |
|             |               |                     |             |                    'WEIGHT_PARALLELISM_DIM_2': 1,                  |
|             |               |                     |             |                    'WEIGHT_PRECISION_0': 32,                       |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_0': 10,                 |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_1': 3136,               |
|             |               |                     |             |                    'WEIGHT_TENSOR_SIZE_DIM_2': 1}}                 |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
| output      | output        | output              | output      | {'device_id': 0, 'is_implicit': True}                              |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------+
```

Software metadata
```
INFO     Inspecting graph [add_common_meta_param_analysis_pass]
INFO:chop.passes.graph.analysis.report.report_node:Inspecting graph [add_common_meta_param_analysis_pass]
INFO     
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| Node name   | Fx Node op    | Mase type           | Mase op     | Software Param                                                                 |
+=============+===============+=====================+=============+================================================================================+
| x           | placeholder   | placeholder         | placeholder | {'results': {'data_out_0': {'stat': {}}}}                                      |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| flatten     | call_function | implicit_func       | flatten     | {'args': {'data_in_0': {'stat': {}}}, 'results': {'data_out_0': {'stat': {}}}} |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| fc1         | call_module   | module_related_func | linear      | {'args': {'bias': {'stat': {}},                                                |
|             |               |                     |             |           'data_in_0': {'stat': {}},                                           |
|             |               |                     |             |           'weight': {'stat': {}}},                                             |
|             |               |                     |             |  'results': {'data_out_0': {'stat': {}}}}                                      |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| relu        | call_function | module_related_func | relu        | {'args': {'data_in_0': {'stat': {}}}, 'results': {'data_out_0': {'stat': {}}}} |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| fc2         | call_module   | module_related_func | linear      | {'args': {'bias': {'stat': {}},                                                |
|             |               |                     |             |           'data_in_0': {'stat': {}},                                           |
|             |               |                     |             |           'weight': {'stat': {}}},                                             |
|             |               |                     |             |  'results': {'data_out_0': {'stat': {}}}}                                      |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| relu_1      | call_function | module_related_func | relu        | {'args': {'data_in_0': {'stat': {}}}, 'results': {'data_out_0': {'stat': {}}}} |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| fc3         | call_module   | module_related_func | linear      | {'args': {'bias': {'stat': {}},                                                |
|             |               |                     |             |           'data_in_0': {'stat': {}},                                           |
|             |               |                     |             |           'weight': {'stat': {}}},                                             |
|             |               |                     |             |  'results': {'data_out_0': {'stat': {}}}}                                      |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| output      | output        | output              | output      | {'args': {'data_in_0': {'stat': {}}}}                                          |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
INFO:chop.passes.graph.analysis.report.report_node:
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| Node name   | Fx Node op    | Mase type           | Mase op     | Software Param                                                                 |
+=============+===============+=====================+=============+================================================================================+
| x           | placeholder   | placeholder         | placeholder | {'results': {'data_out_0': {'stat': {}}}}                                      |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| flatten     | call_function | implicit_func       | flatten     | {'args': {'data_in_0': {'stat': {}}}, 'results': {'data_out_0': {'stat': {}}}} |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| fc1         | call_module   | module_related_func | linear      | {'args': {'bias': {'stat': {}},                                                |
|             |               |                     |             |           'data_in_0': {'stat': {}},                                           |
|             |               |                     |             |           'weight': {'stat': {}}},                                             |
|             |               |                     |             |  'results': {'data_out_0': {'stat': {}}}}                                      |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| relu        | call_function | module_related_func | relu        | {'args': {'data_in_0': {'stat': {}}}, 'results': {'data_out_0': {'stat': {}}}} |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| fc2         | call_module   | module_related_func | linear      | {'args': {'bias': {'stat': {}},                                                |
|             |               |                     |             |           'data_in_0': {'stat': {}},                                           |
|             |               |                     |             |           'weight': {'stat': {}}},                                             |
|             |               |                     |             |  'results': {'data_out_0': {'stat': {}}}}                                      |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| relu_1      | call_function | module_related_func | relu        | {'args': {'data_in_0': {'stat': {}}}, 'results': {'data_out_0': {'stat': {}}}} |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| fc3         | call_module   | module_related_func | linear      | {'args': {'bias': {'stat': {}},                                                |
|             |               |                     |             |           'data_in_0': {'stat': {}},                                           |
|             |               |                     |             |           'weight': {'stat': {}}},                                             |
|             |               |                     |             |  'results': {'data_out_0': {'stat': {}}}}                                      |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
| output      | output        | output              | output      | {'args': {'data_in_0': {'stat': {}}}}                                          |
+-------------+---------------+---------------------+-------------+--------------------------------------------------------------------------------+
```


## Task 2: Read through top/hardware/rtl/top.sv and make sure you understand how our MLP model maps to this hardware design. Explain what each part is doing in the .sv file.

### Parameters
The top.sv modules is the Top level module for the Neural Network instantiated in hardware.
This module contains all the parameters from all nodes from the hardware metadata which it uses to instantiate all the nodes in the computational graph with the correct parameters.

### Input and output ports
The top level module contains input and output ports for the clock because the design is synchronous and contains a reset signal to set the model's state to a known one when needed. It has an array (bus) of input wires for the Neural Network model's data input, along side this it has a valid and ready flags for this bus of data. The output ports are instantiated in the same way but with the appropriate parameters.

### Internal wires
Then the necessary internal wires are instantiated to connect the various modules instantiated within the Top Level module. These connections are achieved through the assign statements that follow the instantiations of the hardware modules.

### Design
The PyTorch design of the the Neural Network consists of the following layers:
* Flatten layer
* Fully Connected layer
* relu layer
* Fully Connected layer
* ReLu Layer
* Fully Connected layer

The RTL design of the Neural Network is different:
* Flatten layer is not needed since the `data_in_0` input port is of dimension `(1 x 784 x 1)` it implies that the input should have been processed before hand such that it is 1 dimensional.
* Fixed Linear layer 1: this layer eats the input to the Top level module: `data_in_0`. It grabs its weights and biases from two other modules `weight_source` and `bias_source`.
* ReLu layer: this layer is fed by the output of Fixed Linear layer 1.
* Fixed Linear layer 2: takes the output from the previous ReLu layer and grabs a set of weights and biases from another pair of `weight_source` and `bias_source`.
* ReLu layer 2: this layer is fed by the output of Fixed Linear layer 2.
* Fixed Linear layer 3: takes the output from the previous ReLu layer and grabs a set of weights and biases from another pair of `weight_source` and `bias_source`.
* Output: 
    * `data_out_0`: this is driven by the output of the Fixed Linear Layer 3.
    * `data_out_valid_0`: driven the output signal `fc3_data_out_0_valid` from the Fixed Lineary layer 3.

## Task 3: Launch the simulation, log and show the simulation results.

100000.00ns INFO     cocotb.regression                  test passed
100000.00ns INFO     cocotb.regression                  **************************************************************************************
                                                        ** TEST                          STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
                                                        **************************************************************************************
                                                        ** mase_top_tb.test.test          PASS      100000.00           9.63      10381.41  **
                                                        **************************************************************************************
                                                        ** TESTS=1 PASS=1 FAIL=0 SKIP=0             100000.00           9.98      10015.59  **
                                                        **************************************************************************************

## Task 4: Choose another layer type from the Pytorch list and write a SystemVerilog file to implement that layer in hardware.

### Design

### Effect
**Throughput**<br>

**Latency**<br>

**Accuracy**<br>

## Extension: cocotb testbench

### Testbench

### Results: make a PR
