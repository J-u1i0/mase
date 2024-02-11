# Lab 2
---
## Task 1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr` ... You might find the doc of torch.fx useful.

NOTE: this function is part of the `analysis` module therefore will not transform the `fx_graph` in the `MaseGraph`.
The function `report_graph_analysis_pass` prints the torch fx graph that is stored in
`MaseGraph` argument, in this case it prints the fx graph from the JSC-Tiny network. 
The output is shown below.
```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    return seq_blocks_3Network overview:
```

Furthermore, the `report_graph_analysis_pass` function also counts the opcodes 
of the graph and displays this information as a dictionary.
The output is shown below:
```
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 4, 'output': 1}
```
The opcodes of an fx graph represent the high level action that each node carries out.

Lastly, the `report_graph_analysis_pass` function prints a list of the metadata 
of each node in the MaseGraph. The metadata consists of PyTorch's operator names 
such as "Linear" and it also contains the nodes' arguments and defaulted arguments.
This is shown below:
```
Layer types
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
```

## Task 2. What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?

In the `profile_statistics_analysis_pass` function the `pass_args` are very important because they indicate which nodes, in the given graph, should be targetted. This is carried out by the `target_weight_nodes` and the `target_activation_nodes` sections of the `pass_args` dictionary.
In addition, the pass args will determine the statistical analysis performed on each targetted node in the inputted graph.
For the nodes that get targetted, the `profile_statistics_analysis_pass` function sets statistical meta data about each node in the graph. This meta data is stored in the inputted `MaseGraph`.
**NOTE:** both of these functions are part of the `analysis` module therefore will not transform the `fx_graph` in the `MaseGraph`.
<br>
The `pass_args` used are shown below:
```
pass_args = {
    "by": "type",                                                            # collect statistics by node name
    "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
    "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
    "weight_statistics": {
        "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
    },
    "activation_statistics": {
        "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
    },
    "input_generator": input_generator,                                      # the input generator for feeding data to the model
    "num_samples": 32,                                                       # feed 32 samples to the model
}
```

The `report_node_meta_param_analysis_pass` function generates a report about the found meta data for each node in the given `MaseGraph`. 
The `pass_args` are also very important for this function since they will determine the headers of the table. 
This function traverses the inputted `MaseGraph` and adds the required meta data for each node to a table (python list).
This table then gets displayed to the console and can get saved to a file if provided in the `pass_args`.
**NOTE:** the `pass_args` for this function are different from the `profile_statistics_analysis_pass` function.
The results for this function are displayed below:

```
+--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
| Node name    | Fx Node op   | Mase type           | Mase op      | Software Param                                                                          |
+==============+==============+=====================+==============+=========================================================================================+
| x            | placeholder  | placeholder         | placeholder  | {'results': {'data_out_0': {'stat': {}}}}                                               |
+--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
| seq_blocks_0 | call_module  | module              | batch_norm1d | {'args': {'bias': {'stat': {}},                                                         |
|              |              |                     |              |           'data_in_0': {'stat': {}},                                                    |
|              |              |                     |              |           'running_mean': {'stat': {}},                                                 |
|              |              |                     |              |           'running_var': {'stat': {}},                                                  |
|              |              |                     |              |           'weight': {'stat': {}}},                                                      |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                               |
+--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
| seq_blocks_1 | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 512,                       |
|              |              |                     |              |                                                     'max': 1.9987810850143433,          |
|              |              |                     |              |                                                     'min': -1.6725506782531738,         |
|              |              |                     |              |                                                     'range': 3.6713318824768066}}}},    |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                               |
+--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
| seq_blocks_2 | call_module  | module_related_func | linear       | {'args': {'bias': {'stat': {'variance_precise': {'count': 5,                            |
|              |              |                     |              |                                                  'mean': -0.010048285126686096,         |
|              |              |                     |              |                                                  'variance': 0.052554499357938766}}},   |
|              |              |                     |              |           'data_in_0': {'stat': {}},                                                    |
|              |              |                     |              |           'weight': {'stat': {'variance_precise': {'count': 80,                         |
|              |              |                     |              |                                                    'mean': -0.009382132440805435,       |
|              |              |                     |              |                                                    'variance': 0.02084079012274742}}}}, |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                               |
+--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
| seq_blocks_3 | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 160,                       |
|              |              |                     |              |                                                     'max': 0.7414377331733704,          |
|              |              |                     |              |                                                     'min': -0.989246129989624,          |
|              |              |                     |              |                                                     'range': 1.7306838035583496}}}},    |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                               |
+--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
| output       | output       | output              | output       | {'args': {'data_in_0': {'stat': {}}}}                                                   |
+--------------+--------------+---------------------+--------------+-----------------------------------------------------------------------------------------+
```

## Task 3. Explain why only 1 OP is changed after the `quantize_transform_pass`
The `QUANTIZEABLE_OP` tuple contains all the layers that can be quantized. 

In the provided graph (from JSC-Tiny) only the Linear layer exists in this tuple.
Moreover, in the `pass_args` given to the `quantize_transform_pass` only the linear 
layer is given `config` arguments. These arguments are used by the `quantize_transform_pass`
to quantize the targetted layer to the required degree.

In the JSC-Tiny graph there is only 1 linear layer and the `pass_args` only target 
linear layers therefore only 1 OP is changed after the `quantize_transform_pass`.

## Task 4. Write some code to traverse both mg and ori\_mg, check and comment on the nodes in these two graphs. You might find the source code for the implementation of summarize\_quantization\_analysis\_pass useful.

The code used to traverse the original Mase Graph and the transformed Mase Graph is shown below.
```python
from chop.passes.graph.transforms.quantize.summary import graph_iterator_compare_nodes
from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
import pandas as pd

def get_type_str(node):
    if node.op == "call_module":
        return type(get_node_actual_target(node)).__name__
    elif get_mase_type(node) in [
        "builtin_func",
        "module_related_func",
        "patched_func",
    ]:
        return get_node_actual_target(node).__name__
    elif get_mase_type(node) in ["implicit_func"]:
        actual_target = get_node_actual_target(node)
        if isinstance(actual_target, str):
            return actual_target
        else:
            return actual_target.__name__
    else:
        return node.target

def my_traversion(original, transformed):
    headers = [
        "Name",
        "MASE_TYPE",
        "Mase_OP",
        "Original type",
        "Quantized type",
        "Changed",
    ]
    rows = []
    for ori_n, n in zip(original.fx_graph.nodes, transformed.fx_graph.nodes):
        rows.append(
            [
                n.name,
                get_mase_type(n),
                get_mase_op(n),
                get_type_str(ori_n),
                get_type_str(n),
                type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n)),
            ]
        )

    df = pd.DataFrame(rows, columns=headers)

    print(f"My summary \n{df.to_markdown()}")
```

The result of running the code the table shown below.

|    | Name         | MASE\_TYPE           | Mase\_OP      | Original type   | Quantized type   | Changed   |
|---:|:-------------|:--------------------|:-------------|:----------------|:-----------------|:----------|
|  0 | x            | placeholder         | placeholder  | x               | x                | False     |
|  1 | seq\_blocks\_0 | module              | batch\_norm1d | BatchNorm1d     | BatchNorm1d      | False     |
|  2 | seq\_blocks\_1 | module\_related\_func | relu         | ReLU            | ReLU             | False     |
|  3 | seq\_blocks\_2 | module\_related\_func | linear       | Linear          | LinearInteger    | True      |
|  4 | seq\_blocks\_3 | module\_related\_func | relu         | ReLU            | ReLU             | False     |
|  5 | output       | output              | output       | output          | output           | False     |

From this table it can be observed that the quantisation pass has only changed the Linear Layer.<br>
The quantisation pass has changed the Linear layer by changing its type from `Linear` to `LinearInteger` as shown in the table.
All other blocks have been unchanged by the quantisation pass as show in the table.
The behaviour displayed in the table matches the expected behaviour from the `pass_args` inputted into the quantisation pass.

## Task 5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the pass\_args for your custom network might be different if you have used more than the Linear layer in your network.

```
| Original type   | OP           |   Total |   Changed |   Unchanged |
|-----------------+--------------+---------+-----------+-------------|
| BatchNorm1d     | batch_norm1d |       5 |         0 |           5 |
| Linear          | linear       |       5 |         5 |           0 |
| ReLU            | relu         |       6 |         0 |           6 |
| output          | output       |       1 |         0 |           1 |
| x               | placeholder  |       1 |         0 |           1 |
INFO:chop.passes.graph.transforms.quantize.summary:
| Original type   | OP           |   Total |   Changed |   Unchanged |
|-----------------+--------------+---------+-----------+-------------|
| BatchNorm1d     | batch_norm1d |       5 |         0 |           5 |
| Linear          | linear       |       5 |         5 |           0 |
| ReLU            | relu         |       6 |         0 |           6 |
| output          | output       |       1 |         0 |           1 |
| x               | placeholder  |       1 |         0 |           1 |
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    %seq_blocks_8 : [num_users=1] = call_module[target=seq_blocks.8](args = (%seq_blocks_7,), kwargs = {})
    %seq_blocks_9 : [num_users=1] = call_module[target=seq_blocks.9](args = (%seq_blocks_8,), kwargs = {})
    %seq_blocks_10 : [num_users=1] = call_module[target=seq_blocks.10](args = (%seq_blocks_9,), kwargs = {})
    %seq_blocks_11 : [num_users=1] = call_module[target=seq_blocks.11](args = (%seq_blocks_10,), kwargs = {})
    %seq_blocks_12 : [num_users=1] = call_module[target=seq_blocks.12](args = (%seq_blocks_11,), kwargs = {})
    %seq_blocks_13 : [num_users=1] = call_module[target=seq_blocks.13](args = (%seq_blocks_12,), kwargs = {})
    %seq_blocks_14 : [num_users=1] = call_module[target=seq_blocks.14](args = (%seq_blocks_13,), kwargs = {})
    %seq_blocks_15 : [num_users=1] = call_module[target=seq_blocks.15](args = (%seq_blocks_14,), kwargs = {})
    return seq_blocks_15Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 16, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=16, bias=True), BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=16, bias=True), BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=16, bias=True), BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=16, bias=True), BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
```

From the results, it can be observed that 5 linear layers are quantized which matches the expected behaviour and design of the JSC-My network.

## Task 6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the Quantized Layers.

```python
# Displaying the quantised and non-quantised weights.
def display_weights(mg: MaseGraph, ori_mg: MaseGraph):
    for quant, ori in zip(mg.model.seq_blocks.named_children(), ori_mg.model.seq_blocks.named_children()):
        _, quant = quant
        _, ori = ori
        if isinstance(quant, torch.nn.modules.linear.Linear):
            print(f"Layer: {quant}")
            print(f"Original weights:\n{ori.weight}")
            print(f"Quantised weights:\n{quant.w_quantizer(quant.weight)}")
display_weights(mg, ori_mg)
```

The code above takes two `MaseGraph` objects as inputs it then traverses them 
and extracts their weights.
**NOTE:** due to the `LinearInteger` layers performing the quantisation of their 
weight on the fly, the `w_quantizer` function from the layer was applied to the 
layer's weights thus replicating the expected behaviour.

The results are shown below:
**Note**: only a part of the results is shown for brevity's sake.
```
Original weights:
Parameter containing:
tensor([[ 0.1229, -0.0122, -0.1395,  0.3080,  0.0464,  0.2593, -0.0608,  0.0120,
         -0.1413, -0.1137,  0.0778,  0.0242, -0.1053,  0.0551,  0.2815,  0.3175],
        [-0.1244,  0.2101, -0.1388, -0.1228,  0.1795,  0.0759, -0.0949, -0.1859,
         -0.0175,  0.1861, -0.2143, -0.2263,  0.0522, -0.0842, -0.1683,  0.0119],
        [-0.0306, -0.1838, -0.2012, -0.0442, -0.2356, -0.1146, -0.2018, -0.2481,
          0.2691, -0.1714, -0.0923,  0.1454,  0.2122,  0.3798, -0.0619, -0.1836],
        [-0.1169, -0.0624,  0.2918, -0.3108, -0.0513,  0.0096, -0.1084, -0.1502,
          0.2431, -0.2372,  0.2161, -0.1961,  0.2619, -0.2912,  0.2729, -0.3240],
        [-0.1789,  0.0406,  0.3311, -0.1322, -0.1155,  0.3726,  0.1100,  0.2577,
         -0.3098, -0.0238, -0.0979,  0.3664,  0.2133, -0.1704,  0.0921,  0.1265]],
       requires_grad=True)
Quantised weights:
tensor([[ 0.1250, -0.0000, -0.1250,  0.3125,  0.0625,  0.2500, -0.0625,  0.0000,
         -0.1250, -0.1250,  0.0625,  0.0000, -0.1250,  0.0625,  0.3125,  0.3125],
        [-0.1250,  0.1875, -0.1250, -0.1250,  0.1875,  0.0625, -0.1250, -0.1875,
         -0.0000,  0.1875, -0.1875, -0.2500,  0.0625, -0.0625, -0.1875,  0.0000],
        [-0.0000, -0.1875, -0.1875, -0.0625, -0.2500, -0.1250, -0.1875, -0.2500,
          0.2500, -0.1875, -0.0625,  0.1250,  0.1875,  0.3750, -0.0625, -0.1875],
        [-0.1250, -0.0625,  0.3125, -0.3125, -0.0625,  0.0000, -0.1250, -0.1250,
          0.2500, -0.2500,  0.1875, -0.1875,  0.2500, -0.3125,  0.2500, -0.3125],
        [-0.1875,  0.0625,  0.3125, -0.1250, -0.1250,  0.3750,  0.1250,  0.2500,
         -0.3125, -0.0000, -0.1250,  0.3750,  0.1875, -0.1875,  0.0625,  0.1250]],
       grad_fn=<IntegerQuantizeBackward>)
```

From the results it can be observed that the original weights are approximated using 
powers of 2 thus proving that they have been quantised.

## Task 7. Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface. 

## Task 8. Implement a pass to count the number of FLOPs (floating-point operations) and BitOPs (bit-wise operations).

### FLOPs pass

