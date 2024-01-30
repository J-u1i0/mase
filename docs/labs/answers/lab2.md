# Lab 2

# TODO: use the MASE documentation: https://deepwok.github.io/mase/

## Explain the functionality of `report\_graph\_analysis\_pass` and its printed jargons such as `placeholder`, `get\_attr` ... You might find the doc of torch.fx useful.

#### Return: graph, {}
Since this an analysis pass it will not transform the inputted graph.

#### Operation:
1. Prints the torch fx graph.
2. Lists the metadata of each node in the MaseGraph and prints it. The metadata 
consists of PyTorch's operator names such as "Linear" and it also contains 
the nodes'/operators'/layers' arguments and defaulted arguments.
```
Layer types
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
```
3. Counts the torch fx opcodes of each node in the graph and displays this 
information. The torch fx opcodes describe the abstract functionality of each 
node in a torch fx graph.
```
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 4, 'output': 1}
```

## What are the functionalities of `profile\_statistics\_analysis\_pass` and `report\_node\_meta\_param\_analysis\_pass` respectively?

#### Return: graph, {}
Since this an analysis pass it will not transform the inputted graph.

#### Operation:
Profile statistics?
Difference between a name and a type?
What is the meta section of the nodes?

graph\_iterator\_register\_stat\_collections\_by\_name
Only works with layers that are "call\_modules" and are in the target\_weight\_nodes
pass arguments.

WeightStatCollection

## Explain why only 1 OP is changed after the `quantize_transform_pass`
* The linear layer is the only OP changed.
    * Reason: only layer that can be quantised.
    * Because it contains weights.
    * Quantisation reduces the precision format of these numbers.
* Any layer that contains learnable parameters can be quantised.
* In the network only the Linear layer satisfies this condition because it 
contains weights and biases.
* The batch norm layer can also be quantised since it contains the scaling parameters and shift parameters.
    * However, in the pass args the only OP that is chosen to be quantised is the linear layer.
* Activation layers can also be quantised.

## Write some code to traverse both mg and ori\_mg, check and comment on the nodes in these two graphs. You might find the source code for the implementation of summarize\_quantization\_analysis\_pass useful.

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


## Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the pass\_args for your custom network might be different if you have used more than the Linear layer in your network.

1. Load the larger JSC model.
2. Prepare the graph for other passes.
3. Run a simple report graph pass for a sanity check.
4. Form the pass args for quantisation.
```python
pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}
```
Need to keep in mind that if I use more layers for the larger JSC model then those need to be quantised as well. I don't think it will change because the network will just have more linear and batch norm layers.
```
QUANTIZEABLE_OP = (
    "add",
    "bmm",
    "conv1d",
    "conv2d",
    "matmul",
    "mul",
    "linear",
    "relu",
    "sub",
)
```

## Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the Quantized Layers.
Print the quantised weights.
Print the original weights.

1. Write a function to access the model parameters after training.
    a. 
2. Train the network.
3. Execute the code.

When a model is trained does it get saved with its weights?
Yes.
So I need to access the weights before quantisation.

Then quantise the model
Then test the model but when testing access the weights?
