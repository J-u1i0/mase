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
