from ptflops import get_model_complexity_info

def get_input_shape(graph):
    node = next(iter(graph.fx_graph.nodes))

    # Input data layer will always have "data_out_0".
    data_out = "data_out"
    for key in node.meta["mase"].parameters["common"]["results"]:
        if data_out in key:
            in_shape = node.meta["mase"].parameters["common"]["results"][key]["shape"]
            return graph, in_shape

    raise KeyError("No input layer was found in the graph due to data_out_0 not being present.")

def report_flops(graph, pass_args={}):
    model = graph.model
    graph, in_shape = get_input_shape(graph)
    # Ignore the bath dimension.
    dims = tuple(in_shape[1:])
    macs, _ = get_model_complexity_info(model, dims, as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"FLOPs: {macs}")
    return graph, {}
