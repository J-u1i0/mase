import torch

#from ..utils import _import_config_from_py_file
#from .deepspeed import estimate_sw_deepspeed
#from .fine_grained import estimate_sw_fine_grained
from .calculator import calculate_modules

#estimator_style_map = {
#    "deepspeed": estimate_sw_deepspeed,
#    # FIXME
#    "fine-grained": estimate_sw_fine_grained,
#}


#def run_flop_estimator(
#    model_name: int,
#    task: str,
#    info: dict,
#    model: torch.nn.Module,
#    data_module,
#    config_path: str = None,
#    save_dir: str = None,
#):
#    config = _import_config_from_py_file(model_name, config_path)
#
#    # set default to deepspeed
#    if "style" in config:
#        estimator_style = config.pop("style")
#    else:
#        estimator_style = config.get("style", "deepspeed")
#
#    estimator_style_map[estimator_style](
#        model_name=model_name,
#        info=info,
#        model=model,
#        task=task,
#        data_module=data_module,
#        save_dir=save_dir,
#        config=config,
#    )

def report_graph_flops_pass(graph, pass_args={"file_name": None}):
    """
        Generates a report for the graph analysis
        and prints out an over the model in a table.

        :param graph: a MaseGraph
        :type graph: MaseGraph
        :param pass_args: this pass can take a string argument named "file_name", defaults to None
        :type pass_args: dict, optional
        :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
        :rtype: tuple(MaseGraph, dict)
    """

    for node in graph.fx_graph.nodes:
        if node.meta["mase"].module is not None:
            #layer_types.append(node.meta["mase"].module)
            calculate_modules(node.meta["mase"].module)
            #print(node.meta["mase"].module)

    return graph, {}
