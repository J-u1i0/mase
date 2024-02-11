import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity

from chop.passes.graph import (
    save_node_meta_param_interface_pass,
    report_node_meta_param_analysis_pass,
    report_node_shape_analysis_pass,
    profile_statistics_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph

from chop.models import get_model_info, get_model

set_logging_verbosity("info")

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()
#print(f"Data module: {data_module}")
#print(f"Train module: {data_module.train_dataset}")
#exit()

JSC_MY_PATH = "../../mase_output/jsc_my.ckpt"
JSC_TINY_PATH = "../../mase_output/jsc_tiny_02_07/software/training_ckpts/best.ckpt"

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=JSC_TINY_PATH, load_type="pl", model=model)

# get the input generator
input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
#print(f"Dummy: {dummy_in}")
#exit()
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# Report the graph's FLOPs.
#from chop.passes.graph.analysis.flop_estimator import report_graph_flops_pass
# TODO: what to do about the inputs and outputs for the flops
#_ = report_graph_flops_pass(mg)
#from ptflops import get_model_complexity_info
#macs, params = get_model_complexity_info(model, (16,), as_strings=True, print_per_layer_stat=True, verbose=True)

#mg, _ = report_node_shape_analysis_pass(mg)

from chop.passes.graph.analysis.report.report_flops import report_flops
mg, _ = report_flops(mg)

#print(f"Macs:\n{macs}")
#print(f"Params:\n{params}")
## report graph is an analysis pass that shows you the detailed information in the graph
#from chop.passes.graph import report_graph_analysis_pass
#_ = report_graph_analysis_pass(mg)
#
#pass_args = {
#    "by": "type",
#    "default": {"config": {"name": None}},
#    "linear": {
#        "config": {
#            "name": "integer",
#            # data
#            "data_in_width": 8,
#            "data_in_frac_width": 4,
#            # weight
#            "weight_width": 8,
#            "weight_frac_width": 4,
#            # bias
#            "bias_width": 8,
#            "bias_frac_width": 4,
#        }
#    },
#}
#
#from chop.passes.graph.transforms import (
#    quantize_transform_pass,
#    summarize_quantization_analysis_pass,
#)
#from chop.ir.graph.mase_graph import MaseGraph
#
#
#ori_mg = MaseGraph(model=model)
#ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
#ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})
#
#mg, _ = quantize_transform_pass(mg, pass_args)
#summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")
