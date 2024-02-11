import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
from logging import getLogger

# figure out the correct path
#machop_path = Path(".").resolve().parent.parent /"mase/machop"
#assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
#sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
    calculate_avg_bits_mg_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model




logger = getLogger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

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

import copy
# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_args))

# grid search
import torch
import numpy as np
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# ========================================================================== #
# Metrics
# ========================================================================== #

class Metrics:

    def __init__(self):
        self.losses = []
        self.accuracies = []
        self.latencies = []
        self.search_space = []
        self.confusion_matrices = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.mem_bits = []

    def calc_mean(self, arr):
        arr = np.array(arr)
        return np.mean(arr, axis=0)



class RunningMetrics(Metrics):

    def add_point(self, val):
        self.search_space.append(val)

    def add_lat(self, val):
        self.latencies.append(val)

    def averages(self):
        acc_avg = self.calc_mean(self.accuracies)
        loss_avg = self.calc_mean(self.losses)
        lat_avg = self.calc_mean(self.latencies)
        return (acc_avg, loss_avg, lat_avg)

    def calculate(self, preds, ys):
        accuracy = MulticlassAccuracy(num_classes=5)
        precision = MulticlassPrecision(num_classes=5)
        recall = MulticlassRecall(num_classes=5)
        f1 = MulticlassF1Score(num_classes=5)
        confusion = ConfusionMatrix(task="multiclass", num_classes=5)

        # Calculate the loss.
        loss = torch.nn.functional.cross_entropy(preds, ys)
        self.losses.append(loss.item())
        
        # Calculate accuracy.
        acc = accuracy(preds, ys)
        self.accuracies.append(acc.item())

        # Calculate recall.
        rec = recall(preds, ys)
        self.recalls.append(rec.item())

        # Calculate F1 score.
        score = f1(preds, ys)
        self.f1_scores.append(score.item())

        # Calculate the confusion matrix.
        matrix = confusion(preds, ys)
        self.confusion_matrices.append(matrix.tolist())

        # Calculate precision.
        precision = precision(preds, ys)
        self.precisions.append(precision.item())

    def show(self):
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"Mean Loss: {self.calc_mean(self.losses)}")
        print(f"Mean Accuracy: {self.calc_mean(self.accuracies)}")
        print(f"Mean Precision: {self.calc_mean(self.precisions)}")
        print(f"Mean Recall: {self.calc_mean(self.recalls)}")
        print(f"Mean F1 score: {self.calc_mean(self.f1_scores)}")
        print(f"Mean Latency (ns): {self.calc_mean(self.latencies)}")
        print(f"Mean Confusion matrix:\n{self.calc_mean(self.confusion_matrices)}")
        print(f"----------------------------------------------")


class AvgMetrics(Metrics):

    def extract(self, metrics: RunningMetrics):
        self.losses.append(self.calc_mean(metrics.losses))
        self.accuracies.append(self.calc_mean(metrics.accuracies))
        self.latencies.append(self.calc_mean(metrics.latencies))
        self.confusion_matrices.append(self.calc_mean(metrics.confusion_matrices))
        self.precisions.append(self.calc_mean(metrics.precisions))
        self.recalls.append(self.calc_mean(metrics.recalls))
        self.f1_scores.append(self.calc_mean(metrics.f1_scores))

    def mem_requirements(self, graph:MaseGraph):
        data_in_cost, weights_cost = 0, 0
        data_in_size, weights_size = 0, 0

        for node in graph.fx_graph.nodes:
            mase_meta = node.meta["mase"].parameters
            mase_op = mase_meta["common"]["mase_op"]
            mase_type = mase_meta["common"]["mase_type"]

            if mase_type in ["module", "module_related_func"]:
                if mase_op in ["linear", "conv2d", "conv1d"]:
                    data_in_0_meta = mase_meta["common"]["args"]["data_in_0"]
                    w_meta = mase_meta["common"]["args"]["weight"]
                    # maybe add bias
                    d_size = np.prod(data_in_0_meta["shape"])
                    w_size = np.prod(w_meta["shape"])
                    data_in_cost += sum(data_in_0_meta["precision"]) * d_size
                    data_in_size += d_size
                    weights_size += w_size
                    weights_cost += sum(w_meta["precision"]) * w_size

        mem = data_in_cost + weights_cost
        self.mem_bits.append(mem)

        return graph

    def show(self):
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"Memory Requirements (bits): {self.mem_bits[-1]}")
        print(f"Mean Loss: {self.calc_mean(self.losses)}")
        print(f"Mean Accuracy: {self.calc_mean(self.accuracies)}")
        print(f"Mean Precision: {self.calc_mean(self.precisions)}")
        print(f"Mean Recall: {self.calc_mean(self.recalls)}")
        print(f"Mean F1 score: {self.calc_mean(self.f1_scores)}")
        print(f"Mean Latency (ns): {self.calc_mean(self.latencies)}")
        print(f"Mean Confusion matrix:\n{self.calc_mean(self.confusion_matrices)}")
        print(f"----------------------------------------------")

# ========================================================================== #
# Search Strategy: grid search
# ========================================================================== #

def search_strategy(search_spaces, runner, data_module, batch_num, mg):
    avg_metrics = AvgMetrics()

    for idx, config in enumerate(search_spaces):
        mg, _ = quantize_transform_pass(mg, config)
        mg = avg_metrics.mem_requirements(mg)
        metrics = runner(data_module, mg, batch_num)
        # Save avg metrics
        avg_metrics.extract(metrics)
        avg_metrics.show()
    return avg_metrics

# ========================================================================== #
# Runner Strategy
# ========================================================================== #
from torchmetrics.classification import MulticlassAccuracy, ConfusionMatrix, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import time

def runner_strategy(data_module, mg, batch_num):
    metrics = RunningMetrics()
    batch_idx = 0
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        # Timing the inference.
        start = time.perf_counter_ns()
        preds = mg.model(xs)
        end = time.perf_counter_ns()
        lat = end - start
        metrics.add_lat(lat)

        # Calculate all metrics.
        metrics.calculate(preds, ys)

        if batch_idx > batch_num:
            break

        batch_idx += 1
    return metrics

# ========================================================================== #
# Execute search strategy
# ========================================================================== #

batch_num = 5
avg_metrics = search_strategy(search_spaces, runner_strategy, data_module, batch_num, mg)
avg_metrics.show()

