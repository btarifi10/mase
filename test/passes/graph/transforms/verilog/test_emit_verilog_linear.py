#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os, sys, logging, traceback, pdb
import toml

import torch
import torch.nn as nn

import chop as chop
import chop.passes as passes
from chop.tools.utils import execute_cli

from pathlib import Path

from chop.actions import simulate
from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


sys.excepthook = excepthook


# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 28 * 28, bias=False)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        return x


def test_emit_verilog_linear():
    mlp = MLP()
    mg = chop.MaseGraph(model=mlp)

    # Provide a dummy input for the graph so it can use for tracing
    batch_size = 1
    x = torch.randn((batch_size, 28, 28))
    dummy_in = {"x": x}

    mg, _ = passes.init_metadata_analysis_pass(mg, None)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    # Quantize to int
    config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "configs",
        "tests",
        "quantize",
        "integer.toml",
    )

    # load toml config file
    with open(config_file, "r") as f:
        quan_args = toml.load(f)["passes"]["quantize"]
    mg, _ = passes.quantize_transform_pass(mg, quan_args)

    # There is a bug in the current quantizzation pass, where the results metadata is not uppdated with the precision.
    # Here we temporarily update the metadata here so we can test the hardware back end.
    for node in mg.fx_graph.nodes:
        for arg, _ in node.meta["mase"].parameters["common"]["args"].items():
            if (
                type(node.meta["mase"].parameters["common"]["args"][arg]) == dict
                and "type" in node.meta["mase"].parameters["common"]["args"][arg].keys()
            ):
                node.meta["mase"].parameters["common"]["args"][arg]["type"] = "fixed"
        for result, _ in node.meta["mase"].parameters["common"]["results"].items():
            if (
                type(node.meta["mase"].parameters["common"]["results"][result]) == dict
                and "type"
                in node.meta["mase"].parameters["common"]["results"][result].keys()
            ):
                node.meta["mase"].parameters["common"]["results"][result][
                    "type"
                ] = "fixed"
                node.meta["mase"].parameters["common"]["results"][result][
                    "precision"
                ] = [8, 3]

    # Increase weight range
    mg.model.fc1.weight = torch.nn.Parameter(
        10 * torch.randn(mg.model.fc1.weight.shape)
    )

    mg, _ = passes.add_hardware_metadata_analysis_pass(mg)
    mg, _ = passes.report_node_hardware_type_analysis_pass(mg)  # pretty print

    mg, _ = passes.emit_verilog_top_transform_pass(mg)
    mg, _ = passes.emit_bram_transform_pass(mg)
    mg, _ = passes.emit_internal_rtl_transform_pass(mg)
    mg, _ = passes.emit_cocotb_transform_pass(mg)

    simulate(skip_build=False, skip_test=True)


if __name__ == "__main__":
    test_emit_verilog_linear()
