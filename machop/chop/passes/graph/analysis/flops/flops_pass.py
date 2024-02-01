
from tabulate import tabulate
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.analysis.flops.calculate_flops import calculate_funcs, calculate_modules, get_flops_per_op
from chop.passes.graph.utils import get_mase_op, get_mase_type
from chop.passes.graph.utils import get_node_actual_target


def analyse_flops_pass(graph: MaseGraph, silent=False): 
    headers = [
        "Node name",
        "Node type",
        "Parameters",
        "Forward FLOPS",
        "Backward FLOPS",
        "Input buffer size",
        "Output buffer size",
    ]

    rows = []

    for node in graph.fx_graph.nodes:
        mase_op = get_mase_op(node)
        mase_type = get_mase_type(node)
 
        try:
            in_data = node.meta['mase'].parameters['common']['args']['data_in_0']['value']
        except KeyError:
            in_data = (None,)
        out_data = node.meta['mase'].parameters['common']['results']['data_out_0']['value']
        
        if not isinstance(in_data, tuple):
            in_data = (in_data,)
        if not isinstance(out_data, tuple):
            out_data = (out_data,)

        target = get_node_actual_target(node)
        if mase_type == "module" or mase_type == "module_related_func":
            data = calculate_modules(target, in_data, out_data)
            node.meta["mase"].parameters["software"]["computations"] = data
        else:
            # TODO: add support for  functions
            node.meta["mase"].parameters["software"]["computations"] = {}

        rows.append([node.name, mase_op, *(node.meta["mase"].parameters["software"]["computations"].values())])

    if not silent:
        print("Computations summary:")
        print("---------------------")
        print(tabulate(rows, headers=headers, tablefmt="github"))
