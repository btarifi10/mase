
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
        try:
            in_data_precision = node.meta['mase'].parameters['common']['args']['data_in_0']['precision']
        except KeyError:
            in_data_precision = None
        try:
            weight_precision = node.meta['mase'].parameters['common']['args']['weights']['precision']
        except KeyError:
            weight_precision = None
        try:    
            bias_precision = node.meta['mase'].parameters['common']['args']['bias']['precision']
        except KeyError:
            bias_precision = None
        out_data = node.meta['mase'].parameters['common']['results']['data_out_0']['value']
        
        if not isinstance(in_data, tuple):
            in_data = (in_data,)
        if not isinstance(out_data, tuple):
            out_data = (out_data,)

        target = get_node_actual_target(node)
        if mase_type == "module" or mase_type == "module_related_func":
            data = calculate_modules(target, in_data, out_data)

            bitops_multiplier = 1
            if in_data_precision:
                bitops_multiplier *= sum(in_data_precision)
            if weight_precision:
                bitops_multiplier *= sum(weight_precision)

            data["forward_bitops"] = data["computations"] * bitops_multiplier
            data["backward_bitops"] = data["backward_computations"] * bitops_multiplier

            node.meta["mase"].parameters["software"]["computations"] = data
        else:
            # TODO: add support for  functions
            node.meta["mase"].parameters["software"]["computations"] = {}

        rows.append([node.name, mase_op, *(node.meta["mase"].parameters["software"]["computations"].values())])

    if not silent:
        print("Computations summary:")
        print("---------------------")
        print(tabulate(rows, headers=headers, tablefmt="github"))

    return graph
