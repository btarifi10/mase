import logging
from math import ceil
from torch import nn
from chop.passes.graph.utils import get_mase_op, get_mase_type, get_parent_name
from chop.tools.logger import get_logger


logger = get_logger("chop")
logger.setLevel(logging.INFO)

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def instantiate_relu(in_features):
    return nn.ReLU(in_features)

def instantiate_batchnorm(in_features):
    return nn.BatchNorm1d(in_features)

def linear_multiplier_transform_pass(graph, pass_args=None):
    default = pass_args.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    by = pass_args.pop('by')
    if not by in ['name', 'type']:
        raise ValueError(f"'by' must be 'name' or 'type' for linear_multiplier_transform_pass.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        if by == 'name':
            config = pass_args.get(node.name, default)['config']
        else:
            config = pass_args.get(get_mase_op(node), default)['config']

        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            if not isinstance(ori_module, nn.Linear):
                raise ValueError(f"Node {node.name} is not a linear layer.")

            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only" or name == "both":
                output_multiplier = config.get("output_multiplier", config.get("channel_multiplier"))
                if not output_multiplier:
                    logger.warning(f"Could not find output_multiplier or channel_multiplier for node {node.name}. Using value of 1.")
                    output_multiplier = 1
                out_features = ceil(out_features * output_multiplier)
            elif name == "input_only" or name == "both":
                input_multiplier = config.get("input_multiplier", config["channel_multiplier"])
                if not input_multiplier:
                    logger.warning(f"Could not find input_multiplier or channel_multiplier for node {node.name}. Using value of 1.")
                    input_multiplier = 1
                in_features = ceil(in_features * input_multiplier)

            # Find the previous linear module
            # All the previous modules should be either Linear, ReLU, or BatchNorm1d
            # The batchnorm1d and relu layers should be resized to the new in_features
            # The previous linear layer's output should be scaled to match the new in_features
            if name == "input_only" or name == "both":
                valid = False
                prev_node = node.prev
                prev_module = graph.modules.get(prev_node.target, None)
                while (prev_node and prev_module and not valid):
                    if isinstance(prev_module, nn.Linear):
                        valid = True
                    prev_node = prev_node.prev
                    prev_module = graph.modules.get(prev_node.target, None)
                
                if valid:
                    prev_node = node.prev
                    prev_module = graph.modules[prev_node.target]
                    while (not isinstance(prev_module, nn.Linear)):
                        if isinstance(prev_module, nn.ReLU):
                            new_prev_module = instantiate_relu(in_features)
                            parent_name, name = get_parent_name(prev_node.target)
                            setattr(graph.modules[parent_name], name, new_prev_module)
                        elif isinstance(prev_module, nn.BatchNorm1d):
                            new_prev_module = instantiate_batchnorm(in_features)
                            parent_name, name = get_parent_name(prev_node.target)
                            setattr(graph.modules[parent_name], name, new_prev_module)
                        prev_node = prev_node.prev
                        prev_module = graph.modules[prev_node.target]
                    assert isinstance(prev_module, nn.Linear)
                    new_prev_module = instantiate_linear(prev_module.in_features, in_features, prev_module.bias)
                    parent_name, name = get_parent_name(prev_node.target)
                    setattr(graph.modules[parent_name], name, new_prev_module)
                else:
                    logger.warning(f"Node {node.name} is not connected to a linear layer on the input side. " + 
                                   "Skipping input transformation.")
                    in_features = ori_module.in_features

            if name == "output_only" or name == "both":
                valid = False
                next_node = node.next
                next_module = graph.modules.get(next_node.target, None)
                while (next_node and not valid):
                    if isinstance(next_module, nn.Linear):
                        valid = True
                    next_node = next_node.prev
                    next_module = graph.modules.get(next_node.target, None)

                if valid:
                    next_node = node.next
                    next_module = graph.modules[next_node.target]
                    while (not isinstance(next_module, nn.Linear)):
                        if isinstance(next_module, nn.ReLU):
                            new_next_module = instantiate_relu(out_features)
                            parent_name, name = get_parent_name(next_node.target)
                            setattr(graph.modules[parent_name], name, new_next_module)
                        elif isinstance(next_module, nn.BatchNorm1d):
                            new_next_module = instantiate_batchnorm(out_features)
                            parent_name, name = get_parent_name(next_node.target)
                            setattr(graph.modules[parent_name], name, new_next_module)
                        next_node = next_node.next
                        next_module = graph.modules[next_node.target]
                    assert isinstance(next_module, nn.Linear)
                    new_next_module = instantiate_linear(out_features, next_module.out_features, next_module.bias)
                    parent_name, name = get_parent_name(next_node.target)
                    setattr(graph.modules[parent_name], name, new_next_module)
                else:
                    logger.warning(f"Node {node.name} is not connected to a linear layer on the output side." + 
                                   "Skipping output transformation.")
                    out_features = ori_module.out_features

             # Finally, set the new linear module.
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)

    return graph, {}
