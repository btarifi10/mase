import logging
from math import ceil
from pprint import pprint
from torch import nn
from chop.passes.graph.utils import deepcopy_mase_graph, get_mase_op, get_mase_type, get_parent_name
from chop.tools.logger import get_logger


logger = get_logger("chop")
logger.setLevel(logging.INFO)

def instantiate_conv2d(in_channels, out_channels, bias, stride, kernel_size, padding):
    if bias is not None:
        bias = True
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        kernel_size=kernel_size,
        padding=padding,
        bias=bias)
    

def instantiate_relu():
    return nn.ReLU()

def instantiate_batchnorm(num_features, momentum=0.9):
    return nn.BatchNorm2d(
        num_features=num_features,
        momentum=momentum
    )

def conv2d_multiplier_transform_pass(graph, pass_args=None):
    default = pass_args.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    by = pass_args.pop('by')
    if not by in ['name', 'type']:
        raise ValueError(f"'by' must be 'name' or 'type' for linear_multiplier_transform_pass.")
    i = 0

    original_graph_dims = {}
    for node in graph.fx_graph.nodes:
        if graph.modules.get(node.target) and isinstance(graph.modules[node.target], nn.Conv2d):
            original_graph_dims[node.name] = {
                "in_channels": graph.modules[node.target].in_channels,
                "out_channels": graph.modules[node.target].out_channels,
            }

    print(original_graph_dims)

    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        if by == 'name':
            config = pass_args.get(node.name, default)['config']
        else:
            config = pass_args.get(get_mase_op(node), default)['config']

        config_name = config.get("name", None)
        if config_name is not None:
            ori_module = graph.modules[node.target]
            if not isinstance(ori_module, nn.Conv2d):
                print(ori_module)
                raise ValueError(f"Node {node.name} is not a Conv2d layer.")

            in_channels = original_graph_dims[node.name]['in_channels']
            out_channels = original_graph_dims[node.name]['in_channels']
            bias = ori_module.bias
            stride = ori_module.stride
            kernel_size = ori_module.kernel_size
            padding = ori_module.padding

            if config_name == "output_only" or config_name == "both":
                output_multiplier = config.get("output_multiplier", config.get("channel_multiplier"))
                if not output_multiplier:
                    logger.warning(f"Could not find output_multiplier or channel_multiplier for node {node.name}. Using value of 1.")
                    output_multiplier = 1
                out_channels = ceil(out_channels * output_multiplier)
            if config_name == "input_only" or config_name == "both":
                input_multiplier = config.get("input_multiplier", config.get("channel_multiplier"))
                if not input_multiplier:
                    logger.warning(f"Could not find input_multiplier or channel_multiplier for node {node.name}. Using value of 1.")
                    input_multiplier = 1
                in_channels = ceil(in_channels * input_multiplier)

            # Find the previous Conv2d module
            # All the previous modules should be either Conv2d, ReLU, or BatchNorm1d
            # The batchnorm1d and relu layers should be resized to the new in_channels
            # The previous conv2d layer's output should be scaled to match the new in_channels
            if config_name == "input_only" or config_name == "both":
                valid = False
                prev_node = node.prev
                prev_module = graph.modules.get(prev_node.target, None)
                while (prev_node and prev_module and not valid):
                    if isinstance(prev_module, nn.Conv2d):
                        valid = True
                    prev_node = prev_node.prev
                    prev_module = graph.modules.get(prev_node.target, None)
                
                if valid:
                    prev_node = node.prev
                    prev_module = graph.modules[prev_node.target]
                    while (not isinstance(prev_module, nn.Conv2d)):
                        if isinstance(prev_module, nn.ReLU):
                            new_prev_module = instantiate_relu()
                            parent_name, name_ = get_parent_name(prev_node.target)
                            setattr(graph.modules[parent_name], name_, new_prev_module)
                        elif isinstance(prev_module, nn.BatchNorm2d):
                            new_prev_module = instantiate_batchnorm(in_channels, prev_module.momentum)
                            parent_name, name_ = get_parent_name(prev_node.target)
                            setattr(graph.modules[parent_name], name_, new_prev_module)
                        prev_node = prev_node.prev
                        prev_module = graph.modules[prev_node.target]
                    assert isinstance(prev_module, nn.Conv2d)
                    new_prev_module = instantiate_conv2d(
                        prev_module.in_channels,
                        in_channels,
                        prev_module.bias,
                        prev_module.stride,
                        prev_module.kernel_size,
                        prev_module.padding
                    )
                    parent_name, name_ = get_parent_name(prev_node.target)
                    setattr(graph.modules[parent_name], name_, new_prev_module)
                else:
                    logger.warning(f"Node {node.name} is not connected to a conv2d layer on the input side. " + 
                                   "Skipping input transformation.")
                    in_channels = original_graph_dims[node.name]['in_channels']

            if config_name == "output_only" or config_name == "both":
                valid = False
                next_node = node.next
                next_module = graph.modules.get(next_node.target, None)
                while (next_node and not valid):
                    if isinstance(next_module, nn.Conv2d):
                        valid = True
                    next_node = next_node.prev
                    next_module = graph.modules.get(next_node.target, None)

                if valid:
                    next_node = node.next
                    next_module = graph.modules[next_node.target]
                    while (not isinstance(next_module, nn.Conv2d)):
                        if isinstance(next_module, nn.ReLU):
                            new_next_module = instantiate_relu()
                            parent_name, name_ = get_parent_name(next_node.target)
                            setattr(graph.modules[parent_name], name_, new_next_module)
                        elif isinstance(next_module, nn.BatchNorm2d):
                            new_next_module = instantiate_batchnorm(out_channels, next_module.momentum)
                            parent_name, name_ = get_parent_name(next_node.target)
                            setattr(graph.modules[parent_name], name_, new_next_module)
                        next_node = next_node.next
                        next_module = graph.modules[next_node.target]
                    assert isinstance(next_module, nn.Conv2d)
                    new_next_module = instantiate_conv2d(
                        out_channels,
                        next_module.out_channels,
                        next_module.bias,
                        next_module.stride,
                        next_module.kernel_size,
                        next_module.padding
                    )
                    parent_name, name_ = get_parent_name(next_node.target)
                    setattr(graph.modules[parent_name], name_, new_next_module)
                else:
                    logger.warning(f"Node {node.name} is not connected to a conv2d layer on the output side." + 
                                   "Skipping output transformation.")
                    out_channels = original_graph_dims[node.name]['out_channels']

             # Finally, set the new conv2 module.
            new_module = instantiate_conv2d(in_channels, out_channels, bias, stride, kernel_size, padding)
            parent_name, name_ = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name_, new_module)

    return graph, {}
