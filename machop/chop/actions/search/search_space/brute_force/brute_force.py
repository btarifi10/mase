from copy import deepcopy
import itertools
from pprint import pprint
from chop.actions.search.search_space.base import SearchSpaceBase
from chop.actions.search.search_space.utils import flatten_dict, unflatten_dict
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.analysis.add_metadata.add_common_metadata import add_common_metadata_analysis_pass
from chop.passes.graph.analysis.init_metadata import init_metadata_analysis_pass
from chop.passes.graph.transforms.quantize.quantize import QUANTIZEABLE_OP
from chop.passes.graph.utils import get_mase_op, get_mase_type


class BruteForceSpace(SearchSpaceBase):
    def _post_init_setup(self):
        pass

    def build_search_space(self):
        # Build a mapping from node name to mase_type and mase_op.
        mase_graph = MaseGraph(self.model)

        mase_graph, _ = init_metadata_analysis_pass(mase_graph, None)
        mase_graph, _ = add_common_metadata_analysis_pass(mase_graph, {"dummy_in": self.dummy_input})
        node_info = {}
        for node in mase_graph.fx_graph.nodes:
            node_info[node.name] = {
                "mase_type": get_mase_type(node),
                "mase_op": get_mase_op(node),
            }
        choices = []
        seed = self.config["seed"]
        print("seed:", seed)

        base_pass_args = {}
        match self.config["setup"]["by"]:
            case "type":
                base_pass_args["by"] = "type"
            case "name":
                # base_pass_args["by"] = "name"
                raise ValueError(
                    f"Brute force only supported by 'type' for now."
                )
            case _:
                raise ValueError(
                    f"Unknown quantization by: {self.config['setup']['by']}"
                )
        
        base_pass_args["default"] = seed["default"]
        base_pass_args["default"]["config"]["name"] = base_pass_args["default"]["config"]["name"][0]

        n_op_choices = {}
        for n_op in QUANTIZEABLE_OP:
            if n_op in seed:
                n_op_choices[n_op] = []
                pass_args = deepcopy(base_pass_args)
                pass_args[n_op] = {
                    "config": {
                        "name": seed[n_op]['config']['name'][0],
                    }
                }

                selected_quantizations = []
                quantization_options = []
                data_in_widths = seed[n_op]['config'].get("data_in_width", None)
                data_in_frac_widths = seed[n_op]['config'].get("data_in_frac_width", None)
                if data_in_frac_widths:
                    if not data_in_frac_widths[0] or not len(data_in_frac_widths) == len(data_in_widths):
                        data_in_frac_widths = [d // 2 for d in data_in_widths]

                if data_in_widths and data_in_frac_widths:
                    quantization_options.append(zip(data_in_widths, data_in_frac_widths))
                    selected_quantizations.append(("data_in_width", "data_in_frac_width"))
                elif data_in_widths:
                    quantization_options.append((data_in_widths,))
                    selected_quantizations.append(("data_in_width",))

                weight_widths = seed[n_op]['config'].get("weight_width", None)
                weight_frac_widths = seed[n_op]['config'].get("weight_frac_width", None)
                if weight_frac_widths:
                    if not weight_frac_widths[0] or not len(weight_frac_widths) == len(weight_widths):
                        weight_frac_widths = [w // 2 for w in weight_widths]

                if weight_widths and weight_frac_widths:
                    quantization_options.append(zip(weight_widths, weight_frac_widths))
                    selected_quantizations.append(("weight_width", "weight_frac_width"))
                elif weight_widths:
                    quantization_options.append((weight_widths,))
                    selected_quantizations.append(("weight_width",))

                bias_widths = seed[n_op]['config'].get("bias_width", None)
                bias_frac_widths = seed[n_op]['config'].get("bias_frac_width", None)
                if bias_frac_widths:
                    if not bias_frac_widths[0] or not len(bias_frac_widths) == len(bias_widths):
                        bias_frac_widths = [b // 2 for b in bias_widths]
                
                if bias_widths and bias_frac_widths:
                    quantization_options.append(zip(bias_widths, bias_frac_widths))
                    selected_quantizations.append(("bias_width", "bias_frac_width"))
                elif bias_widths:
                    quantization_options.append((bias_widths,))
                    selected_quantizations.append(("bias_width",))

                width_combinations = list(itertools.product(*quantization_options))

                for comb in width_combinations:
                    n_op_pass_args = deepcopy(pass_args)
                    for i, q in enumerate(selected_quantizations):
                        n_op_pass_args[n_op]["config"][q[0]] = comb[i][0]
                        if len(q) > 1:
                            n_op_pass_args[n_op]["config"][q[1]] = comb[i][1]
                    n_op_choices[n_op].append(n_op_pass_args)

        ops_combos = list(itertools.product(*n_op_choices.values()))

        for ops in ops_combos:
            pass_args_choices = deepcopy(base_pass_args)
            for i, n_op in enumerate(n_op_choices.keys()):
                pass_args_choices[n_op] = ops[i][n_op]
            choices.append(pass_args_choices)

        # match self.config["setup"]["by"]:
        #     case "name":
        #         # iterate through all the quantizeable nodes in the graph
        #         # if the node_name is in the seed, use the node seed search space
        #         # else use the default search space for the node
        #         for n_name, n_info in node_info.items():
        #             if n_info["mase_op"] in QUANTIZEABLE_OP:
        #                 if n_name in seed:
        #                     choices[n_name] = deepcopy(seed[n_name])
        #                 else:
        #                     choices[n_name] = deepcopy(seed["default"])
        #     case "type":
        #         # iterate through all the quantizeable nodes in the graph
        #         # if the node mase_op is in the seed, use the node seed search space
        #         # else use the default search space for the node
        #         for n_name, n_info in node_info.items():
        #             n_op = n_info["mase_op"]
        #             if n_op in QUANTIZEABLE_OP:
        #                 if n_op in seed:
        #                     choices[n_name] = deepcopy(seed[n_op])
        #                 else:
        #                     choices[n_name] = deepcopy(seed["default"])
        #     case _:
        #         raise ValueError(
        #             f"Unknown quantization by: {self.config['setup']['by']}"
        #         )
            

        self.choices_flattened = choices
        
    def flattened_indexes_to_config(self, indexes):
        return None

    def rebuild_model(self, config, mode="eval"):
        return self.model