import logging
from math import ceil
from os import PathLike

from tqdm import tqdm
from chop.actions.search.strategies.runners.software import get_sw_runner
from chop.dataset import MaseDataModule, get_dataset_info
from chop.ir.graph.mase_graph import MaseGraph
from chop.models import get_model_info
from chop.passes.graph.analysis.add_metadata.add_common_metadata import add_common_metadata_analysis_pass
from chop.passes.graph.analysis.add_metadata.add_software_metadata import add_software_metadata_analysis_pass
from chop.passes.graph.analysis.init_metadata import init_metadata_analysis_pass
from chop.passes.graph.utils import get_mase_op
import copy, torch
from chop.actions import test, train
from chop.passes.graph.transforms import (
    prune_transform_pass,
)
from chop.tools.checkpoint_load import load_model
from chop.tools.config_load import load_config
from chop.tools.utils import parse_accelerator


logger = logging.getLogger(__name__)


def parse_prune_config(prune_config):
    """
    Parse prune config from a dict or a toml file and do sanity check.

    ---
    The prune config must consist of two parts: prune and train.
    """
    if not isinstance(prune_config, dict):
        prune_config = load_config(prune_config)
    prune_config = prune_config["prune"]  # the actual config for action search
    pruning_config = prune_config["iterative_prune"]
    training_config = prune_config["train"]

    return pruning_config, training_config

def prune_iterative(
    model: torch.nn.Module,
    model_info: dict,
    task: str,
    dataset_info: dict,
    data_module: MaseDataModule,
    prune_config: dict | PathLike,
    save_path: PathLike = None,
    accelerator: str = "auto",
    load_name: PathLike = None,
    load_type: str = None,
    visualizer=None,
):
    """
    Args:
        model: the model to be pruned
    """

    pruning_config, training_config = parse_prune_config(prune_config)
    accelerator = parse_accelerator(accelerator)

    # load model if the save_name is provided
    if load_name is not None and load_type in ["pl", "mz", "pt"]:
        model = load_model(load_name=load_name, load_type=load_type, model=model)
        logger.info(f"Loaded model from {load_name}.")
    elif load_name is not None and load_type == "pruned":
        # model = load_model(load_name=load_name, load_type="pt", model=model)
        logger.info(f"Loaded pruned model from {load_name}.")

    model.to(accelerator)
    
    overall_sparsity = pruning_config["sparsity"]
    num_iterations = pruning_config["num_iterations"]

    iteration_sparsity = 1 - ((1 - overall_sparsity)**(1 / num_iterations))

    data_module.prepare_data()
    data_module.setup()
    dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

    mg = MaseGraph(model)
    mg, _ = init_metadata_analysis_pass(mg, dummy_in)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in, "force_device_meta": False})
    mg, _ = add_software_metadata_analysis_pass(mg, None)

    max_epochs = training_config["max_epochs"]
    epochs_per_iteration = ceil(max_epochs / num_iterations)
    training_config["max_epochs"] = epochs_per_iteration

    train_runner = get_sw_runner(
        "basic_train",
        model_info,
        task,
        dataset_info,
        accelerator,
        training_config,
    )

    prune_args = {
        "weight": {
            "scope": pruning_config["scope"],
            "granularity": pruning_config["granularity"],
            "method": pruning_config["method"],
            "sparsity": iteration_sparsity,
        },
        "activation": {
            "scope": pruning_config["scope"],
            "granularity": pruning_config["granularity"],
            "method": pruning_config["method"],
            "sparsity": iteration_sparsity,
        },
    }

    original_w_b = {}

    for node in mg.fx_graph.nodes:
        if get_mase_op(node) in ["linear", "conv2d", "conv1d"]:
            original_w_b[node.name] = {
                "weight": mg.modules[node.target].weight,
                "bias": mg.modules[node.target].bias,
                "meta_weight": node.meta["mase"].parameters["common"]["args"]["weight"]["value"],
                "meta_bias": node.meta["mase"].parameters["common"]["args"]["bias"]["value"],
            }

    train_metrics = []

    num_nodes = len([node for node in mg.fx_graph.nodes if get_mase_op(node) in ["linear", "conv2d", "conv1d"]])

    with tqdm(total=num_iterations*num_nodes) as pbar:
        pbar.set_description(f"Pruning model {num_iterations} times")
        
        for i in tqdm(range(num_iterations)):
            results = train_runner(data_module, model, None)
            train_metrics.append(results)

            logger.info("")
 
            mg, _ = prune_transform_pass(mg, prune_args)

            # copy the weights from the original model to the pruned model
            for node in tqdm(mg.fx_graph.nodes):
                if get_mase_op(node) in ["linear", "conv2d", "conv1d"]:
                    with torch.no_grad():
                        mg.modules[node.target].weight.copy_(original_w_b[node.name]['weight'])
                        # mg.modules[node.target].weight.copy_(original_w_b[node.name]['weight'])

                        mg.modules[node.target].bias.copy_(original_w_b[node.name]['bias'])

                        # update the mase metadata weights
                        node.meta["mase"].parameters["common"]["args"]["weight"]["value"] = original_w_b[node.name]['meta_weight']
                        
                        node.meta["mase"].parameters["common"]["args"]["bias"]["value"] = original_w_b[node.name]['meta_bias']

        results = train_runner(data_module, model, None)
        train_metrics.append(results)
        # test(**train_test_args)

        return model, train_metrics