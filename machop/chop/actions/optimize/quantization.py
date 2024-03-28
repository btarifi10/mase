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
    quantize_transform_pass,
)
from chop.tools.checkpoint_load import load_model
from chop.tools.config_load import load_config
from chop.tools.utils import parse_accelerator


logger = logging.getLogger(__name__)

def parse_quantize_config(quantize_config):
    """
    Parse quantization config from a dict or a toml file and do sanity check.

    ---
    The quantization config must consist of two parts: quantization and train.
    """
    if not isinstance(quantize_config, dict):
        quantize_config = load_config(quantize_config)
    quantize_config = quantize_config["quantization"]  # the actual config for action search
    quantization_config = quantize_config["quantization_config"]
    training_config = quantize_config["train"]

    return quantization_config, training_config


def quantize_model(
    model: torch.nn.Module,
    model_info: dict,
    task: str,
    dataset_info: dict,
    data_module: MaseDataModule,
    quantize_config: dict | PathLike,
    save_path: PathLike = None,
    accelerator: str = "auto",
    load_name: PathLike = None,
    load_type: str = None,
    visualizer=None,
):
    pruning_config, training_config = parse_quantize_config(quantize_config)
    accelerator = parse_accelerator(accelerator)

    # load model if the save_name is provided
    if load_name is not None and load_type in ["pl", "mz", "pt"]:
        model = load_model(load_name=load_name, load_type=load_type, model=model)
        logger.info(f"Loaded model from {load_name}.")
    elif load_name is not None and load_type == "pruned":
        # model = load_model(load_name=load_name, load_type="pt", model=model)
        logger.info(f"Loaded pruned model from {load_name}.")

    model.to(accelerator)

    data_module.prepare_data()
    data_module.setup()
    dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

    mg = MaseGraph(model)
    mg, _ = init_metadata_analysis_pass(mg, dummy_in)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in, "force_device_meta": False})
    mg, _ = add_software_metadata_analysis_pass(mg, None)

    train_runner = get_sw_runner(
        "basic_train",
        model_info,
        task,
        dataset_info,
        accelerator,
        training_config,
    )

    mg, _ = quantize_transform_pass(mg, pruning_config)

    results = train_runner(data_module, mg.model, training_config)

    return model, mg, results