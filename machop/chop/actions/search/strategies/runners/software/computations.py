from chop.actions.search.strategies.runners.software.base import SWRunnerBase
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.analysis.flops.flops_pass import analyse_flops_pass


class RunnerBasicComputations(SWRunnerBase):
    available_metrics = ("total_flops", "total_bitops")

    def __init__(self, model_info, task: str, dataset_info, accelerator, config: dict = None):
        super().__init__(model_info, task, dataset_info, accelerator, config)

    def _post_init_setup(self) -> None:
        pass

    def __call__(self, data_module, model: MaseGraph, sampled_config) -> dict[str, float]:
        assert isinstance(model, MaseGraph)

        model, _ = analyse_flops_pass(model, silent=True)

        computation_data = [
            node.meta['mase'].parameters['software']['computations'] for node in model.nodes
        ]
        total_flops = sum([data.get('computations', 0) + data.get('backward_computations', 0) for data in computation_data if data])
        total_bitops = sum([data.get('forward_bitops', 0) + data.get('backward_bitops', 0) for data in computation_data if data])

        return {
            "total_flops": total_flops,
            "total_bitops": total_bitops
        } 