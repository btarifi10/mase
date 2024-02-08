
import json
import pprint
import random
import time
from numpy import mean

import torch
from chop.actions.search.search_space.brute_force.brute_force import BruteForceSpace
from chop.actions.search.strategies.base import SearchStrategyBase
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.analysis.add_metadata.add_common_metadata import add_common_metadata_analysis_pass
from chop.passes.graph.analysis.add_metadata.add_software_metadata import add_software_metadata_analysis_pass
from chop.passes.graph.analysis.init_metadata import init_metadata_analysis_pass
from chop.passes.graph.transforms.quantize.quantize import quantize_transform_pass
from chop.passes.graph.analysis.flops.flops_pass import analyse_flops_pass
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

class BruteForceStrategy(SearchStrategyBase):
    """
    Brute force search strategy.
    """

    is_iterative: bool = False

    def _post_init_setup(self):
        pass

    def search(self, search_space: BruteForceSpace):
        # sample the search_space.choice_lengths_flattened to get the indexes
        # indexes = {}
        # for k, v in search_space.choice_lengths_flattened.items():
        #     indexes[k] = random.randint(0, v - 1)

        # call search_space.rebuild_model to build a new model with the sampled indexes
        # sampled_config = search_space.flattened_indexes_to_config(indexes)
        
        choices_flattened = search_space.choices_flattened
        
        model = search_space.rebuild_model(None, None)

        mg = MaseGraph(model)
        
        mg, _ = init_metadata_analysis_pass(mg, None)
        mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": search_space.dummy_input})
        mg, _ = add_software_metadata_analysis_pass(mg, None)

        accuracy_metric = MulticlassAccuracy(num_classes=5)
        precision_metric = MulticlassPrecision(num_classes=5)
        recall_metric = MulticlassRecall(num_classes=5)
        f1_metric = MulticlassF1Score(num_classes=5)
        num_batchs = 5
        # This first loop is basically our search strategy,
        # in this case, it is a simple brute force search

        recorded_accs = []
        recorded_latencies = []
        recorded_precisions = []
        recorded_recalls = []
        recorded_f1s = []
        total_flops = []
        total_bitops = []

        print("Num of configs to try: ", len(choices_flattened))

        for i, config in enumerate(choices_flattened):
            mg, _ = quantize_transform_pass(mg, config)

            mg = analyse_flops_pass(mg, silent=True)
            j = 0

            # this is the inner loop, where we also call it as a runner.
            accs, losses = [], []
            latencies = []
            f1s, precisions, recalls = [], [], []

            for inputs in self.data_module.train_dataloader():
                start = time.time()
                xs, ys = inputs
                preds = mg.model(xs)
                loss = torch.nn.functional.cross_entropy(preds, ys)
                latencies.append(time.time() - start)
                acc = accuracy_metric(preds, ys)
                precision = precision_metric(preds, ys)
                recall = recall_metric(preds, ys)
                f1 = f1_metric(preds, ys)

                accs.append(acc)
                losses.append(loss)
                f1s.append(f1)
                precisions.append(precision)
                recalls.append(recall)

                if j > num_batchs:
                    break
                j += 1

            flops = 0
            bitops = 0
            for node in mg.nodes:
                flops_data = node.meta['mase'].parameters['software']['computations']
                if flops_data:
                    flops += flops_data['computations']
                    flops += flops_data['backward_computations']
                    bitops += flops_data['forward_bitops']
                    bitops += flops_data['backward_bitops']

            acc_avg = mean(accs)
            precision_avg = mean(precisions)
            recall_avg = mean(recalls)
            f1_avg = mean(f1s)
            latency_avg = mean(latencies)

            recorded_accs.append(acc_avg.item())
            recorded_latencies.append(latency_avg.item())
            recorded_precisions.append(precision_avg.item())
            recorded_recalls.append(recall_avg.item())
            recorded_f1s.append(f1_avg.item())

            total_flops.append(flops)
            total_bitops.append(bitops)

        results = []
        for i, config in enumerate(choices_flattened):
            result = {
                "config": config,
                "accuracy": recorded_accs[i],
                "latency": recorded_latencies[i],
                "precision": recorded_precisions[i],
                "recall": recorded_recalls[i],
                "f1": recorded_f1s[i],
                "flops": total_flops[i],
                "bitops": total_bitops[i]
            }
            results.append(result)

        if self.save_dir:
            with open(f"{self.save_dir}/results.json", "w") as f:
                json.dump(results, f, indent=4)

        # save the results
        # self.save_results(
        #     search_space,
        #     sampled_config,
        #     indexes,
        #     sw_metrics,
        #     hw_metrics,
        #     is_eval_mode=True,
        # )
        # return sw_metrics, None