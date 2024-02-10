import itertools
import json
import time
import torch
import logging

from chop.actions.search.search_space.base import SearchSpaceBase
from chop.actions.search.search_space.utils import flatten_dict
from ..base import SearchStrategyBase

logger = logging.getLogger(__name__)


class SearchStrategyBruteForce(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        if not self.sum_scaled_metrics:
            self.directions = [
                self.config["metrics"][k]["direction"] for k in self.metric_names
            ]
        else:
            self.direction = self.config["setup"]["direction"]

    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool):
        # note that model can be mase_graph or nn.Module
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.sw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.sw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics

    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool):
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.hw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.hw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics

    def search(self, search_space: SearchSpaceBase):
        all_configs = []
        
        sampled_indexes = {}
        all_config_keys = list(search_space.choices_flattened.keys())
        all_config_key_lengths = list(search_space.choice_lengths_flattened.values())
        
        config_keys = [all_config_keys[i] for i in range(0, len(all_config_keys)) if all_config_key_lengths[i] > 0]
        config_key_indices = [list(range(0, all_config_key_lengths[i])) for i in range(0, len(all_config_keys)) if all_config_key_lengths[i] > 0]

        combinations = list(itertools.product(*config_key_indices))
        for combination in combinations:
            sampled_indexes = dict(zip(config_keys, combination))
            sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)
            all_configs.append(sampled_config)

        is_eval_mode = self.config.get("eval_mode", True)
        
        results = {}
        start_time = time.time()
        for i, config in enumerate(all_configs):
            if i >= self.config["setup"]["n_trials"]:
                logger.info(f"Reached maximum number of trials: {i}")
                break

            if time.time() - start_time > self.config["setup"]["timeout"]:
                logger.info(f"Reached maximum time: {time.time() - start_time}")
                break

            model = search_space.rebuild_model(config, is_eval_mode)

            software_metrics = self.compute_software_metrics(
                model, config, is_eval_mode
            )
            hardware_metrics = self.compute_hardware_metrics(
                model, config, is_eval_mode
            )
            metrics = software_metrics | hardware_metrics
            scaled_metrics = {}
            for metric_name in self.metric_names:
                scaled_metrics[metric_name] = (
                    self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
                )

            self.visualizer.log_metrics(metrics=metrics, step=i)

            flat_config = {}
            flatten_dict(config, flattened=flat_config)
            results[i] = {
                "config": flat_config,
                "metrics": metrics,
                "scaled_metrics": scaled_metrics
            }

        if self.save_dir:
            with open(f"{self.save_dir}/results.json", "w") as f:
                json.dump(results, f, indent=4)
        