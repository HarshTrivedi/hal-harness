import os
from collections import defaultdict
from typing import Any, Dict, List

from .base_benchmark import BaseBenchmark


class AppWorldBenchmark(BaseBenchmark):
    """AppWorld benchmark implementation"""

    def __init__(
        self, agent_dir: str, config: Dict[str, Any], benchmark_name: str = "appworld_test_normal"
    ):
        self.benchmark_name = benchmark_name
        valid_benchmark_names = [
            "appworld_test_normal",
            "appworld_test_challenge",
            "appworld_dev",
            "appworld_train",
        ]
        if benchmark_name not in valid_benchmark_names:
            raise ValueError(
                f"Invalid benchmark name {benchmark_name}. Use one of {valid_benchmark_names}."
            )
        self.split = benchmark_name.removeprefix("appworld_")
        self.setup_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "appworld", "setup.sh"
        )
        self.requires_sandbox = False
        super().__init__(
            agent_dir,
            config,
            requires_sandbox=self.requires_sandbox,
            setup_script=self.setup_script,
        )
        # Load dataset splits
        self.splits = {
            "test_normal": self._read_task_ids("test_normal.txt"),
            "test_challenge": self._read_task_ids("test_challenge.txt"),
            "dev": self._read_task_ids("dev.txt"),
            "train": self._read_task_ids("train.txt"),
        }
        # Create benchmark dictionary
        self.benchmark = {}
        benchmark_directory = os.path.join(os.getcwd(), "hal", "benchmarks", "appworld")
        data_directory = os.path.join(benchmark_directory, "data")
        if not os.path.exists(data_directory):
            raise FileNotFoundError(
                f"Data directory {data_directory} does not exist. "
                "Please download AppWorld data: "
                f"`pip install appworld && appworld download data --root {benchmark_directory}`"
            )
        for task_id in self.splits[self.split]:
            self.benchmark[task_id] = {
                "task_id": task_id,
                "files": {"data": data_directory},
            }

    def _read_task_ids(self, filename: str) -> List[str]:
        """Read task IDs from a given file"""
        file_path = os.path.join(os.getcwd(), "hal", "benchmarks", "appworld", filename)
        with open(file_path) as file:
            return [line.strip() for line in file.readlines()]

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        # NOTE: world.evaluate().to_dict() expected to to be called in the agent code itself.
        return agent_output

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics from evaluation results.

        Args:
            eval_results: Dictionary containing evaluation results

        Returns:
            Dictionary with calculated metrics and task lists
        """
        successful_task_ids = []
        failed_task_ids = []
        scenario_id_to_successes = defaultdict(list)
        for task_id, result in eval_results.items():
            scenario_id = task_id.split("_")[0]
            success = result["success"]
            scenario_id_to_successes[scenario_id].append(success)
            if success:
                successful_task_ids.append(task_id)
            else:
                failed_task_ids.append(task_id)
        scenario_id_to_success = {
            scenario_id: all(successes)
            for scenario_id, successes in scenario_id_to_successes.items()
        }
        scenario_goal_completion = (
            sum(scenario_id_to_success.values()) / len(scenario_id_to_success)
            if scenario_id_to_success
            else 0.0
        )
        task_goal_completion = len(successful_task_ids) / len(eval_results) if eval_results else 0.0
        metrics = {
            "accuracy": task_goal_completion,
            "task_goal_completion": task_goal_completion,
            "scenario_goal_completion": scenario_goal_completion,
            "successful_tasks": successful_task_ids,
            "failed_tasks": failed_task_ids,
        }
        return metrics

    def mount_benchmark(self):
        """Mount AppWorld benchmark environment - not needed since we use pip install"""
        pass
