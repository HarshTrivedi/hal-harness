import os

from appworld import cli
from appworld.common.path_store import path_store
from appworld.common.utils import write_file
from appworld.evaluator import evaluate_dataset
from appworld_experiments.configs._generator.run import make_experiment_config


current_directory = os.path.dirname(os.path.abspath(__file__))
path_store.update_root(current_directory)


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    # assert required keys
    required_keys = ["model_name", "model_file_name", "method_name"]
    for required_key in required_keys:
        assert required_key in kwargs, f"{required_key} is required"
    actual_dataset_name = kwargs["benchmark_name"].removeprefix("appworld_")
    # create a dataset file with only passed task
    task_ids = list(input.keys())
    dataset_name = "sample"
    dataset_file_path = os.path.join(current_directory, "data", "datasets", f"{dataset_name}.txt")
    write_file("\n".join(task_ids), dataset_file_path)
    # create a sample config file for the experiment
    model_file_name = kwargs["model_file_name"]
    agent_name = kwargs["method_name"]
    output_ = make_experiment_config(
        model_name=model_file_name,
        agent_name=agent_name,
        dataset_name=actual_dataset_name,
        save=False,
    )
    reference_experiment_config = output_["config"]
    actual_experiment_config = reference_experiment_config.replace(actual_dataset_name, dataset_name)
    actual_experiment_name = "output"
    actual_experiment_config_file_path = os.path.join(
        path_store.experiment_configs, f"{actual_experiment_name}.jsonnet"
    )
    write_file(actual_experiment_config, actual_experiment_config_file_path)
    # run the experiment
    cli.run(
        experiment_name=actual_experiment_name,
        task_id=None,
        override=None,
        num_processes=1,
        process_index=None,
        root=current_directory,
    )
    # evaluate the results
    evaluation = evaluate_dataset(
        experiment_name=actual_experiment_name,
        dataset_name=dataset_name,
        suppress_errors=True,
        include_details=True,
    )
    return {task_ids[0]: "Completed", "evaluation": evaluation}
