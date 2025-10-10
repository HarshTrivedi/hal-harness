import os

from appworld import cli, update_root
from appworld.common.io import write_file
from appworld.common.path_store import path_store
from appworld.common.random import get_unique_id
from appworld.evaluator import evaluate_dataset
from appworld_experiments.configs._generator.run import make_experiment_config


current_directory = os.path.dirname(os.path.abspath(__file__))
update_root(current_directory)


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    # assert required keys
    required_keys = ["model_name", "model_file_name", "method_name"]
    for required_key in required_keys:
        assert required_key in kwargs, f"{required_key} is required"
    # create a dataset file with only passed task
    task_ids = list(input.keys())
    task_id = task_ids[0]  # has only 1 task_id in practice
    dataset_file_path = os.path.join(current_directory, "data", "datasets", f"{task_id}.txt")
    write_file(task_id, dataset_file_path)
    # create a sample config file for the experiment
    model_file_name = kwargs["model_file_name"]
    agent_name = kwargs["method_name"]
    output_ = make_experiment_config(
        model_name=model_file_name,
        agent_name=agent_name,
        dataset_name=task_id,
        save=False,
    )
    experiment_config = output_["config"]
    unique_id = get_unique_id()
    experiment_config_file_path = os.path.join(
        path_store.experiment_configs, f"{unique_id}_{task_id}.jsonnet"
    )
    write_file(experiment_config, experiment_config_file_path)
    # run the experiment
    cli.run(
        experiment_name=task_id,
        task_id=None,
        override=None,
        num_processes=1,
        process_index=None,
        root=current_directory,
    )
    # evaluate the results
    evaluation = evaluate_dataset(
        experiment_name=task_id,
        dataset_name=task_id,
        suppress_errors=True,
        include_details=True,
    )
    # clean up
    if os.path.exists(experiment_config_file_path):
        os.remove(experiment_config_file_path)
    return evaluation["individual"]
