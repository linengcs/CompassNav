#!/bin/bash

export PYTHONUNBUFFERED=1

MODEL_PATH=""  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./datasets/InstanceImageGoalDataset_problem_replaced@train \
    data.val_files=./datasets/InstanceImageGoalDataset_problem_replaced@test \
    data.format_prompt=./examples/format_prompt/nav.jinja \
    data.max_prompt_length=7000 \
    data.max_response_length=2024 \
    data.max_pixels=5000000 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/nav.py:compute_score \
    trainer.experiment_name=compassNavRL \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \