<div align="center">
<h3>CompassNav: Steering from Path Imitation to Decision Understanding in Navigation</h3>

LinFeng Li `<sup>`1,2`</sup>`&nbsp;[Jian Zhao](https://scholar.google.com.sg/citations?hl=en&user=zdhRJCkAAAAJ&view_op=list_works&gmla=AJsN-F4PURIx5GMQHVpprJJBjTsNC62YCHjxGsKOwVhrkZ1aJsLgBiuKPBbAgbdcE5_KNw3OnLQgOVSjlqmS6gc-6ti0M2K5o-klHgoOywFCbdaaGnpis130zvgoZFJkVfmoNKpo8Krp) `<sup>`2`</sup>`&nbsp;[Yuan Xie](https://scholar.google.com/citations?user=RN1QMPgAAAAJ&hl=en) `<sup>`1`</sup>`&nbsp;[Xin Tan](https://scholar.google.com/citations?hl=zh-CN&user=UY4NCdcAAAAJ&view_op=list_works) `<sup>`1`</sup>`&nbsp;[Xuelong Li](https://scholar.google.com/citations?user=ahUibskAAAAJ&hl=en) `<sup>`2`</sup>`&nbsp;

`<sup>`1 `</sup>`East China Normal University; `<sup>`2 `</sup>`The Institute of Artificial Intelligence (TeleAI), China Telecom&nbsp;

[![ArXiv](https://img.shields.io/badge/ArXiv-<2510.10154>-<COLOR>.svg)](https://arxiv.org/abs/2510.10154) [![Webpage](https://img.shields.io/badge/Webpage-CompassNav-<COLOR>.svg)](https://linengcs.github.io/CompassNav/)[![Dataset](https://img.shields.io/badge/Dataset-CompassData-<COLOR>.svg)](https://huggingface.co/datasets/Lineng/CompassData)[![Model](https://img.shields.io/badge/Model-CompassNav7B-<COLOR>.svg)](https://huggingface.co/datasets/Lineng/CompassData)

<p align="center">
  <img src="assets/crop_compass.svg" width="600">>
</p>

</div>

The top panel contrasts our End-to-End Goal Navigation paradigm with traditional approaches. Unlike Vision-Language Navigation (VLN), which relies on dense, step-by-step instructions, and complex Modular Navigation pipelines, CompassNav directly maps a high-level goal (e.g., "find the plant") to an action through integrated spatial logical reasoning.

The bottom panel details our core contribution-how to stimulate model reasoning ability: a paradigm shift from "Path Imitation" to "Decision Understanding." While traditional methods train agents to replicate a single expert trajectory and penalize any deviation, our agent learns to evaluate the relative quality of all feasible paths at each decision point. This approach cultivates a true "internal compass," enabling the agent to make more intelligent and flexible decisions in unseen environments.

# CompassNav

Official code release for the paper **CompassNav**.

This repository contains two main components:

- `train/`: training code for CompassNav-style policy learning, including SFT and RL stages.
- `Navigation/`: evaluation code for embodied ObjectNav inference (built on top of [VLMnav](https://github.com/Jirl-upenn/VLMnav)).

Repository URL: https://github.com/linengcs/CompassNav

## Project Structure

```text
CompassNav/
├── train/                     # Training code
│   ├── train.sh               # One-command training launcher
│   ├── examples/              # Prompt + reward examples
│   ├── verl/                  # Internal training runtime
│   ├── scripts/model_merger.py
│   ├── Dockerfile             # Recommended training container
│   └── requirements.txt
├── Navigation/                # Navigation evaluation/inference
│   ├── config/ObjectNav.yaml  # Main eval config
│   ├── scripts/main.py        # Single-process entry
│   ├── scripts/aggregator.py  # Parallel logging aggregator
│   ├── parallel.sh            # Multi-instance launcher (tmux)
│   └── src/                   # Env, agent, simulator wrapper, VLM API
```

## 1) Training

The training pipeline in this repository contains two stages:

- `SFT`: based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- `RL`: with configuration referenced from [EasyR1](http://github.com/hiyouga/easyr1), which is a simplified version of veRL

The current recommended RL entry point here is:

- `train/train.sh`

### 1.1 Installation

For environment setup and dependency installation:

- For `SFT`, please follow the official [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) instructions.
- For `RL`, please follow the official [EasyR1](http://github.com/hiyouga/easyr1) instructions first, then return to this repository for task-specific configuration and launch.

This repository also provides `train/Dockerfile` if you prefer preparing the training environment with Docker.

### 1.2 Supervised Fine-Tuning (SFT)

The SFT part is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). The detailed CompassNav-specific SFT configuration is currently being organized and will be added in a future update.

### 1.3 Reinforcement Learning (RL) Configuration

Before launching training, update the following items:

- Set `MODEL_PATH` in `train/train.sh` to your local base model path.
- Prepare the config file referenced by the launcher: `train/examples/config.yaml`.
- Adjust `data.train_files` and `data.val_files` in `train/train.sh`.
- Keep or modify `train/examples/format_prompt/nav.jinja` as the prompt template.
- Keep or modify `train/examples/reward_function/nav.py` as the reward function.

### 1.4 RL Data Format (Key Fields)

The training dataset loader expects fields compatible with `train/verl/utils/dataset.py`.

Required/important keys:

- `prompt` (or custom `prompt_key`)
- `answer` (or custom `answer_key`)
- `action_to_goal_dis` (dict, used by reward function)
- optional `images` for VLM training samples

### 1.5 Launch RL Training

Use the bundled one-command launcher:

```bash
cd train
bash train.sh
```

Model merge utility after distributed checkpointing:

```bash
python scripts/model_merger.py --local_dir /path/to/checkpoint_dir
```

## 2) Navigation (Evaluation)

### 2.1 Environment Setup

For navigation environment setup, please refer to the official [VLMnav](https://github.com/Jirl-upenn/VLMnav) repository. Their installation process is based on a Conda environment with Habitat-Sim:

```bash
conda create -n vlm_nav python=3.9 cmake=3.14.0
conda activate vlm_nav
conda install habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat

cd Navigation
pip install -e .
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

Notes:

- The code is designed for Habitat-Sim-based ObjectNav evaluation.
- VLMnav also requires downloading the Habitat-Matterport 3D Research dataset and benchmark datasets such as ObjectNav/GOAT-Bench. Please follow the dataset instructions in the official [VLMnav](https://github.com/Jirl-upenn/VLMnav) repository.
- VLM inference endpoints are configured in `config/ObjectNav.yaml` (`base_url`, `model`).

### 2.2 Configure Model Endpoints

Edit `Navigation/config/ObjectNav.yaml`:

- `agent_cfg.vlm_cfg.model_kwargs.base_url`
- `agent_cfg.vlm_cfg.model_kwargs.model`
- `agent_cfg.vlm_cfg.stop_model_kwargs.base_url`
- `agent_cfg.vlm_cfg.stop_model_kwargs.model`

### 2.3 Run Evaluation

Single process:

```bash
cd Navigation
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python scripts/main.py --config ObjectNav --name your_run_name -ne 50 -ms 500
```

Parallel (multi-instance, tmux-based):

```bash
cd Navigation
bash parallel.sh
```

Before using `parallel.sh`, set your hardware/runtime values in the script (e.g., `NUM_GPU`, `INSTANCES`, `VENV_NAME`, `PORT`).

## 3) Acknowledgements

This codebase builds on/openly adapts components from:

- [VLMnav](https://github.com/Jirl-upenn/VLMnav) (navigation framework basis for `Navigation/`)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (SFT framework basis for `train/`)
- [EasyR1](http://github.com/hiyouga/easyr1) (RL configuration/reference basis for `train/`)

Please also cite these projects where appropriate.

## 4) Citation

If you find this code useful, please cite our paper:

```bibtex
@article{compassnav2026,
  title={CompassNav: [Paper Title]},
  author={[Authors]},
  journal={[Venue]},
  year={2026}
}
```

(Replace placeholders with final camera-ready metadata.)

## 5) License

The repository currently contains upstream components with their own licenses (e.g., `train/LICENSE`).

Please ensure your redistribution and usage follow the corresponding upstream license terms.

## TODO

- [X] Release CompassNav training code
- [X] Release CompassNav Object Goal Nav/Instance Image-Goal Nav test code
- [ ] Release Compass-Data-22k
- [ ] Release CompassNav-7B
