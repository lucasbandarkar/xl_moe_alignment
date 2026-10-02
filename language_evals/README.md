# Evaluation Launcher

This folder now has a main bash entrypoint:

- `run_eval_only.sh` for evaluation only

Examples:

```bash
./run_eval_only.sh -m microsoft/Phi-tiny-MoE-instruct -l si -g 1
./run_eval_only.sh -m ../contrastive_training/checkpoints/moe_contrastive_training_test/checkpoint-2500/ -l si -g 0
```

Common flags:

- `-m, --model-name`
- `-l, --language`
- `-g, --gpus`
- `-d, --dataset-name` for training
- `-a, --adapter-path` for eval-only

The `-g/--gpus` flag should be a comma-separated list like `0`, `0,1`, or `0,1,2,3`.

## Eval runtime defaults

`run_eval_only.sh` chooses different defaults based on the server:

- On systems with `/data2`, it uses the currently active `python`/conda environment and skips `apply_vllm_phimoe_patch` for Phi-tiny.
- On other systems, it uses `uv run python` and applying
  `apply_vllm_phimoe_patch`.

You can override either default explicitly:

```bash
EVAL_RUNNER=uv APPLY_VLLM_PHIMOE_PATCH=1 ./run_eval_only.sh -m microsoft/Phi-tiny-MoE-instruct -l si -g 0
EVAL_RUNNER=python APPLY_VLLM_PHIMOE_PATCH=0 ./run_eval_only.sh -m microsoft/Phi-tiny-MoE-instruct -l si -g 0
```

## Intermediate checkpoints

When `run_eval_only.sh` evaluates a path ending in `checkpoint-N`, it writes into the
parent run's eval directory and suffixes the summary with the approximate number of
training samples seen. For example, evaluating `.../<run>_200k/checkpoint-5000` in a
run whose last checkpoint is `checkpoint-6250` writes `summary_160k.json` under
`results/eval-<run>_200k-<lang>/`.

## Create environment

Confusingly, different models may require different environments based on transformers, CUDA, pytorch packages at the time of their release.

This env was the main environment used for all models on NVIDIA CUDA remote servers:

```bash
uv python install 3.12.11
uv venv ~/.venvs/moevllm --python 3.12.11
source ~/.venvs/moevllm/bin/activate

uv pip install torch==2.10.0 torchvision torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install transformers==4.57.6
uv pip install vllm==0.19.0
uv pip install datasets==3.6.0 lm-eval==0.4.10 hf_transfer==0.1.9 peft==0.16.0 ray
```

For Qwen3.5, I reused the environment at the bottom of `routing_analysis/README.md`, which i named `qwen35`

## Evaluating a new language

See `language_to_task.py` for instructions on how to add another language.

## Code Organization

- `README.md` documents setup and evaluation commands.
- `run_eval_only.sh` launches an evaluation with the selected model, language, task, and GPUs.
- `launch.sh` runs a configured sequence of evaluations and logs their output.
- `run_eval.py` loads models, runs the selected tasks, and writes results.
- `task_evaluators.py` implements scoring for the supported multilingual benchmarks.
- `language_to_task.py` maps languages to tasks and benchmark-specific names.
- `export_fsdp_checkpoint.py` converts FSDP checkpoints for vLLM evaluation.
- `eval_output_paths.py` names output files for full runs and intermediate checkpoints.
- `vllm_phimoe_patch.py` patches vLLM to load Phi-tiny-MoE models.
- `pyproject.toml` declares the evaluation project's Python dependencies for users on AWS EC2 instances.
- `task_utils/flores.yaml` configures the FLORES translation task.
- `task_utils/global_mgsm.yaml` configures the Global MGSM math task.
- `task_utils/gmmlu_medical_samples_dict.json` lists sampled questions for medical MMLU evaluation.
- `task_utils/multiloko_utils.py` defines language-specific Multiloko prompts.
