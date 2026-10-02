The name of this folder "contrastive_training" is an outdated naming convention (since our published method doesn't include negatives in the loss). The main method and classes (most notably `contrastive_trainer.py` were written in June 2025).

## Quick start

Use case examples: `CUDA_VISIBLE_DEVICES="7" accelerate launch --config_file accelerate_config_1gpu.yaml train.py -l pes -t`

Additionally, can launch `accelerate_config_4gpu.yaml` for multi-gpu (2,3,or 4) with additional `--num_processes X` flag

There are many training flags, all documented at the bottom of the main python entry file, `train.py`.

## Environment details

These trainings were run on A100 or H100 NVIDIA CUDA servers using `accelerate` for multi-GPU training. The `accelerate` configs are documented in this folder.

```
conda create -n xlcl python=3.14
conda activate xlcl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r xlcl_requirements.txt
pip install causal-conv1d==1.6.1 mamba-ssm==2.3.1 --no-build-isolation
pip install liger-kernel==0.8.0
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
pip install --upgrade-strategy only-if-needed torchtitan==0.2.2 ## doesn't work with any models, so doing nothing for now
```

For H100s, the 12.6 cuda and nvcc toolkit required me to start over:
```
conda create -n xlcl python=3.14 uv cuda-toolkit=12.6.0 -c nvidia -c conda-forge -y
conda activate xlcl
uv pip install torch==2.10.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -r xlcl_requirements.txt
uv pip install causal-conv1d==1.6.1 mamba-ssm==2.3.1 --no-build-isolation
uv pip install liger-kernel==0.8.0
MAX_JOBS=4 uv pip install flash-attn==2.8.3 --no-build-isolation
pip install --upgrade-strategy only-if-needed torchtitan==0.2.2 ## doesn't work with any models, so doing nothing for now
```

## Code Organization

- `README.md` documents setup, training commands, and the files in this folder.
- `launch.sh` is the intended entrypoint for launching training runs.
- `train.py` configures and runs contrastive, language modeling, and translation training.
- `contrastive_trainer.py` implements the training losses and trainer classes.
- `modeling.py` loads MoE models and supports partial forward passes.
- `packed_forward.py` computes packed sequence forwards for contrastive training.
- `parallel_dataset.py` loads parallel data and builds training batch collators.
- `dataset_helpers.py` defines parallel data sources and formats their examples.
- `build_filtered_parallel_dataset.py` filters and saves parallel examples by token length.
- `optimizations.py` applies optional model loading and training optimizations.
- `training_configs.json` sets model-specific training batch sizes.
- `accelerate_config_1gpu.yaml` configures single-GPU Accelerate training.
- `accelerate_config_4gpu.yaml` configures multi-GPU FSDP training.
- `accelerate_config.txt` records an Accelerate configuration walkthrough.
- `accelerate_config_fsdp.txt` records an FSDP Accelerate configuration walkthrough.
- `packing_details.txt` documents packed training behavior and performance measurements.
- `xlcl_requirements.txt` lists Python dependencies for training.
