


This folder contains routing-analysis code (no evals, no training).

## Environment details

Confusingly, different models may require different environments based on transformers, CUDA, pytorch packages at the time of their release.

Instructions to recreate the `moe` env for these files:

1. `conda create --name moe python=3.13.4` - I named my env for finetuning "moe"
1. `conda activate moe`
1. `conda install -c conda-forge pip - install pip`
1. `conda install pytorch==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia` - install torch and related packages (and all its dependencies) using conda to ensure right CUDA compatibility. On H100, do pytorch==2.7.1 pytorch-cuda=12.6, on A100, do pytorch==2.5.1 pytorch-cuda=12.4. I haven't tried on A6000 servers.
1. `pip install -r requirements.txt` - use the requirements.txt I provided, but it may not be complete

Note: For models that are newer than transformers==4.52.4, you can either use the xlcl env detailed in `contrastive_training/README.md` or try:

For Qwen3.5 i have been using this:
```bash
conda create -n qwen35 python=3.12.12
conda activate qwen35
pip install torch==2.10.0 torchvision==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.57.6
pip install -r matplotlib pandas seaborn
pip install peft==0.16.0 uv==0.11.6 # for evals
uv pip install vllm==0.18.0 # for evals
pip install lm_eval==0.4.10 # for evals
```

## Code Organization

- `README.md` documents the routing analysis environment and files.
- `requirements.txt` lists Python packages used for routing analysis.
- `get_routing_weights.py` collects model router outputs for multilingual datasets.
- `run_routing_weights_for_checkpoints.sh` collects routing weights for configured checkpoints.
- `models.py` loads MoE models and extracts their router logits.
- `utils.py` has random utils for data.
- `analysis_helpers.py` computes routing entropy and similarity metrics.
- `expert_importance_eda.ipynb` explores expert importance and routing patterns across languages.
- `training_router_analysis.ipynb` compares routing behavior before and after training.
- `training_cka_hiddenstate_analysis.ipynb` plots cross-lingual hidden-state similarity changes after training.
