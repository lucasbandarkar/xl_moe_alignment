[WIP] This repo is currently being adapted for publication & presentation.

# Cross-Lingual Routing Alignment in Decoder-only MoEs


code for designing and experimenting with cross-lingual contrastive learning in MoE LLMs

## Environment

This code was developed and tested on remote Linux servers with NVIDIA GPUs and CUDA. Checkpoint and dataset locations can be customized with the environment variables documented in the relevant scripts.


This repository contains the code used for the experiments described in the paper.


## Code Organization

- `contrastive_training/` is the main code containing implementation and scripts for contrastive training method
- `language_evals/` contains the code for downstream task evaluation (and also SFT+eval)
- `routing_analysis/`

## Environments

See `contrastive_training/README.md`, `language_evals/README.md`, and `routing_analysis/README.md` for environment and execution instructions.
