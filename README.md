# MOSAIC: Multilingual, Taxonomy-Agnostic, and Computationally Efficient Radiological Report Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="mosaic-icon.png" alt="My Image" width="200"/>


MOSAIC is a framework for efficient radiological report classification that is:
- 🌐 **Multilingual**: Works across different languages
- 🎯 **Taxonomy-Agnostic**: Adapts to various classification schemes
- ⚡ **Computationally Efficient**: Optimized for resource usage

## What’s in this repo
- `mosaic/core/finetune.py`: Unsloth + TRL SFT training with optional LoRA, early stopping, and WANDB logging.
- `mosaic/core/inference.py`: VLLM-based evaluation that scores predictions with weighted F1 and saves CSVs.
- `mosaic/core/translate.py`: Translate report datasets into multiple languages using VLLM.
- `mosaic/core/perplexity.py`: Perplexity utilities for MOSAIC and SIB-200 corpora.
- `mosaic/core/preprocess_data.py`: CLI wrappers that mirror the notebook preprocessing blocks.
- `config/`: Model, dataset, and experiment settings (see `config/exp/README.md` for experiment tags).
- `scripts/`: Thin shell wrappers to run training/evaluation with sensible defaults.

## Setup
```bash
conda env create -f environment.yaml
conda activate mosaic
pip install -e .
```
Update `config/paths.yaml` (default base is `/home/alice/work/mosaic`) and `config/datasets.yaml` so the paths point to your local HuggingFace `load_from_disk` datasets.

## Data layout
- Each dataset entry in `config/datasets.yaml` should point to a `datasets`-format directory containing `train/`, `val/`, and `test/` splits.
- Splits are expected to include `report` (text), `labels` (stringified dict of finding → class), `classes` (list of class ids), `findings` (list of strings), and optional `fs_examples` for few-shot prompts.
- To recreate datasets from raw files, use the notebook-aligned CLI:
  ```bash
  python -m mosaic.core.preprocess_data --function mimic \
    --input-dir /path/to/raw/data \
    --output-dir data/mimic
  ```
  Available functions include `mimic`, `padchest`, `casia`, `danskcxr`, `reflacx`, and variants listed inside `mosaic/core/preprocess_data.py`.

## Training
You can call the module directly or use `scripts/run_training.sh` (which activates the `mosaic` conda env and checks data paths).
```bash
python -m mosaic.core.finetune \
  --model_name medgemma-4b \              # key from config/models.yaml
  --config_tag m \                        # experiment folder in config/exp/
  --train_dataset_names "mimic" \         # space-separated keys from config/datasets.yaml
  --valid_dataset_names "mimic" \
  --output_dir outputs \
  --project_name None                     # set to a WANDB project name to enable logging
```
Outputs are saved under `<output_dir>/models/<experiment_name>/` and checkpoints under `<output_dir>/checkpoints/<experiment_name>/`.

## Evaluation and inference
Evaluation runs via VLLM and writes prediction/eval CSVs.
```bash
python -m mosaic.core.inference \
  --model_name medgemma-4b \              # same key used for training
  --zeroshot off \                        # one of: zeroshot | fewshot | ft-fewshot | off
  --train_dataset_names "mimic" \
  --test_dataset_names "mimic padchest_EN" \
  --models_folder outputs \               # where your trained models live
  --output_dir outputs/eval \
  --experiment_tag _m
```
`zeroshot/fewshot` run the base model, `ft-fewshot` adds few-shot prompts to a fine-tuned model, and `off` uses the fine-tuned checkpoints as-is.

## Translate datasets
```bash
python -m mosaic.core.translate \
  --model_tag medgemma-4b \
  --dataset_name mimic \
  --models_folder outputs \
  --output_folder data/mimic_translated \
  --source_language eng_Latn \
  --target_languages "dan_Latn spa_Latn fra_Latn" \
  --train_only True
```
Languages and VLLM sampling defaults come from `config/languages.yaml` and `config/vllm.yaml`.

## Perplexity utilities
```bash
python -m mosaic.core.perplexity \
  -m unsloth/gemma-3-4b-it \
  -d mosaic \
  -o outputs/ppl \
  --debug True
```
Use `-d sib` for the SIB-200 benchmark. Results are saved as CSVs.

## Citation

If you use MOSAIC in your research, please cite:

```bibtex
@misc{schiavone2025mosaicmultilingualtaxonomyagnosticcomputationally,
      title={MOSAIC: A Multilingual, Taxonomy-Agnostic, and Computationally Efficient Approach for Radiological Report Classification}, 
      author={Alice Schiavone and Marco Fraccaro and Lea Marie Pehrson and Silvia Ingala and Rasmus Bonnevie and Michael Bachmann Nielsen and Vincent Beliveau and Melanie Ganz and Desmond Elliott},
      year={2025},
      eprint={2509.04471},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.04471}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
