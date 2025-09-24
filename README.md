# MOSAIC: Multilingual, Taxonomy-Agnostic, and Computationally Efficient Radiological Report Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="mosaic-icon.png" alt="My Image" width="200"/>


MOSAIC is a framework for efficient radiological report classification that is:
- 🌐 **Multilingual**: Works across different languages
- 🎯 **Taxonomy-Agnostic**: Adapts to various classification schemes
- ⚡ **Computationally Efficient**: Optimized for resource usage

## Installation

You can install MOSAIC using pip:

```bash
pip install mosaic
```

Or install from source:

```bash
git clone https://github.com/aliswh/mosaic
cd mosaic
pip install -e .
```

## Quick Start

```python
from mosaic import ConfigLoader, DatasetLoader
from mosaic.core import finetune

# Initialize configuration
config = ConfigLoader()
datasets = DatasetLoader(config)

# Load model configuration
model_config = config.load_yaml('models')
peft_config = config.load_yaml('peft')

# Load dataset
train_data = datasets.load_datasets('mimic', split='train')
val_data = datasets.load_datasets('mimic', split='val')

# Initialize and train model
model, tokenizer = finetune.model_init(
    model_tag="AliceSch/mosaic-4b",
    model_config=model_config,
    peft_config=peft_config
)

# See demo.ipynb for complete training and evaluation example
```

## Features

- 🤖 Support for multiple LLM architectures (Gemma, LLaMA)
- 📚 Efficient fine-tuning with LoRA
- 📊 Comprehensive evaluation metrics
- 🔄 Multi-lingual translation capabilities

## Documentation

For detailed documentation, examples, and API reference, visit our [documentation page](https://mosaic-rad.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## Citation

If you use MOSAIC in your research, please cite:

```bibtex
@article{mosaic2025,
    title={MOSAIC: A Multilingual, Taxonomy-Agnostic, and Computationally Efficient Approach for Radiological Report Classification},
    author={[Author list]},
    journal={[Journal]},
    year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.