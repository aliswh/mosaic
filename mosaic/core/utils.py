"""Utility functions for file and configuration management."""

import os
from pathlib import Path
from typing import Dict, List, Union
import yaml
import datasets
import transformers
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from datasets import load_from_disk, concatenate_datasets


class ConfigLoader:
    """Configuration loader and manager for MOSAIC."""
    
    def __init__(self, config_dir: Union[str, Path] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Path to configuration directory. If None, uses default package config.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "config"
        self.config_dir = Path(config_dir)
        
    def load_yaml(self, yaml_file: str) -> Dict:
        """
        Load configurations from a YAML file.
        
        Args:
            yaml_file: Name of the YAML file (with or without .yaml extension)
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is invalid
        """
        if not yaml_file.endswith('.yaml'):
            yaml_file += '.yaml'
            
        yaml_path = self.config_dir / yaml_file
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
        with yaml_path.open('r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Invalid YAML file {yaml_file}: {str(e)}")


class DatasetLoader:
    """Dataset loader and manager for MOSAIC."""
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize the dataset loader.
        
        Args:
            config_loader: ConfigLoader instance for accessing dataset configurations
        """
        self.config_loader = config_loader
        
    def load_datasets(
        self, 
        dataset_names: Union[str, List[str]], 
        split: str = 'train'
    ) -> 'datasets.Dataset':
        """
        Load and concatenate multiple datasets.
        
        Args:
            dataset_names: Space-separated string or list of dataset names
            split: Dataset split to load ('train', 'val', 'test')
            
        Returns:
            Concatenated dataset
            
        Raises:
            ValueError: If no valid datasets are found
        """
        if isinstance(dataset_names, str):
            dataset_names = dataset_names.split()
            
        datasets_config = self.config_loader.load_yaml('datasets')
        datasets = []
        
        for name in dataset_names:
            if name not in datasets_config:
                print(f"Warning: Dataset '{name}' not found in configuration")
                continue
                
            path = Path(datasets_config[name]['path'])
            if not path.exists():
                print(f"Warning: Dataset path not found: {path}")
                continue
                
            try:
                dataset = load_from_disk(str(path))
                if split in dataset:
                    datasets.append(dataset[split])
                else:
                    print(f"Warning: Split '{split}' not found in dataset '{name}'")
            except Exception as e:
                print(f"Error loading dataset '{name}': {str(e)}")
                
        if not datasets:
            raise ValueError("No valid datasets found to load")
            
        return concatenate_datasets(datasets)


def get_package_root() -> Path:
    """Get the root directory of the MOSAIC package."""
    return Path(__file__).parent.parent.parent

def get_working_dir() -> str:
    """Get the working directory for MOSAIC."""
    return str(get_package_root())

def load_config(working_dir: str, config_file: str) -> Dict:
    """Load a configuration file."""
    config_loader = ConfigLoader(Path(working_dir) / "config")
    return config_loader.load_yaml(config_file)

def load_dataset(dataset_names: Union[str, List[str]], datasets_config: Dict, split: str = "train") -> "datasets.Dataset":
    """Load dataset(s) by name."""
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split()
    
    # Load paths config for base paths
    paths_config = load_config(get_working_dir(), 'paths')
    base_path = Path(paths_config['paths']['base'])
    
    datasets_list = []
    for name in dataset_names:
        if name not in datasets_config:
            raise ValueError(f"Dataset {name} not found in config")
        # Convert relative path to absolute using base path
        path = base_path / datasets_config[name]["path"]
        dataset = load_from_disk(str(path))
        datasets_list.append(dataset[split])
    
    return concatenate_datasets(datasets_list)

def process_dataset(dataset: "datasets.Dataset", tokenizer, model_tag: str) -> "datasets.Dataset":
    """Process dataset for training."""
    # No special processing needed for now - keep text column as is
    return dataset

def process_dataset_vllm(dataset, tokenizer, model_tag: str, include_description: bool = False, description_language: str = "en", few_shot: bool = False):
    """Process dataset for VLLM inference."""
    return dataset

def decode_output_vllm(output: str, empty_json: dict, idx: int, classes: list) -> dict:
    """Decode model output from VLLM."""
    return empty_json.copy()

class MinEpochsEarlyStoppingCallback(TrainerCallback):
    """Early stopping callback that enforces a minimum number of epochs."""
    
    def __init__(self, patience=1, min_epochs=3):
        self.patience = patience
        self.min_epochs = min_epochs
        self.early_stopping_patience_counter = 0
        self.best_metric = None
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """Called after evaluation."""
        epoch = state.epoch
        
        # Don't start early stopping until min_epochs reached
        if epoch < self.min_epochs:
            return
            
        eval_metric = metrics.get("eval_loss")
        if eval_metric is None:
            return
            
        if self.best_metric is None:
            self.best_metric = eval_metric
        elif eval_metric > self.best_metric:
            self.early_stopping_patience_counter += 1
            if self.early_stopping_patience_counter >= self.patience:
                control.should_training_stop = True
        else:
            self.early_stopping_patience_counter = 0
            self.best_metric = eval_metric