"""Utility functions for file and configuration management."""

import os
from pathlib import Path
from typing import Dict, List, Union
import yaml
import datasets
import transformers
import pandas as pd
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
    base_path = Path(paths_config['paths']['base']) / Path(paths_config['paths']['data'])
    print(f"Loading datasets from base path: {base_path}")
    
    datasets_list = []
    for name in dataset_names:
        if name not in datasets_config:
            raise ValueError(f"Dataset {name} not found in config")
            
        dataset_config = datasets_config[name]
        csv_path = base_path / name / f"{name}.csv"
        
        print(f"Loading dataset from: {csv_path}")
        # Read CSV file
        df = pd.read_csv(csv_path)
        # Create train/val/test splits
        # Read full dataset only once
        full_df = df.copy()
        
        # For small datasets (< 1000 rows), use split column if available
        if 'split' in full_df.columns and len(full_df) < 1000:
            print(f"Small dataset detected ({len(full_df)} rows). Using existing split column.")
            available_splits = full_df['split'].unique()
            print(f"Available splits: {available_splits}")
            if split not in available_splits:
                raise ValueError(f"Split '{split}' not found in dataset. Available splits: {available_splits}")
            df = full_df[full_df['split'] == split].copy()
            print(f"Selected {len(df)} rows for split '{split}'")
            if len(df) == 0:
                raise ValueError(f"No rows found for split '{split}'")
        else:
            # Generate consistent train/val/test splits for larger datasets
            if len(full_df) < 10:
                # For very small datasets, use all data for all splits
                print(f"Warning: Dataset too small ({len(full_df)} rows). Using all data for {split} split.")
                df = full_df.copy()
            else:
                # Normal train/val/test split for reasonable sized datasets
                train_idx = full_df.sample(frac=0.8, random_state=42).index
                remain_df = full_df.drop(train_idx)
                val_idx = remain_df.sample(frac=0.5, random_state=42).index
                test_idx = remain_df.drop(val_idx).index
                
                if split == 'train':
                    df = full_df.loc[train_idx]
                elif split == 'val':
                    df = full_df.loc[val_idx]
                elif split == 'test':
                    df = full_df.loc[test_idx]
        
        # Select the requested split
        # Determine which split to use
        print(f"Processing dataset with {len(full_df)} rows")
        
        if 'split' in full_df.columns and len(full_df) < 1000:
            # For small datasets, use the existing split column
            print("Using existing split column")
            df = full_df[full_df['split'] == split].copy()
            print(f"Selected {len(df)} rows for split '{split}'")
        else:
            # For larger datasets or when no split column exists
            if len(full_df) < 10:
                # Use all data for all splits if too small
                print(f"Warning: Dataset too small ({len(full_df)} rows). Using all data for {split} split.")
                df = full_df.copy()
            else:
                # Normal train/val/test split
                print("Generating train/val/test split")
                train_idx = full_df.sample(frac=0.8, random_state=42).index
                remain_df = full_df.drop(train_idx)
                val_idx = remain_df.sample(frac=0.5, random_state=42).index
                test_idx = remain_df.drop(val_idx).index
                
                if split == 'train':
                    df = full_df.loc[train_idx].copy()
                elif split == 'val':
                    df = full_df.loc[val_idx].copy()
                elif split == 'test':
                    df = full_df.loc[test_idx].copy()
                else:
                    raise ValueError(f"Invalid split: {split}. Must be one of: train, val, test")
            
        # Convert DataFrame to Dataset
        dataset = datasets.Dataset.from_pandas(df)
        datasets_list.append(dataset)
    
    return concatenate_datasets(datasets_list)

def process_dataset(dataset: "datasets.Dataset", tokenizer, model_tag: str) -> "datasets.Dataset":
    """Process dataset for training."""
    # Convert 'report' column to 'text' column if it exists
    if 'report' in dataset.column_names and 'text' not in dataset.column_names:
        dataset = dataset.rename_column('report', 'text')
    
    def _format_chat(text):
        # Extract and clean findings/observations
        if "FINAL REPORT:" in text:
            parts = text.split("FINAL REPORT:", 1)
            findings = parts[1].strip()
        else:
            findings = text.strip()
            
        # Clean up the findings text
        findings = findings.replace('\n', ' ').strip()
        findings = ' '.join(findings.split())  # Normalize whitespace
        
        # Create chat messages in Unsloth format
        messages = [
            {"role": "user", "content": "As a radiologist, please analyze these chest x-ray findings in detail:\n" + findings},
            {"role": "assistant", "content": findings}  # Response is the processed findings
        ]
        
        # Apply appropriate chat template based on model type
        # The tokenizer will handle the actual template formatting
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Important: we want the text template, not tokens
            add_generation_prompt=False  # Don't add extra generation tokens
        )
        return formatted
    
    # Format all examples with chat template
    print("Applying chat templates...")
    dataset = dataset.map(
        lambda example: {'text': _format_chat(example['text'])},
        desc="Formatting chat examples",
        load_from_cache_file=False
    )
    
    # Tokenize all examples without padding
    def _tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,  # No padding - let trainer handle it
            return_tensors=None
        )
    
    print("Tokenizing dataset...")
    dataset = dataset.map(
        _tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != 'text'],
        desc="Tokenizing",
        load_from_cache_file=False
    )
    print("Tokenization complete!")
    
    return dataset
    
    return dataset
    
    # Tokenize all texts
    print("Tokenizing dataset...")
    processed = dataset.map(
        _tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != 'text']
    )
    print("Tokenization complete!")
    
    return processed

def process_dataset_vllm(dataset, tokenizer, model_tag: str, include_description: bool = False, description_language: str = "en", few_shot: bool = False):
    """Process dataset for VLLM inference."""
    return dataset

def decode_output_vllm(output: str, empty_json: dict, idx: int, classes: list) -> dict:
    """Decode model output from VLLM."""
    return empty_json.copy()

class MinEpochsEarlyStoppingCallback(TrainerCallback):
    """Early stopping callback that enforces a minimum number of epochs."""
    
    def __init__(self, min_epochs=1, early_stopping_patience=5, early_stopping_threshold=0.001):
        self.patience = early_stopping_patience
        self.min_epochs = min_epochs
        self.threshold = early_stopping_threshold
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
            return
            
        relative_decrease = (self.best_metric - eval_metric) / abs(self.best_metric)
        
        if relative_decrease > self.threshold:
            self.best_metric = eval_metric
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
            if self.early_stopping_patience_counter >= self.patience:
                control.should_training_stop = True