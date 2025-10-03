import os
import torch
import wandb
import shutil, argparse
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from mosaic.core.utils import get_working_dir, load_config, load_dataset, process_dataset, MinEpochsEarlyStoppingCallback


def model_init(model_tag: str, model_config: dict, peft_config: dict, checkpoint: str = None) -> tuple:
    """
    Initializes a model and tokenizer with optional PEFT configuration.

    Args:
        model_tag (str): The model identifier used for loading the pretrained model.
        model_config (dict): Configuration dictionary for the model, including parameters such as 'lora_rank',
                            'max_seq_length', 'load_in_4bit', and 'chat_template'.
        peft_config (dict): Configuration dictionary for PEFT (Parameter-Efficient Fine-Tuning), containing
                            PEFT_KWARGS and optionally GEMMA_PEFT_KWARGS for Gemma models, if applicable.
        checkpoint (str, optional): Path to a specific model checkpoint to load. If not provided, the latest
                                    pretrained model specified by 'model_tag' is loaded.

    Returns:
        tuple: A tuple containing the initialized model and tokenizer.

    Raises:
        Exception: If there are issues loading the model or tokenizer.
    """

    full_finetuning = True if model_config['lora_rank'] == 0 else False
    is_gemma = 'gemma' in model_tag.lower() or 'gemma' in model_config.get('model_family', '').lower()
    
    # Load model with Unsloth optimizations
    model, tokenizer = FastModel.from_pretrained(
        checkpoint if checkpoint else model_tag, 
        max_seq_length = model_config['max_seq_length'],
        load_in_4bit = model_config['load_in_4bit'],
        full_finetuning = full_finetuning,
        dtype = None,
    )
    
    if hasattr(tokenizer, 'padding_side'):
        tokenizer.padding_side = 'right'
    if hasattr(tokenizer, 'truncation_side'):
        tokenizer.truncation_side = 'left'
    
    # Apply LoRA if needed
    if not full_finetuning:
        print("Applying LoRA...")
        peft_kwargs = {**peft_config["PEFT_KWARGS"]}
        if is_gemma:
            peft_kwargs.update(peft_config.get("GEMMA_PEFT_KWARGS", {}))
            
        model = FastModel.get_peft_model(
            model,
            r = model_config['lora_rank'],
            lora_alpha = model_config['lora_alpha'],
            **peft_kwargs,
        )

    # Log configuration
    chat_template = getattr(tokenizer, 'chat_template', None)
    
    config = {
        'model_config': model_config,
        'full_finetuning': full_finetuning,
        'chat_template': chat_template,
        'peft_config': peft_config["PEFT_KWARGS"],
        'checkpoint': checkpoint if checkpoint else None,
    }
    
    # Only add Gemma config if relevant
    if is_gemma:
        config['gemma_peft'] = peft_config.get("GEMMA_PEFT_KWARGS", {})
        
    # Add tokenizer config if available
    if hasattr(tokenizer, 'pad_token_id'):
        config['tokenizer_config'] = {
            'pad_token_id': tokenizer.pad_token_id,
            'bos_token_id': getattr(tokenizer, 'bos_token_id', None),
            'eos_token_id': getattr(tokenizer, 'eos_token_id', None),
            'padding_side': getattr(tokenizer, 'padding_side', None),
            'truncation_side': getattr(tokenizer, 'truncation_side', None),
        }
        
    if wandb_log: wandb.log(config)
    return model, tokenizer


def init_trainer(model_config, training_config, model_training_config, logging_config, model, tokenizer, train_dataset, val_dataset, run_name, checkpoints_path, train_from_checkpoint=False):
    """
    Initializes and returns an SFTTrainer configured for training a model based on the provided configuration and datasets.

    Args:
        model_config (dict): Configuration dictionary containing model-specific parameters such as 'batch_size' and 'model_family'.
        model: The model instance to be trained.
        tokenizer: The tokenizer associated with the model.
        train_dataset: The dataset used for training the model.
        val_dataset: The dataset used for evaluating the model during training.
        run_name (str): A name for the training run, used for logging and output directory naming.
        checkpoints_path (str): The directory path where model checkpoints will be saved.

    Returns:
        SFTTrainer: An instance of SFTTrainer configured with the specified model, tokenizer, datasets, and training arguments.
    """

    EARLY_STOP = training_config["CALLBACK_KWARGS"]['min_epochs'] > 0
    if not EARLY_STOP: print("No early stopping")
    
    gradient_accumulation_steps = model_training_config["device_batch_size"] // model_config['batch_size']
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        callbacks = [MinEpochsEarlyStoppingCallback(**training_config["CALLBACK_KWARGS"])] if EARLY_STOP else [],
        
        args = SFTConfig(
            dataset_text_field = "text",
            output_dir = checkpoints_path,
            run_name = run_name,
            per_device_train_batch_size = model_config['batch_size'],
            gradient_accumulation_steps = gradient_accumulation_steps,
            seed=training_config["SEED"],
            eval_steps = model_training_config["eval_steps"],
            learning_rate = model_training_config["learning_rate"],
            weight_decay = model_training_config["weight_decay"],
            **training_config["TRAINING_KWARGS"],
            **logging_config["TRAINING_LOG_KWARGS"],
        ),
    )

    if wandb_log:
        wandb.log({
            'batch_size': model_config['batch_size'],
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'seed': training_config["SEED"],
            'early_stop_args': training_config["CALLBACK_KWARGS"] if EARLY_STOP else {},
            'training_args': {**training_config["TRAINING_KWARGS"], **model_training_config}
        })

    trainer = train_only_on_responses(trainer, model_config)
    return trainer
    
def train_only_on_responses(trainer: SFTTrainer, model_config: dict) -> SFTTrainer:
    """
    Modifies the trainer to only train on the assistant responses, ignoring the user's inputs.

    Args:
        trainer: An instance of `SFTTrainer`.
        model_config: A dictionary with configuration for the model.

    Returns:
        The modified `SFTTrainer` instance.
    """
    
    model_family = model_config['model_family']
    if model_family == "llama":
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    elif model_family == "gemma":
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<start_of_turn>user\n",
            response_part = "<start_of_turn>model\n",
        )
    else: raise ValueError(f"Model family {model_family} not supported.")
    return trainer


def save_model(model, tokenizer, save_path: str, model_config: dict):
    """
    Saves a model to a directory.

    Args:
        model: The model to save.
        tokenizer: The tokenizer associated with the model.
        save_path: The directory to save the model in.
        model_config: The configuration of the model.
    """
    model.save_pretrained_merged(save_path, tokenizer) # vllm doesn't load gemma 3 lora adapters
    print(f"Model saved to {save_path}")
    # TODO unsloth doesn't generate the right config for gemma 3 models, so copy (on March 27, 2025) 
    if 'gemma' in model_name:
        working_dir = os.path.dirname(os.path.dirname(working_dir))
        config_path = os.path.join(working_dir, f"/config/gemma/{model_name}/config.json")
        shutil.copy(config_path, save_path)


def gpu_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    if wandb_log:
        wandb.log({
            "start_gpu_memory": start_gpu_memory,
            "max_memory": max_memory,
        })
    return start_gpu_memory, max_memory

def training_stats(trainer_stats, start_gpu_memory, max_memory):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    if wandb_log:
        wandb.log({
            "runtime_seconds": trainer_stats.metrics['train_runtime'],
            "runtime_minutes": round(trainer_stats.metrics['train_runtime']/60, 2),
            "used_memory": used_memory,
            "used_memory_for_lora": used_memory_for_lora,
            "used_percentage": used_percentage,
            "lora_percentage": lora_percentage
        })


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-m', '--model_name', help='HF model tag', required=True)
    argparse.add_argument('-ct', '--config_tag', help='Config tag', required=True)
    argparse.add_argument('-p', '--project_name', help='Wandb project name', required=False, default=None)
    argparse.add_argument('-et', '--experiment_tag', help='Additional information', required=False, default='')
    argparse.add_argument('-tds', '--train_dataset_names', help='Dataset name(s)', required=True)
    argparse.add_argument('-vds', '--valid_dataset_names', help='Dataset name(s)', required=True)
    argparse.add_argument('-c', '--checkpoint', help='Checkpoint path', required=False, default=False)
    argparse.add_argument('-rt', '--resume_training', help='Resume from stopped training', required=False, default=False)
    argparse.add_argument('-cs', '--check_prompts_size', help='Check prompts size', required=False, default=False)
    argparse.add_argument('-o', '--output_dir', help='Output directory', required=True)
    args = argparse.parse_args()

    working_dir = get_working_dir()
    logging_config = load_config(working_dir,'logging.yaml')
    model_config = load_config(working_dir,'models.yaml')
    training_config = load_config(working_dir,'training.yaml')
    peft_config = load_config(working_dir,'peft.yaml')
    sweep_config = load_config(working_dir,'sweep.yaml')
    datasets_config = load_config(working_dir,'datasets.yaml')

    model_name = args.model_name
    train_dataset_names = args.train_dataset_names
    valid_dataset_names = args.valid_dataset_names
    checkpoint = args.checkpoint
    config_tag = args.config_tag
    project_name = args.project_name
    output_dir = args.output_dir

    global wandb_log
    if project_name: wandb_log = True
    else: wandb_log = False

    model_config = model_config[model_name]
    model_tag = model_config['model_tag']
    model_training_config = load_config(working_dir,f'exp/{config_tag}/{model_name}.yaml')
    if 'gemma' in model_tag:
        model_config["config_path"] = working_dir + f"/config/gemma/{model_name}/config.json"

    experiment_name = f"{model_name}_{str(train_dataset_names.replace(' ', '-'))}{args.experiment_tag}"
    if checkpoint: 
        experiment_name = f"{experiment_name}_from-{os.path.basename(checkpoint)}"
    if wandb_log: wandb.init(project=project_name, name=experiment_name)
    start_gpu_memory, max_memory = gpu_stats()

    models_save_path = f"{output_dir}/models/{experiment_name}/"
    checkpoints_path = f"{output_dir}/checkpoints/{experiment_name}"
    if not os.path.exists(models_save_path): os.makedirs(models_save_path)
    if not os.path.exists(checkpoints_path): os.makedirs(checkpoints_path)

    # end of experiment initialization
    # load model and data

    model, tokenizer = model_init(model_tag, model_config, peft_config, checkpoint)

    train_dataset = load_dataset(train_dataset_names, datasets_config, split='train')
    val_dataset = load_dataset(valid_dataset_names, datasets_config, split='val')
    print(train_dataset_names, train_dataset)
    train_dataset = process_dataset(train_dataset, tokenizer, model_tag)
    val_dataset = process_dataset(val_dataset, tokenizer, model_tag)
    print("Training example:\n", train_dataset[0]['text'])
    print(f"N. training: {len(train_dataset)}, n. validation; {len(val_dataset)}")

    if args.check_prompts_size:
    # check that prompts are not too long before training
        tokenized_prompts = tokenizer(train_dataset['text'])
        max_length = model_config['max_seq_length']
        assert max([len(x) for x in tokenized_prompts['input_ids']]) <= max_length, f"Prompts are too long. Max length is {max_length}."

    trainer = init_trainer(
        model_config, training_config, model_training_config, logging_config, 
        model, tokenizer, 
        train_dataset, val_dataset, 
        experiment_name, checkpoints_path,
        checkpoint)
    if args.resume_training: trainer.train(resume_from_checkpoint = True)
    trainer_stats = trainer.train()
    training_stats(trainer_stats, start_gpu_memory, max_memory)
    
    save_model(trainer.model, tokenizer, models_save_path, model_config) # save merged 16bit
    print(f"Model saved to {models_save_path}")