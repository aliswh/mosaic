from datasets import load_from_disk
import numpy as np
import pandas as pd
import ast
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from mosaic.core.utils import process_dataset_vllm, decode_output_vllm, load_config, get_working_dir
from mosaic.core.evals import get_F1_scores
import wandb
import argparse, os

# Load configs
wdir = get_working_dir()
vllm_config = load_config(wdir, 'vllm')
paths_config = load_config(wdir, 'paths')

MAX_SEQ_LENGTH = vllm_config['VLLM_KWARGS']['max_model_len']

def model_init(model_tag, is_quantized, load_adapter):
    vllm_config = load_config(get_working_dir(), 'vllm')
    quantization = {
        'quantization': "bitsandbytes",  # TODO is it different for unsloth dynamic quant?
        'load_format': "bitsandbytes"
    } if is_quantized else {}

    # Use max_seq_length from models config
    vllm_kwargs = vllm_config['VLLM_KWARGS'].copy()
    if isinstance(vllm_kwargs.get('max_model_len'), str) and vllm_kwargs['max_model_len'] == 'MAX_SEQ_LENGTH':
        vllm_kwargs['max_model_len'] = 2048  # Default value from models.yaml

    llm = LLM(
        model_tag, 
        **vllm_kwargs,
        **quantization,
        enable_lora=load_adapter
    )
    sampling_params = SamplingParams(**vllm_config['SAMPLING_KWARGS'])
    return llm, sampling_params

def decode(outputs, empty_json, classes):
    outputs = [o.outputs[0].text for o in outputs]
    return [decode_output_vllm(
        o, empty_json, idx, classes
    ) for idx, o in enumerate(outputs)]

def generate_response(model, sampling_params, dataset, empty_json, classes):
    results = []
    
    outputs = model.generate(dataset, sampling_params)
    
    outputs = decode(outputs, empty_json, classes)
    results = pd.DataFrame(outputs)
    # count how many rows have all values as None
    n_none = results.isnull().all(axis=1).sum()
    results = results.fillna(-1)
    print(f'Number of invalid outputs: {n_none} ({n_none/len(results)*100:.2f}% total)')
    return results, n_none

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # env init
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-m', '--model_name', help='Model name', required=True)
    argparse.add_argument('-zs', '--zeroshot', help='Use zeroshot or original model', default=True, required=False)
    argparse.add_argument('-trds', '--train_dataset_names', help='Dataset name(s) used for training', required=True)
    argparse.add_argument('-p', '--project_name', help='Wandb project name', required=False, default=None)
    argparse.add_argument('-et', '--experiment_tag', help='Additional information', required=False, default='')
    argparse.add_argument('-tt', '--test_tag', help='Tag to save test results', required=False, default='')
    argparse.add_argument('-d', '--models_folder', help='Where to find models', required=True)
    argparse.add_argument('-o', '--output_dir', help='Output directory', required=True)
    argparse.add_argument('-teds', '--test_dataset_names', help='Dataset name(s)', required=False)
    argparse.add_argument('-descr', '--include_description', help='Include description of findings', required=False, default=False)
    argparse.add_argument('-descrlang', '--description_language', help='Language of description of findings', required=False, default='en')
    argparse.add_argument('-c', '--checkpoint', help='Checkpoint path', required=False, default=False)
    argparse.add_argument('-tmp', '--trained_model_path', help='Path to model', required=False, default=False)

    args = argparse.parse_args()

    model_name = args.model_name
    zeroshot = args.zeroshot
    assert zeroshot in ['zeroshot', 'fewshot', 'ft-fewshot', 'off'], "Value not supported"
    train_dataset_names = args.train_dataset_names
    models_folder = args.models_folder
    output_dir = args.output_dir
    datasets_names = args.test_dataset_names
    include_description = True if args.include_description == 'true' else False
    description_language = args.description_language # 'en' or 'da'
    checkpoint = args.checkpoint
    project_name = args.project_name

    wdir = get_working_dir()
    datasets_yaml = load_config(wdir, 'datasets.yaml')
    models_yaml = load_config(wdir, 'models.yaml')

    datasets_names = datasets_names.split()
    # Convert relative paths to absolute
    base_path = Path(paths_config['paths']['base'])
    datasets = [load_from_disk(str(base_path / datasets_yaml[name]['path'])) for name in datasets_names]
    datasets_classes = [datasets_yaml[name]['classes'] for name in datasets_names]
    print(datasets_names)

    model_config = models_yaml[model_name]
    if zeroshot == 'zeroshot':
        trained_model_tag = model_name+"_zs"
        trained_model_path = model_config['model_tag']
    elif zeroshot == "fewshot":
        trained_model_tag = model_name+"_fs"
        trained_model_path = model_config['model_tag']
    elif zeroshot == "ft-fewshot":
        trained_model_tag = f"{model_name}_{str(train_dataset_names.replace(' ', '-'))}"
        if checkpoint: 
            trained_model_tag = f"{trained_model_tag}_from-{os.path.basename(checkpoint)}"
        trained_model_path = models_folder + '/models/' + trained_model_tag
    else:
        trained_model_tag = f"{model_name}_{str(train_dataset_names.replace(' ', '-'))}"
        if checkpoint: 
            trained_model_tag = f"{trained_model_tag}_from-{os.path.basename(checkpoint)}"
        trained_model_path = models_folder + '/models/' + trained_model_tag + args.experiment_tag
    if args.trained_model_path: trained_model_path = args.trained_model_path
    print(f"Model: {trained_model_path}")

    global wandb_log
    if project_name: wandb_log = True
    else: wandb_log = False
    if wandb_log:
        wandb.init(project=project_name, name=f"{trained_model_tag}_test{args.experiment_tag}")
        wandb.log(vllm_config['SAMPLING_KWARGS'])

    base_save_path = output_dir + '/' + trained_model_tag + args.test_tag + '/'
    Path(base_save_path).mkdir(parents=True, exist_ok=True)

    model, sampling_params = model_init(
        model_config['model_tag'],
        model_config['load_in_4bit'],
        model_config.get('load_adapter', False)  # Default to False if not specified
        )

    tqdm_datasets = tqdm(enumerate(datasets))
    for idx, dataset in tqdm_datasets:
        tqdm_datasets.set_description(f"Dataset: {datasets_names[idx]}")
        
        if 'test' not in dataset:
            print(f'No test dataset for {datasets_names[idx]}')
            continue
        else:
            dataset = dataset['test']#.select(range(20))
            gt = pd.DataFrame([ast.literal_eval(output) for output in dataset['labels']])

            dataset = process_dataset_vllm(
                dataset, model, model_config['model_tag'], 
                include_description=include_description, description_language=description_language,
                few_shot= True if "fewshot" in zeroshot else False
            )
            if 'report' in dataset.features:
                dataset = dataset['report']
            else:
                dataset = dataset['text'] 
            print(dataset[0])

            tokenizer = AutoTokenizer.from_pretrained(model_config['model_tag'])
            tokenized_prompts = tokenizer(dataset)
            max_seq_length = 2048  # From models.yaml
            assert max([len(x) for x in tokenized_prompts['input_ids']]) <= max_seq_length, f"Prompts are too long. Max length is {max_seq_length}."

            findings = datasets_yaml[datasets_names[idx]]['findings']
            classes = datasets_classes[idx]
            empty_json = {f: None for f in findings}  # invalid predictions are None

            outputs, n_none = generate_response(
                model, sampling_params, 
                dataset, empty_json, classes
                )
            output_evals = get_F1_scores(gt, outputs)
            dataset_name = datasets_names[idx]
            mean_f1 = np.nanmean(output_evals[2])
            print(f'Model: {trained_model_tag}, Dataset: {dataset_name}, mean wF1: {mean_f1:.3f}, Invalid answers: {n_none}')
            if wandb_log:
                wandb.log({'mean_f1_'+dataset_name:mean_f1, 'invalid_answers':n_none, 'invalid_answers_perc':n_none/len(outputs)*100})

            save_path = base_save_path + dataset_name 
            save_path += args.experiment_tag+"/" if args.experiment_tag else "/"
            Path(save_path).mkdir(parents=True, exist_ok=True)

            # save evals to path
            for e, name in zip(output_evals, ['per_clss_scores', 'per_clss_support', 'weighted']):
                e.to_csv(save_path + name + '.csv')
            outputs.to_csv(save_path + f'predictions.csv', index=False)

    print("Evals saved in ", save_path)

