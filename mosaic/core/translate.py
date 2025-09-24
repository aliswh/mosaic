from vllm import LLM, SamplingParams
import argparse
from tqdm import tqdm 
import warnings
import os, yaml
from datasets import load_from_disk
from datasets import concatenate_datasets
from mosaic.core.utils import get_working_dir, load_config

# Load language codes from config
wdir = get_working_dir()
languages_config = load_config(wdir, 'languages')

def model_init(model_tag, is_quantized, load_adapter):
    vllm_config = load_config(get_working_dir(), 'vllm')
    quantization = {
        'quantization': "bitsandbytes",  # TODO is it different for unsloth dynamic quant?
        'load_format': "bitsandbytes"
    } if is_quantized else {}

    llm = LLM(
        model_tag,
        **vllm_config['llm'],
        **quantization,
    )
    sampling_params = SamplingParams(**vllm_config['sampling'])
    return llm, sampling_params


def save_outputs(dataset, dataset_name, output_folder):
    output_path = output_folder + dataset_name + "_translated"
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


def add_prompt(text, lang_code):
    languages_config = load_config(get_working_dir(), 'languages')
    target_language = languages_config['languages'][lang_code]
    prompt = f"Translate this text into {target_language}. Respond only with the translation."
    return {"conversations": [
        {"role": "system", "content": prompt},
        {"role": "user","content": text['report']}
    ], "source":"translation"}


def process_for_translations(dataset, target_lang, llm):
    def inner(examples):
        texts = [apply_template_fn(text, **{'tokenize': False, 'add_generation_prompt': True}) for text in examples['conversations']]
        return {"text": texts}  

    apply_template_fn = llm.llm_engine.tokenizer.tokenizer.apply_chat_template
    dataset = dataset.map(
        add_prompt, batched = False, fn_kwargs={'lang_code':target_lang},
        remove_columns=dataset.column_names
    )
    dataset = dataset.map(inner, batched = True)
    return dataset


def add_translations_to_dataset(dataset, outputs, language):
    dataset = dataset.add_column('language', [language]*len(dataset))
    return dataset.remove_columns('report').add_column('report', outputs)


def translate_vllm(model, sampling_params, dataset):  
    outputs = model.generate(dataset['text'], sampling_params)
    outputs = [o.outputs[0].text for o in outputs]
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCXR: Multilingual Chest X-Ray Report Classification')
    parser.add_argument('-m', '--model_tag', help='HF model tag', required=True)
    parser.add_argument('-i', '--dataset_name', help='dataset name', required=True)
    parser.add_argument('-o', '--output_folder', help='Output folder', required=True)
    parser.add_argument('-d', '--models_folder', help='Where to find models', required=True)
    parser.add_argument('-sl', '--source_language', help='Language tag of source language. (BCP-47 code)', required=False)
    parser.add_argument('-tl', '--target_languages', help='Language tag of target language. (BCP-47 code)', required=False)
    parser.add_argument('-vds', '--dataset_version', help='Tag for dataset directory', required=False)
    parser.add_argument('-to', '--train_only', help='Only translate train split', required=False, default=False)
        
    args = parser.parse_args()
    model_tag = args.model_tag
    dataset_name = args.dataset_name
    output_folder = args.output_folder
    if args.dataset_version: output_folder += "_" + args.dataset_version
    source_language = args.source_language
    target_languages = args.target_languages

    languages_config = load_config(get_working_dir(), 'languages')
    
    if source_language is None:
        source_language = 'eng_Latn'
        warnings.warn('Source language not provided. Defaulting to ' + str(source_language))
    if target_languages is None:
        target_languages = languages_config['target_languages']
        warnings.warn('Target languages not provided. Defaulting to ' + str(target_languages))
    elif isinstance(target_languages, str):
        target_languages = target_languages.split()
        print(target_languages)
    else:
        raise ValueError('Invalid input.')

    wdir = get_working_dir()
    datasets_yaml = load_config(wdir, 'datasets.yaml')
    models_yaml = load_config(wdir, 'models.yaml')
    model_config = models_yaml[model_tag]

    dataset = load_from_disk(datasets_yaml[dataset_name]['path'])
    dataset_classes = datasets_yaml[dataset_name]['classes']
    
    model, sampling_params = model_init(model_config['model_tag'], model_config['load_in_4bit'], model_config['load_adapter'])

    if args.train_only: splits_to_translate = ['train']
    else: splits_to_translate = ['train', 'test', 'validation']

    for split in splits_to_translate:
        original_dataset = dataset[split]
        translated_datasets = []
        for trg_lang in tqdm(target_languages):
            temp_dataset = process_for_translations(original_dataset, trg_lang, model)
            output = translate_vllm(model, sampling_params, temp_dataset)
            translated_datasets.append(add_translations_to_dataset(original_dataset, output, trg_lang))
        dataset[split] = concatenate_datasets(translated_datasets)

    save_outputs(dataset, dataset_name, output_folder)
    print(dataset)
