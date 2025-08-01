from vllm import LLM, SamplingParams
import argparse
from tqdm import tqdm 
import warnings
from constants import FB_TARGET_LANGUAGES, LLM_KWARGS, SAMPLING_KWARGS, SRC_PATH, LANG_CODES
import os, yaml
from datasets import load_from_disk
from datasets import concatenate_datasets
from utils import get_working_dir, load_config

LANG_CODES = {
    # Germanic
    'eng_Latn': 'English',
    'deu_Latn': 'German',
    'nld_Latn': 'Dutch',
    'dan_Latn': 'Danish',
    'nor_Latn': 'Norwegian',
    'swe_Latn': 'Swedish',
    
    # Romance
    'fra_Latn': 'French',
    'ita_Latn': 'Italian',
    'por_Latn': 'Portuguese',
    'spa_Latn': 'Spanish',
    'ron_Latn': 'Romanian',
    
    # Slavic
    'pol_Latn': 'Polish',
    'ces_Latn': 'Czech',
    'slk_Latn': 'Slovak',
    'slv_Latn': 'Slovene',
    'bul_Cyrl': 'Bulgarian',
    'hrv_Latn': 'Croatian',
    
    # Baltic
    'lit_Latn': 'Lithuanian',
    'lav_Latn': 'Latvian',
    'est_Latn': 'Estonian',
    
    # Other Indo-European
    'ell_Grek': 'Greek',
    'gle_Latn': 'Irish',
    'hun_Latn': 'Hungarian',
    'fin_Latn': 'Finnish',
    'mlt_Latn': 'Maltese'
}

def model_init(model_tag, is_quantized, load_adapter):
    quantization = {
        'quantization': "bitsandbytes", # TODO is it different for unsloth dynamic quant?
        'load_format': "bitsandbytes"
    } if is_quantized else {}

    llm = LLM(
        model_tag, 
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        **quantization,
        )
    sampling_params = SamplingParams(
        temperature = 1.0, top_p = 0.95, top_k = 64,
        seed=42, max_tokens=1024
        )
    return llm, sampling_params


def save_outputs(dataset, dataset_name, output_folder):
    output_path = output_folder + dataset_name + "_translated"
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


def add_prompt(text, lang_code):
    prompt = f"Translate this text into {LANG_CODES[lang_code]}. Respond only with the translation."
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

    if source_language is None:
        source_language = 'eng_Latn'
        warnings.warn('Source languages not provided. Defaulting to ' + str(source_language))
    if target_languages is None:
        target_languages = ['spa_Latn']
        warnings.warn('Target languages not provided. Defaulting to ' + str(target_languages))
    elif type(target_languages) == str:
        target_languages = target_languages.split()
        print(target_languages)
    else:
        raise ValueError('Invalid input.')

    wdir = get_working_dir()
    datasets_yaml = load_config(wdir, 'datasets.yaml')
    models_yaml = = load_config(wdir, 'models.yaml')
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
