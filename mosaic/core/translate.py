from vllm import LLM, SamplingParams
import argparse
from tqdm import tqdm
import warnings
import os

from datasets import load_from_disk, concatenate_datasets, Dataset, DatasetDict
from mosaic.core.utils import get_working_dir, load_config

# Load language codes from config
wdir = get_working_dir()
languages_config = load_config(wdir, 'languages')


def model_init(model_tag, is_quantized):
    vllm_config = load_config(get_working_dir(), 'vllm')
    quantization = {
        'quantization': "bitsandbytes",
        'load_format': "bitsandbytes"
    } if is_quantized else {}

    llm = LLM(
        model_tag,
        **vllm_config['VLLM_KWARGS'],
        **quantization,
    )
    sampling_params = SamplingParams(**vllm_config['SAMPLING_KWARGS'])
    return llm, sampling_params


def is_csv_path(path: str) -> bool:
    return isinstance(path, str) and path.lower().endswith(".csv")


def load_input_dataset(input_path: str):
    """
    Returns (kind, data)
      kind: "hf" or "csv"
      data: DatasetDict for hf, Dataset for csv
    """
    if is_csv_path(input_path):
        ds = Dataset.from_csv(input_path)
        return "csv", ds

    # otherwise assume it's a HF dataset saved to disk
    ds = load_from_disk(input_path)
    if isinstance(ds, Dataset):
        # normalize to DatasetDict with a single split for consistent processing
        ds = DatasetDict({"train": ds})
    return "hf", ds


def save_outputs(data, dataset_name, output_folder, input_kind: str):
    """
    If input was hf -> save_to_disk(output_path)
    If input was csv -> write csv(output_path.csv)
    """
    os.makedirs(output_folder, exist_ok=True)

    if input_kind == "hf":
        output_path = os.path.join(output_folder, f"{dataset_name}_translated")
        data.save_to_disk(output_path)
        print(f"Dataset saved to {output_path}")
    elif input_kind == "csv":
        output_path = os.path.join(output_folder, f"{dataset_name}_translated.csv")
        data.to_csv(output_path, index=False)
        print(f"CSV saved to {output_path}")
    else:
        raise ValueError(f"Unknown input_kind: {input_kind}")


def add_prompt(example, lang_code):
    languages_config = load_config(get_working_dir(), 'languages')
    target_language = languages_config['languages'][lang_code]
    prompt = f"Translate this text into {target_language}. Respond only with the translation."
    return {
        "conversations": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": example["report"]},
        ],
        "source": "translation",
    }


def process_for_translations(dataset, target_lang, llm):
    apply_template_fn = llm.llm_engine.tokenizer.tokenizer.apply_chat_template

    def inner(examples):
        texts = [
            apply_template_fn(conv, tokenize=False, add_generation_prompt=True)
            for conv in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(
        add_prompt,
        batched=False,
        fn_kwargs={"lang_code": target_lang},
        remove_columns=dataset.column_names,
    )
    dataset = dataset.map(inner, batched=True)
    return dataset


def add_translations_to_dataset(dataset, outputs, language):
    # keep original columns, but replace report + add language
    if "language" in dataset.column_names:
        dataset = dataset.remove_columns("language")
    dataset = dataset.add_column("language", [language] * len(dataset))

    if "report" in dataset.column_names:
        dataset = dataset.remove_columns("report")
    dataset = dataset.add_column("report", outputs)

    return dataset


def translate_vllm(model, sampling_params, dataset):
    outputs = model.generate(dataset["text"], sampling_params)
    outputs = [o.outputs[0].text for o in outputs]
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCXR: Multilingual Chest X-Ray Report Classification")
    parser.add_argument("-m", "--model_tag", help="HF model tag", required=True)

    # NOTE: now this can be either:
    #  - path to HF dataset on disk
    #  - path to CSV file
    parser.add_argument("-i", "--dataset_name", help="dataset name (key for YAML or a path to CSV/HF dataset)", required=True)

    parser.add_argument("-o", "--output_folder", help="Output folder", required=True)
    parser.add_argument("-d", "--models_folder", help="Where to find models", required=True)
    parser.add_argument("-sl", "--source_language", help="Language tag of source language. (BCP-47 code)", required=False)
    parser.add_argument("-tl", "--target_languages", help="Language tag of target language. (BCP-47 code)", required=False)
    parser.add_argument("-vds", "--dataset_version", help="Tag for dataset directory", required=False)
    parser.add_argument("-to", "--train_only", help="Only translate train split", required=False, default=False, action="store_true")

    args = parser.parse_args()
    model_tag = args.model_tag
    dataset_name = args.dataset_name
    output_folder = args.output_folder
    if args.dataset_version:
        output_folder += "_" + args.dataset_version

    source_language = args.source_language
    target_languages = args.target_languages

    languages_config = load_config(get_working_dir(), "languages")

    if source_language is None:
        source_language = "eng_Latn"
        warnings.warn("Source language not provided. Defaulting to " + str(source_language))

    if target_languages is None:
        target_languages = languages_config["target_languages"]
        warnings.warn("Target languages not provided. Defaulting to " + str(target_languages))
    elif isinstance(target_languages, str):
        target_languages = target_languages.split()
    else:
        raise ValueError("Invalid input.")

    wdir = get_working_dir()
    datasets_yaml = load_config(wdir, "datasets.yaml")
    models_yaml = load_config(wdir, "models.yaml")
    model_config = models_yaml[model_tag]

    # -------- resolve input path --------
    # If dataset_name matches YAML key, use that path; otherwise treat as direct path (CSV or HF dataset dir)
    if dataset_name in datasets_yaml and "path" in datasets_yaml[dataset_name]:
        input_path = datasets_yaml[dataset_name]["path"]
        inferred_name_for_output = dataset_name
    else:
        input_path = dataset_name  # direct path
        inferred_name_for_output = os.path.splitext(os.path.basename(dataset_name))[0]

    input_kind, dataset_obj = load_input_dataset(input_path)

    # Validate CSV has required column
    if input_kind == "csv":
        if "text" in dataset_obj.column_names and "report" not in dataset_obj.column_names:
            dataset_obj = dataset_obj.rename_column("text", "report")

        if "report" not in dataset_obj.column_names:
            raise ValueError(
                f"CSV input must contain a 'report' (or 'text') column. "
                f"Found: {dataset_obj.column_names}"
            )


    model, sampling_params = model_init(model_config["model_tag"], model_config["load_in_4bit"])

    # Determine splits
    if input_kind == "hf":
        if args.train_only:
            splits_to_translate = ["train"]
        else:
            # only translate splits that exist
            splits_to_translate = [s for s in ["train", "test", "val"] if s in dataset_obj]
    else:
        splits_to_translate = ["train"]  # synthetic split for CSV

    # -------- translation loop --------
    if input_kind == "csv":
        original_dataset = dataset_obj
        translated_datasets = []
        for trg_lang in tqdm(target_languages):
            temp_dataset = process_for_translations(original_dataset, trg_lang, model)
            output = translate_vllm(model, sampling_params, temp_dataset)
            translated_datasets.append(add_translations_to_dataset(original_dataset, output, trg_lang))
        out_dataset = concatenate_datasets(translated_datasets)

        save_outputs(out_dataset, inferred_name_for_output, output_folder, input_kind="csv")
        print(out_dataset)

    else:
        # HF DatasetDict
        for split in splits_to_translate:
            original_dataset = dataset_obj[split]
            translated_datasets = []
            for trg_lang in tqdm(target_languages):
                temp_dataset = process_for_translations(original_dataset, trg_lang, model)
                output = translate_vllm(model, sampling_params, temp_dataset)
                translated_datasets.append(add_translations_to_dataset(original_dataset, output, trg_lang))
            dataset_obj[split] = concatenate_datasets(translated_datasets)

        save_outputs(dataset_obj, inferred_name_for_output, output_folder, input_kind="hf")
        print(dataset_obj)
