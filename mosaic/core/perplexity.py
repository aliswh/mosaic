from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from tqdm import tqdm
import argparse
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd, os
from datasets import load_from_disk
from mosaic.core.utils import load_config, get_working_dir


def load_sib():
    concat_dataset = lambda d: concatenate_datasets([d["train"],d["validation"],d["test"]])

    english = load_dataset("mteb/sib200", "eng_Latn")
    french = load_dataset("mteb/sib200", "fra_Latn")
    spanish = load_dataset("mteb/sib200", "spa_Latn")
    danish = load_dataset("mteb/sib200", "dan_Latn")

    data = {
        "english": concat_dataset(english),
        "french": concat_dataset(french),
        "spanish": concat_dataset(spanish),
        "danish": concat_dataset(danish),
    }

    return data, "text"

def load_mosaic_data():
    concat_dataset = lambda d: concatenate_datasets([d["train"],d["val"],d["test"]])

    wdir = get_working_dir()
    datasets_yaml = load_config(wdir, 'datasets.yaml')
    mimic = load_from_disk(datasets_yaml['mimic']['path'])
    padchest = load_from_disk(datasets_yaml['padchest']['path'])
    padchest_EN = load_from_disk(datasets_yaml['padchest_EN']['path'])
    casia = load_from_disk(datasets_yaml['casia']['path'])
    danskcxr = load_from_disk(datasets_yaml['danskcxr']['path'])

    data = {
        "mimic": concat_dataset(mimic),
        "padchest": concat_dataset(padchest),
        "padchest_EN": concat_dataset(padchest_EN),
        "casia": concat_dataset(casia),
        "danskcxr": concat_dataset(danskcxr),
        }
    return data, "report"

def load_model(model_name, load_in_4bit=False, load_in_8bit=False):
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    # Set up tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1" if "llama" in model_name.lower() else "gemma-3",
    )
    model.eval()
    return model, tokenizer



def ppl_model(model, tokenizer, dataset, text_col="text", debug=True):
    """
    Compute perplexity for a dataset.
    If debug=True, returns a DataFrame with per-sentence PPL and tokens.
    Otherwise, returns overall PPL.
    """
    max_length = 2048
    stride = 512
    nlls = []

    # For debug mode
    debug_records = []

    for s in tqdm(range(len(dataset[text_col]))):
        sentence = dataset[text_col][s]
        encodings = tokenizer(sentence, return_tensors="pt")
        input_ids = encodings.input_ids[0]  # single sequence
        special_ids = tokenizer.all_special_ids if hasattr(tokenizer, "all_special_ids") else []

        # Keep only real tokens (skip special tokens)
        real_token_indices = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
        input_ids = input_ids[real_token_indices].unsqueeze(0).to("cuda")  # add batch dim
        seq_len = input_ids.size(1)
        prev_end_loc = 0

        sentence_nlls = []

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_chunk.clone()
            target_ids[:, :-trg_len] = -100

            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            attention_mask = (input_chunk != pad_token_id).long()

            with torch.no_grad():
                outputs = model(input_chunk, labels=target_ids, attention_mask=attention_mask)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            sentence_nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if debug:
            sentence_ppl = torch.exp(torch.stack(sentence_nlls).mean()).item()

            # Convert input_ids to token strings
            if hasattr(tokenizer, "tokenizer"):  # Gemma3Processor has a tokenizer inside
                tokens = tokenizer.tokenizer.convert_ids_to_tokens(input_ids[0])
            else:
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            # Join tokens with space
            tokens_str = " ".join(tokens)

            debug_records.append({
                #"input_ids": input_ids[0].cpu().numpy(),
                "len_input_ids": input_ids.size(1),
                #"sentence": sentence,
                #"tokens": tokens_str,
                "ppl": sentence_ppl
            })

    overall_ppl = torch.exp(torch.stack(nlls).mean()).item()

    if debug:
        df = pd.DataFrame(debug_records)
        return overall_ppl, df
    else:
        return overall_ppl, None

def compute_ppl(model, tokenizer, dataset, text_col, debug=False):
    """
    Implementation by rolandtannous
    https://github.com/unslothai/unsloth/blob/main/tests/saving/language_models/test_merged_model_perplexity_llama-3.1-8b.py
    """

    # Format the dataset
    def formatting_prompts_func(examples):
        messages = [{"role": "user", "content": text} for text in examples[text_col]]
        #texts = [
        #    tokenizer.apply_chat_template([convo], tokenize=False, add_generation_prompt=False)
        #    for convo in messages
        #]
        texts = examples[text_col]

        return {text_col: texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Compute perplexity using the passed dataset
    ppl_value, debug_df = ppl_model(model, tokenizer, dataset, text_col, debug)

    # IMPORTANT: Convert to Python float if it's a tensor
    if torch.is_tensor(ppl_value):
        ppl_value = ppl_value.cpu().item()  # Move to CPU and convert to Python scalar
    elif hasattr(ppl_value, 'item'):
        ppl_value = ppl_value.item()  # Convert numpy or other array types
    else:
        ppl_value = float(ppl_value)  # Ensure it's a float
    
    return ppl_value, debug_df


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-m', '--model_name', help='HF model tag', required=True)
    argparse.add_argument('-o', '--output_dir', help='Output directory', required=True)
    argparse.add_argument('-dv', '--device', help='Device', required=False, default="cuda")
    argparse.add_argument('-d', '--dataset', help='Dataset to use: mteb or mosaic. Default to mteb.', required=False, default="mteb")
    argparse.add_argument('--debug', help='If true, saves per-sentence PPL to ppl_debug.csv', required=False, default=False, type=bool)

    args = argparse.parse_args()
    model_name = args.model_name
    output_dir = args.output_dir
    device = args.device
    dataset = args.dataset
    debug = args.debug

    if dataset == "sib":
        data, text_col = load_sib()
    elif dataset == "mosaic":
        data, text_col  = load_mosaic_data()
    else:
        raise ValueError("Dataset not supported. Choose 'sib' or 'mosaic'.")
    if debug: debug_data = data.copy()
    
    model, tokenizer = load_model(
        model_name, 
        load_in_4bit=True if "mmed" not in model_name.lower() else False, 
        load_in_8bit=False
        )

    for language, dataset in data.items():
        data[language], _ = compute_ppl(model, tokenizer, dataset, text_col, debug)
        if debug:
            _["language"] = language
            debug_data[language] = _

    data = pd.DataFrame.from_dict(data, orient="index", columns=["ppl"])
    model_name = model_name.split("/")[-1]
    # create output dir if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, f"{model_name}.csv"), index=True)

    if debug:
        output_dir = output_dir.replace("outputs", "outputs/ppl_debug")
        os.makedirs(output_dir, exist_ok=True)
        debug_data = pd.concat(debug_data.values(), ignore_index=True)
        debug_data.to_csv(os.path.join(output_dir, f"{model_name}.csv"), index=True)
