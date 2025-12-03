from datasets import concatenate_datasets, load_from_disk
from transformers import EarlyStoppingCallback
import re, ast, os, yaml

def normalize_wandb_project_name(project_name):
    """Return None when a wandb project should be considered disabled."""
    if project_name is None:
        return None
    if isinstance(project_name, str) and project_name.strip().lower() in {"", "none", "null"}:
        return None
    return project_name

def get_working_dir():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    return working_dir

def load_config(working_dir, config_path):
    working_dir = os.path.dirname(os.path.dirname(working_dir))
    config_path = os.path.join(working_dir, "config/" + config_path)
    if config_path.endswith('.yaml') is False:
        config_path += '.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")

def save_dataset_to_disk(ds, file_path):
    ds.save_to_disk(file_path)
    print(f"Saved dataset (n={len(ds)}) to {file_path}")


def load_dataset(dataset_names, datasets_yaml, split='train'):
    datasets = []
    dataset_names = dataset_names.split(' ')
    
    for dataset_name in dataset_names:
        path = datasets_yaml[dataset_name]['path']
        dataset = load_from_disk(path)
        try: 
            if split != 'train':
                datasets.append(dataset[split])
            else:
                #datasets.append(dataset[split].select(range(0, 80)))
                datasets.append(dataset[split])

        except: print(f'No {split} dataset for {dataset_name}')
    
    dataset = concatenate_datasets(datasets)
    return dataset


def clean_string(text):
    cleaned_text =  " ".join(text.split())
    cleaned_text = re.sub(r"_{2,}", "_", cleaned_text)
    return cleaned_text


def format_report_into_prompt(examples, model_tag, is_test=False, few_shot=False, include_description=False, description_language='en'):
    """ BATCHED = FALSE!"""
    report, labels, classes, findings = examples['report'], examples['labels'], examples['classes'], examples['findings']

    report = clean_string(report)

    request = [
        "You are a helpful radiology assistant. ",
        "Given a radiology report, classify each abnormality into a class. ",
        "Output a valid JSON with each abnormality as key, and the class as value. ",
        f"The keys must be {findings}. ",
        f"The values can be one of {classes}. The values have the following interpretation: ",
        
        # positive mentions
        "(1) the abnormality was positively mentioned in the report; " if 0 in classes 
        else "(1) the abnormality was mentioned, even with uncertainty, in the report " + \
        "e.g. 'A large pleural effusion', 'The cardiac contours are stable.', 'The cardiac size cannot be evaluated.'; ",
        
        # negative mentions
        "(2) the abnormality was negatively mentioned in the report; e.g. 'No pneumothorax.'; " if 2 in classes else "", 
        
        # uncertain mention
        "(0) the abnormality was either: mentioned with uncertainty in the report, " + \
        "or mentioned with ambiguous language in the report and it is unclear " + \
        "if the pathology exists or not, e.g. Explicit uncertainty: 'The cardiac size cannot be evaluated.', " + \
        "Ambiguous language: 'The cardiac contours are stable.';" if 0 in classes else "",
        
        # no mention
        " (-1) the abnormality was not mentioned in the report." if 2 in classes 
        else " (-1) the abnormality was not mentioned in the report, or the abnormality was " + \
        "negatively mentioned in the report; e.g. 'No pneumothorax.'. ",
        ]

    request = ''.join(request)

    description_tag = f'description_{description_language}'
    if include_description and description_tag in examples:
        description = examples[description_tag]
        description = '\nEvery abnormality can be described as:' + str(description)
        request += description

    if few_shot:
        fs_prompt = "\nHere are some examples: "
        fs_prompt += clean_string(examples['fs_examples'])
        request += fs_prompt

    assistant = {
            "role": "assistant",
            "content": f"Answer: json {'' if is_test else labels}"
        }
    
    if "llama" in model_tag.lower():
        system = {
                "role": "system",
                "content": request
            }
        user = {
            "role": "user",
            "content": f"\nText: <<<{report}>>>",
        }
        prompt = [system, user, assistant]

    elif "gemma" in model_tag or "mosaic" in model_tag:
        user = {
            'content': f"{request}\nText: <<<{report}>>>",
            'role': 'user'
            }
        prompt = [user, assistant]

    else: raise ValueError(f"Model {model_tag} not supported.")
    
    return_dict = {"conversations": prompt, 'source': model_tag}
    return return_dict


def check_double_sot(example):
    match="<start_of_turn>model\nAnswer: json<end_of_turn>\n<start_of_turn>model"
    corrected_match="<start_of_turn>model\nAnswer: json"
    result = []
    for text in example["text"]:
        result.append(text.replace(match, corrected_match))
    return {'text':result}


def apply_chat_template_inner(examples, apply_template_fn, fn_kwargs):
    # nested functions can't be hashed properly
    texts = apply_template_fn(examples["conversations"], **fn_kwargs)
    return {"text": texts}


def apply_chat_template_fn(dataset, apply_template_fn, fn_kwargs={}, model_tag=""):
    dataset = dataset.map(
        lambda examples: apply_chat_template_inner(examples, apply_template_fn, fn_kwargs),
        batched=True
    )
    if 'gemma' in model_tag: dataset = dataset.map(check_double_sot, batched=True)
    return dataset


def process_dataset(dataset, tokenizer, model_tag, is_test=False, few_shot=False, include_description=False, description_language='en'):
    apply_template_fn = tokenizer.apply_chat_template
    dataset = dataset.map(
        format_report_into_prompt, batched = False, fn_kwargs={
            'model_tag':model_tag, 'is_test':is_test, 'few_shot':few_shot,
            'include_description':include_description, 'description_language':description_language
            },
        remove_columns = dataset.column_names
    )
    dataset = apply_chat_template_fn(dataset, apply_template_fn, fn_kwargs={'tokenize':False, 'add_generation_prompt':is_test})
    return dataset


def process_dataset_vllm(dataset, llm, model_tag, few_shot=False, include_description=False, description_language='en'):
    apply_template_fn = llm.llm_engine.tokenizer.tokenizer.apply_chat_template
    dataset = dataset.map(
        format_report_into_prompt, batched = False, fn_kwargs={
            'model_tag':model_tag, 'is_test':True, 'few_shot': few_shot,
            'include_description':include_description, 'description_language':description_language
            },
        remove_columns = dataset.column_names
    )
    dataset = apply_chat_template_fn(
        dataset, apply_template_fn, model_tag=model_tag,
        fn_kwargs={'tokenize':False, 'add_generation_prompt':True})
    return dataset


class MinEpochsEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, min_epochs, early_stopping_patience=0, early_stopping_threshold=0.0):
        self.min_epochs = min_epochs
        super().__init__(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            print(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return
        
        if state.epoch >=self.min_epochs:
            self.check_metric_value(args, state, control, metric_value)
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                    control.should_training_stop = True


def decode_output_vllm(response, empty_json={}, idx=None, classes=[-1, 0, 1, 2]):
    invalid_response = lambda x, reason, idx: print(f'Invalid generation, {reason} [{idx}]: {x}') or empty_json

    json_response = re.findall(r'\{[^{}]*\}', response)
    if not json_response: return invalid_response(response, 'JSON not found', idx)
    
    try: output = dict(ast.literal_eval(json_response[-1]))
    except: return invalid_response(response, 'invalid JSON', idx)

    # fix key issues. Add 'not mentioned' as default class for missing keys
    if empty_json.keys() != output.keys(): 
        # add keys that are missing
        output = {**{k:None for k in empty_json.keys() if k not in output}, **output}
        # remove keys that are not in the expected format
        output = {k: v for k, v in output.items() if k in empty_json.keys()}

    classes = [None] + [str(c) for c in classes] + [float(c) for c in classes] + [int(c) for c in classes]
    
    if not all([v in classes for v in output.values()]): # check values
        return invalid_response(response, 'invalid prediction', idx)
    else:
        output = {k: int(v) if v is not None else v for k, v in output.items()}
    
    return output
