import os
import yaml

def get_working_dir() -> str:
    """
    Returns:
        str: The directory of the file which calls this function.
    """
    return os.path.dirname(os.path.realpath(__file__))


def load_config(working_dir: str, yaml_file: str) -> dict:
    """
    Loads the configurations from a YAML file.

    Args:
        working_dir (str): The working directory.
        yaml_file (str): The name of the YAML file.

    Returns:
        dict: The configurations.

    Raises:
        Exception: If the YAML file is invalid.
    """
    with open(working_dir + '/config/' + yaml_file, 'r') as f:
        try: configs = yaml.safe_load(f)
        except: raise Exception('Invalid YAML file')
    return configs


def load_dataset(dataset_names, datasets_yaml, split='train'):
    datasets = []
    dataset_names = dataset_names.split(' ')
    
    for dataset_name in dataset_names:
        path = datasets_yaml[dataset_name]['path']
        dataset = load_from_disk(path)
        try: datasets.append(dataset[split])
        except: print(f'No {split} dataset for {dataset_name}')
    
    dataset = concatenate_datasets(datasets)
    return dataset


