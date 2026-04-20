from pathlib import Path
from typing import Dict
import yaml

from model_functions import detect

CONFIG = None

def load_config(config_path: str = "config.yaml") -> Dict:
    global CONFIG
    try:
        with open(config_path, 'r') as f:
            CONFIG = yaml.safe_load(f)
        return CONFIG
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default parameters...")
        CONFIG = {}
        return CONFIG

def get_config() -> Dict:
    global CONFIG
    if CONFIG is None:
        load_config()
    return CONFIG or {}


def find_image_path(filename: str = "test.jpg") -> str:
    possible_paths = [
        filename,
        f"data/{filename}",
        f"../data/{filename}",
        f"../../data/{filename}",
    ]
    for path in possible_paths:
        if Path(path).exists():
            return path
    return "data/test.jpg"