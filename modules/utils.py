import yaml
from typing import Dict


piece_to_index: Dict[str, int] = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
}


def load_config(config_path: str) -> Dict:
    with open(config_path, mode="r") as f:
        config = yaml.safe_load(f)

    return config
