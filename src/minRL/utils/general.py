from typing import Dict


def get_vals(dic: Dict, keys: str):
    return [dic[k.strip()] for k in keys.split(",")]
