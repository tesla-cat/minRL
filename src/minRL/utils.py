import json
from typing import Dict, List

import matplotlib.pyplot as plt


class Recorder:
    data: Dict[str, Dict[str, List]]

    def __init__(s):
        s.path = "recorder.json"
        try:
            with open(s.path) as f:
                s.data = json.load(f)
        except Exception:
            s.data = {}

    def add(s, id, row, info={}):
        if id not in s.data:
            s.data[id] = {"info": info, "rows": []}
        s.data[id]["rows"].append(row)

    def save(s):
        with open(s.path, "w+") as f:
            json.dump(s.data, f)
        s.plot()

    def plot(s):
        data2: Dict[str, Dict] = {}
        for id, x in s.data.items():
            src, env = id.split("|")
            if env not in data2:
                data2[env] = {}
            data2[env][src] = x
        plt.figure(figsize=(8, 4 * len(data2)))
        idx = 0
        for env, v1 in data2.items():
            idx += 1
            plt.subplot(len(data2), 1, idx)
            plt.title(env)
            for src, v2 in v1.items():
                R = [r[0] for r in v2["rows"]]
                plt.plot(R, label=src)
            plt.legend()
        plt.tight_layout()
        plt.savefig("recorder.png")
