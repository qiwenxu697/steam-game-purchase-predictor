import json
import ast
import pandas as pd

def load_user_items(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            d = ast.literal_eval(line)
            uid = d["user_id"]
            for item in d["items"]:
                rows.append({
                    "user_id": uid,
                    "item_id": str(item["item_id"]),
                    "playtime_forever": item.get("playtime_forever", 0)
                })
    return pd.DataFrame(rows)


def load_metadata(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            d = ast.literal_eval(line)
            rows.append(d)
    return pd.DataFrame(rows)
