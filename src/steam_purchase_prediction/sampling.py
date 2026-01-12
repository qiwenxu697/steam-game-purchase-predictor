from __future__ import annotations

import random
import pandas as pd


def make_negative_samples(
    positives: pd.DataFrame,
    all_item_ids: set[str],
    user_col: str = "user_id",
    item_col: str = "item_id",
    label_col: str = "label",
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each user, sample the same number of negative items as positives.
    positives must contain user_id + item_id.
    """
    rng = random.Random(seed)
    neg_rows = []

    for uid, group in positives.groupby(user_col):
        bought = set(group[item_col].astype(str))
        not_bought = list(all_item_ids - bought)

        if len(not_bought) < len(bought):
            continue

        sampled = rng.sample(not_bought, len(bought))
        for it in sampled:
            neg_rows.append({user_col: uid, item_col: str(it), label_col: 0})

    return pd.DataFrame(neg_rows)

