from __future__ import annotations

import argparse
import pandas as pd

from steam_recsys.io import load_user_items, load_metadata
from steam_recsys.sampling import make_negative_samples
from steam_recsys.preprocess import coerce_price, add_user_game_count, to_list_or_empty
from steam_recsys.features import add_genre_ohe
from steam_recsys.models import train_and_eval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user-items", required=True)
    p.add_argument("--metadata", required=True)
    args = p.parse_args()

    user_items = load_user_items(args.user_items)
    metadata = load_metadata(args.metadata)

    # positives
    pos = user_items.merge(metadata, left_on="item_id", right_on="id", how="left")
    pos["label"] = 1

    # negatives
    all_games = set(metadata["id"].astype(str))
    neg = make_negative_samples(pos[["user_id", "item_id", "label"]], all_games)

    neg = neg.merge(metadata, left_on="item_id", right_on="id", how="left")

    full_df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)

    full_df = add_user_game_count(full_df, user_items)
    if "price" in full_df.columns:
        full_df["price"] = coerce_price(full_df["price"])

    # genres -> one-hot
    if "genres" in full_df.columns:
        full_df["genres_clean"] = full_df["genres"].apply(to_list_or_empty)
        full_df = add_genre_ohe(full_df, col="genres_clean", join_key="item_id")

    train_and_eval(full_df)


if __name__ == "__main__":
    main()
