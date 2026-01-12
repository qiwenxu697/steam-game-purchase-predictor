from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def add_genre_features(df, genre_col="genres_clean"):
    mlb = MultiLabelBinarizer()
    genre_ohe = mlb.fit_transform(df[genre_col])
    genre_df = pd.DataFrame(
        genre_ohe,
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index,
    )
    return pd.concat([df, genre_df], axis=1)


def build_user_genre_preferences(user_items):
    exploded = user_items.explode("genres_clean")
    counts = exploded.groupby(["user_id", "genres_clean"]).size().unstack(fill_value=0)
    prefs = counts.div(counts.sum(axis=1), axis=0)
    prefs.columns = [f"user_pref_genre_{c}" for c in prefs.columns]
    return prefs.reset_index()
