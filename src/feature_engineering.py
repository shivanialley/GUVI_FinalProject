def add_features(df):
    df["balance_age"] = df["balance"] / (df["age"] + 1)
    df["campaign_ratio"] = df["campaign"] / (df["previous"] + 1)
    return df
