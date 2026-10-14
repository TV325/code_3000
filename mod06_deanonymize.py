import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    linked_dataframe = pd.merge(anon_df, aux_df, how='inner', on=["age", "zip3", "gender"])[["anon_id", "name"]]
    linked_dataframe = linked_dataframe.rename(columns={"name": "matched_name"})
    linked_dataframe.drop_duplicates()
    linked_dataframe = linked_dataframe[~linked_dataframe.duplicated(subset="anon_id", keep=False)]
    return linked_dataframe.reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df)  / len(anon_df)
