import pandas as pd
import random
import os


def generate_negative_samples(ddi_df, num_negatives=None):
    drug_ids = list(set(ddi_df["drug1_id"]).union(set(ddi_df["drug2_id"])))

    positive_pairs = set(zip(ddi_df["drug1_id"], ddi_df["drug2_id"]))

    if num_negatives is None:
        num_negatives = len(ddi_df)

    negative_pairs = set()

    while len(negative_pairs) < num_negatives:
        d1 = random.choice(drug_ids)
        d2 = random.choice(drug_ids)

        if d1 == d2:
            continue

        if (d1, d2) in positive_pairs or (d2, d1) in positive_pairs:
            continue

        negative_pairs.add((d1, d2))

    neg_df = pd.DataFrame(list(negative_pairs), columns=["drug1_id", "drug2_id"])
    neg_df["label"] = 0

    return neg_df


def build_final_dataset(input_file="DATASETS/processed/ddi_pairs.csv",
                        output_file="DATASETS/processed/final_ddi_dataset.csv"):

    ddi_df = pd.read_csv(input_file)

    # Positive samples already label=1
    pos_df = ddi_df.copy()

    # Generate negative samples (same count as positives)
    neg_df = generate_negative_samples(pos_df, num_negatives=len(pos_df))

    final_df = pd.concat([pos_df, neg_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)

    print("✅ Final dataset created successfully!")
    print("Total samples:", len(final_df))
    print("Positive samples:", len(pos_df))
    print("Negative samples:", len(neg_df))
    print("Saved at:", output_file)


if __name__ == "__main__":
    build_final_dataset()