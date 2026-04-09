import os
import subprocess


def run_script(script_name):
    print("\n" + "=" * 60)
    print(f"🚀 Running: {script_name}")
    print("=" * 60)

    result = subprocess.run(["python3", script_name])

    if result.returncode != 0:
        print(f"❌ Error occurred in {script_name}")
        exit(1)

    print(f"✅ Finished: {script_name}")


def main():
    print("\n==============================")
    print("   GraphRxInsight Pipeline")
    print("==============================\n")

    # Step 1: Extract DDI pairs from DrugBank
    run_script("src/extract_ddi.py")

    # Step 2: Build final dataset with negatives
    run_script("src/build_training_data.py")

    # Step 3: Build drug numeric features
    run_script("src/build_drug_features.py")

    # Step 4: Build Graph for PyTorch Geometric
    run_script("src/build_graph_data.py")

    # Step 5: Train GAT embeddings
    run_script("src/train_gat_embeddings.py")

    # Step 6: Train baseline DDI classifier (GAT embeddings only)
    run_script("src/train_ddi_classifier.py")

    # Step 7: Build ATC multi-hot features
    run_script("src/build_atc_features.py")

    # Step 8: Reduce ATC using PCA
    run_script("src/atc_pca.py")

    # Step 9: Train CLINENSEMBLE PCA model
    run_script("src/train_clinensemble.py")

    # Step 10: Evaluate baseline DDI classifier
    run_script("src/evaluate_ddi_classifier.py")

    print("\n==============================")
    print("✅ FULL PIPELINE COMPLETED!")
    print("Models saved in: models/")
    print("Results saved in: results/")
    print("==============================\n")


if __name__ == "__main__":
    main()