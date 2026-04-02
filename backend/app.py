from collections import Counter
import os
import re

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_BUILD_DIR = os.path.join(BASE_DIR, "..", "frontend", "build")

app = Flask(__name__, static_folder=FRONTEND_BUILD_DIR, static_url_path="/static")
CORS(app)

STATIC_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "final_hard_model.pth")
DYNAMIC_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "dynamic_model.pth")
SCALER_STATS_PATH = os.path.join(BASE_DIR, "..", "models", "final_hard_scaler_stats.npz")
DYNAMIC_X_PATH = os.path.join(BASE_DIR, "..", "DATASETS", "processed", "X_dynamic.npy")
DYNAMIC_Y_PATH = os.path.join(BASE_DIR, "..", "DATASETS", "processed", "y_dynamic.npy")
HARD_TRAIN_FEATURES_PATH = os.path.join(
    BASE_DIR, "..", "DATASETS", "processed", "X_hard_chunk_0.npy"
)
FEATURE_FILE_CSV = os.path.join(
    BASE_DIR, "..", "DATASETS", "processed", "unified_drug_features.csv"
)
FEATURE_FILE_GZ = os.path.join(
    BASE_DIR, "..", "DATASETS", "processed", "unified_drug_features.csv.gz"
)
RAW_DRUGBANK_FILE = os.path.join(BASE_DIR, "..", "DATASETS", "raw", "drugbank_clean.csv")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "..", "DATASETS", "processed", "drug_embeddings.pt")

STATIC_MODEL_DISPLAY_PATH = "models/final_hard_model.pth"
DYNAMIC_MODEL_DISPLAY_PATH = "models/dynamic_model.pth"
SCALER_STATS_DISPLAY_PATH = "models/final_hard_scaler_stats.npz"
EMBEDDINGS_DISPLAY_PATH = "DATASETS/processed/drug_embeddings.pt"

MIN_DYNAMIC_TRAIN_SAMPLES = 8
AUTO_DYNAMIC_RETRAIN_THRESHOLD = 100


def resolve_feature_file():
    if os.path.exists(FEATURE_FILE_CSV):
        return FEATURE_FILE_CSV
    if os.path.exists(FEATURE_FILE_GZ):
        return FEATURE_FILE_GZ
    raise FileNotFoundError(
        "Could not find unified drug feature file. Expected one of:\n"
        f"- {FEATURE_FILE_CSV}\n"
        f"- {FEATURE_FILE_GZ}"
    )


print("Loading drug feature mapping...")

FEATURE_FILE = resolve_feature_file()
features_df = pd.read_csv(FEATURE_FILE)
print(f"Feature source: {os.path.relpath(FEATURE_FILE, os.path.join(BASE_DIR, '..'))}")
valid_rows = features_df[features_df["drug_id"].notna()].copy()
valid_rows["drug_id"] = valid_rows["drug_id"].astype(str).str.strip()

feature_vectors = valid_rows.drop(columns=["drug_id"]).values.astype(np.float32)
feature_ids = valid_rows["drug_id"].tolist()

id_to_index = {}
for idx, drug_id in enumerate(feature_ids):
    id_to_index[drug_id] = idx
    id_to_index[drug_id.lower()] = idx

raw_drugbank_df = pd.read_csv(RAW_DRUGBANK_FILE, low_memory=False).fillna("")
raw_drugbank_df["drugbank-id"] = raw_drugbank_df["drugbank-id"].astype(str).str.strip()
raw_drugbank_df["name"] = raw_drugbank_df["name"].astype(str).str.strip()

for column in [
    "description",
    "toxicity",
    "mechanism-of-action",
    "pharmacodynamics",
    "indication",
    "absorption",
]:
    if column not in raw_drugbank_df.columns:
        raw_drugbank_df[column] = ""

# Some DrugBank IDs appear more than once; keep the row with the richest textual content.
side_effect_source_columns = [
    "description",
    "toxicity",
    "mechanism-of-action",
    "pharmacodynamics",
    "indication",
    "absorption",
]
raw_drugbank_df["_content_score"] = raw_drugbank_df[side_effect_source_columns].astype(str).apply(
    lambda col: col.str.strip().str.len()
).sum(axis=1)

drugbank_by_id = (
    raw_drugbank_df.sort_values(["drugbank-id", "_content_score"], ascending=[True, False])
    .drop_duplicates(subset=["drugbank-id"], keep="first")
    .drop(columns=["_content_score"])
    .set_index("drugbank-id")
)
gnn_embeddings = torch.load(EMBEDDINGS_PATH, map_location="cpu")
gnn_embedding_dim = int(gnn_embeddings.shape[1])

print(f"Loaded {len(id_to_index) // 2} drug feature vectors")

name_to_id = {}
alias_to_id = {}
drug_search_rows = []

SIDE_EFFECT_KEYWORDS = [
    "nausea",
    "vomiting",
    "headache",
    "dizziness",
    "fatigue",
    "diarrhea",
    "constipation",
    "rash",
    "itching",
    "fever",
    "insomnia",
    "drowsiness",
    "bleeding",
    "hemorrhage",
    "anemia",
    "hypertension",
    "hypotension",
    "bradycardia",
    "tachycardia",
    "seizure",
    "convulsion",
    "kidney failure",
    "renal failure",
    "liver damage",
    "hepatotoxicity",
    "stomach pain",
    "gastric ulcer",
    "ulcer",
    "heart attack",
    "stroke",
    "shortness of breath",
    "respiratory depression",
    "hallucination",
    "confusion",
    "depression",
    "anxiety",
    "dry mouth",
    "blurred vision",
    "arrhythmia",
    "chest pain",
    "swelling",
    "edema",
]


def normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def clean_text(text):
    text = str(text).lower()
    if text == "nan":
        return ""
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_side_effects_from_text(text):
    cleaned = clean_text(text)
    if not cleaned:
        return []

    found = []
    for keyword in SIDE_EFFECT_KEYWORDS:
        if keyword in cleaned:
            found.append(keyword)
    return found


def extract_side_effects_full(row):
    combined_text = " ".join(
        str(row.get(column, ""))
        for column in [
            "toxicity",
            "description",
            "pharmacodynamics",
            "mechanism-of-action",
            "indication",
            "absorption",
        ]
    )

    effects = extract_side_effects_from_text(combined_text)
    if not effects:
        words = clean_text(combined_text).split()
        common_words = [word for word in words if len(word) > 6]
        effects = [word for word, _ in Counter(common_words).most_common(10)]

    return sorted(list(dict.fromkeys(effects)))


def build_side_effect_summary(drug_id):
    if drug_id not in drugbank_by_id.index:
        return {"drug_id": drug_id, "name": None, "effects": []}

    row = drugbank_by_id.loc[drug_id]
    return {
        "drug_id": drug_id,
        "name": row.get("name", ""),
        "effects": extract_side_effects_full(row),
    }


def build_overlap_summary(drug1_summary, drug2_summary):
    overlap = sorted(set(drug1_summary["effects"]).intersection(drug2_summary["effects"]))
    return {
        "shared_effects": overlap,
        "shared_effect_count": len(overlap),
    }


for _, row in raw_drugbank_df.iterrows():
    drug_id = row["drugbank-id"]
    name = row["name"]
    if name:
        normalized = normalize_text(name)
        if normalized and normalized not in name_to_id:
            name_to_id[normalized] = drug_id
        if normalized and drug_id in id_to_index:
            drug_search_rows.append(
                {
                    "drug_id": drug_id,
                    "name": name,
                    "normalized_name": normalized,
                }
            )

    description = str(row.get("description", ""))
    for alias in re.findall(r"also known as _([^_]+)_", description, flags=re.IGNORECASE):
        normalized_alias = normalize_text(alias)
        if normalized_alias and normalized_alias not in alias_to_id:
            alias_to_id[normalized_alias] = drug_id

deduped_search_rows = []
seen_search_keys = set()

for row in sorted(drug_search_rows, key=lambda item: item["name"].lower()):
    key = (row["drug_id"], row["normalized_name"])
    if key in seen_search_keys:
        continue
    seen_search_keys.add(key)
    deduped_search_rows.append(row)

drug_search_rows = deduped_search_rows

print(f"Loaded {len(name_to_id)} raw drug names and {len(alias_to_id)} aliases")


def resolve_drug_id(value):
    if value is None:
        return None

    token = str(value).strip()
    if not token:
        return None

    if token in id_to_index:
        return token

    lower_token = token.lower()
    if lower_token in id_to_index:
        return feature_ids[id_to_index[lower_token]]

    normalized = normalize_text(token)
    if normalized in name_to_id:
        return name_to_id[normalized]
    if normalized in alias_to_id:
        return alias_to_id[normalized]

    partial_matches = set()
    for key, drug_id in list(name_to_id.items()) + list(alias_to_id.items()):
        if normalized in key:
            partial_matches.add(drug_id)
    if len(partial_matches) == 1:
        return partial_matches.pop()

    return None


def build_pair_features(drug1_id, drug2_id):
    if drug1_id not in id_to_index or drug2_id not in id_to_index:
        return None

    f1 = feature_vectors[id_to_index[drug1_id]]
    f2 = feature_vectors[id_to_index[drug2_id]]
    return np.concatenate([f1, f2]).astype(np.float32)


class AdaptiveDDIModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        return torch.sigmoid(self.out(x))


print("Loading static base model...")

static_state_dict = torch.load(STATIC_MODEL_PATH, map_location="cpu")
input_dim = int(static_state_dict["fc1.weight"].shape[1])

static_model = AdaptiveDDIModel(input_dim)
static_model.load_state_dict(static_state_dict)
static_model.eval()

print("Static base model loaded")


feature_scaler_warning = None


def compute_scaler_stats_from_training_chunk(expected_dim):
    X = np.load(HARD_TRAIN_FEATURES_PATH, mmap_mode="r")
    if X.ndim != 2 or X.shape[1] != expected_dim:
        raise ValueError(
            f"Training feature chunk shape {X.shape} is incompatible with expected input dim {expected_dim}."
        )

    mean = np.asarray(X.mean(axis=0), dtype=np.float32)
    scale = np.asarray(X.std(axis=0), dtype=np.float32)
    scale[scale == 0] = 1.0

    np.savez(SCALER_STATS_PATH, mean=mean, scale=scale)
    return mean, scale, "computed_from_X_hard_chunk_0"


def load_feature_scaler(expected_dim):
    global feature_scaler_warning

    if os.path.exists(SCALER_STATS_PATH):
        stats = np.load(SCALER_STATS_PATH)
        mean = np.asarray(stats["mean"], dtype=np.float32)
        scale = np.asarray(stats["scale"], dtype=np.float32)
        if mean.shape == (expected_dim,) and scale.shape == (expected_dim,):
            scale[scale == 0] = 1.0
            return mean, scale, "saved_scaler_stats"

    if os.path.exists(HARD_TRAIN_FEATURES_PATH):
        return compute_scaler_stats_from_training_chunk(expected_dim)

    feature_scaler_warning = "Feature scaler stats could not be loaded; using identity scaling."
    return (
        np.zeros((expected_dim,), dtype=np.float32),
        np.ones((expected_dim,), dtype=np.float32),
        "identity_fallback",
    )


feature_scaler_mean, feature_scaler_scale, feature_scaler_source = load_feature_scaler(input_dim)
print(f"Feature scaler ready ({feature_scaler_source})")


dynamic_dataset_warning = None


def save_empty_dynamic_dataset(expected_dim):
    np.save(DYNAMIC_X_PATH, np.empty((0, expected_dim), dtype=np.float32))
    np.save(DYNAMIC_Y_PATH, np.empty((0,), dtype=np.float32))


def prepare_dynamic_dataset_storage(expected_dim):
    global dynamic_dataset_warning

    if not os.path.exists(DYNAMIC_X_PATH) or not os.path.exists(DYNAMIC_Y_PATH):
        save_empty_dynamic_dataset(expected_dim)
        return True

    X = np.load(DYNAMIC_X_PATH, mmap_mode="r")
    y = np.load(DYNAMIC_Y_PATH, mmap_mode="r")

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        if X.shape[0] == 0 and y.shape[0] == 0:
            save_empty_dynamic_dataset(expected_dim)
            return True

        dynamic_dataset_warning = (
            "Dynamic dataset files are not aligned with each other and were not loaded."
        )
        return False

    if X.shape[1] != expected_dim:
        if X.shape[0] == 0:
            save_empty_dynamic_dataset(expected_dim)
            return True

        dynamic_dataset_warning = (
            f"Dynamic dataset feature dimension {X.shape[1]} does not match current model input "
            f"dimension {expected_dim}."
        )
        return False

    return True


dynamic_dataset_ready = prepare_dynamic_dataset_storage(input_dim)


def get_dynamic_dataset():
    if not dynamic_dataset_ready:
        return None, None

    X = np.load(DYNAMIC_X_PATH)
    y = np.load(DYNAMIC_Y_PATH)
    return X.astype(np.float32), y.astype(np.float32)


def get_dynamic_sample_count():
    if not dynamic_dataset_ready:
        return 0

    X = np.load(DYNAMIC_X_PATH, mmap_mode="r")
    return int(X.shape[0])


def dynamic_state_dict_is_compatible(state_dict):
    required_keys = {
        "fc1.weight",
        "fc1.bias",
        "bn1.weight",
        "bn1.bias",
        "fc2.weight",
        "fc2.bias",
        "bn2.weight",
        "bn2.bias",
        "fc3.weight",
        "fc3.bias",
        "bn3.weight",
        "bn3.bias",
        "out.weight",
        "out.bias",
    }

    if not required_keys.issubset(state_dict.keys()):
        return False

    return tuple(state_dict["fc1.weight"].shape) == (512, input_dim)


def bootstrap_dynamic_model():
    model = AdaptiveDDIModel(input_dim)
    model.load_state_dict(static_model.state_dict())
    model.eval()
    return model


def load_dynamic_model():
    if os.path.exists(DYNAMIC_MODEL_PATH):
        candidate_state_dict = torch.load(DYNAMIC_MODEL_PATH, map_location="cpu")
        if dynamic_state_dict_is_compatible(candidate_state_dict):
            model = AdaptiveDDIModel(input_dim)
            model.load_state_dict(candidate_state_dict)
            model.eval()
            return model, "dynamic_checkpoint"

    return bootstrap_dynamic_model(), "bootstrapped_from_static"


dynamic_model, dynamic_model_state = load_dynamic_model()

print(f"Dynamic serving model ready ({dynamic_model_state})")


def build_gnn_summary():
    return {
        "encoder": "GAT",
        "embedding_source": EMBEDDINGS_DISPLAY_PATH,
        "embedding_dimension": gnn_embedding_dim,
        "per_drug_feature_dimension": int(feature_vectors.shape[1]),
        "pair_feature_dimension": int(input_dim),
        "note": "Predictions use unified drug features that include GAT-derived graph embeddings.",
    }


def build_model_summary():
    sample_count = get_dynamic_sample_count()
    return {
        "active_model": "dynamic",
        "model_state": dynamic_model_state,
        "architecture": "AdaptiveDDIModel",
        "checkpoint_path": DYNAMIC_MODEL_DISPLAY_PATH,
        "bootstrap_source": (
            STATIC_MODEL_DISPLAY_PATH if dynamic_model_state == "bootstrapped_from_static" else None
        ),
        "feature_dimension": int(input_dim),
        "dynamic_feedback_samples": sample_count,
        "dataset_status": "ready" if dynamic_dataset_ready else "unavailable",
        "dataset_warning": dynamic_dataset_warning,
        "min_samples_for_retrain": MIN_DYNAMIC_TRAIN_SAMPLES,
        "auto_retrain_threshold": AUTO_DYNAMIC_RETRAIN_THRESHOLD,
        "ready_for_retrain": sample_count >= MIN_DYNAMIC_TRAIN_SAMPLES,
        "scaler_path": SCALER_STATS_DISPLAY_PATH,
        "scaler_source": feature_scaler_source,
        "scaler_warning": feature_scaler_warning,
    }


def scale_feature_batch(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        return (features - feature_scaler_mean) / feature_scaler_scale
    return (features - feature_scaler_mean.reshape(1, -1)) / feature_scaler_scale.reshape(1, -1)


def predict_probability(features):
    scaled_features = scale_feature_batch(features)
    X_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    if X_tensor.ndim == 1:
        X_tensor = X_tensor.view(1, -1)

    with torch.no_grad():
        return float(dynamic_model(X_tensor).item())


def append_dynamic_sample(pair_features, label):
    if not dynamic_dataset_ready:
        raise ValueError(dynamic_dataset_warning or "Dynamic dataset storage is unavailable.")

    X_old, y_old = get_dynamic_dataset()
    new_row = scale_feature_batch(pair_features).astype(np.float32).reshape(1, -1)
    new_label = np.asarray([float(label)], dtype=np.float32)

    if X_old.size == 0:
        X_new = new_row
        y_new = new_label
    else:
        X_new = np.vstack([X_old, new_row])
        y_new = np.concatenate([y_old, new_label])

    np.save(DYNAMIC_X_PATH, X_new.astype(np.float32))
    np.save(DYNAMIC_Y_PATH, y_new.astype(np.float32))


def train_dynamic_model(epochs=6, lr=0.0005):
    global dynamic_model, dynamic_model_state

    X, y = get_dynamic_dataset()
    if X is None or y is None:
        raise ValueError(dynamic_dataset_warning or "Dynamic dataset storage is unavailable.")

    sample_count = len(X)
    if sample_count < MIN_DYNAMIC_TRAIN_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_DYNAMIC_TRAIN_SAMPLES} feedback samples before retraining."
        )

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dynamic_model.train()
    optimizer = optim.Adam(dynamic_model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = dynamic_model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    dynamic_model.eval()
    torch.save(dynamic_model.state_dict(), DYNAMIC_MODEL_PATH)
    dynamic_model_state = "dynamic_checkpoint"

    return {
        "epochs": epochs,
        "learning_rate": lr,
        "samples": sample_count,
        "final_loss": history[-1],
        "loss_history": history,
    }


def parse_binary_label(value):
    if isinstance(value, bool):
        return int(value)

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    if parsed not in (0, 1):
        return None

    return parsed


def resolve_feedback_drugs(data):
    drug1_id = data.get("drug1_id") or resolve_drug_id(data.get("drug1"))
    drug2_id = data.get("drug2_id") or resolve_drug_id(data.get("drug2"))
    return drug1_id, drug2_id


def risk_label(probability):
    return "High Risk" if probability > 0.5 else "Low Risk"


@app.route("/drugs/search", methods=["GET"])
def search_drugs():
    query = request.args.get("q", "").strip()

    try:
        limit = int(request.args.get("limit", 8))
    except ValueError:
        limit = 8

    limit = max(1, min(limit, 20))

    if not query:
        results = drug_search_rows[:limit]
    else:
        normalized_query = normalize_text(query)

        prefix_matches = []
        contains_matches = []
        alias_matches = []

        for row in drug_search_rows:
            normalized_name = row["normalized_name"]
            if normalized_name.startswith(normalized_query):
                prefix_matches.append(row)
            elif normalized_query in normalized_name:
                contains_matches.append(row)

        for alias, drug_id in alias_to_id.items():
            if not (alias.startswith(normalized_query) or normalized_query in alias):
                continue
            if drug_id not in drugbank_by_id.index:
                continue
            alias_matches.append(
                {
                    "drug_id": drug_id,
                    "name": drugbank_by_id.loc[drug_id]["name"],
                    "normalized_name": normalize_text(drugbank_by_id.loc[drug_id]["name"]),
                }
            )

        deduped_results = []
        seen_ids = set()

        for row in prefix_matches + contains_matches + alias_matches:
            if row["drug_id"] in seen_ids:
                continue
            seen_ids.add(row["drug_id"])
            deduped_results.append(row)

        results = deduped_results[:limit]

    return jsonify(
        {
            "results": [
                {
                    "drug_id": row["drug_id"],
                    "name": row["name"],
                }
                for row in results
            ]
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    features = data.get("features")
    if features is not None:
        features = np.array(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2 or features.shape[1] != input_dim:
            return jsonify(
                {
                    "error": f"Expected feature vector length {input_dim}, got shape {features.shape}.",
                }
            ), 400

        probability = predict_probability(features)
        return jsonify(
            {
                "probability": probability,
                "risk": risk_label(probability),
                "gnn": build_gnn_summary(),
                "model": build_model_summary(),
                "input_mode": "features",
            }
        )

    drug1 = data.get("drug1")
    drug2 = data.get("drug2")
    if drug1 is None or drug2 is None:
        return jsonify(
            {
                "error": "Provide either 'features' or both 'drug1' and 'drug2' in JSON.",
            }
        ), 400

    drug1_id = resolve_drug_id(drug1)
    drug2_id = resolve_drug_id(drug2)

    missing = []
    if drug1_id is None or drug1_id not in id_to_index:
        missing.append(drug1)
    if drug2_id is None or drug2_id not in id_to_index:
        missing.append(drug2)

    if missing:
        return jsonify(
            {
                "error": "Drug name or ID not found in feature mapping.",
                "missing": missing,
            }
        ), 400

    pair_features = build_pair_features(drug1_id, drug2_id)
    probability = predict_probability(pair_features)

    drug1_summary = build_side_effect_summary(drug1_id)
    drug2_summary = build_side_effect_summary(drug2_id)

    return jsonify(
        {
            "probability": probability,
            "risk": risk_label(probability),
            "drug1_id": drug1_id,
            "drug2_id": drug2_id,
            "drug1_name": drug1_summary["name"],
            "drug2_name": drug2_summary["name"],
            "input_mode": "drug_lookup",
            "side_effects": {
                "drug1": drug1_summary,
                "drug2": drug2_summary,
                "overlap": build_overlap_summary(drug1_summary, drug2_summary),
            },
            "gnn": build_gnn_summary(),
            "model": build_model_summary(),
        }
    )


@app.route("/dynamic/status", methods=["GET"])
def dynamic_status():
    return jsonify(build_model_summary())


@app.route("/dynamic/feedback", methods=["POST"])
def dynamic_feedback():
    data = request.json
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    label = parse_binary_label(data.get("label"))
    if label is None:
        return jsonify({"error": "Label must be 0 or 1."}), 400

    drug1_id, drug2_id = resolve_feedback_drugs(data)
    if drug1_id is None or drug2_id is None:
        return jsonify(
            {"error": "Could not resolve drug1/drug2 to valid DrugBank IDs."}
        ), 400

    pair_features = build_pair_features(drug1_id, drug2_id)
    if pair_features is None:
        return jsonify({"error": "Could not build pair features for the provided drugs."}), 400

    append_dynamic_sample(pair_features, label)

    retrained = False
    training_result = None
    auto_retrain = bool(data.get("auto_retrain", False))
    sample_count = get_dynamic_sample_count()

    if auto_retrain and sample_count >= AUTO_DYNAMIC_RETRAIN_THRESHOLD:
        training_result = train_dynamic_model()
        retrained = True

    return jsonify(
        {
            "message": "Feedback sample stored in the dynamic dataset.",
            "stored_sample": {
                "drug1_id": drug1_id,
                "drug2_id": drug2_id,
                "label": label,
            },
            "retrained": retrained,
            "training_result": training_result,
            "model": build_model_summary(),
        }
    )


@app.route("/dynamic/retrain", methods=["POST"])
def dynamic_retrain():
    data = request.json or {}
    epochs = int(data.get("epochs", 6))
    lr = float(data.get("learning_rate", 0.0005))

    epochs = max(1, min(50, epochs))
    lr = max(1e-5, min(1e-2, lr))

    try:
        training_result = train_dynamic_model(epochs=epochs, lr=lr)
    except ValueError as exc:
        return jsonify({"error": str(exc), "model": build_model_summary()}), 400

    return jsonify(
        {
            "message": "Dynamic model retrained successfully.",
            "training_result": training_result,
            "model": build_model_summary(),
        }
    )


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    first_segment = path.split("/", 1)[0] if path else ""
    if first_segment in {"predict", "drugs", "dynamic"}:
        return jsonify({"error": "Not found"}), 404

    if path and app.static_folder and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)

    index_path = os.path.join(app.static_folder or "", "index.html")
    if app.static_folder and os.path.exists(index_path):
        return send_from_directory(app.static_folder, "index.html")

    return jsonify(
        {"error": "Frontend build not found. Run `npm run build` in frontend/ before starting backend."}
    ), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting backend on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
