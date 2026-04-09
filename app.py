import streamlit as st
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import re
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="GraphRxInsight", layout="wide")


# ============================================================
# MEDICAL UI THEME (CSS)
# ============================================================
st.markdown("""
<style>
html, body, [class*="css"]  {
    scroll-behavior: smooth;
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background: linear-gradient(to bottom, #f7fbff, #ffffff);
}

h1, h2, h3 {
    color: #0b2c4a;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #eaf4ff, #ffffff);
    border-right: 1px solid #d6eaff;
}

.stButton button {
    background: linear-gradient(to right, #0077ff, #005ecb);
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.6rem 1rem;
    border: none;
    transition: 0.3s ease;
}

.stButton button:hover {
    background: linear-gradient(to right, #005ecb, #004ba3);
    transform: scale(1.02);
}

div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 14px;
    border: 1px solid #e6f0ff;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    transition: 0.3s ease-in-out;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
}

div[data-testid="stMetricValue"] {
    color: #0b2c4a !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

div[data-testid="stMetricLabel"] {
    color: #1a4b78 !important;
    font-weight: 600 !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    border: 1px solid #e6f0ff;
    overflow: hidden;
}

.card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    border: 1px solid #e6f0ff;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
    transition: 0.3s ease-in-out;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0px 8px 22px rgba(0,0,0,0.10);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL DEFINITION
# ============================================================
class CLINENSEMBLE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


# ============================================================
# LOAD FUNCTIONS
# ============================================================
@st.cache_data
def load_drugbank_data():
    df = pd.read_csv("DATASETS/raw/drugbank_clean.csv", low_memory=False)

    required_cols = [
        "name", "drugbank-id", "toxicity", "mechanism-of-action",
        "description", "pharmacodynamics", "indication", "absorption",
        "groups", "state"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("")
    return df


@st.cache_data
def load_drug_features():
    return pd.read_csv("DATASETS/processed/drug_features.csv")


@st.cache_data
def load_atc_pca_features():
    df = pd.read_csv("DATASETS/processed/atc_pca_features.csv")
    df = df.set_index("drug_id")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


@st.cache_resource
def load_embeddings():
    emb = torch.load("DATASETS/processed/drug_embeddings.pt")
    return emb.numpy()


@st.cache_resource
def load_model(input_dim):
    model = CLINENSEMBLE(input_dim=input_dim)
    model.load_state_dict(torch.load("models/clinensemble_pca_model.pth", map_location="cpu"))
    model.eval()
    return model


# ============================================================
# SIDE EFFECTS FUNCTIONS
# ============================================================
SIDE_EFFECT_KEYWORDS = [
    "nausea", "vomiting", "headache", "dizziness", "fatigue", "diarrhea",
    "constipation", "rash", "itching", "fever", "insomnia", "drowsiness",
    "bleeding", "hemorrhage", "anemia", "hypertension", "hypotension",
    "bradycardia", "tachycardia", "seizure", "convulsion",
    "kidney failure", "renal failure", "liver damage", "hepatotoxicity",
    "stomach pain", "gastric ulcer", "ulcer", "heart attack", "stroke",
    "shortness of breath", "respiratory depression", "hallucination",
    "confusion", "depression", "anxiety", "dry mouth", "blurred vision",
    "arrhythmia", "chest pain", "swelling", "edema"
]


def clean_text(text):
    text = str(text).lower()
    if text == "nan":
        return ""
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_side_effects_from_text(text):
    text = clean_text(text)
    if text.strip() == "":
        return []
    found = []
    for keyword in SIDE_EFFECT_KEYWORDS:
        if keyword in text:
            found.append(keyword)
    return found


def extract_side_effects_full(row):
    combined_text = (
        str(row.get("toxicity", "")) + " " +
        str(row.get("description", "")) + " " +
        str(row.get("pharmacodynamics", "")) + " " +
        str(row.get("mechanism-of-action", "")) + " " +
        str(row.get("indication", "")) + " " +
        str(row.get("absorption", ""))
    )

    effects = extract_side_effects_from_text(combined_text)

    if len(effects) == 0:
        words = clean_text(combined_text).split()
        common_words = [w for w in words if len(w) > 6]
        most_common = Counter(common_words).most_common(10)
        effects = [w[0] for w in most_common]

    return sorted(list(set(effects)))


def combined_side_effect_analysis(drug_side_effect_map):
    all_effects = []
    for effects in drug_side_effect_map.values():
        all_effects.extend(effects)

    freq = Counter(all_effects)
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# RISK FUNCTION
# ============================================================
def risk_level(prob):
    if prob >= 0.85:
        return "🔴 Severe Risk"
    elif prob >= 0.70:
        return "🟠 High Risk"
    elif prob >= 0.50:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"


# ============================================================
# PREDICT FUNCTION
# ============================================================
def predict_interaction(drugA_id, drugB_id, embeddings, atc_df, drug_to_idx, model):
    if drugA_id not in drug_to_idx or drugB_id not in drug_to_idx:
        return None

    if drugA_id not in atc_df.index or drugB_id not in atc_df.index:
        return None

    emb1 = np.array(embeddings[drug_to_idx[drugA_id]]).reshape(-1)
    emb2 = np.array(embeddings[drug_to_idx[drugB_id]]).reshape(-1)

    atc1 = np.array(atc_df.loc[drugA_id].values).reshape(-1)
    atc2 = np.array(atc_df.loc[drugB_id].values).reshape(-1)

    feat1 = np.concatenate([emb1, atc1], axis=0)
    feat2 = np.concatenate([emb2, atc2], axis=0)

    pair_feat = np.concatenate([feat1, feat2], axis=0).astype(np.float32)
    X = torch.tensor(pair_feat).unsqueeze(0)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    return float(probs[1])


# ============================================================
# GRAPH FUNCTION
# ============================================================
def draw_single_drug_graph(center_drug, risk_df, threshold=0.10):
    G = nx.Graph()
    G.add_node(center_drug)

    sub_df = risk_df[(risk_df["Drug A"] == center_drug) | (risk_df["Drug B"] == center_drug)]

    for _, row in sub_df.iterrows():
        dA = row["Drug A"]
        dB = row["Drug B"]
        prob = row["Probability"]

        if prob >= threshold:
            G.add_edge(dA, dB, weight=prob)

    if G.number_of_edges() == 0:
        st.info(f"No interaction edges above threshold for {center_drug}")
        return

    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=1500)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos, font_size=9)

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.axis("off")
    st.pyplot(plt)


# ============================================================
# MAIN TITLE
# ============================================================
st.markdown("<h1>🩺 GraphRxInsight</h1>", unsafe_allow_html=True)
st.markdown("<h3>Polypharmacy Risk & Side Effects Clinical Analyzer</h3>", unsafe_allow_html=True)
st.write("A clean medical-style application for Drug Search, Side Effects, Interaction Risk Prediction and Graph Visualization.")
st.divider()


# ============================================================
# LOAD DATA
# ============================================================
drugbank_df = load_drugbank_data()
drug_features_df = load_drug_features()
atc_df = load_atc_pca_features()
embeddings = load_embeddings()

drug_ids = drug_features_df["drug_id"].tolist()
drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}

all_names = sorted(drugbank_df["name"].unique().tolist())
name_to_id = dict(zip(drugbank_df["name"], drugbank_df["drugbank-id"]))


# ============================================================
# SIDEBAR SETTINGS
# ============================================================
st.sidebar.header("🧪 Drug Selection Panel")

selected_drugs = st.sidebar.multiselect(
    "Select Drugs (2 to 10)",
    all_names,
    max_selections=10
)

st.sidebar.divider()

threshold = st.sidebar.slider("Graph Threshold", 0.0, 1.0, 0.10, 0.05)

st.sidebar.divider()

embedding_dim = embeddings.shape[1]
atc_dim = atc_df.shape[1]
pair_input_dim = (embedding_dim + atc_dim) * 2

st.sidebar.metric("Embedding Dim", embedding_dim)
st.sidebar.metric("ATC PCA Dim", atc_dim)
st.sidebar.metric("Final Input Dim", pair_input_dim)


# ============================================================
# LOAD MODEL
# ============================================================
model = load_model(input_dim=pair_input_dim)


# ============================================================
# STOP IF NO DRUGS
# ============================================================
if len(selected_drugs) < 2:
    st.warning("⚠ Please select at least 2 drugs from the sidebar.")
    st.stop()

selected_ids = [name_to_id[name] for name in selected_drugs]


# ============================================================
# SESSION STATE INIT
# ============================================================
if "risk_df" not in st.session_state:
    st.session_state["risk_df"] = None


# ============================================================
# DASHBOARD METRICS
# ============================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected Drugs", len(selected_drugs))
m2.metric("Possible Pairs", len(list(combinations(selected_drugs, 2))))
m3.metric("Graph Threshold", threshold)
m4.metric("Total DrugBank Drugs", len(all_names))

st.divider()


# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Drug Search",
    "⚠ Toxicity & Mechanism",
    "🩺 Side Effects",
    "📊 Risk Prediction",
    "🕸 Graphs"
])


# ============================================================
# TAB 1: DRUG SEARCH
# ============================================================
with tab1:
    st.subheader("🔍 Selected Drug Details")

    for drug_name in selected_drugs:
        drug_id = name_to_id.get(drug_name)

        row = drugbank_df[drugbank_df["drugbank-id"] == drug_id]
        if row.empty:
            continue

        row = row.iloc[0]

        st.markdown(f"""
        <div class="card">
        <h3>💊 {drug_name}</h3>
        <b>DrugBank ID:</b> {drug_id}<br>
        <b>Groups:</b> {row.get("groups", "Not Available")}<br>
        <b>State:</b> {row.get("state", "Not Available")}<br><br>
        <b>Description:</b><br>
        {str(row.get("description", ""))[:400]}...
        </div>
        """, unsafe_allow_html=True)

        st.write("")


# ============================================================
# TAB 2: TOXICITY & MECHANISM
# ============================================================
with tab2:
    st.subheader("⚠ Toxicity / Mechanism Table")

    tox_table = []
    for drug_name in selected_drugs:
        drug_id = name_to_id.get(drug_name)

        row = drugbank_df[drugbank_df["drugbank-id"] == drug_id]
        if row.empty:
            continue

        row = row.iloc[0]
        tox_text = clean_text(row.get("toxicity", ""))
        moa_text = clean_text(row.get("mechanism-of-action", ""))

        tox_table.append({
            "Drug Name": drug_name,
            "DrugBank ID": drug_id,
            "Groups": row.get("groups", "Not Available"),
            "State": row.get("state", "Not Available"),
            "Toxicity (Short)": tox_text[:250] + "..." if len(tox_text) > 250 else (tox_text if tox_text else "Not Available"),
            "Mechanism (Short)": moa_text[:250] + "..." if len(moa_text) > 250 else (moa_text if moa_text else "Not Available")
        })

    st.dataframe(pd.DataFrame(tox_table), use_container_width=True)


# ============================================================
# TAB 3: SIDE EFFECTS
# ============================================================
with tab3:
    st.subheader("🩺 Side Effects Extraction + Overlap")

    drug_side_effect_map = {}
    side_table = []

    for drug_name in selected_drugs:
        drug_id = name_to_id.get(drug_name)

        row = drugbank_df[drugbank_df["drugbank-id"] == drug_id]
        if row.empty:
            continue

        row = row.iloc[0]
        effects = extract_side_effects_full(row)

        drug_side_effect_map[drug_name] = effects

        side_table.append({
            "Drug Name": drug_name,
            "DrugBank ID": drug_id,
            "Extracted Side Effects": ", ".join(effects) if len(effects) > 0 else "Not Available"
        })

    st.markdown("### 📌 Extracted Side Effects Table")
    st.dataframe(pd.DataFrame(side_table), use_container_width=True)

    st.divider()

    st.markdown("### 🧾 Combined Side Effects Overlap (Polypharmacy Risk)")

    combined_effects = combined_side_effect_analysis(drug_side_effect_map)

    if len(combined_effects) == 0:
        st.info("No combined side effects found.")
    else:
        combined_table = []
        for effect, count in combined_effects:
            if count >= 3:
                severity = "🔴 Severe Overlap"
            elif count == 2:
                severity = "🟠 Moderate Overlap"
            else:
                severity = "🟢 Single Drug Only"

            combined_table.append({
                "Side Effect": effect,
                "Appears In (#Drugs)": count,
                "Overlap Severity": severity
            })

        st.dataframe(pd.DataFrame(combined_table), use_container_width=True)


# ============================================================
# TAB 4: RISK PREDICTION
# ============================================================
with tab4:
    st.subheader("📊 Drug Interaction Risk Prediction")

    run_pred = st.button("🚀 Run Risk Prediction")

    if run_pred:
        pairs = list(combinations(selected_ids, 2))
        results = []

        for d1, d2 in pairs:
            prob = predict_interaction(d1, d2, embeddings, atc_df, drug_to_idx, model)

            if prob is None:
                continue

            drugA_name = drugbank_df[drugbank_df["drugbank-id"] == d1]["name"].values[0]
            drugB_name = drugbank_df[drugbank_df["drugbank-id"] == d2]["name"].values[0]

            results.append({
                "Drug A": drugA_name,
                "Drug B": drugB_name,
                "Probability": prob,
                "Risk Level": risk_level(prob)
            })

        if len(results) == 0:
            st.error("❌ No valid drug pairs could be predicted.")
        else:
            risk_df = pd.DataFrame(results).sort_values(by="Probability", ascending=False)
            st.session_state["risk_df"] = risk_df

            risk_df_display = risk_df.copy()
            risk_df_display["Probability"] = risk_df_display["Probability"].apply(lambda x: round(x, 6))

            st.success("✅ Risk Prediction Completed!")
            st.dataframe(risk_df_display, use_container_width=True)

            top = risk_df.iloc[0]
            top_prob = float(top["Probability"])

            st.divider()
            st.markdown("## 🔥 Highest Risk Combination Found")
            st.write(f"**{top['Drug A']} + {top['Drug B']}**")
            st.write("**Probability:**", f"{top_prob:.8f}")
            st.write("**Risk Level:**", top["Risk Level"])


# ============================================================
# TAB 5: GRAPHS
# ============================================================
with tab5:
    st.subheader("🕸 Graph Visualization (Each Drug Graph)")

    if st.session_state["risk_df"] is None:
        st.warning("⚠ Run Risk Prediction first to generate graphs.")
    else:
        risk_df = st.session_state["risk_df"]

        st.success("✅ Graphs generated from latest prediction results.")
        st.write("Graph threshold:", threshold)

        for drug in selected_drugs:
            st.markdown(f"### 🔹 {drug}")
            draw_single_drug_graph(drug, risk_df, threshold=threshold)