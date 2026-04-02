import React, { useEffect, useState } from "react";
import interactionAtlasArt from "./assets/interaction-atlas.svg";
import gnnLatticeArt from "./assets/gnn-lattice.svg";
import safetySignalArt from "./assets/safety-signal.svg";
import "./App.css";

const EXAMPLE_DRUG_1 = {
  drug_id: "DB00945",
  name: "Acetylsalicylic acid",
};

const EXAMPLE_DRUG_2 = {
  drug_id: "DB01050",
  name: "Ibuprofen",
};

const VISUAL_STORIES = [
  {
    kicker: "Interaction Atlas",
    title: "Pairwise Risk, Mapped Visually",
    description:
      "A gallery-like overview that reflects how GraphRxInsight links two selected therapies into one interpretable interaction check.",
    image: interactionAtlasArt,
    alt: "Illustration of capsules, connected nodes, and analysis tiles on a medical dashboard",
    toneClassName: "story-card-mint",
  },
  {
    kicker: "Graph Features",
    title: "GNN Signals Under The Hood",
    description:
      "A network-inspired picture card that echoes the graph embedding pipeline driving the model's learned drug relationships.",
    image: gnnLatticeArt,
    alt: "Illustration of a graph neural network lattice with glowing medical nodes",
    toneClassName: "story-card-indigo",
  },
  {
    kicker: "Safety Lens",
    title: "Side Effects In Context",
    description:
      "A safety-focused visual cue for overlap analysis, clinical caution, and the extra context shown beside each prediction.",
    image: safetySignalArt,
    alt: "Illustration of a shield, heartbeat line, and medication capsules",
    toneClassName: "story-card-amber",
  },
];

const THEME_PRESETS = [
  { id: "minimal-clinical", label: "Minimal Clinical" },
  { id: "dark-neon", label: "Dark Neon" },
  { id: "royal-blue", label: "Royal Blue" },
];

function SideEffectsDetailPanel({ sideEffects }) {
  const drug1 = sideEffects?.drug1 || {};
  const drug2 = sideEffects?.drug2 || {};
  const overlap = sideEffects?.overlap || {};
  const sharedEffects = overlap.shared_effects || [];

  return (
    <div className="detail-card side-effects-main-card">
      <h3>Detailed Side-Effect Map</h3>

      <div className="side-effects-grid">
        <div className="side-effects-drug">
          <div className="side-effects-headline">
            <strong>{drug1.name || "Drug 1"}</strong>
            <span>{(drug1.effects || []).length} extracted effects</span>
          </div>
          {(drug1.effects || []).length ? (
            <div className="tag-list side-effects-tag-list">
              {(drug1.effects || []).map((effect) => (
                <span className="tag" key={`drug1-${effect}`}>
                  {effect}
                </span>
              ))}
            </div>
          ) : (
            <p className="muted-copy">No extracted effects found for Drug 1.</p>
          )}
        </div>

        <div className="side-effects-drug">
          <div className="side-effects-headline">
            <strong>{drug2.name || "Drug 2"}</strong>
            <span>{(drug2.effects || []).length} extracted effects</span>
          </div>
          {(drug2.effects || []).length ? (
            <div className="tag-list side-effects-tag-list">
              {(drug2.effects || []).map((effect) => (
                <span className="tag" key={`drug2-${effect}`}>
                  {effect}
                </span>
              ))}
            </div>
          ) : (
            <p className="muted-copy">No extracted effects found for Drug 2.</p>
          )}
        </div>
      </div>

      <div className="shared-effects-main">
        <div className="shared-effects-heading">
          <span>Shared Side-Effect Signals</span>
          <strong>{overlap.shared_effect_count || 0}</strong>
        </div>
        {sharedEffects.length ? (
          <div className="tag-list">
            {sharedEffects.map((effect) => (
              <span className="tag tag-strong" key={`shared-main-${effect}`}>
                {effect}
              </span>
            ))}
          </div>
        ) : (
          <p className="muted-copy">No overlapping side-effect keywords for this pair.</p>
        )}
      </div>
    </div>
  );
}

function GnnGatVisual({ gnn }) {
  const embeddingDim = Number(gnn?.embedding_dimension) || 0;
  const perDrugDim = Number(gnn?.per_drug_feature_dimension) || 0;
  const pairDim = Number(gnn?.pair_feature_dimension) || 0;
  const maxDim = Math.max(embeddingDim, perDrugDim, pairDim, 1);

  const dimensionBars = [
    { key: "embedding", label: "Embedding Dim", value: embeddingDim, toneClass: "bar-embedding" },
    { key: "per_drug", label: "Per Drug Features", value: perDrugDim, toneClass: "bar-per-drug" },
    { key: "pair", label: "Pair Features", value: pairDim, toneClass: "bar-pair" },
  ];

  const describeSignalBand = (value) => {
    const ratio = value / maxDim;
    if (ratio >= 0.8) {
      return { label: "High", tone: "signal-high" };
    }
    if (ratio >= 0.45) {
      return { label: "Medium", tone: "signal-medium" };
    }
    return { label: "Low", tone: "signal-low" };
  };

  const featureTableRows = dimensionBars.map((bar) => {
    const relativeToMax = (bar.value / maxDim) * 100;
    const relativeToPair = pairDim ? (bar.value / pairDim) * 100 : 0;
    const signal = describeSignalBand(bar.value);
    return {
      ...bar,
      relativeToMax,
      relativeToPair,
      signal,
    };
  });

  return (
    <div className="detail-card gnn-visual-card gnn-visual-card-main">
      <h3>GNN + GAT Visual</h3>
      <div className="gnn-visual-layout">
        <div className="gnn-canvas">
          <svg
            className="gnn-svg"
            viewBox="0 0 420 260"
            role="img"
            aria-label="Graphical representation of GNN layers and GAT attention connections"
          >
            <defs>
              <linearGradient id="gnnStroke" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#60CBB2" />
                <stop offset="100%" stopColor="#5B8AF4" />
              </linearGradient>
            </defs>

            <g className="gnn-links">
              <path d="M64 54L202 44L344 72" />
              <path d="M64 130L202 122L344 134" />
              <path d="M64 206L202 196L344 182" />
              <path d="M64 54L202 122L344 182" />
              <path d="M64 206L202 122L344 72" />
            </g>

            <g className="gat-attention-links">
              <path className="attn-strong" d="M202 122L344 72" />
              <path className="attn-medium" d="M202 122L344 134" />
              <path className="attn-light" d="M202 122L344 182" />
            </g>

            <g className="gnn-node-column">
              <circle cx="64" cy="54" r="15" />
              <circle cx="64" cy="130" r="15" />
              <circle cx="64" cy="206" r="15" />
            </g>
            <g className="gnn-node-column mid">
              <circle cx="202" cy="44" r="17" />
              <circle cx="202" cy="122" r="21" />
              <circle cx="202" cy="196" r="17" />
            </g>
            <g className="gnn-node-column out">
              <circle cx="344" cy="72" r="15" />
              <circle cx="344" cy="134" r="15" />
              <circle cx="344" cy="182" r="15" />
            </g>

            <text x="38" y="242">Input Graph</text>
            <text x="182" y="242">GNN</text>
            <text x="332" y="242">GAT</text>
          </svg>
          <p className="muted-copy">
            Message passing across graph layers with attention-focused output links.
          </p>
        </div>

        <div className="gnn-bars">
          {dimensionBars.map((bar) => (
            <div className="gnn-bar-row" key={bar.label}>
              <div className="gnn-bar-header">
                <span>{bar.label}</span>
                <strong>{bar.value}</strong>
              </div>
              <div className="gnn-bar-track">
                <span
                  className={`gnn-bar-fill ${bar.toneClass}`}
                  style={{ "--bar-width": `${Math.max(6, Math.round((bar.value / maxDim) * 100))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="gnn-core-metadata">
        <div className="gnn-meta-item">
          <span>Encoder</span>
          <strong>{gnn?.encoder || "GAT"}</strong>
        </div>
        <div className="gnn-meta-item">
          <span>Source</span>
          <strong>{gnn?.embedding_source || "Graph Embeddings"}</strong>
        </div>
      </div>

      <div className="gnn-feature-tables">
        {featureTableRows.map((feature) => (
          <div className="gnn-feature-table-card" key={`table-${feature.key}`}>
            <h4>{feature.label}</h4>
            <table className="gnn-feature-table">
              <tbody>
                <tr>
                  <th>Dimension</th>
                  <td>{feature.value}</td>
                </tr>
                <tr>
                  <th>Relative to max</th>
                  <td>{feature.relativeToMax.toFixed(1)}%</td>
                </tr>
                <tr>
                  <th>Relative to pair</th>
                  <td>{feature.relativeToPair.toFixed(1)}%</td>
                </tr>
                <tr>
                  <th>Signal band</th>
                  <td>
                    <span className={`gnn-signal-badge ${feature.signal.tone}`}>
                      {feature.signal.label}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        ))}
      </div>

      <p className="muted-copy gnn-note-copy">{gnn?.note || "Graph-derived embedding features are active."}</p>
    </div>
  );
}

async function parseJsonResponse(res) {
  const rawText = await res.text();

  if (!rawText) {
    return null;
  }

  try {
    return JSON.parse(rawText);
  } catch {
    throw new Error(`Server returned a non-JSON response: ${rawText}`);
  }
}

function DrugSearchField({
  fieldId,
  label,
  query,
  selectedDrug,
  options,
  loading,
  isOpen,
  onChange,
  onFocus,
  onBlur,
  onSelect,
}) {
  const showDropdown = isOpen && (loading || query.trim().length >= 2);

  return (
    <div className="search-field">
      <label className="search-label" htmlFor={fieldId}>
        {label}
      </label>

      <input
        id={fieldId}
        className="search-input"
        type="text"
        value={query}
        onChange={(event) => onChange(event.target.value)}
        onFocus={onFocus}
        onBlur={onBlur}
        placeholder="Search by generic or known drug name"
        autoComplete="off"
      />

      {selectedDrug ? (
        <p className="selection-copy">
          Selected: <strong>{selectedDrug.name}</strong> <span>{selectedDrug.drug_id}</span>
        </p>
      ) : (
        <p className="selection-copy">Choose one of the matching drugs below.</p>
      )}

      {showDropdown ? (
        <div className="search-dropdown">
          {loading ? <p className="search-status">Searching drugs...</p> : null}

          {!loading && options.length === 0 ? (
            <p className="search-status">No matches found. Try another spelling.</p>
          ) : null}

          {!loading
            ? options.map((option) => (
                <button
                  key={`${fieldId}-${option.drug_id}`}
                  className="search-option"
                  type="button"
                  onMouseDown={(event) => {
                    event.preventDefault();
                    onSelect(option);
                  }}
                >
                  <span>{option.name}</span>
                  <small>{option.drug_id}</small>
                </button>
              ))
            : null}
        </div>
      ) : null}
    </div>
  );
}

function useDrugSearch(query) {
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const trimmedQuery = query.trim();

    if (trimmedQuery.length < 2) {
      setOptions([]);
      setLoading(false);
      return undefined;
    }

    const controller = new AbortController();
    const timeoutId = window.setTimeout(async () => {
      setLoading(true);

      try {
        const res = await fetch(
          `/drugs/search?q=${encodeURIComponent(trimmedQuery)}&limit=8`,
          { signal: controller.signal }
        );

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data?.error || "Unable to search drugs.");
        }

        setOptions(data.results || []);
      } catch (err) {
        if (err.name !== "AbortError") {
          setOptions([]);
        }
      } finally {
        setLoading(false);
      }
    }, 180);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
  }, [query]);

  return { options, loading };
}

function useScrollReveal() {
  useEffect(() => {
    const elements = Array.from(document.querySelectorAll("[data-reveal]"));

    if (!elements.length) {
      return undefined;
    }

    const reveal = (element) => {
      element.classList.add("is-visible");
    };

    const prefersReducedMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (prefersReducedMotion || typeof window.IntersectionObserver !== "function") {
      elements.forEach(reveal);
      return undefined;
    }

    const observer = new window.IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) {
            return;
          }

          reveal(entry.target);
          observer.unobserve(entry.target);
        });
      },
      {
        threshold: 0.18,
        rootMargin: "0px 0px -10% 0px",
      }
    );

    elements.forEach((element, index) => {
      element.style.setProperty("--reveal-delay", `${Math.min(index * 70, 360)}ms`);
      observer.observe(element);
    });

    return () => observer.disconnect();
  }, []);
}

function App() {
  const [themePreset, setThemePreset] = useState("minimal-clinical");
  const [drug1Query, setDrug1Query] = useState("");
  const [drug2Query, setDrug2Query] = useState("");
  const [selectedDrug1, setSelectedDrug1] = useState(null);
  const [selectedDrug2, setSelectedDrug2] = useState(null);
  const [openField, setOpenField] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [feedbackState, setFeedbackState] = useState({
    loading: false,
    message: null,
    error: null,
  });

  useScrollReveal();

  const drug1Search = useDrugSearch(drug1Query);
  const drug2Search = useDrugSearch(drug2Query);

  const loadExample = () => {
    setDrug1Query(EXAMPLE_DRUG_1.name);
    setDrug2Query(EXAMPLE_DRUG_2.name);
    setSelectedDrug1(EXAMPLE_DRUG_1);
    setSelectedDrug2(EXAMPLE_DRUG_2);
    setResult(null);
    setError(null);
    setFeedbackState({ loading: false, message: null, error: null });
  };

  const clearSelection = () => {
    setDrug1Query("");
    setDrug2Query("");
    setSelectedDrug1(null);
    setSelectedDrug2(null);
    setResult(null);
    setError(null);
    setOpenField(null);
    setFeedbackState({ loading: false, message: null, error: null });
  };

  const handleDrug1Change = (value) => {
    setDrug1Query(value);
    setSelectedDrug1(null);
    setError(null);
    setFeedbackState({ loading: false, message: null, error: null });
  };

  const handleDrug2Change = (value) => {
    setDrug2Query(value);
    setSelectedDrug2(null);
    setError(null);
    setFeedbackState({ loading: false, message: null, error: null });
  };

  const predict = async () => {
    const chosenDrug1 = selectedDrug1?.name || drug1Query.trim();
    const chosenDrug2 = selectedDrug2?.name || drug2Query.trim();

    if (!chosenDrug1 || !chosenDrug2) {
      setError("Choose two drugs before running the interaction check.");
      return;
    }

    if (
      selectedDrug1?.drug_id &&
      selectedDrug2?.drug_id &&
      selectedDrug1.drug_id === selectedDrug2.drug_id
    ) {
      setError("Choose two different drugs for the interaction check.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setFeedbackState({ loading: false, message: null, error: null });

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          drug1: chosenDrug1,
          drug2: chosenDrug2,
        }),
      });

      const data = await parseJsonResponse(res);

      if (!res.ok) {
        throw new Error(data?.error || `Server error ${res.status}`);
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Error fetching prediction");
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (label) => {
    if (!result?.drug1_id || !result?.drug2_id) {
      setFeedbackState({
        loading: false,
        message: null,
        error: "Run a drug-pair prediction before submitting feedback.",
      });
      return;
    }

    setFeedbackState({ loading: true, message: null, error: null });

    try {
      const res = await fetch("/dynamic/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          drug1_id: result.drug1_id,
          drug2_id: result.drug2_id,
          label,
        }),
      });

      const data = await parseJsonResponse(res);

      if (!res.ok) {
        throw new Error(data?.error || `Server error ${res.status}`);
      }

      setResult((current) => (current ? { ...current, model: data.model } : current));
      setFeedbackState({
        loading: false,
        message:
          label === 1
            ? `Saved as observed interaction. Feedback samples: ${data.model.dynamic_feedback_samples}.`
            : `Saved as observed non-interaction. Feedback samples: ${data.model.dynamic_feedback_samples}.`,
        error: null,
      });
    } catch (err) {
      setFeedbackState({
        loading: false,
        message: null,
        error: err.message || "Could not save feedback.",
      });
    }
  };

  const probability =
    typeof result?.probability === "number"
      ? `${(result.probability * 100).toFixed(2)}%`
      : "Waiting";

  const sideEffects = result?.side_effects;
  const gnn = result?.gnn;
  const model = result?.model;

  return (
    <main className={`app-shell theme-${themePreset}`}>
      <section className="hero-panel">
        <div className="reveal-front" data-reveal>
          <p className="eyebrow">Drug Interaction Workspace</p>
          <h1>GraphRxInsight Search</h1>
          <p className="hero-copy">
            Search and choose two drugs from the catalog, then run the interaction
            prediction. The results panel still shows side effects, GNN features, and
            the live dynamic-model status.
          </p>
        </div>

        <div className="hero-stat reveal-reverse" data-reveal>
          <span className="stat-label">Workflow</span>
          <strong>Search, choose, predict</strong>
          <span className="stat-hint">autocomplete from the backend drug index</span>
        </div>
      </section>

      <section
        className="disclaimer-strip reveal-front"
        data-reveal
        role="note"
        aria-label="Study disclaimer"
      >
        <p>
          <strong>Disclaimer:</strong> GraphRxInsight is for study and research purposes only.
          It is not medical advice, diagnosis, or treatment guidance.
        </p>
      </section>

      <section className="theme-switcher reveal-reverse" data-reveal aria-label="Theme presets">
        <p className="theme-label">Theme Presets</p>
        <div className="theme-switcher-row">
          {THEME_PRESETS.map((themeOption) => (
            <button
              key={themeOption.id}
              type="button"
              className={`theme-button ${
                themePreset === themeOption.id ? "theme-button-active" : ""
              }`}
              onClick={() => setThemePreset(themeOption.id)}
            >
              {themeOption.label}
            </button>
          ))}
        </div>
      </section>

      <section className="story-grid" aria-label="GraphRxInsight visuals">
        {VISUAL_STORIES.map((story, index) => (
          <article
            className={`story-card ${story.toneClassName} ${
              index % 2 === 0 ? "reveal-front" : "reveal-reverse"
            }`}
            data-reveal
            key={story.title}
          >
            <div className="story-media">
              <img src={story.image} alt={story.alt} />
            </div>
            <div className="story-copy">
              <p className="story-kicker">{story.kicker}</p>
              <h3>{story.title}</h3>
              <p>{story.description}</p>
            </div>
          </article>
        ))}
      </section>

      <section className="workspace-grid">
        <div className="panel editor-panel selector-panel reveal-front" data-reveal>
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Drug Chooser</p>
              <h2>Select Drugs</h2>
            </div>

            <div className="panel-actions">
              <button className="ghost-button" type="button" onClick={loadExample}>
                Load Example
              </button>
              <button className="ghost-button" type="button" onClick={clearSelection}>
                Clear
              </button>
            </div>
          </div>

          <div className="search-stack">
            <DrugSearchField
              fieldId="drug-1-search"
              label="Drug 1"
              query={drug1Query}
              selectedDrug={selectedDrug1}
              options={drug1Search.options}
              loading={drug1Search.loading}
              isOpen={openField === "drug1"}
              onChange={handleDrug1Change}
              onFocus={() => setOpenField("drug1")}
              onBlur={() => {
                window.setTimeout(() => setOpenField((current) => (current === "drug1" ? null : current)), 120);
              }}
              onSelect={(option) => {
                setSelectedDrug1(option);
                setDrug1Query(option.name);
                setOpenField(null);
              }}
            />

            <DrugSearchField
              fieldId="drug-2-search"
              label="Drug 2"
              query={drug2Query}
              selectedDrug={selectedDrug2}
              options={drug2Search.options}
              loading={drug2Search.loading}
              isOpen={openField === "drug2"}
              onChange={handleDrug2Change}
              onFocus={() => setOpenField("drug2")}
              onBlur={() => {
                window.setTimeout(() => setOpenField((current) => (current === "drug2" ? null : current)), 120);
              }}
              onSelect={(option) => {
                setSelectedDrug2(option);
                setDrug2Query(option.name);
                setOpenField(null);
              }}
            />
          </div>

          <div className="selection-preview">
            <p className="selection-helper">
              Start typing at least 2 characters, then choose from the suggestion list.
            </p>

            <div className="chosen-drugs">
              <span className="choice-chip">
                {selectedDrug1?.name || "Drug 1 not selected"}
              </span>
              <span className="choice-chip choice-chip-accent">
                {selectedDrug2?.name || "Drug 2 not selected"}
              </span>
            </div>
          </div>

          <div className="selection-footer">
            <p>
              Search is backed by <code>/drugs/search</code>, and prediction is sent to
              <code> /predict</code>.
            </p>

            <button
              className="primary-button"
              type="button"
              onClick={predict}
              disabled={loading || !drug1Query.trim() || !drug2Query.trim()}
            >
              {loading ? "Checking..." : "Check Interaction"}
            </button>
          </div>

          <div className="chooser-main-insights reveal-reverse" data-reveal>
            <div className="section-heading section-heading-light">
              <h3>Main Feature Panels</h3>
              <p>Large GNN/GAT graph view and detailed side-effect analysis for this pair.</p>
            </div>

            <div className="main-insights-grid main-insights-grid-chooser">
              {gnn ? (
                <GnnGatVisual gnn={gnn} />
              ) : (
                <div className="message-card empty-card">
                  <h3>No GNN metadata yet</h3>
                  <p>Run a prediction to render the graph and attention panel.</p>
                </div>
              )}

              {sideEffects ? (
                <SideEffectsDetailPanel sideEffects={sideEffects} />
              ) : (
                <div className="message-card empty-card">
                  <h3>No side-effect analysis yet</h3>
                  <p>Run a prediction to see detailed side-effect extraction and overlap.</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="panel results-panel reveal-reverse" data-reveal>
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Response</p>
              <h2>Prediction Output</h2>
            </div>
          </div>

          <div className="result-metrics reveal-front" data-reveal>
            <div className={`metric-card ${result ? "metric-card-live" : ""}`}>
              <span className="metric-label">Risk</span>
              <strong
                className={`metric-value ${result ? "metric-value-live" : ""}`}
                key={`risk-${result?.risk || "waiting"}`}
              >
                {result?.risk || "Waiting"}
              </strong>
            </div>
            <div className={`metric-card ${result ? "metric-card-live" : ""}`}>
              <span className="metric-label">Probability</span>
              <strong
                className={`metric-value ${result ? "metric-value-live" : ""}`}
                key={`prob-${probability}`}
              >
                {probability}
              </strong>
            </div>
          </div>

          {error ? (
            <div className="message-card error-card" role="alert">
              <h3>Request Error</h3>
              <p>{error}</p>
            </div>
          ) : null}

          <div className="detail-section reveal-front" data-reveal>
            <div className="section-heading">
              <h3>Feedback</h3>
              <p>
                If you know the real outcome for this pair, save it to the dynamic
                feedback dataset for later retraining.
              </p>
            </div>

            <div className="detail-card feedback-card">
              <div className="feedback-actions">
                <button
                  className="feedback-button feedback-positive"
                  type="button"
                  onClick={() => submitFeedback(1)}
                  disabled={feedbackState.loading || !result?.drug1_id || !result?.drug2_id}
                >
                  {feedbackState.loading ? "Saving..." : "Observed Interaction"}
                </button>
                <button
                  className="feedback-button feedback-negative"
                  type="button"
                  onClick={() => submitFeedback(0)}
                  disabled={feedbackState.loading || !result?.drug1_id || !result?.drug2_id}
                >
                  {feedbackState.loading ? "Saving..." : "No Interaction Observed"}
                </button>
              </div>

              {feedbackState.message ? (
                <div className="message-card success-card">
                  <h3>Feedback Saved</h3>
                  <p>{feedbackState.message}</p>
                </div>
              ) : null}

              {feedbackState.error ? (
                <div className="message-card error-card">
                  <h3>Feedback Error</h3>
                  <p>{feedbackState.error}</p>
                </div>
              ) : null}
            </div>
          </div>

          <div className="detail-section reveal-reverse" data-reveal>
            <div className="section-heading">
              <h3>Model Status</h3>
              <p>
                The app now serves a dynamic model that can collect feedback samples and
                retrain over time.
              </p>
            </div>

            {model ? (
              <div className="mini-grid">
                <div className="mini-card">
                  <span className="metric-label">Active Model</span>
                  <strong>{model.active_model}</strong>
                </div>
                <div className="mini-card">
                  <span className="metric-label">Model State</span>
                  <strong>{model.model_state}</strong>
                </div>
                <div className="mini-card">
                  <span className="metric-label">Feedback Samples</span>
                  <strong>{model.dynamic_feedback_samples}</strong>
                </div>
                <div className="mini-card">
                  <span className="metric-label">Retrain Threshold</span>
                  <strong>{model.auto_retrain_threshold}</strong>
                </div>
                <div className="detail-card note-card">
                  <h3>Checkpoint</h3>
                  <p>{model.checkpoint_path}</p>
                  {model.bootstrap_source ? (
                    <p className="muted-copy">
                      Bootstrapped from {model.bootstrap_source} until enough feedback is
                      collected for fine-tuning.
                    </p>
                  ) : null}
                  {model.dataset_warning ? (
                    <p className="muted-copy">{model.dataset_warning}</p>
                  ) : null}
                </div>
              </div>
            ) : (
              <div className="message-card empty-card">
                <h3>No model metadata yet</h3>
                <p>Run a prediction to inspect the active dynamic model state.</p>
              </div>
            )}
          </div>

          {!result ? (
            <div className="message-card empty-card">
              <h3>No response yet</h3>
              <p>Choose two drugs from the search boxes to see the interaction prediction here.</p>
            </div>
          ) : null}
        </div>
      </section>

      <footer className="developer-credit reveal-front" data-reveal>
        <p className="developer-kicker">Developers</p>
        <div className="developer-names">
          <strong>Ganesh V</strong>
          <strong>Sriguru S</strong>
        </div>
        <p className="developer-role">Final Year B.Tech CSE</p>
      </footer>
    </main>
  );
}

export default App;
