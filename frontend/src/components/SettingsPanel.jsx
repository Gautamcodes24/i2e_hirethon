import { useState, useEffect } from "react";
import { Settings, X, Save, RotateCcw } from "lucide-react";
import { getSettings, updateSettings } from "../api";

const PROVIDERS = ["groq", "openai"];

const SETTINGS_SCHEMA = [
  {
    group: "Providers",
    fields: [
      {
        key: "embedding_provider",
        label: "Embedding Provider",
        type: "select",
        options: PROVIDERS,
      },
      {
        key: "llm_provider",
        label: "LLM Provider",
        type: "select",
        options: PROVIDERS,
      },
      {
        key: "vision_provider",
        label: "Vision Provider",
        type: "select",
        options: PROVIDERS,
      },
    ],
  },
  {
    group: "Models",
    fields: [
      { key: "groq_llm_model", label: "Groq LLM Model", type: "text" },
      { key: "openai_llm_model", label: "OpenAI LLM Model", type: "text" },
      {
        key: "openai_embedding_model",
        label: "OpenAI Embedding Model",
        type: "text",
      },
      {
        key: "openai_embedding_dim",
        label: "OpenAI Embedding Dim",
        type: "number",
      },
      { key: "embedding_model", label: "Local Embedding Model", type: "text" },
      { key: "groq_vision_model", label: "Groq Vision Model", type: "text" },
      {
        key: "openai_vision_model",
        label: "OpenAI Vision Model",
        type: "text",
      },
    ],
  },
  {
    group: "Retrieval",
    fields: [
      { key: "retrieval_top_k", label: "Top K Results", type: "number" },
      {
        key: "hybrid_alpha",
        label: "Hybrid Alpha",
        type: "number",
        step: "0.1",
      },
      { key: "chunk_max_tokens", label: "Chunk Max Tokens", type: "number" },
    ],
  },
  {
    group: "Generation",
    fields: [
      {
        key: "llm_temperature",
        label: "Temperature",
        type: "number",
        step: "0.1",
      },
      { key: "llm_max_tokens", label: "Max Tokens", type: "number" },
    ],
  },
  {
    group: "Features",
    fields: [
      {
        key: "enable_ocr",
        label: "OCR Extraction",
        type: "toggle",
        readonly: true,
      },
      {
        key: "enable_vision",
        label: "Vision Descriptions",
        type: "toggle",
        readonly: true,
      },
      {
        key: "enable_parent_child",
        label: "Parent-Child Chunks",
        type: "toggle",
        readonly: true,
      },
      {
        key: "enable_quality_filter",
        label: "Quality Filter",
        type: "toggle",
        readonly: true,
      },
      {
        key: "enable_dedup",
        label: "Deduplication",
        type: "toggle",
        readonly: true,
      },
    ],
  },
  {
    group: "Status",
    fields: [
      { key: "active_llm_model", label: "Active LLM", type: "readonly" },
      {
        key: "active_embedding_model",
        label: "Active Embedding",
        type: "readonly",
      },
      { key: "active_vision_model", label: "Active Vision", type: "readonly" },
      { key: "has_groq_key", label: "Groq API Key", type: "readonly" },
      { key: "has_openai_key", label: "OpenAI API Key", type: "readonly" },
      { key: "n_chunks", label: "Indexed Chunks", type: "readonly" },
      { key: "index_loaded", label: "Index Loaded", type: "readonly" },
    ],
  },
];

export default function SettingsPanel({ isOpen, onClose, onToast }) {
  const [values, setValues] = useState({});
  const [original, setOriginal] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    getSettings()
      .then((data) => {
        setValues(data);
        setOriginal(data);
      })
      .catch((err) =>
        onToast?.(`Failed to load settings: ${err.message}`, "error"),
      )
      .finally(() => setLoading(false));
  }, [isOpen]);

  if (!isOpen) return null;

  const hasChanges = JSON.stringify(values) !== JSON.stringify(original);

  const handleChange = (key, val) => {
    setValues((prev) => ({ ...prev, [key]: val }));
  };

  const handleSave = async () => {
    // Only send changed, writable fields
    const changed = {};
    for (const group of SETTINGS_SCHEMA) {
      for (const field of group.fields) {
        if (field.type === "readonly" || field.readonly) continue;
        if (values[field.key] !== original[field.key]) {
          changed[field.key] = values[field.key];
        }
      }
    }
    if (Object.keys(changed).length === 0) {
      onClose();
      return;
    }

    setSaving(true);
    try {
      const result = await updateSettings(changed);
      onToast?.(result.message || "Settings saved!", "success");
      setOriginal({ ...values });
      onClose();
    } catch (err) {
      onToast?.(`Failed to save: ${err.message}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => setValues({ ...original });

  const renderField = (field) => {
    const val = values[field.key];

    if (field.type === "readonly") {
      const display =
        typeof val === "boolean" ? (val ? "Yes" : "No") : String(val ?? "—");
      return <span className="settings-readonly">{display}</span>;
    }

    if (field.type === "toggle") {
      return (
        <label className="settings-toggle">
          <input
            type="checkbox"
            checked={!!val}
            disabled={field.readonly}
            onChange={(e) => handleChange(field.key, e.target.checked)}
          />
          <span className="toggle-slider" />
        </label>
      );
    }

    if (field.type === "select") {
      return (
        <select
          className="settings-select"
          value={val || ""}
          onChange={(e) => handleChange(field.key, e.target.value)}
        >
          {field.options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      );
    }

    if (field.type === "number") {
      return (
        <input
          className="settings-input"
          type="number"
          step={field.step || "1"}
          value={val ?? ""}
          onChange={(e) =>
            handleChange(field.key, parseFloat(e.target.value) || 0)
          }
        />
      );
    }

    return (
      <input
        className="settings-input"
        type="text"
        value={val ?? ""}
        onChange={(e) => handleChange(field.key, e.target.value)}
      />
    );
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>
            <Settings size={18} /> Configuration
          </h3>
          <button className="btn-icon" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {loading ? (
            <div
              style={{
                textAlign: "center",
                padding: "2rem",
                color: "var(--text-muted)",
              }}
            >
              Loading settings...
            </div>
          ) : (
            SETTINGS_SCHEMA.map((group) => (
              <div key={group.group} className="settings-group">
                <div className="settings-group-title">{group.group}</div>
                {group.fields.map((field) => (
                  <div key={field.key} className="settings-row">
                    <label className="settings-label">{field.label}</label>
                    {renderField(field)}
                  </div>
                ))}
              </div>
            ))
          )}
        </div>

        <div className="modal-footer">
          <button
            className="btn btn-ghost"
            onClick={handleReset}
            disabled={!hasChanges}
          >
            <RotateCcw size={14} /> Reset
          </button>
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={!hasChanges || saving}
          >
            <Save size={14} /> {saving ? "Saving..." : "Save Changes"}
          </button>
        </div>
      </div>
    </div>
  );
}
