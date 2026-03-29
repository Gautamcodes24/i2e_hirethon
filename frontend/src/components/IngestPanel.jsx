import { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload,
  FileText,
  CheckCircle,
  XCircle,
  Loader,
  BookOpen,
  Cpu,
  Database,
  Search,
  Save,
} from "lucide-react";
import { uploadPDF, getIngestStatus, getIngestResult } from "../api";

const PIPELINE_STEPS = [
  {
    label: "Parsing PDF",
    icon: BookOpen,
    desc: "Extracting text, tables, and images",
  },
  {
    label: "Building Embeddings",
    icon: Cpu,
    desc: "Generating vector embeddings",
  },
  { label: "FAISS Index", icon: Database, desc: "Building dense vector index" },
  { label: "BM25 Index", icon: Search, desc: "Building sparse keyword index" },
  { label: "Saving Artefacts", icon: Save, desc: "Persisting to disk" },
];

export default function IngestPanel({ isOpen, onClose, onToast }) {
  const [file, setFile] = useState(null);
  const [dragover, setDragover] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [stepMessage, setStepMessage] = useState("");
  const [stepDetail, setStepDetail] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);
  const pollRef = useRef(null);

  // Reset on close
  useEffect(() => {
    if (!isOpen) {
      if (pollRef.current) clearInterval(pollRef.current);
    }
  }, [isOpen]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragover(false);
    const f = e.dataTransfer?.files?.[0];
    if (f && f.type === "application/pdf") setFile(f);
  }, []);

  const handleFileSelect = (e) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const startIngestion = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setResult(null);
    setCurrentStep(0);
    setStepMessage("Uploading...");
    setStatus("queued");

    try {
      const { task_id } = await uploadPDF(file);
      setTaskId(task_id);
      setStatus("running");

      // Start polling
      pollRef.current = setInterval(async () => {
        try {
          const s = await getIngestStatus(task_id);

          if (s.status === "completed") {
            clearInterval(pollRef.current);
            setStatus("completed");
            setCurrentStep(s.total_steps || 5);
            setStepMessage("Ingestion complete");
            try {
              const r = await getIngestResult(task_id);
              setResult(r);
            } catch {
              // result endpoint may not exist
            }
            onToast?.("PDF ingested successfully!", "success");
          } else if (s.status === "failed") {
            clearInterval(pollRef.current);
            setStatus("failed");
            setError(s.error || "Ingestion failed");
            onToast?.("Ingestion failed", "error");
          } else {
            setCurrentStep(s.current_step || 0);
            setStepMessage(s.step_message || "Processing...");
            setStepDetail(s.step_detail || "");
          }
        } catch {
          clearInterval(pollRef.current);
          setStatus("failed");
          setError("Lost connection to server");
        }
      }, 1500);
    } catch (err) {
      setStatus("failed");
      setError(err.message);
      onToast?.(`Upload failed: ${err.message}`, "error");
    } finally {
      setUploading(false);
    }
  };

  const resetPanel = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setFile(null);
    setTaskId(null);
    setStatus(null);
    setCurrentStep(0);
    setStepMessage("");
    setStepDetail("");
    setResult(null);
    setError(null);
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>
            <Upload size={18} /> Ingest PDF Document
          </h3>
          <button className="btn-icon" onClick={onClose}>
            <span style={{ fontSize: "1.2rem", lineHeight: 1 }}>&times;</span>
          </button>
        </div>

        <div className="modal-body">
          <div className="ingest-panel">
            {!status && (
              <>
                <div
                  className={`upload-zone ${dragover ? "dragover" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragover(true);
                  }}
                  onDragLeave={() => setDragover(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handleFileSelect}
                  />
                  <div className="upload-zone-icon">
                    <Upload size={32} />
                  </div>
                  {file ? (
                    <p>
                      <FileText size={14} style={{ verticalAlign: "middle" }} />{" "}
                      <strong>{file.name}</strong> (
                      {(file.size / 1024 / 1024).toFixed(1)} MB)
                    </p>
                  ) : (
                    <>
                      <p>Drag & drop a PDF here, or click to browse</p>
                      <p className="up-hint">
                        Supports technical manuals, handbooks, and documents
                      </p>
                    </>
                  )}
                </div>

                {file && (
                  <div style={{ marginTop: "1rem", textAlign: "center" }}>
                    <button
                      className="btn btn-primary"
                      onClick={startIngestion}
                      disabled={uploading}
                    >
                      <Upload size={14} />{" "}
                      {uploading ? "Uploading..." : "Start Ingestion"}
                    </button>
                  </div>
                )}
              </>
            )}

            {status && (
              <div className="ingest-progress">
                <div className="ingest-progress-header">
                  <span className="ingest-progress-title">
                    {status === "completed" ? (
                      <>
                        <CheckCircle
                          size={16}
                          style={{
                            color: "var(--accent)",
                            verticalAlign: "middle",
                          }}
                        />{" "}
                        Ingestion Complete
                      </>
                    ) : status === "failed" ? (
                      <>
                        <XCircle
                          size={16}
                          style={{
                            color: "var(--error)",
                            verticalAlign: "middle",
                          }}
                        />{" "}
                        Ingestion Failed
                      </>
                    ) : (
                      <>
                        <Loader
                          size={16}
                          className="spinning"
                          style={{ verticalAlign: "middle" }}
                        />{" "}
                        Processing {file?.name}
                      </>
                    )}
                  </span>
                </div>

                {/* Stepper */}
                <div className="ingest-stepper">
                  {PIPELINE_STEPS.map((step, i) => {
                    const stepNum = i + 1;
                    const isCompleted =
                      currentStep > stepNum || status === "completed";
                    const isActive =
                      currentStep === stepNum && status === "running";
                    const StepIcon = step.icon;

                    return (
                      <div
                        key={i}
                        className={`step-item ${isCompleted ? "completed" : ""} ${isActive ? "active" : ""}`}
                      >
                        <div className="step-indicator">
                          <div className="step-icon">
                            {isCompleted ? (
                              <CheckCircle size={18} />
                            ) : isActive ? (
                              <Loader size={18} className="spinning" />
                            ) : (
                              <StepIcon size={18} />
                            )}
                          </div>
                          {i < PIPELINE_STEPS.length - 1 && (
                            <div
                              className={`step-connector ${isCompleted ? "completed" : ""}`}
                            />
                          )}
                        </div>
                        <div className="step-content">
                          <div className="step-label">{step.label}</div>
                          <div className="step-desc">{step.desc}</div>
                          {isActive && stepDetail && (
                            <div className="step-detail">{stepDetail}</div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {error && (
                  <div
                    style={{
                      color: "var(--error)",
                      fontSize: "0.85rem",
                      marginTop: "0.75rem",
                    }}
                  >
                    {error}
                  </div>
                )}

                {result && (
                  <div className="ingest-result">
                    <div className="ingest-stat">
                      <div className="ingest-stat-value">
                        {result.n_chunks || 0}
                      </div>
                      <div className="ingest-stat-label">Chunks</div>
                    </div>
                    <div className="ingest-stat">
                      <div className="ingest-stat-value">
                        {result.n_parents || 0}
                      </div>
                      <div className="ingest-stat-label">Parents</div>
                    </div>
                    <div className="ingest-stat">
                      <div className="ingest-stat-value">
                        {result.n_tables || 0}
                      </div>
                      <div className="ingest-stat-label">Tables</div>
                    </div>
                    <div className="ingest-stat">
                      <div className="ingest-stat-value">
                        {result.embedding_dim || 0}
                      </div>
                      <div className="ingest-stat-label">Dimensions</div>
                    </div>
                  </div>
                )}

                {(status === "completed" || status === "failed") && (
                  <div style={{ marginTop: "1rem", textAlign: "center" }}>
                    <button className="btn btn-ghost" onClick={resetPanel}>
                      Upload Another
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
