import { CheckCircle, XCircle, Info, X } from "lucide-react";

const ICONS = {
  success: <CheckCircle size={15} />,
  error: <XCircle size={15} />,
  info: <Info size={15} />,
};

export default function ToastContainer({ toasts, onRemove }) {
  if (toasts.length === 0) return null;

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.type}`}>
          {ICONS[t.type] || ICONS.info}
          <span style={{ flex: 1 }}>{t.message}</span>
          <button
            className="btn-icon"
            style={{ padding: "2px" }}
            onClick={() => onRemove(t.id)}
          >
            <X size={13} />
          </button>
        </div>
      ))}
    </div>
  );
}
