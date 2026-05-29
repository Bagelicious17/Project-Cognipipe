import { useCallback, useState, useRef } from "react";

const ACCEPT = ".csv,.xlsx";
const MAX_MB = 50;

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadStage({ onFileSelected }) {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const validate = useCallback((f) => {
    const ext = f.name.split(".").pop().toLowerCase();
    if (!["csv", "xlsx"].includes(ext)) {
      setError("Only .csv and .xlsx files are supported.");
      return false;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`File exceeds ${MAX_MB} MB limit.`);
      return false;
    }
    if (f.size === 0) {
      setError("File is empty.");
      return false;
    }
    setError("");
    return true;
  }, []);

  const handleFile = useCallback(
    (f) => {
      if (validate(f)) setFile(f);
    },
    [validate]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  return (
    <div className="stage-enter flex flex-col items-center gap-8 w-full max-w-xl mx-auto">
      {/* Logo / Title */}
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight text-white">
          🧠 CogniPipe
        </h1>
        <p className="mt-2 text-zinc-400 text-lg">
          Drop a dataset. Get a production-ready ML pipeline.
        </p>
      </div>

      {/* Drop Zone */}
      <div
        role="button"
        tabIndex={0}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        className={`
          w-full rounded-2xl border-2 border-dashed p-12
          flex flex-col items-center justify-center gap-4
          cursor-pointer transition-all duration-300
          ${
            dragOver
              ? "border-blue-500 bg-blue-500/10 scale-[1.02]"
              : file
              ? "border-emerald-500/50 bg-emerald-500/5"
              : "border-zinc-700 bg-zinc-900/50 hover:border-zinc-500 hover:bg-zinc-800/50"
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />

        {file ? (
          <>
            <div className="text-5xl">📄</div>
            <div className="text-center">
              <p className="text-white font-semibold text-lg">{file.name}</p>
              <p className="text-zinc-400 text-sm mt-1">
                {formatSize(file.size)}
              </p>
            </div>
            <p className="text-zinc-500 text-xs">Click to change file</p>
          </>
        ) : (
          <>
            <div className="text-5xl opacity-60">📂</div>
            <p className="text-zinc-300 font-medium">
              Drag & drop your dataset here
            </p>
            <p className="text-zinc-500 text-sm">
              or click to browse — .csv, .xlsx up to {MAX_MB} MB
            </p>
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="w-full rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3 text-red-400 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Submit */}
      <button
        disabled={!file}
        onClick={() => onFileSelected(file)}
        className={`
          w-full py-3.5 rounded-xl font-semibold text-base
          transition-all duration-300
          ${
            file
              ? "bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/25 hover:shadow-blue-500/40 cursor-pointer"
              : "bg-zinc-800 text-zinc-500 cursor-not-allowed"
          }
        `}
      >
        Analyze Dataset
      </button>
    </div>
  );
}
