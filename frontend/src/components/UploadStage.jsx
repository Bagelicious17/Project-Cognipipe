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
      <div className="text-center relative z-10">
        <h1 className="text-4xl font-heading font-semibold text-[#292524] dark:text-[#F3F2ED]">
          CogniPipe
        </h1>
        <p className="mt-3 text-[#78716C] dark:text-[#a8a29e] text-base max-w-md mx-auto">
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
          relative w-full rounded-xl border-2 border-dashed p-14
          flex flex-col items-center justify-center gap-4
          cursor-pointer transition-all duration-300 ease-out overflow-hidden
          ${
            dragOver
              ? "border-[#3B82F6] bg-[#3B82F6]/5 scale-[1.02] shadow-sm"
              : file
              ? "border-[#10B981] dark:border-[#34D399] bg-[#10B981]/5"
              : "border-[#E7E5E4] dark:border-[#44403C] bg-[#FFFFFF] dark:bg-[#292524] hover:border-[#78716C] dark:hover:border-[#a8a29e] hover:shadow-sm"
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
              <p className="text-[#292524] dark:text-[#F3F2ED] font-semibold text-lg">{file.name}</p>
              <p className="text-[#78716C] dark:text-[#a8a29e] text-sm mt-1">
                {formatSize(file.size)}
              </p>
            </div>
            <p className="text-[#78716C] dark:text-[#a8a29e] text-xs">Click to change file</p>
          </>
        ) : (
          <>
            <div className="text-5xl opacity-60">📂</div>
            <p className="text-[#292524] dark:text-[#F3F2ED] font-medium mt-2">
              Drag & drop your dataset here
            </p>
            <p className="text-[#78716C] dark:text-[#a8a29e] text-sm">
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
          w-full py-3 rounded-md font-medium text-base tracking-wide
          transition-all duration-200 ease-out border
          ${
            file
              ? "bg-[#292524] dark:bg-[#a8a29e] text-white dark:text-[#1C1917] border-transparent hover:opacity-90 shadow-sm cursor-pointer"
              : "bg-[#F5F5F4] dark:bg-[#292524] text-[#A8A29E] dark:text-[#78716C] border-[#E7E5E4] dark:border-[#44403C] cursor-not-allowed"
          }
        `}
      >
        Analyze Dataset
      </button>
    </div>
  );
}
