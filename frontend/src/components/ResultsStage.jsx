import { downloadFile } from "../api";
import { useState } from "react";

export default function ResultsStage({ result, onReset }) {
  const totalTime =
    (result.profiling_duration_seconds || 0) +
    (result.orchestration_duration_seconds || 0);

  // Parse pipeline_summary for display
  const summary = result.pipeline_summary || "";

  return (
    <div className="stage-enter flex flex-col items-center gap-6 w-full max-w-2xl mx-auto">
      {/* Success Banner */}
      <div className="w-full rounded-2xl bg-emerald-500/10 border border-emerald-500/30 p-5 text-center">
        <div className="text-3xl mb-2">🎉</div>
        <h2 className="text-xl font-bold text-emerald-400">
          Pipeline Generated Successfully
        </h2>
        <p className="text-zinc-400 text-sm mt-1">
          Total generation time: {totalTime.toFixed(1)}s
        </p>
      </div>

      {/* Download Buttons */}
      <div className="w-full grid grid-cols-1 sm:grid-cols-3 gap-3">
        <DownloadButton
          icon="📄"
          label="pipeline.py"
          sublabel="Python Script"
          onClick={() =>
            downloadFile("script", result.python_script, "pipeline.py")
          }
        />
        <DownloadButton
          icon="📓"
          label="pipeline.ipynb"
          sublabel="Jupyter Notebook"
          onClick={() =>
            downloadFile("notebook", result.notebook_json, "pipeline.ipynb")
          }
        />
        <DownloadButton
          icon="📋"
          label="requirements.txt"
          sublabel="Dependencies"
          onClick={() =>
            downloadFile(
              "requirements",
              result.requirements_txt,
              "requirements.txt"
            )
          }
        />
      </div>

      {/* Pipeline Summary */}
      {summary && (
        <div className="w-full rounded-2xl bg-zinc-900/80 border border-zinc-800 p-6">
          <p className="text-zinc-500 text-xs uppercase tracking-wider font-medium mb-3">
            Pipeline Summary
          </p>
          <p className="text-zinc-300 text-sm leading-relaxed whitespace-pre-line font-mono">
            {summary}
          </p>
        </div>
      )}

      {/* Reset */}
      <button
        onClick={onReset}
        className="w-full py-3.5 rounded-xl font-semibold text-base
                   bg-zinc-800 hover:bg-zinc-700 text-zinc-300
                   border border-zinc-700 hover:border-zinc-600
                   transition-all duration-300 cursor-pointer"
      >
        ← Analyze Another Dataset
      </button>
    </div>
  );
}

function DownloadButton({ icon, label, sublabel, onClick }) {
  const [downloading, setDownloading] = useState(false);

  const handleClick = async () => {
    setDownloading(true);
    try {
      await onClick();
    } catch {
      // Error handled at app level
    } finally {
      setDownloading(false);
    }
  };

  return (
    <button
      onClick={handleClick}
      disabled={downloading}
      className="flex items-center gap-3 p-4 rounded-xl
                 bg-zinc-900/80 border border-zinc-800
                 hover:bg-zinc-800/80 hover:border-zinc-700
                 transition-all duration-200 cursor-pointer
                 disabled:opacity-50 disabled:cursor-wait"
    >
      <span className="text-2xl">{icon}</span>
      <div className="text-left">
        <p className="text-white text-sm font-semibold">{label}</p>
        <p className="text-zinc-500 text-xs">{sublabel}</p>
      </div>
    </button>
  );
}
