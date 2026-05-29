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
      <div className="w-full rounded-lg bg-[#10B981]/10 dark:bg-[#34D399]/10 border-l-4 border-l-[#10B981] dark:border-l-[#34D399] border-y border-r border-y-[#E7E5E4] dark:border-y-[#44403C] border-r-[#E7E5E4] dark:border-r-[#44403C] p-6 text-left relative z-10">
        <h2 className="text-xl font-heading font-semibold text-[#10B981] dark:text-[#34D399] flex items-center gap-2">
          <span className="text-2xl">🎉</span> Pipeline Generated Successfully
        </h2>
        <p className="text-[#78716C] dark:text-[#a8a29e] text-sm mt-2 font-mono">
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
        <div className="w-full rounded-lg bg-[#FFFFFF] dark:bg-[#292524] border border-[#E7E5E4] dark:border-[#44403C] p-6 shadow-sm relative z-10">
          <p className="text-[#78716C] dark:text-[#a8a29e] text-xs uppercase tracking-wider font-semibold mb-4">
            Pipeline Summary
          </p>
          <div className="bg-[#F5F5F4] dark:bg-[#1C1917] rounded-md p-4 border border-[#E7E5E4] dark:border-[#44403C]">
            <p className="text-[#292524] dark:text-[#F3F2ED] text-sm leading-relaxed whitespace-pre-line font-mono">
              {summary}
            </p>
          </div>
        </div>
      )}

      {/* Reset */}
      <button
        onClick={onReset}
        className="w-full py-3 rounded-md font-medium text-base tracking-wide relative z-10
                   bg-[#FFFFFF] dark:bg-[#292524] hover:bg-[#F5F5F4] dark:hover:bg-[#44403C] text-[#292524] dark:text-[#F3F2ED]
                   border border-[#E7E5E4] dark:border-[#44403C] hover:border-[#78716C] dark:hover:border-[#a8a29e]
                   transition-colors duration-200 ease-out cursor-pointer shadow-sm"
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
      className="flex items-center gap-4 p-4 rounded-lg
                 bg-[#FFFFFF] dark:bg-[#292524] border border-[#E7E5E4] dark:border-[#44403C] shadow-sm
                 hover:bg-[#F5F5F4] dark:hover:bg-[#44403C] hover:border-[#3B82F6] dark:hover:border-[#60A5FA]
                 transition-all duration-200 ease-out cursor-pointer hover:-translate-y-0.5
                 disabled:opacity-50 disabled:cursor-wait disabled:hover:translate-y-0 relative z-10"
    >
      <span className="text-2xl drop-shadow-sm">{icon}</span>
      <div className="text-left">
        <p className="text-[#292524] dark:text-[#F3F2ED] text-sm font-semibold tracking-wide">{label}</p>
        <p className="text-[#78716C] dark:text-[#a8a29e] text-xs mt-0.5">{sublabel}</p>
      </div>
    </button>
  );
}
