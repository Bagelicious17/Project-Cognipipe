import { useState, useCallback } from "react";
import { profileDataset, generatePipeline } from "./api";
import UploadStage from "./components/UploadStage";
import ProfilingStage from "./components/ProfilingStage";
import GeneratingStage from "./components/GeneratingStage";
import ResultsStage from "./components/ResultsStage";
import ErrorCard from "./components/ErrorCard";

/**
 * Application stages:
 *  upload    → user selects a file
 *  profiling → Layer 1 running (~1-3s)
 *  profiled  → results shown, awaiting Generate click
 *  generating → Layers 2+3 running (~30-90s)
 *  results   → downloads + summary
 *  error     → something went wrong
 */
const STAGES = {
  UPLOAD: "upload",
  PROFILING: "profiling",
  PROFILED: "profiled",
  GENERATING: "generating",
  RESULTS: "results",
  ERROR: "error",
};

export default function App() {
  const [stage, setStage] = useState(STAGES.UPLOAD);
  const [file, setFile] = useState(null);
  const [profile, setProfile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const reset = useCallback(() => {
    setStage(STAGES.UPLOAD);
    setFile(null);
    setProfile(null);
    setResult(null);
    setError(null);
  }, []);

  // Stage 1 → 2: Upload → Profiling
  const handleFileSelected = useCallback(async (f) => {
    setFile(f);
    setStage(STAGES.PROFILING);
    try {
      const prof = await profileDataset(f);
      setProfile(prof);
      setStage(STAGES.PROFILED);
    } catch (err) {
      setError(err);
      setStage(STAGES.ERROR);
    }
  }, []);

  // Stage 2 → 3: Profiled → Generating
  const handleGenerate = useCallback(async () => {
    setStage(STAGES.GENERATING);
    try {
      const res = await generatePipeline(file);
      setResult(res);
      setStage(STAGES.RESULTS);
    } catch (err) {
      setError(err);
      setStage(STAGES.ERROR);
    }
  }, [file]);

  return (
    <div className="min-h-screen bg-[#0f0f0f] flex flex-col">
      {/* Header */}
      <header className="border-b border-zinc-800/60 py-3 px-6">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <button
            onClick={reset}
            className="text-zinc-400 hover:text-white transition-colors text-sm font-medium cursor-pointer"
          >
            🧠 CogniPipe
          </button>
          <span className="text-zinc-600 text-xs font-mono">v0.1.0</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-4 py-12">
        {stage === STAGES.UPLOAD && (
          <UploadStage onFileSelected={handleFileSelected} />
        )}

        {stage === STAGES.PROFILING && (
          <div className="stage-enter flex flex-col items-center gap-6">
            <div className="w-16 h-16 rounded-full border-4 border-zinc-800 border-t-blue-500 animate-spin" />
            <div className="text-center">
              <h2 className="text-xl font-bold text-white">
                Profiling your dataset...
              </h2>
              <p className="text-zinc-400 mt-1 text-sm">
                Analyzing columns, types, and distributions
              </p>
            </div>
          </div>
        )}

        {stage === STAGES.PROFILED && profile && (
          <ProfilingStage profile={profile} onGenerate={handleGenerate} />
        )}

        {stage === STAGES.GENERATING && <GeneratingStage />}

        {stage === STAGES.RESULTS && result && (
          <ResultsStage result={result} onReset={reset} />
        )}

        {stage === STAGES.ERROR && (
          <ErrorCard error={error} onRetry={reset} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800/60 py-3 px-6">
        <p className="text-center text-zinc-600 text-xs">
          CogniPipe — Automated ML Pipeline Generation
        </p>
      </footer>
    </div>
  );
}
