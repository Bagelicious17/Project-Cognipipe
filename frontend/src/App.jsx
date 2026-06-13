import { useState, useCallback, useEffect } from "react";
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
  const [generatingStatus, setGeneratingStatus] = useState({
    progress: 0,
    message: "",
  });

  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains('dark'));
  }, []);

  const toggleDark = () => {
    if (isDark) {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
      setIsDark(false);
    } else {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
      setIsDark(true);
    }
  };

  const reset = useCallback(() => {
    setStage(STAGES.UPLOAD);
    setFile(null);
    setProfile(null);
    setResult(null);
    setError(null);
    setGeneratingStatus({ progress: 0, message: "" });
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

  // Stage 2 → 3: Profiled → Generating (streaming)
  const handleGenerate = useCallback(async () => {
    setStage(STAGES.GENERATING);
    setGeneratingStatus({ progress: 0, message: "Starting pipeline generation…" });
    try {
      const res = await generatePipeline(file, ({ progress, message }) => {
        setGeneratingStatus({ progress, message });
      });
      setResult(res);
      setStage(STAGES.RESULTS);
    } catch (err) {
      setError(err);
      setStage(STAGES.ERROR);
    }
  }, [file]);

  return (
    <div className="min-h-screen flex flex-col text-[#292524] dark:text-[#F3F2ED] transition-colors duration-300">
      
      {/* Header */}
      <header className="py-4 px-6 relative z-10 border-b border-[#E7E5E4] dark:border-[#44403C]">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <button
            onClick={reset}
            className="hover:opacity-80 transition-opacity flex items-center gap-2 cursor-pointer font-heading font-bold text-xl"
          >
            <span className="text-xl">📓</span> CogniPipe
          </button>
          
          <div className="flex items-center gap-4">
            <span className="text-[#78716C] dark:text-[#a8a29e] text-xs font-mono font-medium">v0.1.0</span>
            <button 
              onClick={toggleDark}
              className="p-1.5 rounded-md hover:bg-[#E7E5E4] dark:hover:bg-[#44403C] transition-colors cursor-pointer text-sm"
              title="Toggle theme"
            >
              {isDark ? "☀️" : "🌙"}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-4 py-12 relative z-10">
        {stage === STAGES.UPLOAD && (
          <UploadStage onFileSelected={handleFileSelected} />
        )}

        {stage === STAGES.PROFILING && (
          <div className="stage-enter flex flex-col items-center gap-6">
            <div className="w-16 h-16 rounded-full border-4 border-[#E7E5E4] dark:border-[#44403C] border-t-[#3B82F6] dark:border-t-[#60A5FA] animate-spin" />
            <div className="text-center">
              <h2 className="text-2xl font-heading font-semibold text-[#292524] dark:text-[#F3F2ED]">
                Profiling your dataset...
              </h2>
              <p className="text-[#78716C] dark:text-[#a8a29e] mt-2 text-sm font-sans">
                Analyzing columns, types, and distributions
              </p>
            </div>
          </div>
        )}

        {stage === STAGES.PROFILED && profile && (
          <ProfilingStage profile={profile} onGenerate={handleGenerate} />
        )}

        {stage === STAGES.GENERATING && (
          <GeneratingStage
            progress={generatingStatus.progress}
            message={generatingStatus.message}
          />
        )}

        {stage === STAGES.RESULTS && result && (
          <ResultsStage result={result} onReset={reset} />
        )}

        {stage === STAGES.ERROR && (
          <ErrorCard error={error} onRetry={reset} />
        )}
      </main>

      {/* Footer */}
      <footer className="py-6 px-6 relative z-10">
        <p className="text-center text-[#78716C] dark:text-[#a8a29e] text-xs font-medium font-sans">
          CogniPipe — Automated ML Pipeline Generation
        </p>
      </footer>
    </div>
  );
}
