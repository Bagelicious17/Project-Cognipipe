import { useEffect, useState, useRef } from "react";

const MESSAGES = [
  "Analyzing feature distributions...",
  "Engineering optimal transformations...",
  "Selecting model architectures...",
  "Tuning hyperparameter recommendations...",
  "Evaluating cross-validation strategies...",
  "Assembling your pipeline...",
];

export default function GeneratingStage() {
  const [progress, setProgress] = useState(0);
  const [msgIndex, setMsgIndex] = useState(0);
  const startRef = useRef(Date.now());

  // Fake progress: ramps quickly to ~85%, then crawls
  useEffect(() => {
    const interval = setInterval(() => {
      const elapsed = (Date.now() - startRef.current) / 1000;
      // Fast initial ramp, then asymptotic approach to 92%
      const p = Math.min(92, 15 * Math.sqrt(elapsed));
      setProgress(p);
    }, 200);
    return () => clearInterval(interval);
  }, []);

  // Rotate messages every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setMsgIndex((i) => (i + 1) % MESSAGES.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="stage-enter flex flex-col items-center gap-8 w-full max-w-xl mx-auto relative z-10">
      {/* Spinner */}
      <div className="relative flex items-center justify-center">
        <div className="w-16 h-16 rounded-full border-4 border-[#E7E5E4] dark:border-[#44403C] border-t-[#3B82F6] dark:border-t-[#60A5FA] animate-spin" />
        <span className="absolute text-2xl z-20">⚙️</span>
      </div>

      {/* Title */}
      <div className="text-center">
        <h2 className="text-3xl font-heading font-semibold text-[#292524] dark:text-[#F3F2ED] tracking-tight">
          Generating Pipeline
        </h2>
        <p className="text-[#78716C] dark:text-[#a8a29e] mt-2 text-sm font-medium">
          This usually takes 30–90 seconds
        </p>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-[#FFFFFF] dark:bg-[#292524] rounded-lg p-6 border border-[#E7E5E4] dark:border-[#44403C] shadow-sm">
        <div className="w-full h-3 rounded-full bg-[#F5F5F4] dark:bg-[#1C1917] overflow-hidden">
          <div
            className="h-full rounded-full bg-[#3B82F6] dark:bg-[#60A5FA] animate-stripes transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-[#78716C] dark:text-[#a8a29e] font-mono text-xs text-right mt-3">
          {Math.round(progress)}%
        </p>
      </div>

      {/* Rotating Message */}
      <p
        key={msgIndex}
        className="text-[#292524] dark:text-[#F3F2ED] text-base font-medium animate-fade-in"
      >
        {MESSAGES[msgIndex]}
      </p>

      {/* Elapsed Timer */}
      <ElapsedTimer startRef={startRef} />
    </div>
  );
}

function ElapsedTimer({ startRef }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [startRef]);

  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;

  return (
    <p className="text-[#78716C] dark:text-[#a8a29e] text-xs font-mono">
      Elapsed: {mins > 0 ? `${mins}m ` : ""}
      {secs}s
    </p>
  );
}
