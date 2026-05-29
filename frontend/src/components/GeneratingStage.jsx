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
    <div className="stage-enter flex flex-col items-center gap-8 w-full max-w-xl mx-auto">
      {/* Spinner */}
      <div className="relative flex items-center justify-center">
        <div className="w-20 h-20 rounded-full border-4 border-zinc-800 border-t-blue-500 animate-spin" />
        <span className="absolute text-2xl">⚙️</span>
      </div>

      {/* Title */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white">Generating Pipeline</h2>
        <p className="text-zinc-400 mt-1 text-sm">
          This usually takes 30–90 seconds
        </p>
      </div>

      {/* Progress Bar */}
      <div className="w-full">
        <div className="w-full h-2 rounded-full bg-zinc-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-zinc-500 text-xs text-right mt-1.5">
          {Math.round(progress)}%
        </p>
      </div>

      {/* Rotating Message */}
      <p
        key={msgIndex}
        className="text-zinc-300 text-base font-medium animate-fade-in"
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
    <p className="text-zinc-600 text-xs font-mono">
      Elapsed: {mins > 0 ? `${mins}m ` : ""}
      {secs}s
    </p>
  );
}
