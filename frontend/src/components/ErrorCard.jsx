export default function ErrorCard({ error, onRetry }) {
  const detail =
    typeof error?.detail === "object" ? error.detail : { detail: String(error?.detail || error) };
  const stage = detail.stage || "unknown";
  const message = detail.detail || "An unexpected error occurred.";

  return (
    <div className="stage-enter flex flex-col items-center gap-6 w-full max-w-xl mx-auto">
      <div className="w-full rounded-2xl bg-red-500/10 border border-red-500/30 p-6 space-y-3">
        <div className="text-center">
          <div className="text-3xl mb-2">❌</div>
          <h2 className="text-xl font-bold text-red-400">
            Pipeline Generation Failed
          </h2>
          <p className="text-zinc-500 text-xs uppercase tracking-wider mt-2">
            Failed during: {stage}
          </p>
        </div>

        <div className="bg-zinc-900/60 rounded-xl p-4">
          <p className="text-zinc-300 text-sm font-mono leading-relaxed break-words">
            {message}
          </p>
        </div>
      </div>

      <button
        onClick={onRetry}
        className="w-full py-3.5 rounded-xl font-semibold text-base
                   bg-zinc-800 hover:bg-zinc-700 text-zinc-300
                   border border-zinc-700 hover:border-zinc-600
                   transition-all duration-300 cursor-pointer"
      >
        ← Try Again
      </button>
    </div>
  );
}
