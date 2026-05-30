const STAGE_LABELS = {
  profiling: "Data Analysis",
  orchestration: "AI Processing",
  assembly: "Code Generation",
  unknown: "Processing",
};

/**
 * Catches any raw API errors that might slip through the backend sanitizer.
 */
function sanitizeMessage(msg) {
  if (!msg) return "An unexpected error occurred. Please try again.";
  const lower = msg.toLowerCase();
  if (lower.includes("429") || lower.includes("resource_exhausted") || lower.includes("quota"))
    return "The AI service has reached its usage limit for today. Please try again later.";
  if (lower.includes("403") || lower.includes("permission_denied"))
    return "The AI service is not authorized. Please contact the administrator.";
  if (lower.includes("fetch") || lower.includes("networkerror") || lower.includes("failed to fetch"))
    return "Unable to reach the server. Please check your connection and try again.";
  return msg;
}

export default function ErrorCard({ error, onRetry }) {
  const detail =
    typeof error?.detail === "object" ? error.detail : { detail: String(error?.detail || error) };
  const stage = STAGE_LABELS[detail.stage] || STAGE_LABELS.unknown;
  const message = sanitizeMessage(detail.detail);

  return (
    <div className="stage-enter flex flex-col items-center gap-6 w-full max-w-xl mx-auto">
      <div className="w-full rounded-lg bg-[#EF4444]/10 dark:bg-[#F87171]/10 border-l-4 border-l-[#EF4444] dark:border-l-[#F87171] border-y border-r border-y-[#E7E5E4] dark:border-y-[#44403C] border-r-[#E7E5E4] dark:border-r-[#44403C] p-6 text-left relative z-10">
        <div className="flex items-center gap-3 mb-4">
          <div className="text-3xl drop-shadow-sm">⚠️</div>
          <div>
            <h2 className="text-xl font-heading font-semibold text-[#EF4444] dark:text-[#F87171] tracking-wide">
              Something Went Wrong
            </h2>
            <p className="text-[#78716C] dark:text-[#a8a29e] text-xs uppercase tracking-wider font-semibold mt-1">
              Failed during: {stage}
            </p>
          </div>
        </div>

        <div className="bg-[#FFFFFF] dark:bg-[#292524] rounded-md p-4 border border-[#E7E5E4] dark:border-[#44403C] shadow-sm">
          <p className="text-[#292524] dark:text-[#F3F2ED] text-sm leading-relaxed break-words">
            {message}
          </p>
        </div>
      </div>

      <button
        onClick={onRetry}
        className="w-full py-3 rounded-md font-medium text-base tracking-wide relative z-10
                   bg-[#FFFFFF] dark:bg-[#292524] hover:bg-[#F5F5F4] dark:hover:bg-[#44403C] text-[#292524] dark:text-[#F3F2ED]
                   border border-[#E7E5E4] dark:border-[#44403C] hover:border-[#78716C] dark:hover:border-[#a8a29e]
                   transition-colors duration-200 ease-out cursor-pointer shadow-sm"
      >
        ← Try Again
      </button>
    </div>
  );
}
