const TASK_BADGE = {
  binary_classification: { label: "Classification", color: "bg-blue-500/20 text-blue-400 border-blue-500/30" },
  multiclass_classification: { label: "Classification", color: "bg-blue-500/20 text-blue-400 border-blue-500/30" },
  regression: { label: "Regression", color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" },
  time_series: { label: "Time Series", color: "bg-purple-500/20 text-purple-400 border-purple-500/30" },
  clustering: { label: "Clustering", color: "bg-amber-500/20 text-amber-400 border-amber-500/30" },
};

export default function ProfilingStage({ profile, onGenerate }) {
  const ds = profile?.dataset || {};
  const badge = TASK_BADGE[ds?.likely_task_type] || {
    label: ds?.likely_task_type || "Unknown",
    color: "bg-zinc-700 text-zinc-300 border-zinc-600",
  };

  // Gather critical issues from the profile
  const issues = [];
  if (ds?.data_leakage_risks?.length > 0) {
    ds.data_leakage_risks.forEach((r) =>
      issues.push(`⚠️ Potential leakage: ${r?.column_name} — ${r?.reason}`)
    );
  }
  if (ds?.high_correlation_pairs?.length > 0) {
    issues.push(
      `🔗 ${ds.high_correlation_pairs.length} highly correlated pair(s) detected`
    );
  }
  if (ds?.duplicate_row_count > 0) {
    issues.push(`♻️ ${ds.duplicate_row_count} duplicate rows found`);
  }
  // Add missing pattern issues
  const highMissing = profile.columns?.filter(
    (c) => c.missing?.missing_percentage > 30
  );
  if (highMissing?.length > 0) {
    issues.push(
      `🕳️ ${highMissing.length} column(s) with >30% missing values`
    );
  }

  const topIssues = issues.slice(0, 3);

  return (
    <div className="stage-enter flex flex-col items-center gap-6 w-full max-w-xl mx-auto">
      {/* Header */}
      <div className="text-center">
        <div className="text-4xl mb-2">📊</div>
        <h2 className="text-2xl font-bold text-white">Profiling Complete</h2>
        <p className="text-zinc-400 mt-1">
          Analyzed in {profile.profiling_duration_seconds?.toFixed(2)}s
        </p>
      </div>

      {/* Summary Card */}
      <div className="w-full rounded-2xl bg-zinc-900/80 border border-zinc-800 p-6 space-y-5">
        {/* Shape & Task */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-zinc-500 text-xs uppercase tracking-wider font-medium">
              Dataset Shape
            </p>
            <p className="text-white text-xl font-semibold mt-1">
              {(ds?.num_rows ?? ds?.n_rows)?.toLocaleString() ?? '...'} × {ds?.num_columns ?? ds?.n_columns ?? '...'}
            </p>
          </div>
          <span
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border ${badge.color}`}
          >
            {badge.label}
          </span>
        </div>

        {/* Divider */}
        <div className="border-t border-zinc-800" />

        {/* Target */}
        <div>
          <p className="text-zinc-500 text-xs uppercase tracking-wider font-medium">
            Suspected Target
          </p>
          <p className="text-white font-mono text-base mt-1">
            {ds?.suspected_target_column || "—"}
          </p>
        </div>

        {/* Issues */}
        {topIssues.length > 0 && (
          <>
            <div className="border-t border-zinc-800" />
            <div>
              <p className="text-zinc-500 text-xs uppercase tracking-wider font-medium mb-2">
                Issues Detected
              </p>
              <ul className="space-y-1.5">
                {topIssues.map((issue, i) => (
                  <li
                    key={i}
                    className="text-sm text-zinc-300 bg-zinc-800/60 rounded-lg px-3 py-2"
                  >
                    {issue}
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}

        {topIssues.length === 0 && (
          <>
            <div className="border-t border-zinc-800" />
            <div className="flex items-center gap-2 text-emerald-400 text-sm">
              <span>✅</span>
              <span>No critical issues detected</span>
            </div>
          </>
        )}
      </div>

      {/* Action */}
      <button
        onClick={onGenerate}
        className="w-full py-3.5 rounded-xl font-semibold text-base
                   bg-blue-600 hover:bg-blue-500 text-white
                   shadow-lg shadow-blue-600/25 hover:shadow-blue-500/40
                   transition-all duration-300 cursor-pointer"
      >
        🚀 Generate Pipeline
      </button>
    </div>
  );
}
