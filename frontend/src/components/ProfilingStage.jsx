const TASK_BADGE = {
  binary_classification: { label: "Classification", color: "bg-[#3B82F6]/10 text-[#3B82F6]" },
  multiclass_classification: { label: "Classification", color: "bg-[#3B82F6]/10 text-[#3B82F6]" },
  regression: { label: "Regression", color: "bg-[#10B981]/10 text-[#10B981]" },
  time_series: { label: "Time Series", color: "bg-[#F59E0B]/10 text-[#F59E0B]" },
  clustering: { label: "Clustering", color: "bg-[#78716C]/10 text-[#78716C]" },
};

export default function ProfilingStage({ profile, onGenerate }) {
  const ds = profile?.dataset || {};
  const badge = TASK_BADGE[ds?.likely_task_type] || {
    label: ds?.likely_task_type || "Unknown",
    color: "bg-[#78716C]/10 text-[#78716C]",
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
      <div className="text-center relative z-10">
        <div className="text-5xl mb-3 drop-shadow-sm">📊</div>
        <h2 className="text-3xl font-heading font-semibold text-[#292524] dark:text-[#F3F2ED] tracking-tight">
          Profiling Complete
        </h2>
        <p className="text-[#78716C] dark:text-[#a8a29e] mt-2 font-mono text-sm">
          Analyzed in {profile.profiling_duration_seconds?.toFixed(2)}s
        </p>
      </div>

      {/* Summary Card */}
      <div className="w-full rounded-lg bg-[#FFFFFF] dark:bg-[#292524] border border-[#E7E5E4] dark:border-[#44403C] p-6 space-y-6 shadow-sm">
        {/* Shape & Task */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[#78716C] dark:text-[#a8a29e] text-xs uppercase tracking-wider font-semibold">
              Dataset Shape
            </p>
            <p className="text-[#292524] dark:text-[#F3F2ED] text-xl font-medium mt-1">
              {(ds?.num_rows ?? ds?.n_rows)?.toLocaleString() ?? '...'} × {ds?.num_columns ?? ds?.n_columns ?? '...'}
            </p>
          </div>
          <span
            className={`px-3 py-1 rounded-full text-xs font-semibold ${badge.color}`}
          >
            {badge.label}
          </span>
        </div>

        {/* Divider */}
        <div className="border-t border-[#E7E5E4] dark:border-[#44403C]" />

        {/* Target */}
        <div>
          <p className="text-[#78716C] dark:text-[#a8a29e] text-xs uppercase tracking-wider font-semibold">
            Suspected Target
          </p>
          <p className="text-[#292524] dark:text-[#F3F2ED] font-mono text-base mt-1">
            {ds?.suspected_target_column || "—"}
          </p>
        </div>

        {/* Issues */}
        {topIssues.length > 0 && (
          <>
            <div className="border-t border-[#E7E5E4] dark:border-[#44403C]" />
            <div>
              <p className="text-[#78716C] dark:text-[#a8a29e] text-xs uppercase tracking-wider font-semibold mb-2">
                Issues Detected
              </p>
              <ul className="space-y-2">
                {topIssues.map((issue, i) => (
                  <li
                    key={i}
                    className="text-sm text-[#292524] dark:text-[#F3F2ED] bg-[#F5F5F4] dark:bg-[#1C1917] border border-[#E7E5E4] dark:border-[#44403C] rounded-md px-3 py-2"
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
            <div className="border-t border-[#E7E5E4] dark:border-[#44403C]" />
            <div className="flex items-center gap-2 text-[#10B981] dark:text-[#34D399] text-sm font-medium">
              <span>✅</span>
              <span>No critical issues detected</span>
            </div>
          </>
        )}
      </div>

      {/* Action */}
      <button
        onClick={onGenerate}
        className="w-full py-3 rounded-md font-medium text-base tracking-wide
                   bg-[#292524] dark:bg-[#a8a29e] text-white dark:text-[#1C1917]
                   transition-all duration-200 ease-out cursor-pointer hover:opacity-90 shadow-sm"
      >
        🚀 Generate Pipeline
      </button>
    </div>
  );
}
