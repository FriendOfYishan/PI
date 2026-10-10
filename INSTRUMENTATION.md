# Instrumentation correction

After the first completed run, review found that whole-trajectory replay's
`collateral_regression_commits` counter omitted regressions occurring during
accepted sandbox commits. Its final collateral-regression flag was already
correct. The counter and per-trial event record were corrected and the complete
run repeated. The original summary is retained as
`summary_before_instrumentation_fix.csv`.

This correction changes diagnostic counts only. Candidate ranking, requests,
seeds, budget limits, accepted programs, learning rules and controller behavior
are unchanged. The report builder verifies that every other summary field agrees
exactly between the original and corrected run. No performance tuning followed
inspection of scores.
