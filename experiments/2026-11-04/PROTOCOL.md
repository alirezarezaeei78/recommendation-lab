# Frozen batch-serving and export protocol

Declared 4 November 2026 before executing the new batch experiment. Results are
unknown. This is a serving-correctness integration, not a new model or speed claim.

## Question and intervention

Can a local batch interface preserve the candidate rules, ranking evidence and
metrics of the existing offline BPR/popularity evaluator across batch boundaries?
Add a documented batch API and CLI with explicit cold-user fallback, all-seen
handling, validation before scoring, and lossless JSON/CSV exports.

The new scorer widens the existing float32 factors to float64, then accumulates
factor products in ascending dimension order. It vectorizes across queries and
items, using a bounded block of at most128 queries. It deliberately fixes reduction
order across batch sizes. It is not a claim that a BLAS matrix-matrix call is
bitwise equivalent to a matrix-vector call, nor that this implementation is faster.
Do not refit or tune on the benchmark. Popularity stays the existing train-only
positive-user count. Exclude every past-rated item, including disliked items;
additional user-declared exclusions are allowed in the app but absent in the
matched benchmark. Sort by descending score, descending popularity, ascending
ASCII item ID. Unknown valid users and known users without fitted positive factors
use an explicitly reported popularity fallback for a BPR request. All-seen users
return an empty ranking and retained query status. No personal user selection task.

## Data, frozen artifacts and exposure

Reuse the 19/23 October MovieLens100K chronological task: cutoff889237269,
80003 past rows,19997 strictly later rows; rating>=4 is observed relevance.
Verify the complete source/train/test hashes against results/2026-10-23.json.
Reload all three original BPR factor archives (seeds13,29,47), verify their hashes
and canonical train-derived row mappings. No new benchmark training, seed choice,
new data split, candidate sampling or hyperparameter selection.

Primary cohort95 eligible warm users; secondary192 eligible cold users. Preserve
all287 eligible and14 excluded future-user counts,95 future-positive pairs outside
the1616-item training catalog. This repeatedly exposed task is exploratory
engineering verification, not independent predictive validation. Missing feedback
is not a known dislike. Frozen popularity is the baseline, BPR the collaborative
reference. No satisfaction, revenue or causal claim.

GroupLens research terms require attribution and prohibit redistribution without
permission; commercial use needs separate permission. Public artifacts contain
code, hashes, aggregate numerical results and independently authored synthetic
examples only. Never publish ratings, demographics, provider item records, fitted
benchmark factors, per-user scores/rankings or reconstructable artifacts. Keep
those in the private audit directory. The default CLI trains on authored synthetic
interactions and serves synthetic profiles, separate from this research task.

## Prespecified conditions and acceptance checks

For each seed, both BPR and popularity, run batch sizes1,7,32,128 over the same
ordered287 eligible users and candidate catalogs. Compare every canonical score
against an independent Python math.fsum dot-product oracle (absolute tolerance
1e-12), and compare BPR scores against the original float32 matrix-vector scorer
(absolute tolerance2e-5 plus relative2e-6). The latter is numerical compatibility,
not an exact-equality promise. Record maximum errors and any rank mismatches.
Never hide a mismatch by rounding scores or silently swapping scorers.

Require exact score/order/status/export equality across batch sizes for the new
fixed-order scorer. Compare top10 order and NDCG@10/Recall@10/catalog coverage
against the old evaluator separately. If ranks differ, retain the discrepancy and
its metric consequences; do not assert parity. Independent set membership, binary
DCG/recall and CSV round-trip oracles cover candidate masks and full-precision
exports. Hand-authored exact tie/near-tie fixtures, unknown/no-positive profiles,
empty candidates, invalid query collections/IDs/items/counts and unchanged model
arrays are required. CLI workflow tests must execute a subprocess and compare its
actual JSON and CSV files; an HTTP test exercises the loopback batch route.

Save protocol hash before execution, actual stdout, source/factor/data hashes,
software/hardware versions and private audit hashes. Report all seeds, conditions,
negative results and uncertainty limits. Across-seed ranges are descriptive, not
confidence intervals. No measured latency comparison; elapsed run time is only an
audit value. Evaluation wall budget20minutes, stop and preserve incomplete work if
exceeded. No LinkedIn publication in this preparation run.

## Sources

- Rendle et al., BPR, UAI2009; arXiv1205.2618v1 posted9May2012.
  https://arxiv.org/abs/1205.2618v1 . The existing minibatch Adam variant is not the
  original paper's exact optimizer or a reproduction of paper-reported results.
- NumPy1.26 matmul reference, matrix product signature and BLAS implementation
  note: https://numpy.org/doc/1.26/reference/generated/numpy.matmul.html . Numeric
  compatibility is measured here, not guaranteed by this documentation.
- MovieLens100K provider README, accessed4November2026:
  https://files.grouplens.org/datasets/movielens/ml-100k/README . Attribution:
  Harper and Konstan2015, The MovieLens Datasets: History and Context,
  DOI10.1145/2827872, GroupLens Research, University of Minnesota.
