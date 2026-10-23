# Frozen BPR integration protocol — 23 October 2026

Declared before any BPR fit or new evaluation. This reuses the development-visible
19 October MovieLens100K task; all results are exploratory, not fresh independent
validation. No paper-result reproduction or scientific novelty is claimed.

Question: does a compact BPR-MF variant improve the same chronological ranking
task, and can a user switch to its learned ranking and export the exact output?
The result is unknown. Negative or mixed outcomes remain publishable.

Research data: original GroupLens MovieLens100K, research terms read23 October.
Fetch locally from provider; no ratings/demographics/catalog records, learned
benchmark factors or per-user predictions in GitHub or a demo. Cite Harper and
Konstan2015, DOI10.1145/2827872. No commercial data rights or endorsement implied.
The app fits an independently authored synthetic catalog by default; users may
supply their own appropriately licensed local CSV. No paid API or outbound upload.

Preserve the old global chronological split exactly: cutoff at sorted timestamp
position79999; all ties at/before cutoff train, strictly later events evaluate.
Training catalog includes every train-observed item. Rating>=4 is a binary positive.
Every training-rated item, including dislikes, is excluded from recommendations
and from sampled training negatives. Unknown interactions are not known dislikes.
No future event or label enters training, catalog, similarity, popularity or factors.

Baselines: same existing popularity and item-kNN (20neighbors, shrinkage10).
New variant: dot-product user/item factors, dimension32, normal initialization
std0.05, no biases. Minimize mean softplus(-(p_u dot(q_i-q_j))) plus1e-4 times
mean sampled squared L2 norms of p_u,q_i,q_j. Adam lr0.01, batch2048,30epochs.
Each epoch shuffles all train-positive pairs and samples one uniform unobserved
training-catalog negative per pair. This is a declared minibatch Adam variant,
not the paper's exact bootstrapped SGD implementation. Users with no positives
or no unseen item receive no BPR updates. Seeds13,29,47; keep every run. No sweep,
early stopping, validation tuning or test-based model selection. Default appseed13
is fixed prospectively. Existing CPU,2torch threads,20minutes maximum per fit;
stop after the first epoch crossing the limit and record the actual stop.

Evaluation: full train catalog minus all seen items; identical candidates,
warm/cold eligibility, relevance and lexical tie rules across all methods.
Primary same95 eligible warm users; macro NDCG@10. Recall@10 denominator is
all eligible positives, not10. Secondary catalog coverage and all eligible users;
cold BPR users explicitly fall back to popularity. Report denominators, eachseed,
mean/range (not a CI), exclusions and unchanged baseline reproduction.

App: optional training-user selector uses that user's past liked/seen sets for
all methods. BPR scores the fitted user vector; custom anonymous histories remain
supported by item-kNN/popularity, with a clear BPR validation error. Empty/no-fitted
profiles explicitly fall back. No claim of new-user personalization or production
scalability. Match displayed item/rank/scores and CSV/JSON export with actual outputs.

Checks: independent finite-difference pair-loss gradient oracle, finite loss and
factors, negative-sampling exclusion including disliked items/full histories,
same seed/reordered input, no future-data influence on fits, metric hand oracles,
baseline metric reproduction, identical candidates and cohorts, no seen/duplicate
recommendations, unknown/empty/malformed inputs, API/export parity, browser workflow,
mobile/keyboard labels, full precision exports and save/reload scores.

Retain restricted inputs, all seed factors/predictions/configs/stdout and hashes
in the date package's restricted audit. Only authored code, synthetic fixtures,
aggregate results, figures, protocol and provenance are eligible public files.
Offline observed ratings do not establish clicks, revenue, causal impact or broad
model superiority. Stochastic training objective values are not validation curves.
