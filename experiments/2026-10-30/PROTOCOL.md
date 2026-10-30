# Frozen catalog-cold workflow protocol — 30 October 2026

Declared before fitting/evaluation. Question: can a local catalog workflow score
an item with metadata but no training interactions, and what ranking trade-offs
does its explicit hybrid make? No winner assumed, no tuning after inspection.

Use existing locally downloaded provider MovieLens100K u.data and u.item.
Research terms/no redistribution/no commercial data rights. Harper & Konstan
(2015), DOI10.1145/2827872, GroupLens Research. Public code/hashes/aggregate
metrics only. No movie/user records, models, per-user scores or metadata mirror.
Default CLI uses independently authored synthetic inputs.

Reuse global cutoff at sorted timestamp position79999 (including ties), as in
19/23/26 October. This already exposed benchmark is exploratory, not independent
validation. Catalog is every item observed by cutoff; metadata is the provider's
fixed genre snapshot, assumed available at that cutoff, not historically verified.
Do not use future-only items. Choose simulated cold items by
int(SHA256('catalog-cold-v1:'+item_id),16) modulo5 ==0. Remove ALL their past
interactions before fitting counts, BPR and positive query histories. Keep every
remaining past interaction, rating>=4 as positive. Retain full original past-seen
exclusion for every method, including masked items: this is deliberately simulated
missing training interactions with an oracle seen-item filter, NOT genuine first
launch. No timestamps/labels from future interactions affect models or candidates.
Fit genre-IDF on known catalog metadata only, including catalog-cold metadata.

Train three BPR models from scratch on retained interactions: seeds13/29/47,
32 factors,30 epochs, batch2048, Adam .01, regularization1e-4, CPU2threads;
existing bpr.py implementation unchanged. Save actual objective/epochs/stops.
Warm items require at least one retained interaction, even if no positive ratings.

Content: binary genre vector, smooth IDF log((1+N)/(1+df))+1, item L2 norm.
User profile is mean of retained-positive genre vectors then L2 normalized.
Score by cosine. Empty/zero-signal content falls back to train popularity.
Collaborative reference: fitted BPR user dot products on warm items; cold items
have score0, sorted AFTER warm items, never random untrained vectors. Unknown
or untrained users fall back to popularity. All score ties: popularity then ID.
Popularity has0 for cold items. Content-only ranks all known candidates.

Hybrid: equal-weight reciprocal-rank sum 1/(60+rank) of content positive-score
list and collaborative warm-item list. Lists cover all candidates available to
that component, not a top10 truncation. Cold items receive content contribution
only. No positive metadata signal -> collaborative; no collaborative profile ->
content; neither -> popularity. This authored heuristic is untuned and not a
reproduction/new algorithm. CLI custom histories use a mean of learned warm item
factors for the collaborative component; the benchmark uses fitted user vectors.
Expose that distinction, method, per-component ranks/scores, cold flag, fallback
reason and candidate count in JSON/CSV. Scores are ranking signals, not chances.

Primary cohort: future users with >=1 retained training positive and >=1 future
positive available unseen item. Exact same full catalog-minus-original-seen
candidates across all four methods. Positives rating>=4. No negative sampling
at evaluation. Primary macro NDCG@10; Recall@10 denominator all eligible positives.
Report same rankings against cold-only and warm-only relevance (separate eligible
user cohorts with >=1 corresponding positive); IDCG uses min(10,relevant_count).
These subgroup metrics are NOT rerankings or directly interchangeable cohorts.
Report denominators, positive counts, unscorable items, item coverage all/cold/warm,
all excluded users/future-only positives, per-seed metrics and arithmetic means.
Seed variation is optimizer sensitivity, not a confidence interval. Keep failures.

Save local restricted factors, predictions, partition IDs, input/model/code hashes,
environment/stdout. Independently recompute ranking metrics and replay outputs.
Check IDF/cosine and hybrid arithmetic with scalar oracles, no seen/duplicate ranks,
unknown/empty/malformed/all-seen inputs, deterministic ties, cold metadata without
interactions, unchanged fitted IDF after adding item, JSON/CSV exact export parity.
Actual CLI command workflow/error/export checks required. Local runner only;
existing browser app remains unchanged. Static measured plot/table plus workflow.
Existing CPU,20minute total benchmark budget, no paid keys or compute. No latency,
CTR/revenue/causal/deployment claims. Freeze protocol/code hashes before fitting.
