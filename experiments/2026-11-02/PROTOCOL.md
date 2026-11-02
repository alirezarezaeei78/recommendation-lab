# Frozen relevance–diversity app experiment

Declared2November2026 before executing the new sweep. No result is predetermined.

Question: what does a transparent MMR control change when relevance scores and the eligible catalog stay fixed? Ship a local CLI/API which exports the actual decision score, relevance, redundancy penalty and base rank. No hosted demo or adoption claim.

## Data and existing exposure

Use locally downloaded MovieLens100K ratings and19 binary genre flags from the original GroupLens provider. Research-only terms; no redistribution/commercial rights implied. Cite Harper and Konstan2015, DOI10.1145/2827872. Never publish original rows, item metadata/catalog, per-user rankings, factors or reconstructable artifacts. Public outputs are code, hashes and aggregates. Synthetic CLI catalog/interactions are independently authored and separate.

Reuse the known19/23/26October task exactly: cutoff889237269, all rows at/before cutoff training, strictly later rows evaluation; rating>=4 is observed-positive relevance. Use the training catalog only, exclude all training-seen items, including ratings below4. Reproduce95warm-user primary and192cold-user secondary cohorts; retain excluded denominators. Unobserved items are not known dislikes.

Reload the three original23October BPR fits (seeds13,29,47) after verifying their hashes and train-row mapping. No new optimization or tuning. These train-only factors previously informed exposed results; this extension is exploratory, not independent validation. Static genre metadata is assumed available at cutoff; its historical availability is not verified, a limitation. No future interactions or label-based similarity.

## Fixed intervention

Every lambda receives the same full unseen candidate set and raw relevance scores. Fitted warm profiles use BPR dot product; cold profiles use past-only popularity, explicitly marked fallback. Binary-genre cosine similarity (including the provider's unknown flag), not interaction overlap or future labels. Validate at least one genre tag per item. Rank all candidates; no top100 prefilter.

Per query, min–max normalize raw relevance to[0,1] using all unseen candidates; constants map to0. First choose highest relevance, including at lambda0. Then greedily maximize lambda*relevance - (1-lambda)*maximum cosine to any already selected item. Break ties by raw relevance, training popularity, ascending ASCII item ID. This is an adapted MMR decision rule, not the original paper's document-retrieval experiment or an optimal set optimizer.

Freeze lambda in{1,.75,.5,.25,0},k=10. Lambda1 must exactly reproduce original BPR/popularity ranks and metrics. No selection of a winning lambda from these outputs. Same candidate hashes/users/denominators throughout.

## Outcomes and checks

Report mean per-user NDCG@10/Recall@10, catalog coverage (unique recommended items / full training catalog), and mean intra-list diversity: mean of45pairwise binary-genre cosine distances per10-item list. Larger diversity here means a genre-distance proxy; no satisfaction, fairness, novelty or business-utility claim. Report warm primary and all287eligible secondary separately. Across-seed means/min/max are descriptive ranges, not confidence intervals. Coverage is a set measure; do not call an averaged seed count a single catalog count.

Independent scalar ranking-metric oracle; independent set-overlap/sqrt genre-cosine oracle for every selected pair; hand-calculated MMR example; candidate/no-seen/unique/tie/lambda-endpoint checks; finite/exact exported scores. Save complete restricted rankings and hashes privately. Public aggregates suppress row-level IDs. Retain negative results and all five prespecified settings.20minute evaluation budget; no timing superiority claim. No new fits except a compact default synthetic CLI demo, not part of the benchmark comparison.

## Sources

- Carbonell and Goldstein,SIGIR1998, *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*, section2. Original document method; our score normalization/genre representation/first-item rule are declared implementation choices. https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
- GroupLens MovieLens100K README, rechecked2November2026. https://files.grouplens.org/datasets/movielens/ml-100k/README
- Existing BPR fit protocol/code/results at immutable commit92a1c41b23453db4dd53a33724b3a36dfafcb6f5. Existing data SHA256 and split/factor hashes are mandatory checks, not a new test set.
