# Relevance/diversity control with inspectable exports

This release adds a greedy MMR reranker to the existing trained recommendation model. A local CLI lets a reader adjust a declared relevance weight and export each selected item's original rank, raw/normalized relevance, maximum similarity to earlier selections and final decision utility. The original app and earlier experiments remain available.

## Run the safe local workflow

Python3.12; install CPU PyTorch2.4.0 from its official index, then the pinned dependencies:

```sh
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python diversity_cli.py --user-id D01 --lambda-value 1 --n 5 --output private/relevance
python diversity_cli.py --user-id D01 --lambda-value 0.5 --n 5 --output private/diverse
python -m unittest -v
```

Defaults fit the small, independently authored synthetic interaction example and use authored synthetic tags. This is a working local runner, not a hosted demo. JSON stdout and saved JSON/CSV match. The parameter can reorder items without changing the set (observed for this five-item D01 example); it need not improve every displayed list metric. D00's five-item list was unchanged at1,.5,0. No synthetic example is used to claim real-user improvement.

Supply both `--data interactions.csv` and `--catalog tags.json` to use your own licensed local data. CSV columns are exactly `user,item,rating,timestamp`; catalog JSON maps every item ID to a nonempty string-tag list. No paid key/network processing. Unknown user IDs produce an explicit training-popularity fallback which is then reranked. Invalid lambda, malformed catalog, missing tags, invalid IDs and invalid list lengths are rejected. Empty candidate sets produce an empty ranking and undefined/null diversity. CSV still has its header. This runner's pairwise similarity matrix is deliberately limited by the existing3000-item local model cap; it is not a scalable production service.

## Decision rule

All unseen items in the fitted training catalog remain candidates. Each query's fixed raw relevance scores are minâ€“max normalized to[0,1]; constant scores map to0. The first item is the highest base relevance, including at lambda0. Later steps maximize `lambda*normalized_relevance - (1-lambda)*maximum_tag_cosine_to_selected`. Ties use raw relevance, training popularity, then ascending ASCII ID. Lambda1 preserves the original ranking exactly. Lower lambda emphasizes a declared metadata redundancy penalty; there is no guarantee that the final diversity or catalog coverage changes monotonically.

## Reproduce the restricted research analysis

Download MovieLens100K from the original GroupLens provider locally and observe its research-only licence. Do not commit its ratings, item catalog, demographics, individual predictions or learned factors. See the full [protocol](PROTOCOL.md) and [sources](sources.json). Genre metadata is assumed known at cutoff; its actual historical availability is not established.

The published analysis reloads the exact23October factor arrays after SHA256 validation against `results/2026-10-23.json`. If those private fitted arrays are unavailable, `benchmark_bpr.py` documents how to train them from the declared provider input; verify hashes before reproducing this frozen-factor sweep. A new fit with differing factors must be reported as a separate run, not substituted silently.

```sh
python benchmark_diversity.py --data /local/ml-100k/u.data --items /local/ml-100k/u.item --factors /local/original23October-audit --audit /local/new-private-audit --output experiments/2026-11-02/results.json
python plot_diversity.py
```

No new benchmark fitting/tuning; three previously fitted seeds, five preset lambda values, same seen-item exclusion and full candidates.95warm primary users,192additional cold users; all287eligible secondary results are retained separately. One warm user has no eligible fitted positive profile and also falls back, explicitly counted. Reused known chronological evaluation is exploratory, not independent validation. Static genre distance is a proxy, not relevance labels or user satisfaction. All negative outcomes remain in `results.json`; across-seed minâ€“max are descriptive ranges, not confidence intervals.

Checks include exact lambda1 baseline reproduction, candidate hashes/no seen items, independent scalar NDCG/Recall, independent pairwise set-cosine diversity, hand-computed MMR decisions, deterministic ties and exact CLI exports. Actual executed commands/results and checks accompany the release. Code follows the repository MIT licence; source paper and dataset retain their respective rights. No scientific novelty, SOTA, deployment, commercial data licence or business-outcome claim.

## Actual warm-primary results

Seed means on95warm users; all287-user secondary results are retained in JSON.

| λ | NDCG@10 | Recall@10 | Genre ILD | Catalog coverage |
| --- | --- | --- | --- | --- |
| 1 | 0.1710 | 0.1146 | 0.7114 | 21.60% |
| 0.75 | 0.1288 | 0.0736 | 0.9106 | 20.46% |
| 0.5 | 0.0997 | 0.0456 | 0.9929 | 17.47% |
| 0.25 | 0.0946 | 0.0397 | 0.9999 | 16.81% |
| 0 | 0.0946 | 0.0397 | 0.9999 | 16.81% |

The declared genre diversity proxy increased as relevance and catalog coverage fell. Lambda.25 and0 produced identical results here. No lambda was selected as best. At lambda1, original BPR ranking scores/metrics reproduce exactly.71local tests passed; actual logs are included. The restricted benchmark sweep used zero new fits and completed in the declared20minute budget.
