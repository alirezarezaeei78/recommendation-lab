# Recommendation Explorer: frozen protocol

Declared before downloading/fitting/evaluation on 19 October 2026.

Question: can a simple learned item-neighborhood recommender improve a fixed
future-interaction ranking task over training-only popularity, and can a user
inspect the same ranking through a local app and export?

This is an established-method local experiment, not a new algorithm or a claimed
reproduction of the original paper's exact implementation/results.

## Data and chronology

MovieLens100K from GroupLens, research use only. Read the provider README first.
Do not redistribute ratings, users, movie records, model matrices, row-level
predictions or reconstructable artifacts. Local private audit only. Public code,
hashes, this protocol and aggregate results are permitted artifacts. Acknowledge
Harper and Konstan (2015), DOI10.1145/2827872. No commercial data rights implied.
The default runnable app uses an independently authored synthetic catalog.

Read only u.data. Determine cutoff as the timestamp at position79999 of100000
sorted timestamps, without looking at ratings. All rows at or before cutoff train;
strictly later rows evaluate. No random row split. Fix this one cutoff and the
following parameters before seeing results; no tuning or model selection.

Train catalog = all items observed in training, including items with no positive
rating. Positive interaction = rating>=4; each user-item like is binary. Exclude
every training-rated item from that user's candidates, including ratings<4.
Unobserved items are unknown, not established dislikes. Candidate set is the
entire train catalog minus seen items, identical for both methods; no sampling.
Future item availability uses train observation only. Cold test items are outside
the catalog and separately counted. Do not use movie metadata or demographics.

## Fixed models and metrics

Popularity: distinct training-positive-user count per item.
Item-kNN: binary-positive user-item matrix B; c=B.T@B; similarity(i,j)=
c(i,j)/sqrt(c(i,i)c(j,j)) * c(i,j)/(c(i,j)+10), zero diagonal. Keep the20 largest
positive source similarities per candidate column, with lexical item-ID tie break.
Score(candidate)=sum of retained similarities from the user's liked training items.
If all eligible scores are zero, use popularity and explicitly flag fallback.
All ranking ties: score descending, popularity descending, lexical ID ascending.

Primary: macro NDCG@10 over warm users with at least one future-positive,
train-catalog, unseen item. Binary gain, discount1/log2(rank+1), IDCG over
min(10, number of eligible positives). Recall@10 denominator is all eligible
positives, not min(10, positives). Report the same eligible-user count for both.
Secondary: Recall@10 and recommendation catalog coverage; all-eligible-user
results including cold users; cold-user/item and excluded-positive counts.
Report the warm restriction prominently: the session-oriented chronological split
can exclude many cold users. This is offline observed-rating relevance, not CTR,
conversion, causal benefit or deployment performance. No universal winner assumed.

Deterministic fits, no stochastic seeds or repeated fits needed for uncertainty;
same-input repeat is a reproducibility check, not a confidence interval. Save private
row-level predictions, train/test hashes, configuration, convergence status (closed
form/no iterative optimization), dependencies, hardware and actual stdout.
After the first evaluation, further inspection is development-visible/exploratory.

Budget: CPU, existing hardware, max20minutes including fit/evaluation; no paid compute.
Dense item similarity is quadratic in catalog size: this implementation caps input
at3000items and500000rows, rejects larger tasks, and is not a production-scale engine.

## Gates

Independent tiny cosine/shrinkage and ranking-metric oracles; deterministic ties;
same candidates; strict train/test timestamps; seen/unknown/empty input handling;
finite values; reordering invariance; reproducibility; local API/browser workflow;
JSON/CSV parity; malformed input, cross-origin and oversized input checks; responsive
layout and keyboard labels. Public assets contain authored synthetic examples only.
