# Score scale, rank and probability: a worked example

A ranking score of 0.8 does not by itself establish an 80% chance of a click.
This short tutorial executes hand-set vector arithmetic to separate ordering,
score margins, a pairwise preference model and event-probability calibration.

## Reproduce

From the repository root, Python 3.12 (standard library only for the arithmetic):

```sh
python education/2026-11-14/tutorial.py --output education/2026-11-14/results.json
python -m unittest discover -s education/2026-11-14 -v
```

For the original static figure, install Matplotlib 3.10.0 and run:

```sh
python education/2026-11-14/plot_tutorial.py
```

The actual run used Python 3.12.9 on the CPU. Plotting additionally used
Matplotlib 3.10.0, NumPy 1.26.4 and the existing local font installation.
No model fitting, provider dataset, restricted MovieLens artifact, network
access, random seed or paid key is required by these commands.

## What the arithmetic establishes

For the hand-set user [1, 2], take the dot product with A=[0.4, 0.2],
B=[-0.1, 0.2], C=[0.2, -0.2]. The scores are 0.8, 0.3 and -0.2
(rounded for presentation; JSON retains the actual floating-point outputs).
Applying 10*s+7 gives 15, 10 and 5. Both rank A > B > C.

A common strictly increasing transform preserves a fixed user's ordering in
real arithmetic. The checked example retains all pairwise comparisons.
This says nothing about score comparability between users, models or candidate
policies. Numerical saturation and rounding can create floating-point ties.

The BPR formulation models a pair preference with sigmoid(s_A-s_B). Here the
margin 0.5 maps to approximately 0.6225. After positive affine rescaling, the
margin 5 maps to approximately 0.9933, while the order is unchanged. A common
shift cancels in the pair difference; positive scaling does not. The negative
log-sigmoid term changes from about 0.4741 to 0.0067. Rescaling scores after
scoring therefore preserves this ranking, **not the BPR training objective**.
We did not refit a model or claim equivalent regularized parameter estimates.

A sigmoid applied to the single-item score is a different calculation:
sigmoid(0.8) is about 0.6900. Merely putting numbers in [0,1] does not validate
them as click probabilities. The BPR pair-preference assumption also does not
establish a calibrated probability of clicking an individual displayed item.

## Common mistake and practical next step

Do not append a percent sign to a ranking score merely because it lies in
[0,1], or assume a sigmoid fixes the meaning. If an application needs a click
probability, define the click event, prediction window and exposure conditions.
Fit an appropriate probabilistic model or calibrator and evaluate on separate,
representative outcomes; account for the actual logging/selection policy.
Inspect reliability with its sample support rather than treating one rounded
score as a guarantee. No such outcome data or calibration result exists here.

## Verification, sources and limits

Six checks cover exact rational dot/affine arithmetic, all pairwise signs and
rank order, independent 60-digit Decimal sigmoid arithmetic, common-shift
cancellation versus scale-dependent likelihood, item-versus-pair sigmoid,
and ties/extremes/nonfinite rejection. See actual stdout, results and frozen
protocol. The tests validate arithmetic, not recommender quality.

Primary sources checked 14 November 2026:

- Rendle, Freudenthaler, Gantner and Schmidt-Thieme, **BPR: Bayesian Personalized
  Ranking from Implicit Feedback**, UAI 2009, sections 4.1 and 4.3/4.3.1.
  [arXiv v1](https://arxiv.org/abs/1205.2618v1) was deposited 9 May 2012.
  Used for pair-preference likelihood, score differences and factor dot products;
  no paper-reported benchmark or reproduction claim is made.
- [scikit-learn 1.4.2: Probability calibration](https://scikit-learn.org/1.4/modules/calibration.html),
  definition, calibration curves and separation of model/calibrator data.

All vectors, calculations, code and figure design are original authored
teaching material under the repository MIT licence. Papers/documentation retain
their own rights. This is established teaching with synthetic inputs, no learned
accuracy result, scientific novelty, click/revenue outcome or deployment claim.
