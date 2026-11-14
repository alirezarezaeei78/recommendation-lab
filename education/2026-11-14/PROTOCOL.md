# A ranking score is not automatically a probability

Frozen teaching protocol, 14 November 2026. This is an authored deterministic
arithmetic example, not a fitted recommender, benchmark, calibration study or
reproduction of paper-reported results. No provider data, clicks, customers,
learned weights, random draws or paid service are used.

Question: what changes when the score scale changes after scoring, while the
candidate ordering stays fixed?

Inputs: one hand-set user vector [1, 2], and item vectors A=[0.4, 0.2],
B=[-0.1, 0.2], C=[0.2, -0.2]. Compute each dot product. Compare the same
three candidates before and after the strictly increasing transform 10*s+7.
Compare all pairwise signs and descending item order, retaining deterministic
item-ID tie breaks. This is a transformation of computed scores, not a claim
that another regularized model fit would produce equivalent parameters.

Also compute the stable logistic sigmoid of each single-item score and the
sigmoid of the A-minus-B margin, before and after the affine transform. The
pairwise form illustrates the BPR model's preference assumption. Neither a
hand-set item sigmoid nor a model-assumed pair preference is an observed click
rate. Changing positive scale changes margins and the BPR log-likelihood;
preserving ranking does not preserve the training objective. A common shift
cancels in pairwise differences, while it changes single-item sigmoids.

Verification before calling the illustration checked: independently compute
dot products and affine values with exact rational arithmetic; compare sigmoid
values with 60-digit Decimal exponential arithmetic; check every pairwise sign,
ranking and common-shift cancellation. Exercise negative/zero/positive inputs,
ties, nonfinite rejection and floating-point sigmoid saturation. Strict
monotonicity is a real-arithmetic property: finite precision can create ties
at extreme values. No runtime or predictive improvement claim is planned.

Save input values, actual output, check stdout, platform/dependency versions,
script/protocol hashes and an original static PNG/SVG generated from the output.
Use large type and inspect the full graphic and a phone-sized rendering.
No calibration curve will be drawn: there are no outcome labels to estimate one.

Sources: Rendle et al., UAI 2009, sections 4.1 and 4.3/4.3.1 (arXiv v1,
9 May 2012); scikit-learn 1.4.2 official probability-calibration documentation.
These are established concepts, not recent-research or novelty claims.

All teaching code, input vectors and original figures use the repository MIT
licence. Primary papers and dependency documentation retain their own rights;
no source artwork or dataset is redistributed.
