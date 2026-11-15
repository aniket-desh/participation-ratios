# Batch 0: generator, filter and covariance-pipeline validation (Gate A)

Gate A passes. Two independent filters agree on every one of the 4,096 contexts at T = 6 to within floating-point round-off. The plan's generator-only table is reproduced to its six printed digits, sampled moments agree with exact moments within sampling error at $q_2=0.25$, and every prescribed encoding produces the covariance $E\Sigma_bE^\top$ to $10^{-14}$. These are calculations on the generator and on imposed encodings, with no training, so a pass here certifies the measurement pipeline and implies nothing about transformers (plan §5).

Everything below comes from [`results.json`](results.json), produced by [`run.py`](run.py) on the coordinator CPU in well under a minute (`uv run python experiments/batch0-generator-validation/run.py`). The generator is the two-factor binary process of plan §3.1 with $(\rho_1,q_1)=(0.8,0.5)$ and $\rho_2=0.8$, using token permutation `[2,0,3,1]`.

## Generator-only benchmark, exact enumeration at T = 6

| $q_2$ | $v_1$ | $v_2$ | factored PR | joint (Walsh) PR | isometric-$u$ PR | role |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 0.414752 | 0.139888 | 1.605662 | 1.925329 | 1.157320 | main sweep |
| 0.35 | 0.414752 | 0.245232 | 1.876218 | 2.392460 | 1.513331 | **held out** |
| 0.50 | 0.414752 | 0.414752 | 2.000000 | 2.684611 | 2.000000 | main sweep |
| 0.65 | 0.414752 | 0.579200 | 1.946712 | 2.694961 | 1.675545 | **held out** |
| 0.80 | 0.414752 | 0.736687 | 1.854991 | 2.626951 | 1.340662 | main sweep |

The $q_2\in\{0.25,0.5,0.8\}$ rows match the plan's §3.4 table to all printed digits. The held-out reliabilities 0.35 and 0.65 are locked here, before any model was analyzed. The "isometric-$u$ PR" column is the PR of the covariance of the readout coordinates $u_a=\operatorname{atanh}(\rho_aq_am_a)$. It differs sharply from the $m$ column in the anisotropic conditions, which makes the coordinate system of a learned metric testable (Batch 2 uses this).

At the Batch 2 context length T = 16 the same quantities come from 2,000,000 Monte Carlo contexts (10 batches; standard errors from batch means). The factored PR is 1.6202 ± 0.0004, 1.8823 ± 0.0003, 1.99998 ± 0.00001, 1.9481 ± 0.0002 and 1.8579 ± 0.0003 for $q_2$ = 0.25, 0.35, 0.5, 0.65 and 0.8. At T = 16 the q2 = 0.25 value is therefore 1.620 rather than the T = 6 value of 1.606, which is why Batch 2 recomputes its references on its own evaluation contexts.

## Checks and what they would have caught

- **Independent fact-check of this write-up** against `results.json`: all headline numbers held. It corrected the moment count (three distinct moments, not four), scoped the sampled-moment agreement to $q_2=0.25$, and removed an unsupported comparison with Batch 2, whose results did not exist yet.

- **Two filters.** The matrix HMM filter and the scalar magnetization recursion (derivations §1.3) agree to 0 at printed precision (max difference below $10^{-15}$), and the scalar filter's product of predictive probabilities reproduces every context probability. This would catch a wrong emission or transition convention, or a token-label mix-up.
- **Normalization and tower property.** $\sum_cp_c-1$ and $\max|\mathbb E[b]-\pi|$ are 0 at printed precision for every $q_2$.
- **Walsh spectrum.** The direct 4-state posterior covariance has eigenvalues $v_1/4$, $v_2/4$, $v_1v_2/4$ and 0 to round-off, confirming derivations §1.5.
- **Exact vs sampled moments (q2 = 0.25).** z-scores of the 2×2 covariance entries (the off-diagonal one appears twice) are (−0.1, 0.5, 0.5, 1.6) at N = 10⁴, (0.1, −2.5, −2.5, 1.1) at N = 10⁵ and (1.0, −0.2, −0.2, 2.0) at N = 10⁶. Those are three distinct correlated moments at three sample sizes, and the largest |z| is 2.5, so this is consistent with sampling error. Sampled PR is 1.6063 at N = 10⁶ against an exact value of 1.6057.
- **Prescribed encodings (exact T = 6 beliefs, q2 = 0.25).** An isometric embedding into d = 64 keeps PR at 1.605662 and lowers nPR to 0.0251. A diagonal stretch diag(3, 1) changes PR to 1.0748 at unchanged rank 2. A rotation leaves PR unchanged. The maximum deviation from $E\Sigma_bE^\top$ is at most $7\times10^{-15}$. The nonlinear encoding $(m_1,m_2,m_1m_2)$ has linear rank 3 (PR 1.9253, the joint value), showing that a two-variable manifold need not have linear rank two.
- **Imposed positional filters (plan §7.1).** With a fixed random output map, the identity $C_h=\sum_{k\ell}B_k\Gamma_{k\ell}B_\ell^\top$ matches exact enumeration to $2\times10^{-15}$ for both an exponential kernel ($\rho=0.8$, PR 1.4870) and a uniform last-3 kernel (PR 1.4528). Dropping the off-diagonal temporal correlations gives PR 1.3906 in both cases, so the independent-token approximation is visibly wrong for HMM data, as the derivations warn. The rank is 2 because the output geometry is imposed, not learned.
- **Weak-signal benchmark.** The relative errors of the leading formula and of the exact linear-filter variance against the exact posterior variance grow as $q^2$, about 4× per doubling of q. At $\rho=0.5$, T = 16 the leading formula's error is 0.17%, 0.66% and 2.6% at q = 0.05, 0.1 and 0.2. At $\rho=0.8$, T = 12 it reaches 12.7% at q = 0.2, so the expansion is already poor there. This confirms the $O(q^4)$ absolute error (and the test suite checks the slope).
- **Coupled generators (plan §4.2, c = 0.8, q1 = q2 = 0.5).** The stationary law moves from uniform to (0.276, 0.224, 0.224, 0.276) at ε = 0.05 and (0.332, 0.168, 0.168, 0.332) at ε = 0.2. The interaction coordinate $\mathbb E[S_1S_2\mid c]-m_1m_2$ has variance 0, 0.0010 and 0.0074 at ε = 0, 0.05 and 0.2. The loss gap between the exact next-token predictor and the product of its two next-token marginals is 0, 0.00017 and 0.0023 nats/token. So the coupling does change next-token prediction, but by a small amount. Whether a transformer can resolve gaps this small should be judged against the Batch 2 models' excess KL once those runs finish, before spending on the coupled-factor experiments.

## Not yet checked

- **Mess3 (plan §4.1).** Not implemented. It needs the exact matrices from the paper and its repository, which could not be fetched and verified tonight.
- **A matched-statistics pair of generators (plan §4.3).** Constructing two generators with similar belief covariance but different update dynamics is a numerical design problem that was not attempted.
