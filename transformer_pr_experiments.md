# Predicting participation ratio in small transformers: experimental plan

*A staged test of whether known predictive-state statistics, together with a specified learning theory, predict the covariance spectrum of trained transformer activations.*

Prepared 15 November 2026. This is a research plan. All results below are hypotheses, analytical benchmarks, or proposed acceptance criteria; no training results are being reported. The companion document is [the theory roadmap](transformer_pr_theory.md). The scope is small transformers and controlled sequence generators.

## 1. Scientific question and the contribution to aim for

The question is whether we can predict which directions vary in a trained residual stream, and how strongly they vary, from the statistics of the prediction task and the model's learning dynamics. The principal observable is the activation covariance spectrum. Participation ratio (PR) summarizes that spectrum, but does not identify a belief geometry on its own.

Simplex's existing work already establishes much of the background: known generators predict belief geometries that are linearly recoverable from transformers [1]; architectural constraints explain some intermediate attention geometries [2]; and factored generators lead to approximately factored activation subspaces, with dimensional compression and later expansion during training [3]. Their factored-representations code reports cumulative explained variance and dimensions at variance thresholds [4]. Replicating these findings with PR would be a useful baseline and possibly a good explanatory blog post. A stronger contribution requires a prediction of a nonconstant spectrum or PR curve that survives tests on conditions not used to fit it.

The progression is deliberately conditional. We first verify statistics of the generator, then a solved learning model, then a restricted attention model, then a small nonlinear transformer. The large-width project begins only after these lower levels identify a concrete unanswered covariance question.

| Target claim | Evidence needed | What would be insufficient |
| --- | --- | --- |
| Belief statistics are computed correctly | Exact enumeration agrees with independent filtering and Monte Carlo | A plausible-looking simplex plot |
| Beliefs explain activation variance | Held-out prediction of activations from beliefs, a covariance decomposition, and spectrum comparisons | A high-scoring probe from activations to beliefs alone |
| A learning theory predicts PR | Predictions fixed before the tested training runs, including relative mode strengths | Fitting a separate encoding metric after every run |
| A transformer theory predicts nonlinear feature learning | Measured feature evolution and covariance trajectories converge under a specified width scaling | Increasing width under an unspecified parameterization |
| The geometry matters for prediction | Controlled interventions change prediction in the theoretically expected way | A PR match or an unvalidated steering result |

## 2. Observable definitions and the prediction contract

### 2.1 Specify the population before computing a covariance

The primary population is independently generated token contexts of a fixed length $T$, initialized from the generator's stationary hidden-state distribution. Filtering starts from that prior at the beginning of each context. The belief is therefore the posterior available to a finite-window predictor, not the posterior of an observer who has seen the infinite past.

The primary activation is the final-position residual stream, after a specified block and before any subsequent normalization. Other token positions and pre/post-normalization locations are separate observables. Do not silently pool them. A secondary analysis may average position-specific covariances with explicit weights; it must also account for covariance of the position-specific means if it claims to describe the pooled distribution.

Let $b(x)\in\mathbb R^r$ denote independent belief coordinates for context $x$, and $h_{\ell}(x)\in\mathbb R^d$ the measured activation. Define

$$
\Sigma_b=\mathbb E_x[(b-\bar b)(b-\bar b)^\top],
\qquad
C_{h,\ell}=\mathbb E_x[(h_\ell-\mu_\ell)(h_\ell-\mu_\ell)^\top].
$$

The expectation uses the actual context probabilities. Uniform enumeration over all possible strings is a different experiment and can give a different spectrum. Monte Carlo sampling from the generator automatically supplies probability weighting, provided contexts are drawn independently.

For nonzero positive semidefinite $C$, report

$$
\operatorname{PR}(C)=\frac{(\operatorname{tr}C)^2}{\operatorname{tr}(C^2)},
\qquad
\operatorname{nPR}(C)=\frac{\operatorname{PR}(C)}{d}.
$$

Also report the trace, the full normalized eigenvalue spectrum $p_i=\lambda_i/\sum_j\lambda_j$, and $d_{95}=\min\{k:\sum_{i=1}^{k}p_i\geq0.95\}$. PR is undefined at exactly zero covariance; mark such conditions as degenerate rather than assigning zero or one. Numerical eigenvalue clipping must be limited to a documented floating-point tolerance.

### 2.2 Separate belief content from other activation content

Fit the forward regression from beliefs to activations, rather than relying only on the familiar reverse probe from activations to beliefs. At the population least-squares optimum,

$$
h-\mu=E(b-\bar b)+\varepsilon,
\qquad
\mathbb E[\varepsilon(b-\bar b)^\top]=0,
$$

where $E\in\mathbb R^{d\times r}$. This gives

$$
C_h=C_{\mathrm{belief}}+C_{\mathrm{res}},
\quad
C_{\mathrm{belief}}=E\Sigma_bE^\top,
\quad
C_{\mathrm{res}}=\mathbb E[\varepsilon\varepsilon^\top].
$$

Fit $E$ and the mean on a probe-training set and evaluate on independent probe-test contexts. The population orthogonality condition will not hold exactly out of sample. Report the held-out cross term

$$
C_{\mathrm{cross}}
=\operatorname{Cov}(\hat h,\varepsilon)
+\operatorname{Cov}(\varepsilon,\hat h),
\qquad
C_h=C_{\mathrm{belief}}+C_{\mathrm{res}}+C_{\mathrm{cross}},
$$

using the fitted predictor $\hat h$. This checks whether apparent variance explanation is a regression overfit. In an exhaustive finite population one may compute the exact population projection instead, while retaining an independently sampled validation check.

Compare PR and spectra of all three relevant positive semidefinite covariances: $C_h$, $C_{\mathrm{belief}}$, and $C_{\mathrm{res}}$. Do not subtract PRs to obtain a residual dimension. Even when $C_{\mathrm{cross}}=0$,

$$
\operatorname{tr}(C_h^2)
=\operatorname{tr}(C_{\mathrm{belief}}^2)
+\operatorname{tr}(C_{\mathrm{res}}^2)
+2\operatorname{tr}(C_{\mathrm{belief}}C_{\mathrm{res}}).
$$

Uncorrelated random components need not occupy orthogonal activation subspaces.

The residual is not automatically noise. It may encode nonlinear functions of beliefs, token identity, positions, algorithmic working variables, or predictive distinctions omitted by the chosen belief coordinates. Test these alternatives before interpreting residual variance as irrelevant.

### 2.3 Three strengths of prediction

Use the following labels in every result table:

1. **Specified metric benchmark.** The experiment imposes an encoding metric, or computes PR of beliefs in a declared coordinate system. This tests the statistics and measurement pipeline.
2. **Conditional reconstruction.** A metric $G=E^\top E$ is estimated after training and inserted into the covariance identity. This tests whether the representation model accounts for observations.
3. **Independent learning prediction.** The metric or the activation covariance is predicted from the generator, architecture, initialization, optimizer, and training time. No per-test-condition metric is fitted. This is the strongest target.

These levels answer different questions. Success at level 2 is not evidence that level 3 has been achieved.

## 3. Primary generator: two binary hidden processes

### 3.1 A small generator with exact filtering

For factor $a$, let the hidden state and observation be $S_t^{(a)},X_t^{(a)}\in\{-1,+1\}$. Use persistence $0<\rho_a<1$ and observation reliability $0<q_a<1$:

$$
P(S_t^{(a)}=s'\mid S_{t-1}^{(a)}=s)
=\frac{1+\rho_a ss'}{2},
$$

$$
P(X_t^{(a)}=x\mid S_t^{(a)}=s)
=\frac{1+q_a xs}{2}.
$$

The stationary hidden-state prior is uniform. Let $m_t^{(a)}=\mathbb E[S_t^{(a)}\mid X_{1:t}^{(a)}]$. Starting from $m_0^{(a)}=0$, the exact update is

$$
m_t^{(a)}
=\frac{\rho_a m_{t-1}^{(a)}+q_a X_t^{(a)}}
{1+\rho_a q_a X_t^{(a)}m_{t-1}^{(a)}}.
$$

The next observation probability is

$$
P(X_{t+1}^{(a)}=+1\mid X_{1:t}^{(a)})
=\frac{1+\rho_a q_a m_t^{(a)}}{2}.
$$

Two independent factors emit an observed token representing their pair $(X_t^{(1)},X_t^{(2)})$. Randomly permute the four token labels once per dataset specification and record the permutation. The model receives a four-token vocabulary with no factor labels. Permutation controls test whether a particular token indexing creates the result.

For short contexts, enumerate $4^T$ strings and compute each string's probability and posterior exactly. At $T=6$ this is only 4,096 contexts. At $T=16$ or $32$, use independent Monte Carlo contexts; keep $T=6$ as the exact reference. Do not choose $\rho=0$ as a primary case: it removes temporal predictive value even though observations can still reveal the current hidden state.

### 3.2 Two competing covariance geometries

With $b=(m^{(1)},m^{(2)})$, symmetry and factor independence give

$$
\Sigma_b=\begin{pmatrix}v_1&0\\0&v_2\end{pmatrix},
\qquad v_a=\mathbb E[(m^{(a)})^2].
$$

For an isometric factored encoding,

$$
\operatorname{PR}_{\mathrm{factored}}
=\frac{(v_1+v_2)^2}{v_1^2+v_2^2}.
$$

This changes continuously between one and two as the variance ratio changes, despite fixed belief rank two. If the two factors have different learned gains $g_a$, replace $v_a$ by $g_a^2v_a$. Those gains must be imposed or predicted for an independent activation-PR prediction.

There is an exact alternative benchmark. Represent the joint posterior over four hidden states as the probability vector

$$
\pi(s_1,s_2)=\frac{1}{4}(1+s_1m^{(1)})(1+s_2m^{(2)}).
$$

In an orthonormal Walsh basis, its centered covariance has eigenvalues $v_1/4$, $v_2/4$, and $v_1v_2/4$. Therefore

$$
\operatorname{PR}_{\mathrm{joint}}
=\frac{(v_1+v_2+v_1v_2)^2}
{v_1^2+v_2^2+v_1^2v_2^2}.
$$

The factored and joint representations contain the same predictive information on this product-state manifold but have different linear covariance spectra. This is precisely why information sufficiency alone does not fix activation PR.

### 3.3 A crucial probability-versus-logit control

For next-token cross-entropy, an exact additive logit representation can use

$$
u_a=\operatorname{atanh}(\rho_aq_am^{(a)}).
$$

The logits for pair token $(x_1,x_2)$ can be $x_1u_1+x_2u_2$ up to a context-dependent common shift. Thus the natural coordinates for a linear softmax readout may be nonlinear functions of belief probabilities. Compare regressions using $b=(m_1,m_2)$, $u=(u_1,u_2)$, and a small preregistered augmented feature set including $m_1m_2$. Their covariances are all computable from the same generator.

A poor affine fit in $m$ combined with a good fit in $u$ can identify a readout-induced coordinate change. It should not be called a failure of belief-based prediction. Conversely, an arbitrarily flexible post-hoc coordinate transformation would remove the hypothesis's falsifiability. Declare candidate features before analyzing the held-out sweep.

### 3.4 Proposed first parameter sweep

Use $(\rho_1,q_1)=(0.8,0.5)$ and $\rho_2=0.8$, with $q_2\in\{0.25,0.5,0.8\}$. Exact finite-context enumeration at $T=6$ gives the following generator-only benchmark. The two PR columns refer to the specified probability-coordinate encodings above, not trained activations.

| $q_2$ | $v_1$ | $v_2$ | Factored belief PR | Joint belief PR |
| --- | --- | --- | --- | --- |
| $0.25$ | $0.414752$ | $0.139888$ | $1.605662$ | $1.925329$ |
| $0.50$ | $0.414752$ | $0.414752$ | $2.000000$ | $2.684611$ |
| $0.80$ | $0.414752$ | $0.736687$ | $1.854991$ | $2.626951$ |

These numbers were computed by exact context enumeration while preparing this plan. Direct four-state posterior covariance agrees with the stated Walsh eigenvalues to floating-point precision, and the additive-logit construction reproduces the next-token probabilities. This verifies the analytical benchmark and gives a nontrivial predicted PR range before training. Recompute it independently in the experiment implementation, and recompute the reference when changing context length or population. Lock the final sweep before model analysis.

This generates one symmetric case and two anisotropic cases, with fixed latent rank and vocabulary. Persistence sweeps come later so the first experiment changes one interpretable property at a time. Reserve two intermediate reliability values, such as $q_2=0.35$ and $0.65$, as unseen conditions for a learned-metric theory. The precise reserved values should be locked along with the main sweep.

## 4. Secondary generators and stress tests

### 4.1 Mess3 as a direct connection to Simplex

Reproduce one published Mess3 setting from [2], using its exact matrices and token convention. Begin with the setting $x=0.15$, $\alpha=0.6$ used in their attention analysis, verifying the implementation against the paper and repository. A later $x=0.5$ condition tests the oscillatory regime where their analysis finds a role for multiple attention heads. Do not transfer these head-count predictions to a different HMM definition without deriving the corresponding update.

Mess3 establishes continuity with the published belief-geometry experiments. It is a secondary benchmark because symmetry can make PR insensitive to changes in geometric shape, and the binary product process gives smaller exact populations and clearer nonconstant covariance predictions.

### 4.2 Controlled violations of factor independence

After the independent case works, introduce an explicit mixture in the four-state hidden transition matrix:

$$
P_\epsilon=(1-\epsilon)(P_1\otimes P_2)+\epsilon P_{\mathrm{corr}},
$$

where a concrete starting choice is a correlated reset,

$$
P_{\mathrm{corr}}(s_1',s_2'\mid s_1,s_2)
=\frac{1+c s_1's_2'}{4},\qquad c=0.8.
$$

Every entry is positive. Compute the stationary distribution of the resulting four-state chain explicitly; it need not remain the independent uniform prior. Keep the observation model unchanged. Start with $\epsilon\in\{0,0.05,0.2\}$ at one fixed reliability pair.

Before training, compute exact joint filtering and the loss gap between the full predictor and a specified factorized approximation. Verify that the proposed $P_{\mathrm{corr}}$ actually induces posterior dependence and that this dependence changes next-token probabilities. Hidden-state coupling does not by itself guarantee an observable or predictively necessary interaction.

Candidate hypotheses are that the model first represents marginals and later develops interaction directions, or that its encoding continuously changes without a sharp stage. Both are possibilities. Measure the interaction coordinate $\mathbb E[S_1S_2\mid x]-m_1m_2$, its decodability, covariance contribution, and causal effect. An increase in PR by itself does not identify an interaction mode.

### 4.3 Nuisance and matched-statistics controls

Use a shuffled-token sequence with the same one-token frequencies to remove temporal prediction. Use a token-label permutation to test arbitrary embedding conventions. Later, add an independent nuisance observation process whose future is unpredictable; characterize whether the model retains its current value in residual variance despite no forecasting benefit. These controls distinguish useful predictive structure from embedding and residual-memory effects.

A stronger test constructs two generators with similar belief covariance but different belief-update dynamics, or matched rank with different covariance. The first asks whether PR misses meaningful geometry; the second asks whether the theory predicts more than a rank bound. Exact matching is a numerical design problem and should not be assumed achievable without checking.

## 5. Stage A: validate the generator and covariance pipeline

This stage requires no neural training. Implement two independent posterior calculations: direct matrix multiplication over hidden states and the scalar recursion above. Check probability normalization and agreement for every context at $T\leq6$. Compare exact moments with independently sampled contexts, increasing sample size until reported uncertainties explain the differences.

Apply known linear encodings to beliefs: an isometry, a diagonal stretch, a rotation, and an embedding into a larger ambient dimension. The measured activation spectra must match the prescribed covariance $E\Sigma_bE^\top$. Isometric expansion should preserve raw PR and lower PR divided by ambient dimension. Diagonal stretching should change PR while preserving algebraic rank. This is a calibration of the observable, not a learning result.

Verify the joint-posterior Walsh spectrum independently. Check an intentionally nonlinear encoding, such as $(m_1,m_2,m_1m_2)$, to establish that a two-variable manifold need not have linear rank two.

**Gate A:** proceed only when exact and sampled covariances agree within sampling error, numerical tolerances are documented, and all prescribed-encoding checks pass. A failure here is an implementation failure, not evidence against transformer theory.

## 6. Stage B: a solved trained-model benchmark

Use whitened belief coordinates $z=\Sigma_b^{-1/2}(b-\bar b)$ on the nonzero covariance support. Supply these coordinates directly to a two-layer linear network, with centered teacher target $y=Tz$ and loss

$$
\mathcal L=\frac12\mathbb E\|DEz-Tz\|^2.
$$

This is a squared-loss learning benchmark, not a token transformer and not an exact model of cross-entropy training. It isolates the difficulty of predicting the learned encoding metric. Start with $T=\operatorname{diag}(1,0.25)$ and $T=\operatorname{diag}(1,1)$, using hidden width $d\geq2$.

For a balanced exact factorization selected by minimum total factor norm, the benchmark predicts

$$
E^\top E=(T^\top T)^{1/2},
\qquad
\operatorname{PR}(C_h)
=\frac{(\sum_i\sigma_i(T))^2}{\sum_i\sigma_i(T)^2}.
$$

Test the finite-time prediction first in the aligned balanced initialization where each mode obeys the known scalar gradient-flow equation [5]. Use a small Euler step and reduce it to verify convergence to the predicted continuous-time trajectory. Then test random small balanced initialization and an explicitly regularized factorization. Do not conflate these initialization and objective conditions with one theorem about arbitrary gradient descent.

Include one deliberately imbalanced initialization to expose departures from the balanced prediction. Record $EE^\top-D^\top D$, singular values of $DE$, and covariance eigenvalues through training. Use an unwhitened-input condition only after the whitened version works, deriving its changed dynamics rather than reusing the same formula.

**Support:** independently predicted time courses and endpoint spectra match in the stated regime. **Refutation within the regime:** persistent disagreement after step-size and optimization-error checks. **Outside the regime:** disagreement under imbalanced initialization or a different loss limits transfer of the theory; it does not refute the balanced calculation.

**Gate B:** do not advance a claim of trained-metric prediction until this known calculation is reproduced. A transformer result that resembles the balanced endpoint is initially an analogy, not a validated derivation.

## 7. Stage C: restricted attention with a computable covariance

The first attention experiment should expose which theoretical assumptions are doing the work. Use one causal attention block, no MLP, and no normalization. Separate a fixed value/output map from the attention pattern. Record both attention output and the residual sum so token and position embeddings cannot masquerade as unexplained attention variance.

### 7.1 An imposed-filter calibration

For deterministic position-dependent coefficients $B_k$, construct

$$
h_t=\sum_{k=0}^{T-1}B_k\phi(X_{t-k}).
$$

The generator determines

$$
C_h=\sum_{k,\ell}B_k\operatorname{Cov}[\phi(X_{t-k}),\phi(X_{t-\ell})]B_\ell^\top.
$$

Test a declared exponential memory kernel and a short uniform kernel, with fixed token features and fixed output geometry. This validates the attention covariance calculation. If low rank is built in through fixed output vectors, report it as imposed rather than learned.

A generator-specific analytical case is weak observation signal. At fixed $T$ and fixed $0<\rho<1$, expanding the binary filter at small $q$ gives

$$
m_T=q\sum_{k=0}^{T-1}\rho^kX_{T-k}+O(q^3),
\qquad
v=q^2\frac{1-\rho^{2T}}{1-\rho^2}+O(q^4).
$$

A positional softmax head with weights $\rho^k/Z_T$, where $Z_T=\sum_{k=0}^{T-1}\rho^k$, and value/output gain $qZ_T$, implements the leading-order filter. Compare its exact generator-averaged covariance, the weak-signal formula, and exact posterior variance. A small sweep such as $q\in\{0.05,0.1,0.2\}$ tests the predicted error order before learning is involved. This asymptotic benchmark does not apply without qualification to the primary reliability values $0.5$ and $0.8$. The predictive signal is weak in this auxiliary test, so report numerical precision and excess loss rather than inferring success from nearly constant next-token probabilities. The companion theory document's weak-signal derivation supplies the approximation assumptions.

### 7.2 A learned-filter test

Next train the permitted positional query/key parameters, keeping the declared value/output geometry fixed. Derive a prediction for their optimum or dynamics from the restricted task, or use an architecture-specific constrained-belief calculation. Until that derivation exists, this is an exploratory experiment, not an end-to-end theoretical prediction.

Two comparisons are useful. Predict the covariance from the learned filter on independent contexts to test the conditional calculation. Separately compare the learned filter to the theoretically predicted one. The former can succeed while the latter fails.

If content-dependent attention is released, the deterministic-$B_k$ formula is no longer generally applicable: coefficients and token features are correlated random functions of the same context. Either integrate the complete forward map over the finite context population or derive the joint statistics. Replacing attention by its mean without an error analysis is not a controlled theory.

**Gate C:** either obtain an independent trained-filter prediction, or identify precisely what remains unclosed. A useful negative result could show that attention learns accurate predictions while its encoding metric varies across seeds, ruling out a generator-only PR law.

## 8. Stage D: the first small nonlinear transformers

### 8.1 Initial architecture and optimizer

Use a decoder-only causal transformer with $L=1$, residual width $d=64$, $H=2$ heads, and $d_h=32$. The MLP has hidden width $4d=256$ and GELU activation. Use a learned token embedding and learned positional embedding, an untied linear softmax readout, and no dropout. Start without normalization to keep the measurement map inspectable, while verifying stable training. Add pre-LayerNorm as a separate ablation after the baseline.

For the first finite-width discovery runs, a conventional attention scaling $1/\sqrt{d_h}$ and Adam are reasonable defaults. Use a three-value pilot learning-rate check, for example $10^{-4},3\times10^{-4},10^{-3}$, on the symmetric generator only. Choose the largest reliably stable value with good validation learning and freeze it across the reliability sweep. These are finite-width pilot settings, not the large-width feature-learning parameterization.

Train on independently generated contexts with batch size 256 and $T=16$, using next-token cross-entropy over all valid positions. Evaluate all covariance claims at the final position unless a separate position analysis is named. Use online training data so repeated finite-dataset memorization is not the primary effect. Save the generator state and seed so runs can be reproduced.

Use 20,000 optimizer steps as an initial cap and record loss versus tokens and steps. This is a budget proposal, not an assertion that every condition converges by then. If validation loss has not stabilized, extend only the relevant conditions after documenting the reason. Evaluate against the finite-context Bayes predictor at the same positions, and report the excess cross-entropy or mean conditional KL divergence in nats per token.

### 8.2 What should be measured

At each selected checkpoint, measure embedding, attention output, post-attention residual, MLP output, and post-MLP residual. For the normalized model, also measure after normalization. Record all covariance spectra, traces, PRs, $d_{95}$, reverse belief-probe accuracy, forward activation-prediction accuracy, and estimated metric $G$ on the same declared context population.

Compare the following hypotheses:

| Hypothesis | Observation supporting it | Observation limiting or refuting it |
| --- | --- | --- |
| H1: an approximately isometric belief encoding explains PR | Belief-explained variance is high and normalized $G$ is nearly scalar on the belief support | Beliefs are well decoded but $G$ has large condition-dependent anisotropy |
| H2: task/readout coordinates explain anisotropy | The declared $u_a$ features predict spectra better than raw beliefs on held-out conditions | Neither coordinate set explains the dominant covariance or their metric is still unpredicted |
| H3: attention and MLP have distinct predicted geometries | Measured stage-specific covariance changes match an independent constrained-update calculation | A claimed filter geometry fails despite accurate predictions, or the distinction disappears after controlling embeddings |
| H4: extra covariance is mostly nuisance memory | Token/position controls explain residual variance and removing the corresponding subspace preserves task behavior | Residual directions carry conditional predictive information or their removal harms prediction |
| H5: the model represents factor interactions when necessary | Interaction coordinates emerge when the Bayes-versus-factored loss gap is nonzero and their removal increases loss | PR rises without interaction content, or the factorized approximation already achieves the Bayes prediction |

Normalize $G$ by its trace before comparing anisotropy, since uniform rescaling does not change PR. A generator-only PR law is unsupported if independently trained models with equivalent prediction quality develop systematically different metrics and PRs.

### 8.3 Causal checks

Use probes fitted on independent training contexts to define candidate subspaces. Compare removal or patching of a belief-related subspace to removal of random subspaces of equal dimension and comparable removed activation energy. Include untouched and reconstruction-only baselines. Interventions should be local to a named layer and position.

For steering, choose donor and recipient contexts with known posterior changes, predict the sign of the change in next-token probabilities, and compare to shuffled donors. Patching is more interpretable when donor/recipient activations remain near the observed population. A successful off-distribution steering manipulation is not by itself evidence that the network naturally uses the proposed coordinates.

**Gate D:** advance only with a verified representation model and at least one nontrivial prediction that generalizes across unfit generator conditions or seeds. If only conditional reconstruction works, that can still make a useful blog or diagnostic study. It does not yet justify the large-width prediction claim.

## 9. Stage E: width limits and nonlinear feature learning

The width project asks whether a limiting dynamical theory predicts covariance as features learn. Bordelon, Chaudhry, and Pehlevan analyze distinct transformer limits in head dimension, head count, and depth, with parameterizations that keep attention updates nontrivial [6]. Implementing the relevant parameterization and learning-rate scaling from that work is a prerequisite. A conventional finite-width Adam sweep is not a substitute.

The full many-head feature-learning DMFT is not an immediately available tractable numerical predictor: the authors describe its full computation as essentially uncomputable and use wide-network ensembles as a practical proxy in their numerical comparisons. Such an ensemble is a simulation baseline, not an independent solution of DMFT for our task. The missing milestone here is a small-context, symmetry-reduced or otherwise controlled closure whose numerical predictions can be evaluated without training the tested networks.

For the first numerical closure, start at $T=2$ or $3$, giving 16 or 64 contexts for the four-token process. Test the one-step and two-step predictions before integrating a full training trajectory. Context-pair and training-history order parameters can make even a modest context population expensive; exact filtering at $T=6$ does not establish feasibility of its DMFT closure. Use symmetry reduction and record memory/time scaling before increasing $T$. The $T=6$ population is an exact evaluation reference and a later extension, subject to this computational gate.

Use exact context probabilities, final-position cross-entropy, and full-batch gradient descent approximating the stated gradient-flow limit. This is a separate controlled objective from Stage D's online, all-position training. Verify convergence under step-size reduction. Then introduce minibatch noise only when the theory specifies how it enters. Parameterization checks for Adam do not by themselves supply a closed theory of Adam's adaptive dynamics.

### 9.1 Two separate width sequences

Keep depth fixed at $L=1$ initially, MLP width $4d$, vocabulary fixed, and context length fixed. Study two sequences separately:

| Sequence | Fixed quantities | Values to start with | Interpretation |
| --- | --- | --- | --- |
| A: grow per-head dimension | $H=2$ | $d_h\in\{16,32,64,128\}$, so $d\in\{32,64,128,256\}$ | Approach a large-key/query-dimension regime |
| B: grow head count | $d_h=16$ | $H\in\{2,4,8,16\}$, so $d\in\{32,64,128,256\}$ | Approach an increasing-head regime |

Equal residual widths in these sequences do not define the same asymptotic model. Document the scaling of every learned matrix, query/key score, output/readout, residual branch, and parameter-group learning rate. Use the theoretical time convention, including the conversion from optimizer steps where applicable. In [6]'s relevant valid large-head-dimension scaling with $1/d_h$ score normalization, fixed-head-count limits can collapse head diversity; the increasing-head-count limit can retain it. This is a statement about the specified limits and parameterization, not a prediction for every conventional $1/\sqrt{d_h}$ transformer.

Begin with one symmetric and one anisotropic generator, three widths, and three seeds. Extend to a fourth width only when the limiting prediction is numerically stable and the initial sequence is informative. Study $L=2$ later; there is no need for a simultaneous infinite-depth project.

### 9.2 Controls that establish feature learning

Compare a frozen-representation/readout-only baseline, an appropriate tangent or linearized-model baseline where feasible, and the full feature-learning parameterization. Match the optimizer and time convention to the theory for each comparison. Do not obtain a supposedly lazy control merely by using an arbitrarily small learning rate and stopping before it learns.

Record normalized activation Gram matrices on a fixed probe population through training, attention-pattern changes, query/key score statistics, and prediction performance. For a centered probe Gram matrix $K_\ell(t)$, report for example

$$
\Delta K_\ell(t)=\frac{\|K_\ell(t)-K_\ell(0)\|_F}{\|K_\ell(0)\|_F},
$$

with its normalization declared. Also measure trace-normalized shape change,

$$
\Delta K_{\mathrm{shape},\ell}(t)
=\left\|\frac{K_\ell(t)}{\operatorname{tr}K_\ell(t)}
-\frac{K_\ell(0)}{\operatorname{tr}K_\ell(0)}\right\|_F,
$$

when both traces are nonzero. Uniform gain changes can produce a large raw $\Delta K$ without changing PR. Persistent $O(1)$ kernel change as width grows is evidence relevant to feature learning, but should be accompanied by task-aligned spectral changes and the theoretical scaling checks. Attention movement alone is not the full covariance prediction.

### 9.3 Compare the predicted population object

The theoretical target should name the covariance or centered kernel whose spectrum determines PR. A large-width theory of losses alone is insufficient. For each time $t$ and layer $\ell$, obtain predicted first and second spectral moments, and ideally a predicted kernel or leading eigenvalues, from the generator/task statistics and the selected dynamical closure. The reduction from an activation covariance to the weighted context Gram spectrum is exact for a deterministic finite model. Replacing that Gram matrix by a limiting or disorder-averaged kernel additionally requires the relevant self-averaging argument; taking the PR of an averaged covariance is not generally the same as averaging the PR.

Use a genuinely fixed prediction convention: one set of theory inputs, no separate gain fit at every width, time, or generator setting. If a global time scale or finite-width correction is calibrated on one discovery condition, lock it and test it elsewhere. Report this as calibrated prediction rather than parameter-free prediction.

### 9.4 Avoid a false finite-sample width limit

An empirical covariance from $N$ contexts has rank at most $N-1$. At large width, insufficient $N$ can create an apparent PR ceiling. Sample covariance noise can also inflate the second spectral moment and bias plug-in PR downward. Therefore every width comparison needs a context-sample convergence check, not just repeated model seeds.

For $d\leq256$, begin with 65,536 independent final-position contexts for broad covariance estimation and compare 16,384, 65,536, and 262,144 contexts on selected checkpoints. Stream first and second moments rather than keeping all activations in memory. These sample counts are proposals; increase them if PR or the leading spectrum does not stabilize to the required accuracy. Low-rank data do not automatically require $N\gg d$, but using that margin avoids a major avoidable ambiguity in this small-model project.

On exact $T=6$ context populations, probability-weighted Gram matrices give an independently checkable finite-context operator. On a population supported on $M$ contexts, the centered activation rank is at most $M-1$, independently of width. Therefore raw PR can approach a finite value while PR divided by $d$ tends to zero; that alone is not representational collapse. A fixed finite probe Gram spectrum is a well-defined object, but its convergence with model width must not be mistaken for convergence of a longer-context population activation PR. Check context-population convergence separately and keep the order of limits explicit. If using a finite-sample correction or a cross-sample estimator of $\operatorname{tr}(C^2)$, derive its assumptions, use independent data for centering where needed, and validate it against exact populations before relying on it.

**Gate E:** claim support for the large-width theory only if the specified feature-learning regime is observed, the predicted covariance object is correct, finite-sample effects are controlled, and prediction errors decrease or stabilize at the theoretically justified level across widths. A persistent error can identify a missing closure, a wrong scaling regime, or a finite-width effect; those explanations require distinct tests.

## 10. Data splits, uncertainty, and quantitative comparison

Use independent random seeds for training data, initialization, validation data, probe-fitting data, and final evaluation data. For extrapolation claims, reserve generator parameters as well as contexts. A probe-test set with a new context seed does not establish generalization of the theory to a new generator.

The unit of replication for training variability is the model seed. Start with three seeds for discovery; confirm main claims with at least five independent initializations and training streams. Report individual runs as well as mean and uncertainty. An ensemble-average spectrum need not equal the typical spectrum or the average PR, so compute PR per model before averaging.

For independently sampled contexts, bootstrap contexts within each model only to quantify measurement uncertainty. For contexts cut from a long trajectory, use block resampling with blocks longer than the measured dependence time, or preferably switch the primary evaluation back to independent contexts. Do not treat overlapping token windows as independent observations.

Compare spectra after normalization by total variance as well as in absolute units. Proposed errors are

$$
e_{\mathrm{PR}}=\frac{|\operatorname{PR}_{\mathrm{pred}}-\operatorname{PR}_{\mathrm{obs}}|}{\operatorname{PR}_{\mathrm{obs}}},
\qquad
e_{\mathrm{spec}}=\frac12\sum_i|p_i^{\mathrm{pred}}-p_i^{\mathrm{obs}}|,
$$

padding the shorter spectrum with zeros. For repeated or nearly repeated eigenvalues compare eigenspaces rather than individual eigenvectors. Use principal angles for subspace comparisons and state any dimension threshold.

As an initial practical criterion, a leading prediction should aim for less than 10% relative PR error and less than 0.1 normalized spectrum error on held-out conditions, with discrepancies assessed against model-to-model and measurement uncertainty. These thresholds are proposed decision rules, not statistical significance levels or universal standards. Lock them before the confirmatory sweep. A theory that correctly orders conditions but misses absolute values should be reported as qualitative prediction.

Record checkpoints at step 0; 1, 2, 5; then approximately logarithmically at 10, 20, 50, 100, 200, 500, 1,000, 2,000, 5,000, 10,000, and 20,000. Compute cheap loss and scale diagnostics frequently; perform full covariance analysis at selected checkpoints. Denser checkpoints are justified near a reproducible dimensional transition. Avoid selecting checkpoints after seeing which one best matches the theory.

## 11. First run batches and stop rules

The initial plan is intentionally small. It should reveal the dominant scientific obstruction before a large compute allocation.

| Batch | Work | Approximate scope | Decision |
| --- | --- | --- | --- |
| 0 | Generator/filter/metric validation | Exact populations, synthetic encodings, no training | Pass Gate A |
| 1 | Balanced linear learning | Two teacher spectra, aligned and perturbed initialization; cheap CPU/GPU runs | Pass Gate B; isolate initialization effects |
| 2 | Small transformer discovery | Three binary-generator conditions, $L=1,d=64,H=2$, three seeds: nine main runs | Assess $b$ versus $u$, metric anisotropy, residual variance |
| 3 | Focused architecture test | Chosen informative condition(s): no-MLP and pre-LayerNorm ablations, three seeds | Identify where covariance departures arise |
| 4 | Confirmatory curve | Five seeds on locked sweep and held-out intermediate generator conditions | Decide whether an independent spectrum law is supported |
| 5 | Width pilot | Two conditions, three widths, three seeds, one declared scaling sequence: 18 runs | Pass regime and sampling checks before a second sequence |

The imposed-filter test can be done alongside Batch 0. A learned-filter experiment belongs between Batches 1 and 2 when its theory is available. Mess3 replication and coupled-factor experiments should be additions selected to answer an identified question, not automatic multipliers of every grid.

Stop adding width when the theory has no prediction for the measured covariance, when the scaling implementation fails its unit-update checks, or when finite-sample uncertainty is larger than the claimed convergence. Stop expanding generator complexity if the binary case's encoding metric remains unexplained. Extend training only after determining whether a loss plateau reflects optimization, approximation limits, or irreducible finite-context error.

## 12. Figures that would make the argument inspectable

1. **Generator and prediction.** Hidden transition/emission model, exact belief geometry, and predicted belief/factored/joint spectra. Separate schematics from computed quantities.
2. **A nonconstant PR curve.** Predicted PR versus reliability or variance ratio, with per-seed observations on held-out parameter values. Show the associated full spectra so a scalar coincidence is visible.
3. **The learned metric.** Eigenvalues of normalized $G$ and their time courses, alongside belief-explained and residual variance. This reveals whether the missing theory is geometry or extra content.
4. **Training trajectory.** Loss, covariance eigenvalues, PR, and candidate belief coordinates at the same checkpoints. Distinguish a predicted mode-growth law from descriptive dimensional compression.
5. **Attention versus MLP.** Stage-specific covariance and geometry, with the proposed constrained-update prediction and the explicit embedding contribution.
6. **Width and feature learning.** Separate panels for per-head-width and head-count sequences; prediction errors, kernel change, and finite-context-sample convergence.
7. **Intervention and refutation.** Belief-related versus energy-matched random subspace interventions, plus the condition where a simpler hypothesis fails.

No synthetic plots should be presented as observed training results. Every theory curve must identify whether it is exact, numerically evaluated from the generator, a controlled approximation, or fitted after training.

## 13. Interpreting outcomes and choosing the final output

| Outcome | Scientific interpretation | Appropriate next output |
| --- | --- | --- |
| Beliefs are decodable, but activation PR varies strongly with seed and metric | Belief sufficiency does not fix neural covariance; a generator-only law is inadequate | Blog explaining the distinction, with a reproducible counterexample |
| A specified metric predicts PR; learned metrics remain unpredicted | The generator calculation works, but the optimization closure is missing | Blog or short methods note with clearly conditional claims |
| A tractable trained attention model has a derived metric and held-out spectrum prediction | A concrete bridge from task to learned covariance, subject to literature comparison | Potential mini paper, with finite-model scope |
| A nonlinear transformer follows a calibrated but transferable covariance law | Evidence for a useful approximation, not necessarily a first-principles solution | Potential empirical/theory short paper depending on novelty and robustness |
| A large-width feature-learning theory predicts covariance trajectories across conditions and widths | A substantive explanation beyond decoding and dimensionality measurement | Potential full theory–experiment paper, contingent on derivation, novelty, and evidence |
| The limiting theory fails systematically under verified assumptions | A missing mechanism or closure is exposed; failure may be informative | A negative-result analysis if the failure is reproducible and discriminating |

Publication potential cannot be inferred from the use of DMFT or the presence of a PR plot. The contribution should be an independently derived and tested explanation, or a precise counterexample to a plausible explanation. Before framing a paper, update the nearest-neighbor literature review and identify which claim is new relative to existing belief geometry, factored representation, and transformer-limit work.

The first useful milestone is small: one generator sweep, one declared activation location, a prediction made before training, and a clear account of why it succeeds or fails. That result determines which layer of theory is worth pursuing next.

## References and implementation starting points

[1] Shai, Marzen, Teixeira, Oldenziel, and Riechers (2024), [Transformers Represent Belief State Geometry in their Residual Stream](https://arxiv.org/abs/2405.15943). Establishes the controlled belief-geometry program and linear probing methodology. [Code](https://github.com/adamimos/epsilon-transformers).

[2] Piotrowski, Riechers, Filan, and Shai (2025), [Constrained Belief Updates Explain Geometric Structures in Transformer Representations](https://arxiv.org/abs/2502.01954). See the Mess3 attention construction, spectral treatment of updates, architectural ablations, and Appendix G's activation-dimensionality analysis. Its mechanistic formulas should be checked against the exact model variant used here.

[3] Shai et al. (2026), [Transformers learn factored representations](https://arxiv.org/abs/2602.02385). Tests factored versus joint predictive representations, dimensionality, approximate orthogonality, and expansion when factoring is lossy. The initial binary product experiment is a deliberately smaller analytical benchmark rather than a claim to originate factorization experiments.

[4] [Astera-org/factored-reps](https://github.com/Astera-org/factored-reps), especially [weighted PCA](https://github.com/Astera-org/factored-reps/blob/master/fwh_core/analysis/pca.py), [independent-factor training configuration](https://github.com/Astera-org/factored-reps/blob/master/experiments/training/configs/repro_independent.yaml), and the belief-grid, orthogonality, and noise analysis directories. Pin a commit before importing code or reproducing a result; record licensing and local modifications.

[5] Saxe, McClelland, and Ganguli (2013/2014), [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120). Source for mode-aligned deep-linear learning dynamics; the present squared-loss baseline must retain the required initialization and data assumptions.

[6] Bordelon, Chaudhry, and Pehlevan (2024), [Infinite Limits of Multi-head Transformer Dynamics](https://arxiv.org/abs/2405.15712), [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3eff068e195daace49955348de9f8398-Abstract-Conference.html). Source for distinct feature-learning transformer limits and associated scaling requirements. The width experiments depend on an explicit reproduction of the selected parameterization, not a general appeal to infinite width.
