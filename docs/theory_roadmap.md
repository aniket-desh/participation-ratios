# From belief geometry to transformer participation ratio

*A theory roadmap for predicting activation spectra in small transformers, from exact filtering to nonlinear feature learning. Research plan, 15 November 2026.*

The question is whether the statistical structure of a known sequence generator, together with a specified learning rule, predicts the covariance of a transformer's internal representations. Participation ratio (PR) is one observable of that covariance. A useful result should predict a nonconstant curve across generator parameters, training time, or model width, and should explain why that curve changes.

There are three progressively stronger targets. First, calculate the covariance of the generator's belief states. Second, predict how a tractable trained model encodes those beliefs. Third, calculate the evolving activation kernel of a nonlinear feature-learning transformer in a controlled large-width limit. The first is accessible immediately; the second supplies a bridge between data and learning; the third is a substantive research program with a numerical-closure problem of its own.

**The proposed contribution is a predictive connection between generator statistics and learned covariance, rather than the observation that HMM-trained transformers have low-dimensional activations.** Simplex already provides close precedents for belief geometry, constrained attention updates, and factored activation spectra [1–3]. This document contains exact identities and proposed calculations, not new experimental results or a completed transformer DMFT solution.

The companion [experiment plan](experiment_plan.md) turns these calculations into staged tests. The models throughout are small sequence transformers and controlled learning baselines.

## 1. Scientific contract and levels of ambition

| Level | Inputs available before the test | Predicted object | What would remain unexplained |
| --- | --- | --- | --- |
| A: belief statistics | Generator, prior, context length and sampling distribution | Belief covariance, its spectrum and PR | How a neural network encodes it |
| B: specified encoding | Level A plus an independently specified encoding metric | Signal activation spectrum and PR | Why training selects that metric; extra activity |
| C: solvable trained model | Data/task statistics, architecture, loss, initialization and optimizer | Encoding metric or feature kernel over training | Transfer to unrestricted attention and cross-entropy |
| D: restricted attention | Generator plus a derived attention algorithm and specified parameterization | Attention-output and residual covariance | Nonlinear recurrent inference and unrestricted learning |
| E: nonlinear large-width training | Generator, finite training protocol, parameterization and a solved statistical closure | Evolving residual kernel, covariance spectrum and PR | Finite-width/long-time corrections and generality |

Levels A–B are conditional geometry calculations. Levels C–E add a learning mechanism. A fitted encoding at every test condition verifies a representation model, but does not constitute an independent prediction of the learned geometry.

The near-term target should be one Level C calculation and one Level D approximation that survive a held-out parameter sweep. Attempt Level E after establishing which covariance feature actually needs a statistical learning theory.

## 2. Define the population and observable before calculating

Let $c=x_{1:T}$ be a context of fixed length $T$ drawn from the known distribution $p_T(c)$. At layer $\ell$, token position $T$, and training time $\tau$, let

$$
h_{\ell,\tau}(c)\in\mathbb R^d,
\qquad
\mu_{\ell,\tau}=\mathbb E_{c\sim p_T}[h_{\ell,\tau}(c)].
$$

For one fixed trained network, define the population covariance

$$
C_{\ell,\tau}
=\mathbb E_c\left[(h_{\ell,\tau}(c)-\mu_{\ell,\tau})
(h_{\ell,\tau}(c)-\mu_{\ell,\tau})^\top\right].
$$

The averaging is over contexts, not random initializations. Report the distribution of this observable across training seeds separately. For nonzero covariance,

$$
\boxed{
\operatorname{PR}(C)=\frac{[\operatorname{tr}C]^2}{\operatorname{tr}(C^2)}
=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
}
$$

Also record $\operatorname{tr}C$, the full eigenvalue spectrum, and

$$
d_{95}=\min\left\{k:\frac{\sum_{i=1}^k\lambda_i}{\sum_i\lambda_i}\geq0.95\right\},
\qquad
\operatorname{nPR}=\frac{\operatorname{PR}}{d}.
$$

PR equals rank only for an equal nonzero spectrum. A zero covariance has undefined PR and must be marked explicitly. Neither PR nor $d_{95}$ identifies the topology, fractal structure, or semantic interpretation of a point cloud.

**Hold the measurement distribution fixed.** Equal weighting of all token strings generally differs from weighting by $p_T(c)$. Combining positions also adds between-position covariance:

$$
\operatorname{Cov}(h)
=\mathbb E_t[\operatorname{Cov}(h\mid t)]
+\operatorname{Cov}_t(\mathbb E[h\mid t]).
$$

Start with a fixed position. Treat pooled-position results as a separate observable. Measure before and after normalization separately: LayerNorm is a context-dependent nonlinear transformation, and its effect cannot generally be removed by one global rescaling.

## 3. Level A: calculate belief covariance from the generator

### 3.1 Exact finite-context filtering

For an edge-emitting HMM with $K$ states, define token-labeled matrices

$$
T^{(x)}_{ij}=P(x_t=x,s_t=j\mid s_{t-1}=i),
\qquad
P=\sum_xT^{(x)}.
$$

Let $\pi$ be the chosen initial row distribution; use $\pi P=\pi$ when the process begins in stationarity. For a context $c=x_{1:T}$,

$$
p_T(c)=\pi T^{(x_1)}\cdots T^{(x_T)}\mathbf1,
\qquad
b(c)=\frac{\pi T^{(x_1)}\cdots T^{(x_T)}}{p_T(c)}.
$$

Contexts of zero probability are excluded. This is the posterior over the state after observing the context. The next-token probability is

$$
P(x_{T+1}=y\mid c)=b(c)T^{(y)}\mathbf1.
$$

Choose $Q\in\mathbb R^{K\times(K-1)}$ with orthonormal columns spanning $\mathbf1^\perp$. Write the independent centered belief coordinates as

$$
z(c)=Q^\top[b(c)^\top-\bar b],
\qquad
\bar b=\mathbb E_c[b(c)^\top].
$$

Then

$$
\boxed{
\Sigma_b(T)=\sum_c p_T(c)z(c)z(c)^\top.
}
$$

For small alphabets and $T$, enumerate contexts exactly using a prefix tree carrying unnormalized state probabilities. This gives an exact finite sum up to floating-point error. For larger $T$, sample independent contexts and apply the same filter; this is a numerical estimate with sampling uncertainty, not an analytic solution.

For a stationary hidden process, $\mathbb E[b(c)^\top]=\pi^\top$ by the tower property. This provides a strong normalization check. Finite-window beliefs initialized at $\pi$ differ from beliefs conditioned on an infinitely long observed past. Use the former when the model only sees the former.

### 3.2 Belief dimension is an upper bound, not a universal PR

The covariance rank is at most $K-1$, and can be smaller. Distinct hidden states may have identical predictive consequences; a nonminimal HMM can inflate the apparent state count. Moreover, a fractal subset of the simplex can still have covariance rank $K-1$. Distinguish hidden-state count, minimal predictive dimension, affine span, and local/information dimension.

Permutation symmetry can make the belief covariance isotropic on $\mathbf1^\perp$. For example, if the distribution is invariant under all three-state permutations, then on its two-dimensional centered subspace $\Sigma_b=vI_2$. Whenever $v>0$, PR is exactly two even if the distribution within that plane changes substantially. This is a useful check but a poor sole research target. Introduce anisotropy or unequal factor strengths to obtain a nonconstant PR prediction.

### 3.3 A primary toy: two binary hidden processes

For each factor $a$, let $S_t^{(a)},X_t^{(a)}\in\{-1,+1\}$, with

$$
P(S_t=s\mid S_{t-1}=s')=\frac{1+\rho_a ss'}{2},
\qquad
P(X_t=x\mid S_t=s)=\frac{1+q_a xs}{2}.
$$

Here $0<\rho_a<1$ controls persistence and $0<q_a<1$ controls emission reliability. Use the stationary symmetric prior and define the posterior magnetization

$$
m_t^{(a)}=\mathbb E[S_t^{(a)}\mid X_{1:t}^{(a)}].
$$

Prediction gives a prior magnetization $\rho_a m_{t-1}^{(a)}$. Bayes' rule then gives the exact recursion

$$
\boxed{
m_t^{(a)}
=\frac{\rho_a m_{t-1}^{(a)}+q_a X_t^{(a)}}
{1+\rho_aq_aX_t^{(a)}m_{t-1}^{(a)}},
\qquad m_0^{(a)}=0.
}
$$

The next observation has conditional mean $\rho_aq_am_t^{(a)}$, so

$$
P(X_{t+1}^{(a)}=+1\mid c)
=\frac{1+\rho_aq_am_t^{(a)}}{2}.
$$

Two independent factors yield observed tokens $(X_t^{(1)},X_t^{(2)})$, encoded as four arbitrary integer symbols. Let

$$
v_a(T)=\mathbb E[(m_T^{(a)})^2].
$$

Independent sign-symmetric factors have zero mean and zero cross covariance. In the direct-sum magnetization coordinates,

$$
\Sigma_m=\operatorname{diag}(v_1,v_2),
\qquad
\boxed{\operatorname{PR}_{\mathrm{factored}}
=\frac{(v_1+v_2)^2}{v_1^2+v_2^2}.}
$$

This runs from two at equal strengths toward one when a single factor dominates. It is a belief-coordinate prediction; neural gains must still be specified.

There is also a distinct joint encoding. The four-state posterior is

$$
b(s_1,s_2)=\frac14(1+s_1m_1)(1+s_2m_2).
$$

In the orthonormal Walsh basis on four states, the centered coordinates are $m_1/2,m_2/2,m_1m_2/2$. Independence and sign symmetry therefore give three covariance eigenvalues

$$
\frac{v_1}{4},\quad\frac{v_2}{4},\quad\frac{v_1v_2}{4},
$$

and hence

$$
\boxed{
\operatorname{PR}_{\mathrm{joint}}
=\frac{(v_1+v_2+v_1v_2)^2}
{v_1^2+v_2^2+v_1^2v_2^2}.
}
$$

This is an exact distinction between two coordinate representations of the same independent predictive information. The joint representation's third coordinate is a nonlinear function of the first two; its covariance can have rank three while the points require only two scalar coordinates and lie on a two-dimensional product-state surface. At fixed context length the actual support is finite; for infinite histories it can be fractal. A weak product mode may lie below a 95% variance threshold. Inspect its predicted eigenvalue rather than only thresholded dimension.

### 3.4 A controlled attention approximation from the same toy

For fixed $T$, $\rho$ bounded away from one, and weak emission reliability $q$, expand the exact filter around $q=0$. Since $m_t=O(q)$,

$$
m_t=\rho m_{t-1}+qX_t+O(q^3),
\qquad
m_T=q\sum_{k=0}^{T-1}\rho^kX_{T-k}+O(q^3).
$$

This is an exponentially weighted attention-like statistic. The observed correlations are exactly

$$
\Gamma_{k\ell}=\mathbb E[X_{T-k}X_{T-\ell}]
=\begin{cases}1,&k=\ell,\\q^2\rho^{|k-\ell|},&k\neq\ell.\end{cases}
$$

Thus the linear-filter statistic has exactly calculable variance

$$
v_{\mathrm{lin}}=q^2\sum_{k,\ell=0}^{T-1}\rho^{k+\ell}\Gamma_{k\ell},
$$

whereas the exact posterior variance obeys the leading weak-signal expansion

$$
v(T)=q^2\sum_{k=0}^{T-1}\rho^{2k}+O(q^4)
=q^2\frac{1-\rho^{2T}}{1-\rho^2}+O(q^4).
$$

The exact variance of the approximate filter is not the exact variance of the nonlinear filter. In particular, the expansion is not uniform as $\rho\to1$ or $T\to\infty$.

Normalized attention can implement weights $\alpha_k=\rho^k/Z_T$, with $Z_T=\sum_k\rho^k$; the value/output gain must supply $qZ_T$. A frozen implementation verifies this algebra. Explaining why training finds that kernel and gain is the next learning problem. This deliberately restricted model predicts the leading weak-signal belief statistic, not full Bayes inference at arbitrary reliability.

## 4. Level B: transport belief covariance into activation covariance

Suppose centered activations satisfy

$$
h-\mu=Ez+\epsilon,
\qquad E\in\mathbb R^{d\times r},
\qquad \mathbb E[z\epsilon^\top]=0.
$$

Then

$$
C_h=E\Sigma_bE^\top+C_\epsilon.
$$

For the explained component $C_{\mathrm{bel}}=E\Sigma_bE^\top$, define $G=E^\top E$. Cyclicity of the trace gives

$$
\operatorname{tr}C_{\mathrm{bel}}=\operatorname{tr}(G\Sigma_b),
\qquad
\operatorname{tr}(C_{\mathrm{bel}}^2)=\operatorname{tr}(G\Sigma_bG\Sigma_b),
$$

so

$$
\boxed{
\operatorname{PR}(C_{\mathrm{bel}})
=\frac{[\operatorname{tr}(G\Sigma_b)]^2}
{\operatorname{tr}(G\Sigma_bG\Sigma_b)}.
}
$$

The nonzero covariance eigenvalues are those of $\Sigma_b^{1/2}G\Sigma_b^{1/2}$. Isometric encoding, $G=cI$, preserves belief PR. An arbitrary invertible linear change of belief coordinates does not change the activation prediction if $E$ and $\Sigma_b$ are transformed consistently; it can change belief PR considered on its own. Therefore always declare the coordinate metric used for belief calculations.

For two orthogonal factor encodings with gains $g_1,g_2$, substitute $g_1^2v_1,g_2^2v_2$ in the factored formula. Symmetry may constrain gains within a factor without equating gains across factors of unequal predictability. Do not assume the latter.

### 4.1 Extra activity can dominate a full-stream measurement

The exact second trace is

$$
\operatorname{tr}(C_h^2)
=\operatorname{tr}(C_{\mathrm{bel}}^2)
+2\operatorname{tr}(C_{\mathrm{bel}}C_\epsilon)
+\operatorname{tr}(C_\epsilon^2).
$$

Statistically uncorrelated components need not occupy orthogonal neuron subspaces. The mixed trace does not generally vanish. For an illustrative isotropic residual $C_\epsilon=\sigma^2I_d$, let $s_1=\operatorname{tr}C_{\mathrm{bel}}$ and $s_2=\operatorname{tr}(C_{\mathrm{bel}}^2)$. Then

$$
\operatorname{PR}(C_h)
=\frac{(s_1+d\sigma^2)^2}
{s_2+2\sigma^2s_1+d\sigma^4}.
$$

A small per-coordinate residual variance can accumulate over many coordinates. This toy expression is a diagnostic, not an assumption that actual transformer residuals are isotropic or meaningless. Extra activity can encode nonlinear belief coordinates, current-token information, or useful intermediate computations.

Population least-squares regression from beliefs to activations gives the orthogonality condition above. A probe from activations to beliefs establishes decodability and is a different regression. On held-out samples, fitted forward-regression residuals need not be exactly orthogonal; measure their cross terms rather than imposing the population decomposition numerically.

## 5. Level C: predict a learned metric in a solvable model

### 5.1 Balanced two-layer linear learning

Whiten the nondegenerate belief coordinates so $\mathbb E[z]=0$ and $\mathbb E[zz^\top]=I_r$. For a deterministic linear target $y=Tz$, train

$$
h=Ez,\qquad \hat y=Dh,
\qquad
\mathcal L=\frac12\mathbb E\|DEz-Tz\|^2
=\frac12\|DE-T\|_F^2.
$$

Here $T\in\mathbb R^{p\times r}$ is a target matrix, not context length; in this section its meaning is local. Let $T=U\operatorname{diag}(s_i)V^\top$. Assume hidden width at least $\operatorname{rank}(T)$ and equal Euclidean training rates for both factors.

Among exact factorizations $DE=T$, the minimum of

$$
\frac12(\|D\|_F^2+\|E\|_F^2)
$$

is attained by balanced factors

$$
D=U\operatorname{diag}(\sqrt{s_i})R^\top,
\qquad
E=R\operatorname{diag}(\sqrt{s_i})V^\top,
\qquad R^\top R=I.
$$

Consequently,

$$
G=V\operatorname{diag}(s_i)V^\top=(T^\top T)^{1/2},
\qquad
\boxed{\operatorname{PR}(C_h)=\frac{(\sum_is_i)^2}{\sum_is_i^2}.}
$$

This is an exact optimization benchmark. It is a dynamical prediction only when the training assumptions select these balanced factors. Do not assume arbitrary initialization, unequal layer learning rates, Adam, or next-token cross-entropy automatically do so.

### 5.2 Dynamics provide a stronger test than the endpoint

Under balanced, mode-aligned gradient flow, write the two factor amplitudes as $e_i=d_i=\sqrt{a_i}$. The equations are

$$
\dot e_i=(s_i-d_ie_i)d_i,
\qquad
\dot d_i=(s_i-d_ie_i)e_i,
$$

and therefore

$$
\dot a_i=2a_i(s_i-a_i),
\qquad
a_i(\tau)=\frac{s_i}
{1+[s_i/a_i(0)-1]e^{-2s_i\tau}}.
$$

For positive target modes the hidden covariance eigenvalues are $a_i(\tau)$, giving

$$
\boxed{
\operatorname{PR}(\tau)
=\frac{[\sum_i a_i(\tau)]^2}{\sum_i a_i(\tau)^2}.
}
$$

This follows the controlled mode-dynamics tradition of Saxe et al. [4]. For a paired balanced zero-target channel, $\dot a_i=-2a_i^2$; its initial variance decays rather than disappearing instantly. For rectangular targets, prescribe any unpaired null-direction initialization and calculate its covariance separately. The first proposed benchmark uses square full-rank targets. Exactly zero factor initialization cannot learn; use a specified small nonzero initialization.

With factor penalty $\lambda(\|D\|_F^2+\|E\|_F^2)/2$, replace $s_i$ in the amplitude growth term by $s_i-\lambda$. The equilibrium amplitudes become $(s_i-\lambda)_+$. This yields a further falsifiable threshold without changing target rank.

### 5.3 What the baseline does and does not transfer

Use soft next-token probability targets with squared loss when they are affine in beliefs, or use an explicitly constructed linear teacher. Categorical cross-entropy instead acts through logits and softmax. For the binary generator, the minimal signed logit coordinate is

$$
u_a(c)=\operatorname{atanh}(\rho_aq_am_a(c)),
$$

because $P(X_{T+1}^{(a)}=x\mid c)\propto e^{xu_a(c)}$. A linear unembedding can read $u_a$ directly; it cannot generally read exact probabilities from an affine $m_a$ representation through softmax without a nonlinear transformation elsewhere. Test both $m$ and $u$ coordinate hypotheses. In the weak-signal regime $u_a\approx \rho_aq_am_a$, making the difference perturbatively small.

A careful extension would derive the metric selected by a local quadratic expansion of cross-entropy, with an explicit Fisher/Hessian weighting and a controlled approximation regime. The word-embedding work of Karkada, Simon, Bahri and DeWeese [5] is a useful methodological precedent: solve a simplified language-learning objective and compare its representation dynamics to the original objective. Its solution is not a transformer theorem.

## 6. Level D: tractable attention calculations

### 6.1 Known positional mixing gives an exact covariance identity

For centered token features $\phi(X)$ and fixed position-dependent matrices $B_k$, define

$$
h_T=\sum_{k=0}^{T-1}B_k\phi(X_{T-k}).
$$

Then, without independence assumptions,

$$
\boxed{
C_h=\sum_{k,\ell=0}^{T-1}B_k\Gamma_{k\ell}B_\ell^\top,
\qquad
\Gamma_{k\ell}=\operatorname{Cov}[\phi(X_{T-k}),\phi(X_{T-\ell})].
}
$$

For a stationary edge-emitting HMM with scalar observables $f,g$, let $M_f=\sum_xf(x)T^{(x)}$ and $M_g=\sum_xg(x)T^{(x)}$. At lag $n\geq1$,

$$
\mathbb E[f(X_t)g(X_{t+n})]
=\pi M_fP^{n-1}M_g\mathbf1.
$$

Subtract the means; compute zero-lag terms separately using the single-token distribution. Vector features follow componentwise. Thus the data side of the attention covariance is exactly available without a neural approximation.

If attention is content-dependent, the $B_k$ vary with the context. They cannot be pulled outside the expectation. For small $T$, compute the full context sum; for larger cases, derive a conditional or perturbative closure. The fixed-kernel formula does not solve trained softmax attention by itself.

### 6.2 A practical derivation ladder

1. Fix an exponential positional kernel and output metric; verify the weak-signal calculation in Section 3.4.
2. Train only a low-dimensional kernel parameter, such as a decay coefficient, with fixed token features and value/output geometry. Derive the population loss as a weighted context sum and obtain its stationarity condition or gradient-flow ODE.
3. Unfreeze the value/output factors. Determine whether a balanced-factor argument survives and whether attention/value gains trade off.
4. Allow content-dependent attention, first as a perturbation around the positional solution. Predict the first covariance correction and its parameter dependence.
5. Add a nonlinear MLP and compare post-attention and post-MLP spectra. This is the first point at which a different statistical closure may be necessary.

At step 2, a low-dimensional numerical ODE evaluated from the known generator is an independent theory prediction if its initial conditions and coefficients are specified before the test. An analytic closed form is preferable when available, but is not required. Fitting the final decay or metric separately to each trained model is an explanatory reconstruction.

Simplex's constrained-belief work [2] supplies an architecture-aware precedent. A useful project extension must calculate a new covariance response, learning trajectory, or controlled failure boundary beyond reproducing its predicted geometry.

## 7. An exact bridge: a context kernel determines PR

This identity is the main connection to wide-network theory. For a fixed network, define the population-centered feature kernel

$$
k_d(c,c';\tau)
=\frac1d[h_\tau(c)-\mu_\tau]^\top[h_\tau(c')-\mu_\tau].
$$

Then

$$
\mathbb E_c[k_d(c,c;\tau)]=\frac{\operatorname{tr}C_\tau}{d}.
$$

For independent contexts $c,c'$, expand the squared inner product:

$$
\begin{aligned}
\mathbb E_{c,c'}[k_d(c,c';\tau)^2]
&=\frac1{d^2}\sum_{i,j}
\mathbb E_c[h_{c,i}h_{c,j}]
\mathbb E_{c'}[h_{c',i}h_{c',j}]\\
&=\frac{\operatorname{tr}(C_\tau^2)}{d^2},
\end{aligned}
$$

where $h_c$ denotes centered activations in this display. Therefore

$$
\boxed{
\operatorname{PR}(C_\tau)
=\frac{\{\mathbb E_c[k_d(c,c;\tau)]\}^2}
{\mathbb E_{c,c'}[k_d(c,c';\tau)^2]}.
}
$$

**Predicting an encoding matrix is sufficient but unnecessary. Predicting this learned kernel also closes the PR calculation.**

For finite support $c_1,\ldots,c_M$ with probabilities $p_i$, let $F_{i,:}=h(c_i)-\mu$, $P_p=\operatorname{diag}(p_i)$, and $K_c=FF^\top/d$. The nonzero eigenvalues of

$$
\frac{C_h}{d}=\frac{F^\top P_pF}{d}
$$

equal those of

$$
\boxed{\mathcal K=P_p^{1/2}K_cP_p^{1/2}.}
$$

Thus the kernel gives the full covariance spectrum, not just its first two traces. To center a predicted uncentered kernel $k$, subtract both probability-weighted row/column means and add its weighted grand mean. This is population centering over contexts, distinct from subtracting the coordinate mean in LayerNorm.

## 8. What the tractable transformer literature actually supplies

| Reference | Controlled object | Role in this project | Boundary |
| --- | --- | --- | --- |
| Hron, Bahri, Sohl-Dickstein, Novak [6] | Infinite-attention NNGP/NTK | Initialization and lazy/kernel baselines | Does not explain learned nonlinear belief features |
| Yang and Hu [7] | Feature-learning parameterizations via Tensor Programs | Specify a limit in which representations can move | A parameterization is not a solved covariance calculation |
| Bordelon and Pehlevan [8] | Dynamical feature/gradient kernels and response functions | Methodological basis for a self-consistent training calculation | Generic-network equations require an architecture-specific adaptation |
| Bordelon, Chaudhry, Pehlevan [9] | Transformer feature-learning limits | Closest direct foundation for Level E | Full closure is computationally difficult |
| Bordelon and Pehlevan [10] | Finite-width kernel fluctuations | Guidance for corrections and seed variation | Not an existing finite-head transformer PR solution |
| Karkada et al. [5,11] | Solvable embedding learning and symmetry-controlled geometry | Middle-level dynamics and symmetry ideas | Word-embedding theory is not full transformer dynamics |

Hron et al. [6] distinguish single-head non-Gaussian behavior from Gaussian-process limits obtained through many heads, and discuss positional encodings and normalization. The forward activation kernel and NTK are different objects: the former determines representation covariance, while the latter controls output learning in the lazy limit. A successful NTK fit to loss is not evidence of learned belief geometry.

Bordelon, Chaudhry and Pehlevan [9] analyze feature-learning transformers, including nonlinear MLPs and pre-normalization. Their dimension-per-head and number-of-head limits differ: in their admissible large-head-dimension scaling, attention heads collapse; increasing head count at fixed head dimension preserves a distribution of heads. Sample size, context length and training duration stay fixed in the limits. The paper notes that the full infinite-head dynamics is computationally prohibitive in its experiments and uses a widest-model ensemble proxy. A proxy of that kind is useful empirical evidence, not an independently solved DMFT.

For the present plan, the literature establishes a possible route. The new work is to make a restricted instance calculable, connect it to known predictive states, and test its spectral predictions.

## 9. Level E: nonlinear feature learning at large width

### 9.1 Choose one limit rather than saying “large width”

Write model width as $d=Hn$, where $H$ is head count and $n$ is dimension per head. Keep depth $L$ and context length $T$ small and fixed.

The recommended first route is $H\to\infty$ at fixed modest $n$. The comparison route is $n\to\infty$ at fixed $H$, with its separately specified attention scaling. These are different model sequences and should never be pooled into one width curve. Depth scaling is outside the initial scope.

Record a full parameterization contract: every matrix variance and forward multiplier; attention-logit scaling; residual-branch multiplier; readout multiplier; parameter-group learning rates; loss normalization; normalization epsilon and affine parameters; embeddings and biases; optimizer and initialization seeds. Preserve a literal copy of the implemented forward equations alongside the theoretical variables.

As an example in the raw-weight convention of [9], the readout is scaled by $1/(\gamma_0d)$ and bulk SGD rate by $\eta_0dL^{2\alpha_L-1}$. These factors must be used with that paper's other forward/initialization factors. They are not learning rates to paste onto an ordinary fan-in-normalized PyTorch transformer. For fixed $L$, implement one convention end to end and audit first-step and early-step update sizes across widths before claiming a feature-learning limit.

### 9.2 The required order parameters

For contexts $c,c'$, positions $s,s'$, training times $\tau,\tau'$, and layer $\ell$, introduce a residual feature kernel of the form

$$
K^\ell_{ss'}(c,c';\tau,\tau')
=\lim_{d\to\infty}\frac1d
h^\ell_s(c;\tau)^\top h^\ell_{s'}(c';\tau').
$$

The learning calculation generally needs more than this kernel: backward/gradient correlations, attention statistics, and response functions describing how a perturbation at an earlier training time changes later features. They capture dependencies created by repeatedly reusing learned weights. Replacing trained weights by fresh independent Gaussians each step discards the learning correlations the project is trying to explain.

The calculation should output the equal-time, fixed-position, context-centered kernel $k_\infty$. If it concentrates over initialization, its squared moments converge, and the limiting squared-kernel expectation is nonzero, Section 7 gives

$$
\boxed{
\operatorname{PR}_\infty(\ell,\tau,T)
=\frac{\{\mathbb E_c[k_\infty(c,c;\tau)]\}^2}
{\mathbb E_{c,c'}[k_\infty(c,c';\tau)^2]}.
}
$$

If the displayed kernel normalization instead gives a zero kernel or a vanishing denominator, this formula cannot determine the limit: rescale the kernel consistently or retain the required subleading terms. At fixed finite context support, a deterministic two-context kernel with nonzero squared moment is sufficient at leading order. A new four-point closure is not automatically required merely because PR contains $\operatorname{tr}(C^2)$. However, averaging a random kernel before squaring is invalid without concentration, and finite-width corrections can require higher-order correlations.

### 9.3 A feasible reduced solver is itself a milestone

Start with one layer, a short context, a small fixed training set or exact finite-support population objective, and full-batch gradient descent. Use a fixed finite update horizon. Treat output logits as a small vector and include their loss gradients explicitly.

1. Derive or faithfully specialize the attention model's forward and backward effective processes. Check the scalar-output to categorical-output extension rather than assuming it changes nothing.
2. Derive the initialization kernel independently and verify it against random finite networks.
3. Solve one update, then two updates. The second update can expose dependencies that an initialization-only approximation misses.
4. Add history-dependent responses and implement a small-horizon self-consistent solver. Iterate effective-process statistics until order parameters converge.
5. Evaluate held-out contexts as additional probe indices in the same process. Avoid fitting the test kernel from a large trained network.
6. Increase process-sampling count, numerical accuracy and update horizon separately. Record solver residuals and numerical uncertainty independently of network seed uncertainty.
7. Only after the attention-only closure works, add a nonlinear MLP; then add LayerNorm. A deliberately simplified first closure is valid if its finite networks implement exactly that simplification.

Use sign-flip, factor-exchange and token-permutation symmetries to reduce context-pair classes when they are genuinely preserved by the generator, objective, architecture and initialization law. The optimizer trajectory of a finite random model need not preserve each symmetry exactly. For two symmetric binary factors, a four-context probe set at $T=1$ is a debugging case; $T=2$ or $3$ gives a first nontrivial memory test. Keep larger-context finite-network experiments separate from this initial solver.

If the full solver remains impractical, derive a weak-signal or early-training perturbation for the kernel. A controlled expansion with an error regime is preferable to labeling an empirical widest-model curve as DMFT.

### 9.4 Feature learning must be demonstrated in the measured geometry

Track a centered, scale-normalized kernel change such as

$$
\Delta_K(\tau)
=\left\|\frac{\mathcal K_\tau}{\operatorname{tr}\mathcal K_\tau}
-\frac{\mathcal K_0}{\operatorname{tr}\mathcal K_0}\right\|_F,
$$

together with raw kernel norm, belief alignment, attention-pattern change and loss. An order-one uniform rescaling changes activity magnitude but not PR or normalized kernel shape. Attention can also change without a large change in PR. Compare to a deliberately lazy or frozen-feature control at matched prediction quality where feasible.

The strongest result is a kernel/spectrum trajectory predicted before finite-network testing, with error decreasing along the declared width sequence, while the lazy baseline fails to explain the same representation changes.

## 10. Limits, finite width, and what is actually averaged

### 10.1 Quenched and ensemble observables

For weights $W$, population covariance is $C_W=\operatorname{Cov}_c[h_W(c)]$. The project should ordinarily predict $\operatorname{PR}(C_W)$ and its seed variability. In general,

$$
\mathbb E_W[\operatorname{PR}(C_W)]
\neq\operatorname{PR}(\mathbb E_W[C_W]).
$$

Likewise $\mathbb E_W[k_W(c,c')^2]\neq\mathbb E_W[k_W(c,c')]^2$. A common deterministic limit can justify replacing these random objects only after checking concentration in the relevant regime.

### 10.2 Leading kernels can miss extensive dimensions

If $C_d=\lambda I_d$, then

$$
\frac{\operatorname{tr}C_d}{d}=\lambda,
\qquad
\frac{\operatorname{tr}(C_d^2)}{d^2}=\frac{\lambda^2}{d},
\qquad
\operatorname{PR}(C_d)=d.
$$

The denominator is a subleading term in width. If the leading limiting kernel has zero squared off-context expectation, one must retain finite-width terms to determine PR scaling. A nonzero limiting denominator instead gives finite limiting PR under suitable convergence assumptions.

For fixed finite context support of size $M$, covariance rank is at most $M-1$ for every width. Therefore this setting cannot establish a truly extensive $\operatorname{PR}\propto d$ asymptotic with fixed $M$. It can still test informative low-dimensional feature kernels. A joint support/width limit is a separate theory problem.

### 10.3 Sampling can imitate a width effect

An empirical covariance from $N_{\mathrm{eval}}$ contexts has rank at most $N_{\mathrm{eval}}-1$. Keeping that sample count small while increasing width can manufacture apparent dimension saturation. Exact finite-support weighting, sufficiently large independent evaluation sets, and convergence checks in evaluation size are required. Token positions from one trajectory are not independent bootstrap samples.

If using pairwise kernel estimates, off-diagonal context pairs can estimate the squared-kernel expectation when centering is population-known or estimated independently. State the centering and diagonal treatment; deleting diagonals from a finite-support population sum is not correct merely because the contexts can coincide.

## 11. What would count as progress

| Outcome | Claim justified | Claim not yet justified |
| --- | --- | --- |
| Exact belief spectra and controlled encoding counterexamples | Clear account of what belief geometry does and does not predict | A learning theory of transformer PR |
| Solved balanced baseline and weak-signal filter predictions | A reproducible bridge between task statistics and covariance | Generalization to arbitrary cross-entropy transformers |
| Independently predicted restricted-attention spectra across held-out parameters | A scoped theory of learned representational dimension | A full nonlinear feature-learning theory |
| Solved reduced large-head closure with finite-network spectral convergence | A concrete nonlinear transformer theory–experiment result | Universal behavior or guaranteed publication |
| Persistent failure despite verified scaling and solver accuracy | A falsified closure and identified missing variables | Automatic failure of the entire belief-state program |

A strong blog post can explain and reproduce the gap between belief rank, PR, learned metric and useful prediction. A plausible short paper needs a genuinely new predictive mechanism or sharp negative result, beyond substituting PR for existing dimension measures. The ambitious paper target is a tractable nonlinear training calculation that predicts representation spectra and their dependence on generator structure, with controlled finite-size validation. Publication value depends on novelty, strength of comparison, and results; none of these outcomes guarantees acceptance.

## 12. Work packages and decision points

**Package 1: exact generator calculations.** Implement binary filtering, probability-weighted enumeration, direct-sum/joint covariance, and weak-signal predictions. Deliver a parameterized spectrum before training any neural model. Advance when exact sums and independent sampling agree and degenerate cases behave correctly.

**Package 2: learned-metric baseline.** Derive and validate the balanced linear trajectory on whitened beliefs, including nonuniform target singular values. Deliberately violate balance and initialization assumptions. Advance when observed departures can be assigned to a specific broken condition.

**Package 3: restricted trained attention.** Derive the decay/gain ODE or a generator-evaluated population optimum. Freeze predictions for held-out persistence/reliability combinations. Advance if a nonconstant spectral curve is predicted, or revise the approximation if a specific term fails.

**Package 4: small nonlinear models.** Use the companion experiment plan to locate whether distortions arise at attention, the MLP, normalization, or residual accumulation. This tells the theory which variables require closure.

**Package 5: reduced width theory.** Reproduce initialization and early-step limits, then attempt a finite-horizon large-head solver. Add nonlinearities only after each simpler specialization passes. If the full closure is intractable, pursue an explicitly controlled early-time or weak-signal expansion.

The first theory commitment should be: **derive the unequal-factor weak-signal spectrum, then predict how a specified trained attention model changes its gains.** That yields a useful discriminating test while the large-width machinery is being investigated.

## References and reading order

Primary sources were checked on 15 November 2026. The equations proposed in this plan are scoped derivations; source citations identify precedents rather than attributing every proposed extension to those authors.

1. Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel, Paul M. Riechers. *Transformers Represent Belief State Geometry in their Residual Stream* (NeurIPS 2024). [Paper](https://arxiv.org/abs/2405.15943). Read for predictive states and linear probes.
2. Mateusz Piotrowski, Paul M. Riechers, Daniel Filan, Adam Shai. *Constrained Belief Updates Explain Geometric Structures in Transformer Representations* (ICML 2025). [Paper](https://proceedings.mlr.press/v267/piotrowski25a.html), [analysis repository](https://github.com/adamimos/epsilon-transformers). Read for architecture-dependent intermediate beliefs.
3. Adam Shai et al. *Transformers learn factored representations* (ICML 2026). [Paper](https://proceedings.mlr.press/v306/shai26a.html), [code](https://github.com/Astera-org/factored-reps), [weighted PCA implementation](https://github.com/Astera-org/factored-reps/blob/master/fwh_core/analysis/pca.py). Read for the closest spectral/factorization comparisons.
4. Andrew M. Saxe, James L. McClelland, Surya Ganguli. *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks* (ICLR 2014). [Paper](https://arxiv.org/abs/1312.6120). Read for the mode-aligned gradient-flow baseline.
5. Dhruva Karkada, James B. Simon, Yasaman Bahri, Michael R. DeWeese. *Closed-Form Training Dynamics Reveal Learned Features and Linear Structure in Word2Vec-like Models* (2025). [Paper](https://arxiv.org/abs/2502.09863). Earlier versions use a different title; this is the checked v3 title.
6. Jiri Hron, Yasaman Bahri, Jascha Sohl-Dickstein, Roman Novak. *Infinite attention: NNGP and NTK for deep attention networks* (ICML 2020). [Paper](https://proceedings.mlr.press/v119/hron20a.html). Read for initialization and lazy attention kernels.
7. Greg Yang, Edward J. Hu. *Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks* (ICML 2021). [Paper](https://proceedings.mlr.press/v139/yang21c.html). Read for the parameterization distinction.
8. Blake Bordelon, Cengiz Pehlevan. *Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks* (NeurIPS 2022; later journal version). [Paper](https://arxiv.org/abs/2205.09653). Read for effective processes, kernels and responses.
9. Blake Bordelon, Hamza Tahir Chaudhry, Cengiz Pehlevan. *Infinite Limits of Multi-head Transformer Dynamics* (NeurIPS 2024). [Paper](https://arxiv.org/abs/2405.15712), [full text](https://arxiv.org/html/2405.15712v2). Read Sections 2–3 and Appendices B, E.3, E.5, E.6 before implementing a width sequence.
10. Blake Bordelon, Cengiz Pehlevan. *Dynamics of Finite Width Kernel and Prediction Fluctuations in Mean Field Neural Networks* (NeurIPS 2023; later journal version). [Paper](https://arxiv.org/abs/2304.03408). Read if leading-width predictions work but seed/width corrections matter.
11. Dhruva Karkada, Daniel James Korchinski, Andres Nava, Matthieu Wyart, Yasaman Bahri. *Symmetry in language statistics shapes the geometry of model representations* (ICML 2026). [Paper](https://proceedings.mlr.press/v306/karkada26a.html). Read for symmetry-based geometry and as a nearby novelty comparison.

Suggested first reading sequence: [3], [2], [4], [9], then [6], [8] and [5]. The project should be positioned against both Simplex's existing spectra and the existing learned-kernel theory before claiming a new prediction of low-dimensional structure.
