# Predicting representation spectra during nonlinear transformer learning

*Step-by-step derivations for five research packages: generator statistics, learned metrics, attention, nonlinear finite models, and a tractable many-head training limit.*

Prepared 15 November 2026. This is a derivation document and research foundation, not a report of completed training experiments. Its target is a tractable nonlinear training calculation that predicts representation spectra as generator structure changes, with controlled finite-size validation.

The five packages of the earlier plan are retained. Their calculations have different logical status. The HMM identities, covariance algebra, and finite-network derivatives are exact. The balanced linear solution is exact under its explicit initialization and loss assumptions. Weak-signal formulas are asymptotic expansions at fixed context length. The nonlinear continuum calculation below closes for an explicitly restricted transformer architecture; it is not a solution of the full densely mixed transformer DMFT. Its finite-horizon approximation argument is given with boundedness and smoothness assumptions. A finite-size central-limit calculation is identified separately as a formal fluctuation expansion where a full probability theorem is not proved.

**A complete calculation needs an architecture that is specific enough to close.** Package 5 therefore introduces a one-block model with content-dependent softmax attention, a trainable nonlinear residual MLP, and a many-head mean readout. The simplification is block-diagonal processing between heads, not frozen attention. The resulting transport equation can be evaluated by deterministic quadrature over a two-dimensional initialization law. This provides an actionable nonlinear calculation while making the extension to dense cross-head mixing a separate research question.

Existing work already demonstrates belief geometry, activation-variance concentration, and factorization-dependent spectral signatures [R1–R3]. General transformer feature-learning limits also exist [R4]. The proposed research must contribute quantitative learning-dependent spectra, a controlled approximation, or a discriminating failure mechanism. Neither using PR nor writing a mean-field equation establishes novelty or publication value by itself.

## Reading map and conventions

| Package | Calculation completed here | Research work after the derivation |
| --- | --- | --- |
| 1. Generator statistics | Filtering, belief moments, factored/joint spectra, weak-signal expansion | Evaluate generator sweeps and select discriminating cases |
| 2. Learned metric | Covariance transport, balanced linear optimization and trajectories | Test transfer beyond the solvable assumptions |
| 3. Restricted attention | Correlation-based covariance and trainable positional-filter dynamics | Integrate the low-dimensional equations and test their regime |
| 4. Small nonlinear models | Explicit attention/MLP/normalization derivatives and exact covariance evolution | Implement finite networks and identify which variables matter |
| 5. Reduced width theory | Nonlinear head-distribution transport, spectral map, onset expansion, finite-head error argument | Solve the continuum numerically, validate error scaling, and assess extension to dense mixing |

Throughout, $T$ denotes context length, $\tau$ continuous training time, $c$ a context, and $p_c$ its population probability. A representation is a column vector. Expectations over contexts and averages over initialization are different operations. The generator is fixed during one training trajectory. Changes in its parameters compare distinct trajectories, unless a derivative with respect to a generator parameter is explicitly taken.

For a nonzero positive-semidefinite covariance $C$,

$$
\operatorname{PR}(C)=\frac{[\operatorname{tr}C]^2}{\operatorname{tr}(C^2)}.
$$

All spectra in the document refer to covariance of activations across contexts at a specified token position and layer. An ensemble covariance over network seeds is a different object. A zero covariance has undefined PR.

The derivations below give the intermediate algebra for each included model. They do not turn unproved statements about an unrestricted transformer into assumptions hidden inside a calculation. Background results used without a full proof, such as the spectral theorem, singular-value decomposition, and eigenvalue perturbation inequalities, are named where invoked.

## Package 1. Exact generator statistics and the covariance they imply

The first package concerns the probability distribution of contexts and predictive states. No neural-network training assumption enters this calculation. Throughout this package, $T$ is context length, $c=x_{1:T}$ is a context, and all context averages use its actual probability $p_T(c)$. The posterior is conditioned only on this finite context, starting from a declared prior. This differs from conditioning on an infinitely long observed past.

### 1.1 From the joint HMM distribution to the belief update

Consider an edge-emitting hidden Markov model with hidden state $S_t\in\{1,\ldots,K\}$ and token $X_t\in\mathcal X$. Define one nonnegative $K\times K$ matrix per token:

$$
T^{(x)}_{ij}=P(X_t=x,S_t=j\mid S_{t-1}=i).
$$

The unobserved hidden transition matrix is

$$
P_{ij}=\sum_{x\in\mathcal X}T^{(x)}_{ij},
\qquad
P\mathbf1=\mathbf1.
$$

Let the initial hidden distribution be the row vector $\pi$. For one hidden path $s_0,\ldots,s_T$, the Markov property gives

$$
P(s_0,x_1,s_1,\ldots,x_T,s_T)
=\pi_{s_0}\prod_{t=1}^{T}T^{(x_t)}_{s_{t-1}s_t}.
$$

Summing over the unobserved states except the final state gives

$$
\begin{aligned}
P(c,S_T=j)
&=\sum_{s_0,\ldots,s_{T-1}}
\pi_{s_0}T^{(x_1)}_{s_0s_1}\cdots T^{(x_T)}_{s_{T-1}j}\\
&=\left[\pi T^{(x_1)}\cdots T^{(x_T)}\right]_j.
\end{aligned}
$$

Consequently, the probability of the context is

$$
\begin{aligned}
p_T(c)
&=\sum_{j=1}^{K}P(c,S_T=j)\\
&=\pi T^{(x_1)}\cdots T^{(x_T)}\mathbf1.
\end{aligned}
$$

For every context with $p_T(c)>0$, Bayes' rule gives the row belief vector

$$
\boxed{
b_T(c)
=\frac{\pi T^{(x_1)}\cdots T^{(x_T)}}
{\pi T^{(x_1)}\cdots T^{(x_T)}\mathbf1}.
}
$$

The update can be performed recursively. If $b_{t-1}$ is the posterior after the first $t-1$ tokens, then

$$
\begin{aligned}
P(X_t=x,S_t=j\mid x_{1:t-1})
&=\sum_iP(S_{t-1}=i\mid x_{1:t-1})T^{(x)}_{ij}\\
&=[b_{t-1}T^{(x)}]_j.
\end{aligned}
$$

Summing over $j$ gives $P(X_t=x\mid x_{1:t-1})=b_{t-1}T^{(x)}\mathbf1$. Dividing the joint expression by this marginal gives

$$
b_t=\frac{b_{t-1}T^{(x_t)}}{b_{t-1}T^{(x_t)}\mathbf1}.
$$

The next-token distribution follows from the same calculation:

$$
P(X_{T+1}=y\mid c)=b_T(c)T^{(y)}\mathbf1.
$$

More generally, for a future word $w=y_1\cdots y_L$,

$$
P(w\mid c)=b_T(c)T^{(y_1)}\cdots T^{(y_L)}\mathbf1.
$$

Thus the posterior determines every future-word probability. It need not be a minimal representation: if hidden states have identical future distributions, the generator includes redundant predictive coordinates.

### 1.2 Mean belief and covariance

Write the column belief as $B(c)=b_T(c)^\top$ and its mean as $\bar B$. Its $j$th coordinate satisfies

$$
\begin{aligned}
\bar B_j
&=\sum_c p_T(c)P(S_T=j\mid c)\\
&=\sum_c P(c,S_T=j)\\
&=P(S_T=j).
\end{aligned}
$$

If $\pi P=\pi$, the hidden process starts in stationarity and $\bar B=\pi^\top$. This equality holds at every finite context length; no infinite-context approximation is involved.

The full probability-coordinate covariance is

$$
\begin{aligned}
\Sigma_B
&=\mathbb E_c[(B(c)-\bar B)(B(c)-\bar B)^\top]\\
&=\sum_c p_T(c)B(c)B(c)^\top-\bar B\bar B^\top.
\end{aligned}
$$

The second line follows by expanding all four terms and using $\mathbb E[B]=\bar B$:

$$
\mathbb E[BB^\top-B\bar B^\top-\bar BB^\top+\bar B\bar B^\top]
=\mathbb E[BB^\top]-\bar B\bar B^\top.
$$

Since $\mathbf1^\top B(c)=1$, we have $\mathbf1^\top(B(c)-\bar B)=0$. Multiplying the covariance by $\mathbf1$ therefore gives $\Sigma_B\mathbf1=0$, and

$$
\operatorname{rank}(\Sigma_B)\leq K-1.
$$

To remove this known null direction, choose $Q\in\mathbb R^{K\times(K-1)}$ such that $Q^\top Q=I$ and $Q^\top\mathbf1=0$. Define

$$
z(c)=Q^\top(B(c)-\bar B),
\qquad
\Sigma_b=\mathbb E[z(c)z(c)^\top]=Q^\top\Sigma_BQ.
$$

Because $QQ^\top$ is the identity on $\mathbf1^\perp$, $B(c)-\bar B=Qz(c)$ and $\Sigma_B=Q\Sigma_bQ^\top$. The nonzero eigenvalues of $\Sigma_B$ and $\Sigma_b$ agree. An orthonormal reduction preserves PR; an arbitrary change of belief coordinates need not.

For every nonzero covariance $C$, define

$$
\operatorname{PR}(C)=\frac{[\operatorname{tr}C]^2}{\operatorname{tr}(C^2)}.
$$

If $C=U\operatorname{diag}(\lambda_1,\ldots,\lambda_r,0,\ldots)U^\top$ with orthogonal $U$, then

$$
\operatorname{tr}C=\sum_{i=1}^{r}\lambda_i,
\qquad
\operatorname{tr}(C^2)=\sum_{i=1}^{r}\lambda_i^2.
$$

Cauchy–Schwarz gives $(\sum_i\lambda_i)^2\leq r\sum_i\lambda_i^2$, so $\operatorname{PR}(C)\leq r$. Equality requires all nonzero eigenvalues to be equal. A rank bound alone does not determine PR. At $C=0$, the ratio is undefined.

For finite alphabet and context length, there are at most $|\mathcal X|^T$ contexts. Exact enumeration therefore gives a finite sum for $\Sigma_b$. The covariance rank is also at most the number of positive-probability contexts minus one. The support is finite even when its points lie on a curved surface; it should not be described as a continuous manifold at fixed $T$.

### 1.3 Binary HMM: derive the scalar filter

Let $S_t,X_t\in\{-1,+1\}$, with persistence $0<\rho<1$ and emission reliability $0<q<1$:

$$
P(S_t=s\mid S_{t-1}=s')=\frac{1+\rho ss'}{2},
\qquad
P(X_t=x\mid S_t=s)=\frac{1+qxs}{2}.
$$

The stationary hidden prior is uniform. A distribution on $\{-1,+1\}$ is specified by its mean: if $m=\mathbb E[S]$, then $P(S=s)=(1+sm)/2$. Define the posterior magnetization

$$
m_t=\mathbb E[S_t\mid X_{1:t}].
$$

First propagate the previous posterior through the transition. The predictive prior is

$$
\begin{aligned}
P(S_t=s\mid X_{1:t-1})
&=\sum_{s'\in\{-1,+1\}}
\frac{1+\rho ss'}{2}\frac{1+s'm_{t-1}}{2}\\
&=\frac14\sum_{s'}
[1+s'm_{t-1}+\rho ss'+\rho s(s')^2m_{t-1}]\\
&=\frac14[2+2\rho sm_{t-1}]\\
&=\frac{1+\rho sm_{t-1}}{2}.
\end{aligned}
$$

The terms odd in $s'$ cancel and $(s')^2=1$. Set $a_t=\rho m_{t-1}$ temporarily. Multiplying prior and likelihood gives

$$
P(X_t=x,S_t=s\mid X_{1:t-1})
=\frac14(1+qxs)(1+a_ts).
$$

The normalization is

$$
\begin{aligned}
P(X_t=x\mid X_{1:t-1})
&=\frac14\sum_s[1+a_ts+qxs+qxa_ts^2]\\
&=\frac{1+qxa_t}{2}.
\end{aligned}
$$

The unnormalized first moment is

$$
\begin{aligned}
\sum_s sP(X_t=x,S_t=s\mid X_{1:t-1})
&=\frac14\sum_s[s+a_ts^2+qxs^2+qxa_ts^3]\\
&=\frac{a_t+qx}{2}.
\end{aligned}
$$

Dividing the first moment by the normalization and substituting $a_t=\rho m_{t-1}$ yields

$$
\boxed{
m_t=\frac{\rho m_{t-1}+qX_t}{1+\rho qX_tm_{t-1}},
\qquad m_0=0.
}
$$

The predicted next hidden mean is $\rho m_t$ and the emission mean given $S_{t+1}$ is $qS_{t+1}$. Applying conditional expectation twice gives

$$
\mathbb E[X_{t+1}\mid X_{1:t}]=\rho q m_t.
$$

Therefore

$$
P(X_{t+1}=x\mid X_{1:t})=\frac{1+x\rho qm_t}{2}.
$$

The probability of each context can also be accumulated with the scalar filter:

$$
p_T(x_{1:T})
=\prod_{t=1}^{T}\frac{1+\rho qx_tm_{t-1}(x_{1:t-1})}{2}.
$$

This expression and the matrix-HMM expression give independent implementations of the same exact distribution.

### 1.3a An exact belief-distribution recursion and its moment hierarchy

The scalar filter does not imply a closed scalar ODE for its variance. Let $\nu_{t-1}$ be the probability distribution of $m_{t-1}$ across contexts. At a given value $m$, write

$$
F_x(m)=\frac{\rho m+qx}{1+\rho qxm},
\qquad
w_x(m)=\frac{1+\rho qxm}{2}.
$$

The probability of the new token is $w_x(m)$, and its updated belief is $F_x(m)$. Therefore for every test function $g$,

$$
\int g(m')\,\nu_t(dm')
=\int\sum_{x\in\{-1,+1\}}w_x(m)g(F_x(m))\,\nu_{t-1}(dm).
$$

This is an exact distributional update, with $\nu_0=\delta_0$. In particular, setting $g(m')=(m')^2$ and temporarily writing $u=\rho m$ gives

$$
\begin{aligned}
\mathbb E[(m_t)^2\mid m_{t-1}=m]
&=\frac12\left[\frac{(u+q)^2}{1+uq}+\frac{(u-q)^2}{1-uq}\right]\\
&=\frac{(u+q)^2(1-uq)+(u-q)^2(1+uq)}{2(1-u^2q^2)}\\
&=\frac{2u^2+2q^2+uq[(u-q)^2-(u+q)^2]}{2(1-u^2q^2)}\\
&=\frac{2u^2+2q^2-4u^2q^2}{2(1-u^2q^2)}\\
&=\frac{u^2+q^2-2u^2q^2}{1-u^2q^2}.
\end{aligned}
$$

Consequently the exact variance obeys

$$
v_t=\int\frac{\rho^2m^2+q^2-2\rho^2q^2m^2}
{1-\rho^2q^2m^2}\,\nu_{t-1}(dm).
$$

The right side involves a rational function of $m^2$, not only its mean $v_{t-1}$. Expanding $(1-\rho^2q^2m^2)^{-1}=\sum_{n\geq0}(\rho^2q^2m^2)^n$ introduces all even moments. This series converges for $\rho q<1$ since $|m|\leq1$. Replacing the rational function's expectation by the same function of $v_{t-1}$ is generally an uncontrolled closure. Exact enumeration, the distributional recursion, or a controlled weak-signal expansion avoids that substitution.

### 1.4 Correlations and the weak-observation expansion

The hidden transition gives $\mathbb E[S_{t+1}\mid S_t]=\rho S_t$. Repeated conditioning yields

$$
\mathbb E[S_{t+n}\mid S_t]=\rho^nS_t.
$$

Multiplying by $S_t$, averaging, and using $S_t^2=1$ gives

$$
\mathbb E[S_tS_{t+n}]=\rho^n.
$$

For $n\geq1$, emissions at the two times are independent conditional on their hidden states. Hence

$$
\begin{aligned}
\mathbb E[X_tX_{t+n}]
&=\mathbb E[\mathbb E[X_t\mid S_t]\mathbb E[X_{t+n}\mid S_{t+n}]]\\
&=q^2\mathbb E[S_tS_{t+n}]\\
&=q^2\rho^n.
\end{aligned}
$$

At zero lag, $X_t^2=1$, so the answer is one rather than $q^2$. The means vanish by sign symmetry. Consequently,

$$
\Gamma_{k\ell}=\operatorname{Cov}(X_{T-k},X_{T-\ell})
=\begin{cases}
1,&k=\ell,\\
q^2\rho^{|k-\ell|},&k\neq\ell.
\end{cases}
$$

Now fix $T$ and keep $\rho$ away from one while taking $q\to0$. Starting from $m_0=0$, induction in the exact recursion gives $m_t=O(q)$. Thus $u_t=\rho qX_tm_{t-1}=O(q^2)$ and

$$
\frac1{1+u_t}=1-u_t+O(u_t^2).
$$

Substituting into the exact filter gives

$$
\begin{aligned}
m_t
&=(\rho m_{t-1}+qX_t)(1-\rho qX_tm_{t-1}+O(q^4))\\
&=\rho m_{t-1}+qX_t+O(q^3).
\end{aligned}
$$

Substitute the same relation for $m_{t-1}$, then $m_{t-2}$, until reaching $m_0=0$:

$$
\begin{aligned}
m_T
&=qX_T+\rho(qX_{T-1}+\rho m_{T-2})+O(q^3)\\
&=qX_T+\rho qX_{T-1}+\rho^2qX_{T-2}+\cdots+\rho^{T-1}qX_1+O(q^3)\\
&=q\sum_{k=0}^{T-1}\rho^kX_{T-k}+O(q^3).
\end{aligned}
$$

For the linear statistic $\widetilde m_T=q\sum_k\rho^kX_{T-k}$, its variance is exactly

$$
\begin{aligned}
\mathbb E[\widetilde m_T^2]
&=q^2\mathbb E\left[\sum_k\rho^kX_{T-k}\sum_\ell\rho^\ell X_{T-\ell}\right]\\
&=q^2\sum_{k,\ell=0}^{T-1}\rho^{k+\ell}\Gamma_{k\ell}\\
&=q^2\sum_{k=0}^{T-1}\rho^{2k}
+q^4\sum_{k\neq\ell}\rho^{k+\ell+|k-\ell|}.
\end{aligned}
$$

The difference $m_T-\widetilde m_T$ is $O(q^3)$, so expanding $m_T^2$ shows that its variance differs at order $q^4$:

$$
m_T^2=\widetilde m_T^2+2\widetilde m_TO(q^3)+O(q^6)
=\widetilde m_T^2+O(q^4).
$$

The leading exact posterior variance is therefore

$$
\boxed{
v(T)=\mathbb E[m_T^2]
=q^2\frac{1-\rho^{2T}}{1-\rho^2}+O(q^4).
}
$$

This last step sums the geometric series: $(1-\rho^2)\sum_{k=0}^{T-1}\rho^{2k}=1-\rho^{2T}$. It does not justify exchanging $T\to\infty$ or $\rho\to1$ with the weak-$q$ expansion. The exact variance of the approximate linear filter should also not be confused with the exact nonlinear posterior variance.

### 1.5 Two factors: direct-sum and joint covariance spectra

Take two independent copies, with parameters $(\rho_1,q_1)$ and $(\rho_2,q_2)$. Encode the observation pair as one of four arbitrary tokens. Sign symmetry gives $\mathbb E[m_1]=\mathbb E[m_2]=0$, and factor independence gives $\mathbb E[m_1m_2]=\mathbb E[m_1]\mathbb E[m_2]=0$. Defining $v_a=\mathbb E[m_a^2]$, the covariance in magnetization coordinates is

$$
\Sigma_m=\begin{pmatrix}v_1&0\\0&v_2\end{pmatrix}.
$$

Its traces are $v_1+v_2$ and $v_1^2+v_2^2$, so

$$
\boxed{\operatorname{PR}_{\mathrm{direct}}=\frac{(v_1+v_2)^2}{v_1^2+v_2^2}.}
$$

For one binary factor, orthonormal coordinates of its probability vector are $m/\sqrt2$. Concatenating two such probability-coordinate embeddings multiplies both eigenvalues by $1/2$ and leaves this PR unchanged. This observation is about a specified Euclidean metric, not a universal neural encoding.

The joint posterior over $(s_1,s_2)\in\{-1,+1\}^2$ is

$$
B(s_1,s_2)=\frac14(1+s_1m_1)(1+s_2m_2)
=\frac14[1+s_1m_1+s_2m_2+s_1s_2m_1m_2].
$$

Use the four orthonormal Walsh vectors $e_0(s)=1/2$, $e_1(s)=s_1/2$, $e_2(s)=s_2/2$, and $e_{12}(s)=s_1s_2/2$. Their inner products are sums over the four hidden states. Every nonconstant sign monomial sums to zero and every squared monomial sums to four, which proves orthonormality.

The joint posterior's coordinates are

$$
\begin{aligned}
e_0^\top B&=\frac12\sum_sB(s)=\frac12,\\
e_1^\top B&=\frac12\mathbb E[S_1\mid c]=\frac{m_1}{2},\\
e_2^\top B&=\frac{m_2}{2},\\
e_{12}^\top B&=\frac12\mathbb E[S_1S_2\mid c]=\frac{m_1m_2}{2}.
\end{aligned}
$$

The first coordinate is constant. The remaining coordinates have zero mean. Their cross moments vanish:

$$
\mathbb E[m_1m_2]=0,
\quad
\mathbb E[m_1^2m_2]=\mathbb E[m_1^2]\mathbb E[m_2]=0,
\quad
\mathbb E[m_1m_2^2]=0.
$$

Their variances are $v_1/4$, $v_2/4$, and $\mathbb E[m_1^2m_2^2]/4=v_1v_2/4$. These are therefore the nonzero joint-probability covariance eigenvalues, and

$$
\boxed{
\operatorname{PR}_{\mathrm{joint}}
=\frac{(v_1+v_2+v_1v_2)^2}
{v_1^2+v_2^2+v_1^2v_2^2}.
}
$$

The third mode is a nonlinear product of two sufficient coordinates. The joint belief points lie on a two-parameter product-state surface while their linear covariance may have rank three. At finite context length their support is finite; at infinite history it may be fractal. Neither observation forces the intrinsic support dimension to equal covariance rank.

### 1.6 From belief covariance to neural covariance

Let $z\in\mathbb R^r$ be centered belief coordinates, and consider the representation model

$$
\widetilde h=h-\mu=Ez+\epsilon,
\qquad E\in\mathbb R^{d\times r}.
$$

Expand its covariance without discarding any terms:

$$
\begin{aligned}
C_h
&=\mathbb E[(Ez+\epsilon)(Ez+\epsilon)^\top]\\
&=E\mathbb E[zz^\top]E^\top
+E\mathbb E[z\epsilon^\top]
+\mathbb E[\epsilon z^\top]E^\top
+\mathbb E[\epsilon\epsilon^\top].
\end{aligned}
$$

When $E$ is the population least-squares predictor of activations from beliefs, the cross moments vanish. To see this, let $C_{hz}=\mathbb E[\widetilde hz^\top]$ and minimize

$$
J(E)=\frac12\mathbb E\|\widetilde h-Ez\|^2.
$$

Differentiating componentwise gives

$$
\frac{\partial J}{\partial E_{i\alpha}}
=-\mathbb E\left[\left(\widetilde h_i-\sum_\beta E_{i\beta}z_\beta\right)z_\alpha\right]
=[E\Sigma_b-C_{hz}]_{i\alpha}.
$$

At the optimum, $E\Sigma_b=C_{hz}$, or $E=C_{hz}\Sigma_b^{-1}$ on the nondegenerate support. With $\epsilon=\widetilde h-Ez$,

$$
\mathbb E[\epsilon z^\top]=C_{hz}-E\Sigma_b=0.
$$

It follows that $C_h=C_{\mathrm{bel}}+C_\epsilon$, where $C_{\mathrm{bel}}=E\Sigma_bE^\top$. Population orthogonality does not imply exact held-out orthogonality for a fitted finite-sample probe. Evaluate those cross moments rather than declaring them zero by construction.

Define the encoding metric $G=E^\top E$. The first trace is

$$
\begin{aligned}
\operatorname{tr}C_{\mathrm{bel}}
&=\sum_i\sum_{\alpha,\beta}E_{i\alpha}(\Sigma_b)_{\alpha\beta}E_{i\beta}\\
&=\sum_{\alpha,\beta}(\Sigma_b)_{\alpha\beta}G_{\beta\alpha}\\
&=\operatorname{tr}(\Sigma_bG).
\end{aligned}
$$

For the second trace, multiply the matrices first:

$$
\begin{aligned}
C_{\mathrm{bel}}^2
&=(E\Sigma_bE^\top)(E\Sigma_bE^\top)\\
&=E\Sigma_b(E^\top E)\Sigma_bE^\top\\
&=E\Sigma_bG\Sigma_bE^\top.
\end{aligned}
$$

Taking the trace and moving the last factor to the front gives

$$
\operatorname{tr}(C_{\mathrm{bel}}^2)
=\operatorname{tr}(E^\top E\Sigma_bG\Sigma_b)
=\operatorname{tr}(G\Sigma_bG\Sigma_b).
$$

Therefore

$$
\boxed{
\operatorname{PR}(C_{\mathrm{bel}})
=\frac{[\operatorname{tr}(G\Sigma_b)]^2}
{\operatorname{tr}(G\Sigma_bG\Sigma_b)}.
}
$$

The full nonzero spectrum is that of $\Sigma_b^{1/2}G\Sigma_b^{1/2}$. Indeed, set $A=E\Sigma_b^{1/2}$, so $C_{\mathrm{bel}}=AA^\top$ and $A^\top A=\Sigma_b^{1/2}G\Sigma_b^{1/2}$. The two matrices have the same nonzero squared singular values of $A$.

An isometric encoding has $G=g^2I$, in which case both traces acquire factors $g^2$ and $g^4$ and PR equals belief PR. A general metric changes the result. For $\Sigma_b=I_2$ and $G=\operatorname{diag}(g^2,1)$,

$$
\operatorname{PR}=\frac{(g^2+1)^2}{g^4+1},
$$

which approaches one for large $g$ while the representation remains rank two.

Extra activity must be included when predicting the full residual stream:

$$
\begin{aligned}
\operatorname{tr}(C_h^2)
&=\operatorname{tr}[(C_{\mathrm{bel}}+C_\epsilon)^2]\\
&=\operatorname{tr}(C_{\mathrm{bel}}^2)
+2\operatorname{tr}(C_{\mathrm{bel}}C_\epsilon)
+\operatorname{tr}(C_\epsilon^2).
\end{aligned}
$$

Uncorrelated random components need not occupy orthogonal neuron subspaces. If, solely as an illustrative model, $C_\epsilon=\sigma^2I_d$, write $s_1=\operatorname{tr}C_{\mathrm{bel}}$ and $s_2=\operatorname{tr}(C_{\mathrm{bel}}^2)$. Then

$$
\operatorname{tr}C_h=s_1+d\sigma^2,
\qquad
\operatorname{tr}(C_h^2)=s_2+2\sigma^2s_1+d\sigma^4,
$$

and

$$
\operatorname{PR}(C_h)=\frac{(s_1+d\sigma^2)^2}{s_2+2\sigma^2s_1+d\sigma^4}.
$$

This completes a conditional representation calculation. To predict activations before training, the metric and extra covariance must be fixed by the model or predicted from learning, rather than fitted anew after each run.

## Package 2. A solvable calculation of the metric selected by training

This package is a controlled learning benchmark. It takes beliefs as inputs and uses a linear network with squared loss. It does not claim that arbitrary cross-entropy attention training selects the same metric.

### 2.1 Whitening the input distribution

Let the centered raw belief coordinates be $b\in\mathbb R^r$ with positive-definite covariance $\Sigma_b$. If the covariance is singular, restrict to its positive-eigenvalue support first. Write

$$
\Sigma_b=V\Lambda V^\top,
\qquad
\Sigma_b^{1/2}=V\Lambda^{1/2}V^\top,
\qquad
\Sigma_b^{-1/2}=V\Lambda^{-1/2}V^\top.
$$

Define $z=\Sigma_b^{-1/2}b$. Then

$$
\begin{aligned}
\mathbb E[zz^\top]
&=\Sigma_b^{-1/2}\mathbb E[bb^\top]\Sigma_b^{-1/2}\\
&=V\Lambda^{-1/2}\Lambda\Lambda^{-1/2}V^\top\\
&=VV^\top=I_r.
\end{aligned}
$$

A linear task $y=Ab$ becomes $y=Mz$ with $M=A\Sigma_b^{1/2}$. Thus even this simplest learned-metric problem depends on the generator through the whitening transformation and task matrix. The symbols $M$ and $T$ are kept separate: $M$ is the teacher map, while $T$ remains context length.

### 2.2 Matrix-factorization loss and exact gradients

Take $E\in\mathbb R^{d\times r}$, $D\in\mathbb R^{p\times d}$, hidden representation $h=Ez$, and prediction $\hat y=DEz$. With target $y=Mz$,

$$
\begin{aligned}
\mathcal L(D,E)
&=\frac12\mathbb E\|(DE-M)z\|^2\\
&=\frac12\mathbb E[z^\top(DE-M)^\top(DE-M)z]\\
&=\frac12\operatorname{tr}[(DE-M)^\top(DE-M)\mathbb E[zz^\top]]\\
&=\frac12\|DE-M\|_F^2.
\end{aligned}
$$

Let $R=DE-M$. A differential of the product is $dR=(dD)E+D(dE)$, so

$$
\begin{aligned}
d\mathcal L
&=\operatorname{tr}(R^\top dR)\\
&=\operatorname{tr}(R^\top(dD)E)+\operatorname{tr}(R^\top D(dE))\\
&=\operatorname{tr}[(RE^\top)^\top dD]
+\operatorname{tr}[(D^\top R)^\top dE].
\end{aligned}
$$

Identifying coefficients in the Frobenius inner product gives

$$
\nabla_D\mathcal L=RE^\top,
\qquad
\nabla_E\mathcal L=D^\top R.
$$

With equal unit gradient-flow rates,

$$
\dot D=(M-DE)E^\top,
\qquad
\dot E=D^\top(M-DE).
$$

These are exact finite-width ODEs for this objective. They require no mean-field limit.

### 2.3 Why balanced factors are preserved

Define the hidden-space imbalance matrix $B=EE^\top-D^\top D$. Differentiate each term:

$$
\begin{aligned}
\frac{d}{d\tau}(EE^\top)
&=\dot EE^\top+E\dot E^\top\\
&=D^\top(M-DE)E^\top+E(M-DE)^\top D,
\end{aligned}
$$

whereas

$$
\begin{aligned}
\frac{d}{d\tau}(D^\top D)
&=\dot D^\top D+D^\top\dot D\\
&=E(M-DE)^\top D+D^\top(M-DE)E^\top.
\end{aligned}
$$

The right sides are identical, so $\dot B=0$. Thus exact balance $EE^\top=D^\top D$ is invariant when imposed initially. Generic independently sampled initial weights do not satisfy this equality exactly.

### 2.4 Minimum factor norm selects a specific hidden covariance

Let the rank-$r_*$ compact SVD of $M$ be

$$
M=U\Sigma V^\top,
\qquad
\Sigma=\operatorname{diag}(s_1,\ldots,s_{r_*}),\quad s_i>0.
$$

Here $U^\top U=V^\top V=I_{r_*}$. Assume $d\geq r_*$. Among exact factorizations $DE=M$, consider minimizing

$$
\frac12(\|D\|_F^2+\|E\|_F^2).
$$

First obtain a lower bound. Using $U^\top MV=\Sigma$,

$$
\begin{aligned}
\sum_i s_i
&=\operatorname{tr}(U^\top MV)\\
&=\operatorname{tr}(U^\top DEV)\\
&=\langle D^\top U,EV\rangle_F\\
&\leq\|D^\top U\|_F\|EV\|_F\\
&\leq\|D\|_F\|E\|_F\\
&\leq\frac12(\|D\|_F^2+\|E\|_F^2).
\end{aligned}
$$

The first inequality is Cauchy–Schwarz. The second follows because multiplying by an orthonormal-column matrix projects onto a subspace and cannot increase the Frobenius norm. The last inequality is $(\|D\|_F-\|E\|_F)^2\geq0$.

Now choose any $R\in\mathbb R^{d\times r_*}$ with $R^\top R=I_{r_*}$ and construct

$$
D_*=U\Sigma^{1/2}R^\top,
\qquad
E_*=R\Sigma^{1/2}V^\top.
$$

Their product is

$$
D_*E_*=U\Sigma^{1/2}(R^\top R)\Sigma^{1/2}V^\top
=U\Sigma V^\top=M.
$$

Their squared norms are both $\operatorname{tr}\Sigma=\sum_i s_i$, so the lower bound is attained. Their input-coordinate encoding metric is

$$
\begin{aligned}
G_*
&=E_*^\top E_*\\
&=V\Sigma^{1/2}(R^\top R)\Sigma^{1/2}V^\top\\
&=V\Sigma V^\top\\
&=(M^\top M)^{1/2}.
\end{aligned}
$$

The last equality follows from $M^\top M=V\Sigma^2V^\top$ on its nonzero support. To show that every minimum-factor-norm solution has this same metric, inspect equality in the lower-bound chain. Equality in the projection norm bounds requires $D=UU^\top D$ and $E=EVV^\top$. Equality in Cauchy–Schwarz and in the arithmetic–geometric mean bound then gives $D^\top U=EV=:A$, since the two factor norms are equal and the inner product is positive. Therefore $D=UA^\top$, $E=AV^\top$, and the product constraint yields $A^\top A=\Sigma$. Thus $E^\top E=V\Sigma V^\top$ for any minimizer, up to a hidden-coordinate isometry. Because $z$ is whitened,

$$
C_h=E_*\mathbb E[zz^\top]E_*^\top=E_*E_*^\top=R\Sigma R^\top.
$$

The nonzero activation eigenvalues are therefore $s_i$, not $s_i^2$:

$$
\boxed{
\operatorname{PR}(C_h)=\frac{(\sum_i s_i)^2}{\sum_i s_i^2}.
}
$$

This is a minimum-norm exact-factorization result. It becomes a prediction of a training run only after establishing that its learning dynamics select this solution, with no additional initialized covariance remaining in unused hidden directions.

### 2.5 Mode-aligned dynamics and every step of the sigmoid solution

Choose an orthonormal hidden basis $R$ and initialize the factors in the teacher singular directions:

$$
D(\tau)=U\operatorname{diag}(d_i(\tau))R^\top,
\qquad
E(\tau)=R\operatorname{diag}(e_i(\tau))V^\top.
$$

This is an invariant family of the ODE because products of these factors remain diagonal in the same singular bases. Their effective output amplitudes are $a_i=d_ie_i$ and the loss is

$$
\mathcal L=\frac12\sum_i(d_ie_i-s_i)^2.
$$

Differentiating each scalar gives

$$
\frac{\partial\mathcal L}{\partial d_i}=(d_ie_i-s_i)e_i,
\qquad
\frac{\partial\mathcal L}{\partial e_i}=(d_ie_i-s_i)d_i.
$$

Hence

$$
\dot d_i=(s_i-d_ie_i)e_i,
\qquad
\dot e_i=(s_i-d_ie_i)d_i.
$$

If $d_i(0)=e_i(0)>0$, equality is preserved. Since $a_i=e_i^2=d_i^2$,

$$
\begin{aligned}
\dot a_i
&=\dot d_ie_i+d_i\dot e_i\\
&=(s_i-a_i)(e_i^2+d_i^2)\\
&=2a_i(s_i-a_i).
\end{aligned}
$$

For $s_i>0$ and $a_i(0)>0$, separate variables:

$$
\frac{da_i}{a_i(s_i-a_i)}=2\,d\tau.
$$

The partial fraction decomposition is

$$
\frac1{a(s-a)}=\frac1s\left(\frac1a+\frac1{s-a}\right).
$$

Integrating both sides between $0$ and $\tau$ gives

$$
\frac1{s_i}\left[\log\left|\frac{a_i(\tau)}{s_i-a_i(\tau)}\right|
-\log\left|\frac{a_i(0)}{s_i-a_i(0)}\right|\right]=2\tau.
$$

For the usual small-initialization regime $0<a_i(0)<s_i$, the absolute values can be removed. Exponentiating and solving for $a_i(\tau)$ gives

$$
\begin{aligned}
\frac{a_i(\tau)}{s_i-a_i(\tau)}
&=\frac{a_i(0)}{s_i-a_i(0)}e^{2s_i\tau},\\
\frac{s_i-a_i(\tau)}{a_i(\tau)}
&=\left[\frac{s_i}{a_i(0)}-1\right]e^{-2s_i\tau},\\
\frac{s_i}{a_i(\tau)}
&=1+\left[\frac{s_i}{a_i(0)}-1\right]e^{-2s_i\tau},\\
a_i(\tau)
&=\frac{s_i}{1+[s_i/a_i(0)-1]e^{-2s_i\tau}}.
\end{aligned}
$$

The hidden covariance has eigenvalues $e_i^2=a_i$, so the complete covariance and PR trajectory are

$$
C_h(\tau)=R\operatorname{diag}(a_i(\tau))R^\top,
\qquad
\boxed{
\operatorname{PR}(\tau)=\frac{[\sum_i a_i(\tau)]^2}{\sum_i a_i(\tau)^2}.
}
$$

The prediction uses target singular values, initialization amplitudes, and the declared time convention. It contains no fit to the later activation covariance. Early in training, $a_i\ll s_i$ gives $\dot a_i\approx2s_ia_i$, so $a_i(\tau)\approx a_i(0)e^{2s_i\tau}$. Stronger modes initially grow faster. PR need not be monotone when unequal mode growth, saturation, and initial background are combined.

For a paired balanced zero-target mode, $s_i=0$ gives $\dot a_i=-2a_i^2$. Integrating $da_i/a_i^2=-2d\tau$ gives

$$
a_i(\tau)=\frac{a_i(0)}{1+2a_i(0)\tau}.
$$

This statement concerns an explicitly paired scalar channel. Rectangular architectures can also have unpaired null directions; prescribe their initialization and calculate their covariance separately. Exactly zero initialization cannot leave the zero solution, so a learning experiment needs nonzero initial amplitudes.

### 2.6 Weight decay and unequal layer rates

Add factor penalty $\lambda(\|D\|_F^2+\|E\|_F^2)/2$. The scalar dynamics become

$$
\dot d_i=(s_i-a_i)e_i-\lambda d_i,
\qquad
\dot e_i=(s_i-a_i)d_i-\lambda e_i.
$$

For initially balanced modes, $d_i=e_i$ remains true and

$$
\begin{aligned}
\dot a_i
&=\dot d_ie_i+d_i\dot e_i\\
&=(s_i-a_i)(e_i^2+d_i^2)-2\lambda d_ie_i\\
&=2a_i(s_i-\lambda-a_i).
\end{aligned}
$$

Thus a mode with $s_i>\lambda$ saturates at $a_i=s_i-\lambda$; a mode with $s_i\leq\lambda$ decays toward zero. The equilibrium hidden spectrum is $(s_i-\lambda)_+$ in this controlled aligned regime. At $s_i=\lambda$ the decay is algebraic as in the zero-target calculation.

For unequal constant layer rates without weight decay,

$$
\dot d_i=\eta_D(s_i-a_i)e_i,
\qquad
\dot e_i=\eta_E(s_i-a_i)d_i.
$$

Differentiation gives

$$
\frac{d}{d\tau}\left(\frac{e_i^2}{\eta_E}-\frac{d_i^2}{\eta_D}\right)
=2e_id_i(s_i-a_i)-2d_ie_i(s_i-a_i)=0.
$$

If this scaled imbalance is initialized to zero, then $e_i^2/\eta_E=d_i^2/\eta_D$. Combining this equality with $a_i=d_ie_i>0$ yields

$$
e_i^2=a_i\sqrt{\frac{\eta_E}{\eta_D}},
\qquad
d_i^2=a_i\sqrt{\frac{\eta_D}{\eta_E}}.
$$

Consequently,

$$
\begin{aligned}
\dot a_i
&=\eta_D(s_i-a_i)e_i^2+\eta_E(s_i-a_i)d_i^2\\
&=2\sqrt{\eta_D\eta_E}\,a_i(s_i-a_i).
\end{aligned}
$$

The sigmoid time scale is rescaled by $\sqrt{\eta_D\eta_E}$. The hidden spectrum is multiplied by a common factor $\sqrt{\eta_E/\eta_D}$, which leaves PR unchanged if the same rates apply to every mode. More general mode-dependent rates or nonzero scaled imbalance can alter relative hidden mode strengths and require their own calculation.

### 2.7 The boundary with next-token cross-entropy

For the binary HMM, define the conditional future mean $r(c)=\rho qm_T(c)$. The next-token probabilities are $(1\pm r)/2$. A signed scalar logit coordinate $u$ yields $P(X=x\mid c)=e^{xu}/(e^u+e^{-u})$. Its signed mean is $\tanh u$, so matching the Bayes distribution requires

$$
\tanh u=r,
\qquad
u=\operatorname{atanh}r
=\frac12\log\frac{1+r}{1-r}.
$$

For two independent factors, pair-token logits can be $x_1u_1+x_2u_2$. Their normalization factorizes:

$$
\sum_{x_1,x_2}e^{x_1u_1+x_2u_2}
=(e^{u_1}+e^{-u_1})(e^{u_2}+e^{-u_2}).
$$

Thus the softmax distribution is exactly the product of the two correct binary distributions. These are nonlinear coordinates of raw beliefs. A balanced linear squared-loss theorem cannot be applied to them by merely renaming the loss. It supplies a solved benchmark and a candidate mechanism to compare against the subsequent nonlinear training calculation.

**Primary references for these two packages.** The HMM/belief setup follows the predictive-state framework used by [Shai et al., 2024](https://arxiv.org/abs/2405.15943), with generator-dependent architectural questions developed by [Piotrowski et al., 2025](https://arxiv.org/abs/2502.01954) and factorization comparisons by [Shai et al., 2026](https://arxiv.org/abs/2602.02385). The controlled deep-linear mode dynamics are developed in [Saxe, McClelland and Ganguli, 2014](https://arxiv.org/abs/1312.6120). The specific binary-generator algebra and covariance calculations above are explicit calculations for the present setup; they are not a claim that those papers prove a general transformer-PR theorem.

## Package 3. Restricted attention: derive the statistic, its covariance, and its learning equation

### 3.1 Fix the distribution and the feature map

Let $c=(x_1,\ldots,x_T)$ denote a context in a finite set $\mathcal C$, with exact generator probability $p_c$. Throughout a training run, $p_c$ is fixed. Let $\phi(x)\in\mathbb R^r$ be a fixed token feature map, and let $\mu_\phi=\mathbb E[\phi(X_t)]$ under stationarity. We first use centered token features $\widetilde\phi(x)=\phi(x)-\mu_\phi$. A content-dependent model will be treated separately.

Consider a positional attention statistic at the final position,

$$
z(c;\beta)=\sum_{k=0}^{T-1}\alpha_k(\beta)\widetilde\phi(x_{T-k}),
\qquad
\alpha_k(\beta)=\frac{e^{\beta d_k}}{Z(\beta)},
\qquad
Z(\beta)=\sum_{j=0}^{T-1}e^{\beta d_j}.
$$

The numbers $d_k$ are prescribed relative-position features. For $d_k=-k$, setting $\beta=-\log\rho$ gives $\alpha_k\propto\rho^k$. The scalar $\beta$ is a trainable attention parameter; no infinite-width assumption is involved.

We need the derivative of this statistic because it determines both its learning and its covariance response. Differentiating the normalizer first gives

$$
Z'(\beta)=\sum_jd_je^{\beta d_j},
\qquad
\frac{Z'(\beta)}{Z(\beta)}=\sum_j\alpha_jd_j\equiv\bar d.
$$

The quotient rule then gives

$$
\begin{aligned}
\partial_\beta\alpha_k
&=\frac{d_ke^{\beta d_k}Z-e^{\beta d_k}Z'}{Z^2}\\
&=\frac{e^{\beta d_k}}{Z}\left(d_k-\frac{Z'}Z\right)\\
&=\alpha_k(d_k-\bar d).
\end{aligned}
$$

As a normalization check, $\sum_k\partial_\beta\alpha_k=\sum_k\alpha_kd_k-\bar d\sum_k\alpha_k=0$. Since the token features are fixed,

$$
z_\beta(c)\equiv\partial_\beta z(c)
=\sum_k\alpha_k(d_k-\bar d)\widetilde\phi(x_{T-k}).
$$

The second derivative is also explicit. First,

$$
\partial_\beta\bar d
=\sum_jd_j\alpha_j(d_j-\bar d)
=\sum_j\alpha_jd_j^2-\bar d^2
\equiv\operatorname{Var}_{\alpha}(d).
$$

Therefore

$$
\partial_\beta^2\alpha_k
=\alpha_k\left[(d_k-\bar d)^2-\operatorname{Var}_{\alpha}(d)\right].
$$

These formulas provide exact first and second derivatives of the finite-context attention map. They can be checked against finite differences without training a model.

### 3.2 Calculate the output covariance without discarding temporal correlations

For a more general prescribed positional map, let $B_k\in\mathbb R^{d\times r}$ and

$$
h(c)=\sum_kB_k\widetilde\phi(x_{T-k}).
$$

Stationarity implies $\mathbb E[\widetilde\phi(X_{T-k})]=0$, so $\mathbb E[h]=0$. Expand the outer product before taking the expectation:

$$
\begin{aligned}
hh^\top
&=\left(\sum_kB_k\widetilde\phi(X_{T-k})\right)
\left(\sum_\ell B_\ell\widetilde\phi(X_{T-\ell})\right)^\top\\
&=\sum_{k,\ell}B_k\widetilde\phi(X_{T-k})
\widetilde\phi(X_{T-\ell})^\top B_\ell^\top.
\end{aligned}
$$

Define $\Gamma_{k\ell}=\mathbb E[\widetilde\phi(X_{T-k})\widetilde\phi(X_{T-\ell})^\top]$. Linearity of expectation then gives

$$
\boxed{C_h=\sum_{k,\ell}B_k\Gamma_{k\ell}B_\ell^\top.}
$$

Off-diagonal $k,\ell$ terms are generally nonzero. The formula with only $k=\ell$ is an independent-token approximation, not an identity for HMM data. For $h=z$, substitute $B_k=\alpha_kI_r$. For a fixed output map $A$, substitute $B_k=\alpha_kA$.

**The HMM supplies every temporal covariance.** Let $T^{(x)}$ be the edge-emitting matrices, $P=\sum_xT^{(x)}$, and stationary row distribution $\pi P=\pi$. For scalar token functions $f,g$, define

$$
M_f=\sum_xf(x)T^{(x)},\qquad M_g=\sum_xg(x)T^{(x)}.
$$

For lag $n\geq1$, the joint observation probability is

$$
P(X_t=x,X_{t+n}=y)=\pi T^{(x)}P^{n-1}T^{(y)}\mathbf1.
$$

This expression sums the $n-1$ unobserved intermediate emissions through $P^{n-1}$. Multiplying by $f(x)g(y)$ and summing gives

$$
\begin{aligned}
\mathbb E[f(X_t)g(X_{t+n})]
&=\sum_{x,y}f(x)g(y)\pi T^{(x)}P^{n-1}T^{(y)}\mathbf1\\
&=\pi\left(\sum_xf(x)T^{(x)}\right)P^{n-1}
\left(\sum_yg(y)T^{(y)}\right)\mathbf1\\
&=\pi M_fP^{n-1}M_g\mathbf1.
\end{aligned}
$$

Subtract $\mathbb E[f]\mathbb E[g]$ for covariance, where $\mathbb E[f]=\pi M_f\mathbf1$. At zero lag, the correct expression is

$$
\mathbb E[f(X_t)g(X_t)]=\pi\left(\sum_xf(x)g(x)T^{(x)}\right)\mathbf1.
$$

Apply these scalar formulas to each pair of feature components to construct $\Gamma_{k\ell}$. If the temporal order is reversed, transpose the appropriate vector covariance. This completes the generator-to-covariance calculation for prescribed positional mixing.

### 3.3 Train the positional kernel and readout

Now let a scalar target $y(c)$ be fixed by the generator or a specified teacher, and train

$$
f(c;w,\beta)=w^\top z(c;\beta),
\qquad
\mathcal L(w,\beta)=\frac12\sum_cp_c[f(c)-y(c)]^2.
$$

Define $e_c=f(c)-y(c)$. For a perturbation $dw$,

$$
df=z(c)^\top dw,
\qquad
d\mathcal L=\sum_cp_ce_cz(c)^\top dw.
$$

Consequently $\nabla_w\mathcal L=\sum_cp_ce_cz(c)$. For a perturbation $d\beta$, the chain rule gives $df=w^\top z_\beta(c)d\beta$, and hence

$$
\partial_\beta\mathcal L=\sum_cp_ce_cw^\top z_\beta(c).
$$

Euclidean gradient flow with declared learning-rate factors $\eta_w,\eta_\beta>0$ is therefore

$$
\boxed{
\dot w=-\eta_w\sum_cp_ce_cz(c),\qquad
\dot\beta=-\eta_\beta\sum_cp_ce_cw^\top z_\beta(c).
}
$$

The dot denotes training time $\tau$. These are closed finite-dimensional ODEs whose coefficients are exact weighted sums over the known generator. They predict the filter and its gains before any tested finite-network run. Numerically integrating this ODE is a theory calculation; fitting $\beta$ to each trained network afterward is a different test.

For fixed $\beta$, write $S_z=\sum_cp_cz(c)z(c)^\top$ and $r_{zy}=\sum_cp_cz(c)y(c)$. Expanding the first gradient yields

$$
\begin{aligned}
\nabla_w\mathcal L
&=\sum_cp_c[w^\top z(c)-y(c)]z(c)\\
&=\left(\sum_cp_cz(c)z(c)^\top\right)w-\sum_cp_cz(c)y(c)\\
&=S_zw-r_{zy}.
\end{aligned}
$$

If $S_z$ is invertible, the stationary readout is $w_\star=S_z^{-1}r_{zy}$. If it is singular, use the pseudoinverse and track any initialized null-space component. This does not determine $\beta_\star$; its stationarity condition remains the explicit scalar equation above.

At each ODE time, calculate $C_z(\beta)$ using the covariance sum. Differentiating that sum gives

$$
\partial_\beta C_z
=\sum_{k,\ell}\left[(\partial_\beta\alpha_k)\alpha_\ell
+\alpha_k(\partial_\beta\alpha_\ell)\right]\Gamma_{k\ell}.
$$

Thus $\dot C_z=\dot\beta\,\partial_\beta C_z$. A trainable linear output map adds its own product-rule terms. The learned spectrum follows by diagonalizing this predicted covariance, rather than by measuring a network's encoding matrix.

### 3.4 Content dependence changes the expectation being calculated

If $B_k=B_k(c;\theta)$, then

$$
\mu_h=\sum_cp_c\sum_kB_k(c;\theta)\phi(x_{T-k}),
$$

and the exact second moment is

$$
\mathbb E[hh^\top]
=\sum_cp_c\sum_{k,\ell}B_k(c;\theta)\phi(x_{T-k})
\phi(x_{T-\ell})^\top B_\ell(c;\theta)^\top.
$$

Subtract $\mu_h\mu_h^\top$ for covariance. Replacing each $B_k(c;\theta)$ by its mean does not give this expression because the coefficients and tokens depend on the same context. On a finite support, the exact sum remains a valid calculation; the challenge is determining the trained $\theta$ efficiently. The nonlinear model below makes this distinction explicit.

## Package 4. Exact nonlinear transformer training and covariance dynamics

### 4.1 Declare one complete finite architecture

Use column vectors and a decoder-only block. At position $s$, set

$$
h_s^0=E_{\mathrm{tok}}e_{x_s}+p_s\in\mathbb R^d,
$$

where $e_{x_s}$ is a vocabulary one-hot vector and $p_s$ is a positional embedding. Let $u_s=N_1(h_s^0)$, where $N_1$ is either identity or LayerNorm. For head $a$ with key/query dimension $n$ and value dimension $r$,

$$
q_s^a=W_Q^au_s+b_Q^a,\quad
k_j^a=W_K^au_j+b_K^a,\quad
v_j^a=W_V^au_j+b_V^a.
$$

The matrix shapes are $W_Q^a,W_K^a\in\mathbb R^{n\times d}$ and $W_V^a\in\mathbb R^{r\times d}$. For allowed sources $j\leq s$,

$$
L_{sj}^a=\frac{(q_s^a)^\top k_j^a}{\sqrt n},\qquad
\alpha_{sj}^a=\frac{e^{L_{sj}^a}}{\sum_{i\leq s}e^{L_{si}^a}},\qquad
z_s^a=\sum_{j\leq s}\alpha_{sj}^av_j^a.
$$

Masked positions are omitted from these sums; their derivative is zero. The attention residual and MLP are

$$
r_s=h_s^0+\sum_aW_O^az_s^a+b_O,
\qquad y_s=N_2(r_s),
$$

$$
t_s=W_1y_s+b_1,\qquad v_s^{\mathrm{MLP}}=\sigma(t_s),\qquad
h_s=r_s+W_2v_s^{\mathrm{MLP}}+b_2.
$$

Here $W_O^a\in\mathbb R^{d\times r}$, $W_1\in\mathbb R^{m\times d}$ and $W_2\in\mathbb R^{d\times m}$. The elementwise activation $\sigma$ can be GELU or a smooth function such as $\tanh$. A final-position linear readout is

$$
f(c)=Uh_T+b_U\in\mathbb R^V,\qquad
\widehat\eta_v(c)=\frac{e^{f_v(c)}}{\sum_we^{f_w(c)}}.
$$

This is a fully specified finite model. The $1/\sqrt n$ score factor is the declared finite-model convention, not a claim that it gives every desired infinite-width feature-learning limit.

### 4.2 Derive the population cross-entropy gradient

Let $\eta_v(c)=P(X_{T+1}=v\mid c)$ be the HMM's exact soft target. The conditional loss is

$$
\begin{aligned}
\ell_c
&=-\sum_v\eta_v(c)\log\widehat\eta_v(c)\\
&=-\sum_v\eta_v(c)f_v(c)
+\left(\sum_v\eta_v(c)\right)\log\sum_we^{f_w(c)}\\
&=-\eta(c)^\top f(c)+\log\sum_we^{f_w(c)}.
\end{aligned}
$$

Differentiating each logit gives

$$
\frac{\partial\ell_c}{\partial f_v}
=-\eta_v+\frac{e^{f_v}}{\sum_we^{f_w}}
=\widehat\eta_v-\eta_v\equiv\delta_v.
$$

The Hessian is

$$
\frac{\partial\delta_v}{\partial f_w}
=\widehat\eta_v(\mathbf1_{v=w}-\widehat\eta_w),
\qquad
H_f=\operatorname{diag}(\widehat\eta)-\widehat\eta\widehat\eta^\top.
$$

The full-batch loss is $\mathcal L(\theta)=\sum_cp_c\ell_c(\theta)$. Thus

$$
\boxed{\dot\theta=-\nabla_\theta\mathcal L
=-\sum_cp_cJ_f(c;\theta)^\top\delta(c;\theta),}
$$

where $J_f=\partial f/\partial\theta$. The following backpropagation calculation makes that Jacobian action explicit. All bars below denote derivatives of one conditional loss, for example $\bar h_s=\partial\ell_c/\partial h_s$.

### 4.3 Readout and MLP derivatives

From $df=(dU)h_T+U\,dh_T+db_U$,

$$
\nabla_U\ell_c=\delta h_T^\top,\qquad
\nabla_{b_U}\ell_c=\delta,\qquad
\bar h_s=\mathbf1_{s=T}U^\top\delta.
$$

For a loss summed over multiple output positions, replace the last expression by the sum of their readout contributions. Because each MLP is positionwise, its parameter derivatives are

$$
\nabla_{W_2}\ell_c=\sum_s\bar h_s(v_s^{\mathrm{MLP}})^\top,
\qquad
\nabla_{b_2}\ell_c=\sum_s\bar h_s.
$$

The upstream derivative at the nonlinear activation is $\overline{v_s^{\mathrm{MLP}}}=W_2^\top\bar h_s$. Since $dv_s^{\mathrm{MLP}}=\operatorname{diag}(\sigma'(t_s))dt_s$,

$$
\bar t_s=\sigma'(t_s)\odot(W_2^\top\bar h_s),
$$

where $\odot$ is the componentwise product. The affine map $t_s=W_1y_s+b_1$ then gives

$$
\nabla_{W_1}\ell_c=\sum_s\bar t_sy_s^\top,
\qquad
\nabla_{b_1}\ell_c=\sum_s\bar t_s,
\qquad
\bar y_s=W_1^\top\bar t_s.
$$

There are two paths from $r_s$ to the loss: the identity residual and the MLP. Therefore

$$
\boxed{\bar r_s=\bar h_s+J_{N_2}(r_s)^\top\bar y_s.}
$$

With no normalization, $J_{N_2}=I_d$. The additive identity term cannot be omitted.

### 4.4 Derive attention backpropagation through every path

For each head,

$$
\nabla_{W_O^a}\ell_c=\sum_s\bar r_s(z_s^a)^\top,
\qquad
\bar z_s^a=(W_O^a)^\top\bar r_s,
\qquad
\nabla_{b_O}\ell_c=\sum_s\bar r_s.
$$

Differentiate $z_s^a=\sum_{j\leq s}\alpha_{sj}^av_j^a$:

$$
dz_s^a=\sum_{j\leq s}(d\alpha_{sj}^a)v_j^a
+\sum_{j\leq s}\alpha_{sj}^a\,dv_j^a.
$$

Matching coefficients in $(\bar z_s^a)^\top dz_s^a$ gives

$$
\bar\alpha_{sj}^a=(\bar z_s^a)^\top v_j^a,
\qquad
\bar v_j^a=\sum_{s\geq j}\alpha_{sj}^a\bar z_s^a.
$$

The value vector at source $j$ influences every later query, which explains the second sum. For each fixed query $s$, the quotient rule applied to softmax gives

$$
\frac{\partial\alpha_{si}^a}{\partial L_{sj}^a}
=\alpha_{si}^a(\mathbf1_{i=j}-\alpha_{sj}^a).
$$

Hence

$$
\begin{aligned}
\bar L_{sj}^a
&=\sum_{i\leq s}\bar\alpha_{si}^a
\alpha_{si}^a(\mathbf1_{i=j}-\alpha_{sj}^a)\\
&=\alpha_{sj}^a\bar\alpha_{sj}^a
-\alpha_{sj}^a\sum_{i\leq s}\alpha_{si}^a\bar\alpha_{si}^a\\
&=\alpha_{sj}^a\left[\bar\alpha_{sj}^a
-\sum_{i\leq s}\alpha_{si}^a\bar\alpha_{si}^a\right].
\end{aligned}
$$

Next, $dL_{sj}^a=[(dq_s^a)^\top k_j^a+(q_s^a)^\top dk_j^a]/\sqrt n$. Summing all appearances gives

$$
\bar q_s^a=\frac1{\sqrt n}\sum_{j\leq s}\bar L_{sj}^ak_j^a,
\qquad
\bar k_j^a=\frac1{\sqrt n}\sum_{s\geq j}\bar L_{sj}^aq_s^a.
$$

The projection derivatives now follow from the same affine-map rule used above:

$$
\begin{aligned}
\nabla_{W_Q^a}\ell_c&=\sum_s\bar q_s^au_s^\top,
&\nabla_{b_Q^a}\ell_c&=\sum_s\bar q_s^a,\\
\nabla_{W_K^a}\ell_c&=\sum_j\bar k_j^au_j^\top,
&\nabla_{b_K^a}\ell_c&=\sum_j\bar k_j^a,\\
\nabla_{W_V^a}\ell_c&=\sum_j\bar v_j^au_j^\top,
&\nabla_{b_V^a}\ell_c&=\sum_j\bar v_j^a.
\end{aligned}
$$

The input $u_j$ is used as a query, a key and a value in every head. Its total upstream derivative is therefore

$$
\bar u_j=\sum_a\left[(W_Q^a)^\top\bar q_j^a
+(W_K^a)^\top\bar k_j^a+(W_V^a)^\top\bar v_j^a\right].
$$

There is also a direct residual path from $h_j^0$ to $r_j$. Consequently,

$$
\bar h_j^0=\bar r_j+J_{N_1}(h_j^0)^\top\bar u_j.
$$

Finally,

$$
\nabla_{E_{\mathrm{tok}}}\ell_c=\sum_j\bar h_j^0e_{x_j}^\top,
\qquad
\nabla_{p_j}\ell_c=\bar h_j^0.
$$

Repeated occurrences of a token accumulate in the same embedding column. Full-batch gradients of every parameter are obtained by weighting these conditional gradients with $p_c$ and summing over contexts.

### 4.5 Derive the LayerNorm Jacobian rather than treating normalization as a constant

For $x\in\mathbb R^d$, define

$$
\mu_x=\frac1d\mathbf1^\top x,\qquad
P_0=I_d-\frac1d\mathbf1\mathbf1^\top,\qquad
u=x-\mu_x\mathbf1=P_0x,
$$

$$
v=\frac1du^\top u,\qquad s=\sqrt{v+\epsilon},\qquad
N(x)=\gamma\odot\frac us+\beta_{\mathrm{LN}}.
$$

The differentials are

$$
du=P_0dx,\qquad
dv=\frac2du^\top du=\frac2du^\top dx,
$$

where $u^\top P_0=u^\top$ because $\mathbf1^\top u=0$. Then

$$
ds=\frac{dv}{2s}=\frac{u^\top dx}{d\,s},
\qquad
d(s^{-1})=-s^{-2}ds=-\frac{u^\top dx}{d\,s^3}.
$$

Here $d$ in the denominator is the width, while $ds$ is the differential of $s$. Applying the product rule gives

$$
d(u/s)=\frac{P_0}{s}dx-\frac{uu^\top}{d\,s^3}dx,
$$

and therefore

$$
\boxed{
J_N(x)=\operatorname{diag}(\gamma)
\left[\frac{P_0}{s}-\frac{uu^\top}{d\,s^3}\right].
}
$$

The corresponding gain and bias derivatives for upstream vector $\bar y$ are $\bar\gamma=\bar y\odot(u/s)$ and $\bar\beta_{\mathrm{LN}}=\bar y$, summed over all positions using the same LayerNorm parameters. A fixed $\epsilon>0$ prevents a zero-variance singularity. LayerNorm has a context-dependent Jacobian; it cannot generally be folded into one fixed linear map for the covariance calculation.

### 4.6 Exact activation dynamics expose the missing statistical closure

Choose a measured representation $h(c;\theta)\in\mathbb R^d$, which can be the post-attention or post-MLP residual at the final position. Define $J_h(c)=\partial h(c;\theta)/\partial\theta$. The chain rule and gradient flow give

$$
\begin{aligned}
\dot h(c)
&=J_h(c)\dot\theta\\
&=-\sum_{c'}p_{c'}J_h(c)J_f(c')^\top\delta(c')\\
&=-\sum_{c'}p_{c'}S(c,c';\theta)\delta(c'),
\end{aligned}
$$

where the hidden-output cross-Jacobian kernel $S(c,c')=J_h(c)J_f(c')^\top$ has shape $d\times V$. The output kernel instead is $\Theta(c,c')=J_f(c)J_f(c')^\top$, with

$$
\dot f(c)=-\sum_{c'}p_{c'}\Theta(c,c')\delta(c').
$$

Knowing the output NTK is not enough to recover hidden activity: it does not specify $S$, and both objects normally evolve during feature learning.

To obtain covariance dynamics, define $\mu=\sum_cp_ch(c)$, $\widetilde h(c)=h(c)-\mu$, and $C=\sum_cp_c\widetilde h(c)\widetilde h(c)^\top$. Since $p_c$ is fixed,

$$
\dot\mu=\sum_cp_c\dot h(c),\qquad
\sum_cp_c\widetilde h(c)=0.
$$

Apply the product rule:

$$
\begin{aligned}
\dot C
&=\sum_cp_c\left[(\dot h(c)-\dot\mu)\widetilde h(c)^\top
+\widetilde h(c)(\dot h(c)-\dot\mu)^\top\right]\\
&=\sum_cp_c\left[\dot h(c)\widetilde h(c)^\top
+\widetilde h(c)\dot h(c)^\top\right].
\end{aligned}
$$

The terms containing $\dot\mu$ vanish because the weighted centered mean is zero. Substituting the hidden-output kernel yields

$$
\boxed{
\dot C=-\sum_{c,c'}p_cp_{c'}\left[
S(c,c')\delta(c')\widetilde h(c)^\top
+\widetilde h(c)\delta(c')^\top S(c,c')^\top\right].
}
$$

This equation is exact at every finite width. It is not closed in $C$: different parameter configurations can have the same covariance but different cross-Jacobian kernels and different subsequent evolution.

The centered feature kernel $k(c,c')=\widetilde h(c)^\top\widetilde h(c')/d$ obeys

$$
\dot k(c,c')=\frac1d\left[
(\dot h(c)-\dot\mu)^\top\widetilde h(c')
+\widetilde h(c)^\top(\dot h(c')-\dot\mu)\right].
$$

Define $S_0(c,a)=S(c,a)-\sum_bp_bS(b,a)$. Substitution gives the completely explicit identity

$$
\dot k(c,c')=-\frac1d\sum_ap_a\left[
\delta(a)^\top S_0(c,a)^\top\widetilde h(c')
+\widetilde h(c)^\top S_0(c',a)\delta(a)\right].
$$

A tractable nonlinear training theory must calculate these evolving objects or replace them by a controlled equivalent. Assuming them constant imposes a lazy approximation; it does not derive nonlinear feature learning.

### 4.7 Derive the PR and eigenvalue response

Let $A=\operatorname{tr}C$ and $B=\operatorname{tr}(C^2)>0$. Differentiating the traces gives

$$
\dot A=\operatorname{tr}\dot C,
\qquad
\dot B=\operatorname{tr}(\dot CC+C\dot C)
=2\operatorname{tr}(C\dot C).
$$

Since $\operatorname{PR}=A^2/B$, the quotient rule gives

$$
\begin{aligned}
\frac{d}{d\tau}\operatorname{PR}
&=\frac{2A\dot AB-A^2\dot B}{B^2}\\
&=\frac{2A\operatorname{tr}\dot C}{B}
-\frac{2A^2\operatorname{tr}(C\dot C)}{B^2}.
\end{aligned}
$$

For a simple eigenvalue $\lambda_i$ with normalized eigenvector $u_i$, differentiate $Cu_i=\lambda_iu_i$:

$$
\dot Cu_i+C\dot u_i=\dot\lambda_iu_i+\lambda_i\dot u_i.
$$

Multiplication by $u_i^\top$ cancels the $\dot u_i$ terms because $u_i^\top C=\lambda_i u_i^\top$, giving $\dot\lambda_i=u_i^\top\dot Cu_i$. At an eigenvalue degeneracy, first-order splitting is determined by the restriction of $\dot C$ to the degenerate eigenspace. Tracking arbitrary individual eigenvectors through that crossing is not meaningful.

These identities convert a prediction of nonlinear activation dynamics into predictions of spectral growth and PR. Their derivation alone does not solve the trained model. The next package supplies a deliberately restricted nonlinear architecture for which a distribution over heads closes the dynamics, while identifying the extension still needed for dense cross-head transformer mixing.

## Package 5. A closed nonlinear training calculation and finite-head validation

The exact finite-network equations in Package 4 do not by themselves reduce computational complexity. We now choose a restricted transformer for which the entire nonlinear training law closes through a distribution over heads. The following is a derivation for that architecture. It does not specialize the full dense-transformer DMFT by silently dropping its response terms.

### 5.1 Specify the architecture before taking a limit

Let each context contain two binary observed components, $x_s=(x_{s1},x_{s2})^\top\in\{-1,+1\}^2$, $s=1,\ldots,T$. The generator supplies the finite set of contexts $\mathcal C$, their probabilities $p_c$, and the two next-token conditional means

$$
\mu_j(c)=\mathbb E[Y_j\mid c],\qquad j=1,2.
$$

For the independent binary generators in Package 1, $\mu_j(c)=\rho_jq_jm_j(c)$. Define

$$
w=\frac{(1,1)^\top}{\sqrt2},\qquad t_s=s-T,\qquad
d_s(c)=(w^\top x_T)(w^\top x_s).
$$

A head has parameters $\theta=(a_1,a_2,\beta,\kappa,g)\in\mathbb R^5$. Its forward pass is

$$
\begin{aligned}
\ell_s&=\beta t_s+\kappa d_s(c),\\
\alpha_s&=\frac{e^{\ell_s}}{\sum_{v=1}^T e^{\ell_v}},\\
z_j&=\sum_{s=1}^T\alpha_s x_{sj},\\
r_j&=x_{Tj}+z_j,\\
\psi_j(c;\theta)&=r_j+\tanh(g r_j).
\end{aligned}
$$

The score is a rank-one content query–key product plus a relative-position bias. For example, a query $\kappa w^\top x_T$ and key $w^\top x_s$ realize the content term exactly. Values are fixed two-component token features. The first residual addition is $r=x_T+z$; the second is an elementwise residual MLP with trainable gain $g$ and fixed unit output coefficient. Heads have no dense learned mixing between them. At other positions, the same block uses the causal prefix; the loss and representation below concern the final position.

This is a deliberately restricted one-block transformer. Its attention, MLP gain and readout learn nonlinearly. It is not a generic dense GPT block, and the supplied two-component input embedding already exposes the chosen token coordinates. A result here cannot claim spontaneous discovery of latent factors from arbitrary token embeddings. The mixed content score contains cross-factor terms, so separate factor orthogonality is not forced: $(w^\top x_T)(w^\top x_s)=(x_{T1}x_{s1}+x_{T1}x_{s2}+x_{T2}x_{s1}+x_{T2}x_{s2})/2$.

For $H$ heads, define signed marginal logits

$$
f_{H,j}(c)=\frac1H\sum_{i=1}^H a_{ij}\psi_j(c;\theta_i).
$$

The four categorical token logits are $L_y=y_1f_{H,1}+y_2f_{H,2}$ for $y\in\{-1,+1\}^2$. This output family factorizes; Section 5.14 adds the interaction needed for dependent targets.

The measured residual representation is the concatenation

$$
h_H(c)=\big(\psi(c;\theta_1)^\top,\ldots,\psi(c;\theta_H)^\top\big)^\top
\in\mathbb R^{2H}.
$$

Readout amplitudes $a_j$ do not directly rescale the measured residual coordinates. They influence those coordinates through training of $\beta,\kappa,g$. This distinction prevents readout amplification from being misidentified as hidden-feature movement.

### 5.2 Differentiate every trainable component

Write $Z=\sum_v e^{\ell_v}$. For any scalar parameter $u$,

$$
\begin{aligned}
\partial_u\alpha_s
&=\partial_u(e^{\ell_s}Z^{-1})\\
&=e^{\ell_s}(\partial_u\ell_s)Z^{-1}
-e^{\ell_s}Z^{-2}\partial_uZ\\
&=\alpha_s\partial_u\ell_s
-\alpha_s\sum_v\frac{e^{\ell_v}}Z\partial_u\ell_v\\
&=\alpha_s\left(\partial_u\ell_s-\sum_v\alpha_v\partial_u\ell_v\right).
\end{aligned}
$$

Since $\partial_\beta\ell_s=t_s$ and $\partial_\kappa\ell_s=d_s$,

$$
\begin{aligned}
z_{j,\beta}
&=\sum_sx_{sj}\partial_\beta\alpha_s\\
&=\sum_s\alpha_sx_{sj}t_s
-\left(\sum_s\alpha_sx_{sj}\right)\left(\sum_v\alpha_vt_v\right)\\
&=\sum_s\alpha_s(x_{sj}-z_j)t_s,\\
z_{j,\kappa}&=\sum_s\alpha_s(x_{sj}-z_j)d_s.
\end{aligned}
$$

The residual obeys $r_{j,u}=z_{j,u}$ for $u=\beta,\kappa$ and $r_{j,g}=0$. Use $d\tanh v/dv=1-\tanh^2v=\operatorname{sech}^2v$ to obtain

$$
\begin{aligned}
\psi_{j,u}
&=r_{j,u}+\operatorname{sech}^2(gr_j)g r_{j,u}\\
&=D_jz_{j,u},\qquad
D_j=1+g\operatorname{sech}^2(gr_j),\qquad u=\beta,\kappa,\\
\psi_{j,g}&=r_j\operatorname{sech}^2(gr_j).
\end{aligned}
$$

Let $\phi_j(c;\theta)=a_j\psi_j(c;\theta)$. The five coordinates of its gradient are

$$
\nabla_\theta\phi_j
=\big(\delta_{j1}\psi_j,\delta_{j2}\psi_j,
a_j\psi_{j,\beta},a_j\psi_{j,\kappa},a_j\psi_{j,g}\big)^\top.
$$

These formulas completely specify the nonlinear head gradients without automatic differentiation.

### 5.3 Reduce population cross-entropy exactly

The partition function factorizes because the two signed logits are additive:

$$
\begin{aligned}
Z_f
&=\sum_{y_1=\pm1}\sum_{y_2=\pm1}e^{y_1f_1+y_2f_2}\\
&=(e^{f_1}+e^{-f_1})(e^{f_2}+e^{-f_2})\\
&=4\cosh f_1\cosh f_2.
\end{aligned}
$$

Taking the conditional expectation of $-\log P_f(Y\mid c)$ gives

$$
\begin{aligned}
\ell_c(f)
&=\log Z_f-\mathbb E[Y_1\mid c]f_1-\mathbb E[Y_2\mid c]f_2\\
&=\sum_{j=1}^2\left[\log(2\cosh f_j)-\mu_j(c)f_j\right].
\end{aligned}
$$

Therefore

$$
\mathcal L_H=\sum_cp_c\ell_c(f_H(c)),\qquad
e_j(c)=\partial_{f_j}\ell_c=\tanh f_{H,j}(c)-\mu_j(c),
$$

and

$$
\partial_{f_j}\partial_{f_k}\ell_c
=\delta_{jk}\operatorname{sech}^2 f_j.
$$

The conditional means are enough to compute the loss in this factorized output family, even for dependent targets, but they are not enough to achieve the dependent Bayes predictor. For independent targets the optimum signed logit is $f_j^*(c)=\operatorname{atanh}\mu_j(c)$, obtained by setting $e_j(c)=0$.

### 5.4 Choose the training-time scaling explicitly

Because $f_H=H^{-1}\sum_i\phi(\theta_i)$,

$$
\begin{aligned}
\nabla_{\theta_i}\mathcal L_H
&=\sum_cp_c\sum_j
\partial_{f_j}\ell_c\,\nabla_{\theta_i}f_{H,j}(c)\\
&=\frac1H\sum_cp_c\sum_je_j(c)\nabla_\theta\phi_j(c;\theta_i).
\end{aligned}
$$

Use the declared mean-field gradient flow

$$
\dot\theta_i=-H\nabla_{\theta_i}\mathcal L_H
=-\sum_cp_c\sum_je_j(c)\nabla_\theta\phi_j(c;\theta_i).
$$

The $H$ multiplier cancels the $1/H$ derivative. An Euler implementation uses $\theta_i^{n+1}=\theta_i^n-\Delta\tau H\nabla_{\theta_i}\mathcal L_H$ with the displayed forward parameterization. It is not obtained by applying this learning rate to a differently normalized dense transformer.

Coordinatewise the exact finite-head equations are

$$
\begin{aligned}
\dot a_{ij}&=-\sum_cp_ce_j(c)\psi_j(c;\theta_i),\\
\dot u_i&=-\sum_cp_c\sum_je_j(c)a_{ij}\psi_{j,u}(c;\theta_i),
\qquad u\in\{\beta,\kappa,g\}.
\end{aligned}
$$

This is nonlinear feature learning whenever the last three coordinates move in a way that changes the centered representation kernel.

### 5.5 Derive the distributional closure

Define $\nu_H=H^{-1}\sum_i\delta_{\theta_i}$. For any probability measure $\nu$, set

$$
f_j(c;\nu)=\int\phi_j(c;\theta)\,\nu(d\theta),
\qquad
v(\theta;\nu)=-\sum_cp_c\sum_j
[\tanh f_j(c;\nu)-\mu_j(c)]\nabla_\theta\phi_j(c;\theta).
$$

Then the finite heads obey $\dot\theta_i=v(\theta_i;\nu_H)$. For a smooth test function $F$,

$$
\begin{aligned}
\frac{d}{d\tau}\int F\,d\nu_H
&=\frac1H\sum_i\frac{d}{d\tau}F(\theta_i)\\
&=\frac1H\sum_i\nabla F(\theta_i)\cdot v(\theta_i;\nu_H)\\
&=\int\nabla F(\theta)\cdot v(\theta;\nu_H)\,\nu_H(d\theta).
\end{aligned}
$$

The continuum equation has the same weak form with $\nu_H$ replaced by $\nu$. Integration by parts, with no boundary flux or with decaying/compactly supported test functions, gives

$$
\int\nabla F\cdot(v\nu)=-\int F\nabla_\theta\cdot(v\nu).
$$

Consequently

$$
\boxed{\partial_\tau\nu_\tau+\nabla_\theta\cdot[\nu_\tau v(\theta;\nu_\tau)]=0.}
$$

For singular measures this is a weak continuity equation; it does not require a five-dimensional density. It is closed: $v$ depends only on the current measure, known context weights, and known target means. No learned metric is fitted to held-out networks. The dependence on training history is carried by the transported measure itself.

This measure-valued approach is related to established neural-network mean-field theory [R7–R9]. We derive the particular head model here; we do not import global convergence claims that require other architectures or initialization conditions.

### 5.6 Energy descent and the output-learning kernel

For the continuum loss $\mathcal L(\nu)=\sum_cp_c\ell_c(f(c;\nu))$, a signed perturbation $\delta\nu$ of total mass zero gives

$$
\begin{aligned}
\delta\mathcal L
&=\sum_cp_c\sum_je_j(c)\delta f_j(c)\\
&=\int\left[\sum_cp_c\sum_je_j(c)\phi_j(c;\theta)\right]\delta\nu(d\theta).
\end{aligned}
$$

Thus a functional derivative is $V(\theta;\nu)=\sum_cp_c\sum_je_j(c)\phi_j(c;\theta)$ and $v=-\nabla_\theta V$. Applying the weak equation to the first variation yields

$$
\begin{aligned}
\dot{\mathcal L}
&=\int V\,\partial_\tau\nu\\
&=-\int V\nabla_\theta\cdot(\nu v)\\
&=\int\nabla_\theta V\cdot v\,d\nu\\
&=-\int\|\nabla_\theta V\|^2\,d\nu\leq0.
\end{aligned}
$$

The analogous finite identity is $\dot{\mathcal L}_H=-H\sum_i\|\nabla_{\theta_i}\mathcal L_H\|^2$. Loss descent does not imply monotone PR descent; the PR derivative contains differently weighted spectral terms.

Differentiate the output along the transport:

$$
\begin{aligned}
\dot f_j(c)
&=\int\nabla_\theta\phi_j(c;\theta)\cdot v(\theta;\nu)\,d\nu\\
&=-\sum_{c'}p_{c'}\sum_ke_k(c')
\int\nabla_\theta\phi_j(c;\theta)\cdot\nabla_\theta\phi_k(c';\theta)\,d\nu.
\end{aligned}
$$

Define the evolving output-training kernel

$$
\begin{aligned}
\Theta_{jk}(c,c';\nu)
&=\int\nabla\phi_j(c)\cdot\nabla\phi_k(c')\,d\nu\\
&=\delta_{jk}\int\psi_j(c)\psi_k(c')\,d\nu
+\int a_ja_k\sum_{u\in\{\beta,\kappa,g\}}
\psi_{j,u}(c)\psi_{k,u}(c')\,d\nu.
\end{aligned}
$$

Then $\dot f_j(c)=-\sum_{c'}p_{c'}\sum_k\Theta_{jk}(c,c';\nu)e_k(c')$. This has a kernel form but is not a fixed-NTK approximation: $\nu$, $\psi$ and $\Theta$ evolve together.

### 5.7 Convert the head law into the representation spectrum

At fixed generator parameters, center each head feature across contexts:

$$
\bar\psi(\theta)=\sum_cp_c\psi(c;\theta),\qquad
\widetilde\psi(c;\theta)=\psi(c;\theta)-\bar\psi(\theta).
$$

For a finite network,

$$
\begin{aligned}
K_H(c,c')
&=\frac{[h_H(c)-\bar h_H]^\top[h_H(c')-\bar h_H]}{2H}\\
&=\frac1{2H}\sum_i\widetilde\psi(c;\theta_i)^\top\widetilde\psi(c';\theta_i).
\end{aligned}
$$

Define the observable

$$
\Xi_{cc'}(\theta)=\frac12\widetilde\psi(c;\theta)^\top\widetilde\psi(c';\theta).
$$

The continuum kernel is simply $K_\nu(c,c')=\int\Xi_{cc'}\,d\nu$. Although trained heads interact statistically, no head-independence assumption is needed for this finite sum identity. The cross-head covariance entries in $C_h$ are accounted for by the dual Gram spectrum.

To see this spectrum correspondence, enumerate $M$ contexts and form the $M\times2H$ centered feature matrix $F$, with $F_{c,:}=(h_H(c)-\bar h_H)^\top$. Let $P=\operatorname{diag}(p_c)$ and $B=P^{1/2}F/\sqrt{2H}$. Then

$$
B^\top B=\frac{F^\top PF}{2H}=\frac{C_H}{2H},
\qquad
BB^\top=P^{1/2}K_HP^{1/2}=:S_H.
$$

If $B^\top Bv=\lambda v$ with $\lambda>0$, then $Bv\neq0$ and

$$
BB^\top(Bv)=B(B^\top Bv)=\lambda Bv.
$$

The reverse implication follows with $B^\top$. Thus the nonzero eigenvalues coincide. In the continuum, use $S_\nu=P^{1/2}K_\nu P^{1/2}$. Its eigenvalues predict the covariance eigenvalues divided by $2H$; their normalized proportions and PR require no amplitude rescaling.

Writing $A=\operatorname{tr}S$ and $B_2=\operatorname{tr}(S^2)$,

$$
\operatorname{PR}=A^2/B_2,\qquad
\dot A=\operatorname{tr}\dot S,
\qquad
\dot B_2=\operatorname{tr}(\dot SS+S\dot S)=2\operatorname{tr}(S\dot S),
$$

and the quotient rule gives

$$
\boxed{\dot{\operatorname{PR}}
=\frac{2A\operatorname{tr}\dot S}{B_2}
-\frac{2A^2\operatorname{tr}(S\dot S)}{B_2^2}.}
$$

For a simple eigenvalue, differentiating $Su_i=\lambda_i u_i$ and multiplying by $u_i^\top$ cancels the terms $\lambda_i u_i^\top\dot u_i$ on both sides, giving $\dot\lambda_i=u_i^\top\dot Su_i$. Repeated eigenvalues require an eigenspace perturbation calculation, not individually labeled eigenvectors.

The transport equation gives an exact kernel derivative:

$$
\begin{aligned}
\dot K(c,c')
&=\int\nabla\Xi_{cc'}\cdot v\,d\nu\\
&=-\sum_up_u\sum_je_j(u)
\int\nabla\Xi_{cc'}(\theta)\cdot\nabla\phi_j(u;\theta)\,d\nu.
\end{aligned}
$$

The required centered-feature derivative is

$$
\partial_v\widetilde\psi(c;\theta)
=\partial_v\psi(c;\theta)-\sum_up_u\partial_v\psi(u;\theta),
$$

and the product rule gives

$$
\partial_v\Xi_{cc'}=
\frac12\sum_j\left[
\partial_v\widetilde\psi_j(c)\widetilde\psi_j(c')
+\widetilde\psi_j(c)\partial_v\widetilde\psi_j(c')\right].
$$

It vanishes for $v=a_1,a_2$. Thus every quantity needed for $\dot K$ is specified above. A kernel-only equation generally introduces further moments and does not close autonomously; retaining $\nu$ supplies the closure.

### 5.8 Make the continuum solver two-dimensional

Use the explicit initial law

$$
a_1(0)=a_2(0)=0,\quad g(0)=1,\quad
\xi=(\beta(0),\kappa(0))\sim\operatorname{Unif}([-1,1]\times[-1/2,1/2]).
$$

Only two parameters are random initially, although all five parameters subsequently train. Define the characteristic map $\theta_\tau(\xi)$ through

$$
\frac{d\theta_\tau(\xi)}{d\tau}
=v(\theta_\tau(\xi);\nu_\tau),\qquad
\nu_\tau=(\theta_\tau)_\#\rho_0.
$$

The pushforward means $\int F\,d\nu_\tau=\int F(\theta_\tau(\xi))\rho_0(d\xi)$. It solves the weak continuity equation by the ordinary chain rule. The distribution can remain singular in five-dimensional parameter space; no density grid in that space is required.

Choose deterministic quadrature nodes $\xi_n$ and positive weights $w_n$ summing to one. The numerical continuum equations are

$$
\begin{aligned}
f_j^{(Q)}(c)&=\sum_{n=1}^Qw_n a_{nj}\psi_j(c;\theta_n),\\
e_j^{(Q)}(c)&=\tanh f_j^{(Q)}(c)-\mu_j(c),\\
\dot\theta_n&=-\sum_cp_c\sum_je_j^{(Q)}(c)\nabla\phi_j(c;\theta_n),\\
K^{(Q)}(c,c')&=\sum_nw_n\Xi_{cc'}(\theta_n).
\end{aligned}
$$

These are a deterministic cubature approximation to the nonlinear continuum problem. Their structure resembles a weighted head ensemble because characteristics discretize a measure; this equivalence is explicit, not hidden. Independence from finite-network tests comes from fixing the initial law, using deterministic quadrature and refinement, and never fitting the prediction to tested random networks.

Start at $T=2$ or $3$, giving $M=16$ or $64$ contexts, and compare quadrature orders such as $12^2,24^2,48^2$. Evolve $5Q$ scalar coordinates and accumulate logits in $O(QMT)$ work per right-hand-side evaluation; form the full kernel in $O(QM^2)$ work only at checkpoints. These sizes are proposed feasible starting points, not measured runtimes. Reduce the ODE step or tolerance independently of quadrature refinement.

If the cubature error over initial labels is uniformly bounded by $\varepsilon_Q$ for the relevant evolved integrands throughout the time interval, stability of the characteristic equations propagates it with a finite-horizon factor controlled by Gronwall. An error bound only on initial kernel/logit moments is insufficient. Integrands depend on the evolving characteristics, so a quadrature rule's nominal polynomial exactness does not establish this error automatically. Compare successive orders and, for a rigorous numerical bound, bound the derivatives of the evolved integrands or use validated quadrature. A widest random network is not used as the limiting prediction.

### 5.9 Derive the first nonzero feature-learning term

Zero initial readouts give $f_j(c;0)=0$ and $e_j(c;0)=-\mu_j(c)$. They do not freeze all training. Define, for each initial label $\xi$,

$$
A_j(\xi)=\sum_cp_c\mu_j(c)\psi_j(c;\theta_0(\xi)).
$$

Then $\dot a_j(0,\xi)=A_j(\xi)$. For $u\in\{\beta,\kappa,g\}$,

$$
\dot u(0,\xi)=-\sum_cp_c\sum_je_j(c;0)a_j(0,\xi)\psi_{j,u}(c;\theta_0)=0.
$$

Differentiate the product in its evolution:

$$
\begin{aligned}
\ddot u
=-\sum_cp_c\sum_j\big[
\dot e_j a_j\psi_{j,u}
+e_j\dot a_j\psi_{j,u}
+e_j a_j\tfrac{d}{d\tau}\psi_{j,u}\big].
\end{aligned}
$$

At zero time, the first and third terms vanish because $a_j=0$, leaving

$$
\boxed{
B_u(\xi):=\ddot u(0,\xi)
=\sum_j A_j(\xi)\sum_cp_c\mu_j(c)\psi_{j,u}(c;\theta_0(\xi)).
}
$$

Taylor expansion therefore gives

$$
a_j(\tau,\xi)=\tau A_j(\xi)+O(\tau^2),\qquad
u(\tau,\xi)=u(0,\xi)+\frac{\tau^2}{2}B_u(\xi)+O(\tau^3).
$$

Because $\Xi$ is independent of the readout amplitudes, $\dot K(0)=0$ and

$$
\ddot K(c,c';0)
=\int\sum_{u\in\{\beta,\kappa,g\}}
\partial_u\Xi_{cc'}(\theta_0(\xi))B_u(\xi)\,\rho_0(d\xi).
$$

Let $S_2=P^{1/2}\ddot K(0)P^{1/2}$. Since $\dot S(0)=0$, a simple eigenvalue and PR obey

$$
\begin{aligned}
\lambda_i(\tau)&=\lambda_i(0)+\frac{\tau^2}{2}u_i^\top S_2u_i+O(\tau^3),\\
\operatorname{PR}(\tau)&=\operatorname{PR}(0)
+\frac{\tau^2}{2}\left[
\frac{2A_0\operatorname{tr}S_2}{B_{2,0}}
-\frac{2A_0^2\operatorname{tr}(S_0S_2)}{B_{2,0}^2}\right]+O(\tau^3).
\end{aligned}
$$

There is a simple interpretation of the feature acceleration. Since $A_j(u)=\sum_cp_c\mu_j(c)\psi_j(c;u)$, differentiation gives $\partial_u A_j=\sum_cp_c\mu_j(c)\psi_{j,u}(c;u)$. Consequently

$$
B_u=\sum_jA_j\partial_uA_j
=\frac12\partial_u\sum_jA_j^2.
$$

The initial hidden-feature acceleration points uphill in squared target–feature correlation. This is a concrete generator-dependent nonlinear training prediction: all coefficients are finite context sums and initial-law integrals. No trained network is required to evaluate them. Symmetry can make a coefficient vanish; this is a prediction to check, not a reason to force a nonzero result.

There is also a controlled weak-signal scaling. Scale both emission reliabilities as $q_j=\epsilon\bar q_j$ with fixed $T$ and persistence away from one. Package 1 gives $m_j=O(\epsilon)$ and therefore $\mu_j=\rho_jq_jm_j=O(\epsilon^2)$. Bounded initial features imply

$$
A_j=O(\epsilon^2),\qquad
B_u=O(\epsilon^4),\qquad
\ddot K(0)=O(\epsilon^4).
$$

For nondegenerate $S_0$, the training-induced PR acceleration is also at most $O(\epsilon^4)$. This is an order bound; the leading coefficient may cancel. Compare training-induced changes against each generator's own $K_0$, since the initial data-dependent kernel can itself vary with $\epsilon$. The statement is local in training time, not a uniform estimate at times growing as a power of $1/\epsilon$.

### 5.10 Differentiate dependence on generator structure

Let $\gamma$ be a persistence, reliability, or coupling parameter. Both $p_c(\gamma)$ and $\mu_j(c;\gamma)$ vary. For an edge-emitting HMM path product $A_c=T^{(x_1)}\cdots T^{(x_T)}$,

$$
\partial_\gamma A_c
=\sum_{s=1}^T T^{(x_1)}\cdots T^{(x_{s-1})}
(\partial_\gamma T^{(x_s)})T^{(x_{s+1})}\cdots T^{(x_T)},
$$

$$
\partial_\gamma p_c
=(\partial_\gamma\pi)A_c\mathbf1+\pi(\partial_\gamma A_c)\mathbf1.
$$

If $\pi$ is stationary and depends on $\gamma$, differentiate $\pi P=\pi$ and $\pi\mathbf1=1$:

$$
(\partial_\gamma\pi)(I-P)=\pi\partial_\gamma P,
\qquad (\partial_\gamma\pi)\mathbf1=0.
$$

For an ergodic finite chain these equations determine the derivative on the normalized subspace. Posterior derivatives follow the quotient rule from $b_c=\pi A_c/p_c$. Conditional-mean derivatives can then be computed without finite differences.

For the cubature solver, collect all parameters in $\Theta=(\theta_1,\ldots,\theta_Q)$ and write $\dot\Theta=F(\Theta;\gamma)$. If the same initialization law is used across generators, $Z:=\partial_\gamma\Theta$ satisfies

$$
\dot Z=D_\Theta F(\Theta;\gamma)Z+\partial_\gamma F(\Theta;\gamma),\qquad Z(0)=0.
$$

This is obtained by differentiating each side of the ODE. To make the right-hand side explicit, set $J_{nj}(c)=\nabla\phi_j(c;\theta_n)$ and $H_{nj}(c)=\nabla^2\phi_j(c;\theta_n)$. Then

$$
\begin{aligned}
\partial_\gamma f_j(c)&=\sum_nw_nJ_{nj}(c)^\top Z_n,\\
\partial_\gamma e_j(c)&=\operatorname{sech}^2f_j(c)\sum_nw_nJ_{nj}(c)^\top Z_n
-\partial_\gamma\mu_j(c),\\
\dot Z_n&=-\sum_{c,j}\left[
(\partial_\gamma p_c)e_jJ_{nj}
+p_c(\partial_\gamma e_j)J_{nj}
+p_ce_jH_{nj}Z_n\right].
\end{aligned}
$$

All $J$ entries were derived above. To evaluate the Hessian without an unspecified derivative, first differentiate the softmax once more:

$$
\frac{\partial^2\alpha_s}{\partial\ell_v\partial\ell_w}
=\alpha_s\left[(\delta_{sw}-\alpha_w)(\delta_{sv}-\alpha_v)
-\alpha_v(\delta_{vw}-\alpha_w)\right],
$$

This follows by applying the product rule to $\alpha_s(\delta_{sv}-\alpha_v)$. For direct parameter derivatives write $e_s^\beta=t_s$, $e_s^\kappa=d_s$, $\bar e^u=\sum_s\alpha_se_s^u$. Differentiating $\partial_u\alpha_s=\alpha_s(e_s^u-\bar e^u)$ gives

$$
\partial_v\bar e^u=\sum_s\alpha_s(e_s^v-\bar e^v)e_s^u
=\operatorname{Cov}_\alpha(e^u,e^v),
$$

$$
z_{j,uv}=\sum_sx_{sj}\alpha_s\left[
(e_s^u-\bar e^u)(e_s^v-\bar e^v)
-\operatorname{Cov}_\alpha(e^u,e^v)\right],\qquad u,v\in\{\beta,\kappa\}.
$$

Let $J_j=\operatorname{sech}^2(gr_j)$ and $T_j=\tanh(gr_j)$. Since $dJ_j/d(gr_j)=-2J_jT_j$, differentiating the first derivatives of $\psi$ yields all its nonzero second derivatives:

$$
\begin{aligned}
\psi_{j,uv}&=(1+gJ_j)z_{j,uv}-2g^2J_jT_jz_{j,u}z_{j,v},\\
\psi_{j,ug}&=(J_j-2gr_jJ_jT_j)z_{j,u},\\
\psi_{j,gg}&=-2r_j^2J_jT_j,\qquad u,v\in\{\beta,\kappa\}.
\end{aligned}
$$

For $\phi_j=a_j\psi_j$, the readout–readout Hessian block is zero, the readout–feature block is $\partial_{a_k}\partial_u\phi_j=\delta_{jk}\psi_{j,u}$, and the feature–feature block is $\partial_u\partial_v\phi_j=a_j\psi_{j,uv}$. These specify every entry of $H_{nj}$ in the sensitivity ODE.

When evaluating $\partial_\gamma K$, differentiate both the characteristics and context-centering weights. For example,

$$
\partial_\gamma\bar\psi(\theta_n)
=\sum_c\left[(\partial_\gamma p_c)\psi(c;\theta_n)
+p_cD_\theta\psi(c;\theta_n)Z_n\right].
$$

Finally differentiate $S=P^{1/2}KP^{1/2}$, using $(\partial_\gamma P^{1/2})_{cc}=(\partial_\gamma p_c)/(2\sqrt{p_c})$ for positive-probability contexts. This yields predicted spectral/PR response to generator structure rather than a separate gain fit at each parameter value.

### 5.11 Derive a finite-horizon finite-head error estimate

Fix finite context support and a finite horizon $0\leq\tau\leq\tau_*$. Assume compactly supported initialization and bounded derivatives of the head observables and vector field on the region reached by the flow. The displayed smooth model has $|z_j|\leq1$, $|r_j|\leq2$, $|\psi_j|\leq3$, $|e_j|\leq2$, hence $|\dot a_j|\leq6$. On a finite horizon the $a_j$ are bounded; $|\dot g|\leq4\sum_j|a_j|$ bounds $g$, and the remaining derivatives then bound $\beta,\kappa$. Thus a finite reachable compact region exists. Lipschitz constants can grow with horizon; no uniform long-time claim follows.

On the finite reachable region the velocity is Lipschitz in its head coordinate and in the finite vector of output integrals. Picard iteration on characteristic paths is therefore a contraction on a sufficiently short time interval, using the supremum path norm and the bound $\|\int\phi(\theta)-\phi(\theta')\,d\rho_0\|\leq L_\phi\sup_\xi\|\theta(\xi)-\theta'(\xi)\|$. Repeating short intervals and using the finite-horizon bounds constructs a unique deterministic characteristic law for this initialization. This establishes the continuum reference needed below; it is not an assertion of convergence as training time tends to infinity.

Couple finite heads $\theta_i$ with independent nonlinear characteristics $\bar\theta_i$ having the same iid initial labels. Each $\bar\theta_i$ is driven by the deterministic continuum law $\nu$, so they remain independent across $i$. Let

$$
D_H=\frac1H\sum_i\mathbb E\|\theta_i-\bar\theta_i\|^2,
\quad
\bar f_H(c)=\frac1H\sum_i\phi(c;\bar\theta_i).
$$

Split $f_H-f_\nu=(f_H-\bar f_H)+(\bar f_H-f_\nu)$. If $L_\phi$ is a feature Lipschitz constant, Jensen's inequality gives

$$
\mathbb E\|f_H(c)-\bar f_H(c)\|^2
\leq L_\phi^2D_H.
$$

For the independent average,

$$
\begin{aligned}
\mathbb E\|\bar f_H(c)-f_\nu(c)\|^2
&=\frac1{H^2}\sum_{i,k}
\mathbb E[(\phi_i-\bar\phi)^\top(\phi_k-\bar\phi)]\\
&=\frac1{H^2}\sum_i\mathbb E\|\phi_i-\bar\phi\|^2
\leq\frac{C}{H}.
\end{aligned}
$$

Off-diagonal terms vanish by independence and zero means. Write $\delta_i=\theta_i-\bar\theta_i$. The vector-field difference obeys

$$
\|\dot\delta_i\|
\leq L_1\|\delta_i\|+L_2\|f_H-f_\nu\|_{p},
$$

where $\|f\|_p^2=\sum_cp_c\|f(c)\|^2$. Differentiate the mean squared discrepancy and apply $2ab\leq a^2+b^2$:

$$
\begin{aligned}
\dot D_H
&=\frac2H\sum_i\mathbb E[\delta_i^\top\dot\delta_i]\\
&\leq(2L_1+L_2)D_H+L_2\mathbb E\|f_H-f_\nu\|_p^2\\
&\leq CD_H+C/H.
\end{aligned}
$$

Since $D_H(0)=0$, multiply by $e^{-C\tau}$ and integrate:

$$
\frac{d}{d\tau}(e^{-C\tau}D_H)\leq \frac{C}{H}e^{-C\tau},
\qquad
D_H(\tau)\leq\frac{e^{C\tau}-1}{H}.
$$

Apply the same split to $K_H(c,c')=H^{-1}\sum_i\Xi_{cc'}(\theta_i)$. Boundedness, Lipschitzness and independence of the barred characteristics give, for fixed finite $M$,

$$
\mathbb E\|S_H-S_\nu\|_F^2\leq C_{\tau_*,M}/H.
$$

This is a coupled-observable bound, not a generic dimension-free Wasserstein rate for arbitrary five-dimensional empirical measures. It predicts RMS spectral-matrix error $O(H^{-1/2})$ at fixed horizon and support.

The eigenvalue perturbation inequality $|\lambda_i(A+E)-\lambda_i(A)|\leq\|E\|_{\mathrm{op}}\leq\|E\|_F$ transfers the bound to ordered spectra. For PR, differentiate $R(S)=(\operatorname{tr}S)^2/\operatorname{tr}(S^2)$ in direction $E$:

$$
DR_S[E]=\frac{2\operatorname{tr}S\operatorname{tr}E}{\operatorname{tr}(S^2)}
-\frac{2(\operatorname{tr}S)^2\operatorname{tr}(SE)}{[\operatorname{tr}(S^2)]^2}.
$$

Use $|\operatorname{tr}E|\leq\sqrt M\|E\|_F$ and $|\operatorname{tr}(SE)|\leq\|S\|_F\|E\|_F$. Where the second trace is bounded below and matrix norms are bounded above, $|DR_S[E]|\leq C\|E\|_F$. Integrating this derivative along the matrix segment yields local Lipschitz continuity. On the high-probability nondegenerate event where this lower bound holds, PR error has the same RMS order; handle the exceptional event with the matrix error bound and $1\leq\operatorname{PR}\leq M-1$. Constants worsen near zero covariance. No $O(H^{-1})$ bias or uniform long-training-time estimate has been established by this argument.

### 5.12 Formal fluctuation theory gives a sharper validation target

The previous bound is a finite-horizon estimate. A central-limit description needs an additional tightness and differentiability argument; we derive its candidate linear equation rather than assert that those probability-theorem steps have been completed for this model.

Let $\zeta_H=\sqrt H(\nu_H-\nu)$. Since $f$ is linear in $\nu$,

$$
f_j(c;\nu_H)=f_j(c;\nu)+H^{-1/2}\langle\phi_j(c),\zeta_H\rangle.
$$

Taylor-expand $\tanh$ around $f_j(c;\nu)$:

$$
e_j(c;\nu_H)=e_j(c;\nu)
+H^{-1/2}\operatorname{sech}^2f_j(c;\nu)
\langle\phi_j(c),\zeta_H\rangle+O(H^{-1}).
$$

The linearized velocity perturbation is

$$
\delta v[\zeta](\theta)
=-\sum_cp_c\sum_j\operatorname{sech}^2f_j(c)
\langle\phi_j(c),\zeta\rangle\nabla\phi_j(c;\theta).
$$

Insert $\nu_H=\nu+H^{-1/2}\zeta_H$ and this velocity expansion into the weak continuity equation, subtract the continuum equation, and retain order $H^{-1/2}$:

$$
\frac{d}{d\tau}\langle F,\zeta\rangle
=\langle\nabla F\cdot v,\zeta\rangle
+\langle\nabla F\cdot\delta v[\zeta],\nu\rangle.
$$

This equation is linear in $\zeta$. For iid initialization, its candidate initial Gaussian field has covariance

$$
\mathbb E[\langle F,\zeta_0\rangle\langle G,\zeta_0\rangle]
=\langle FG,\nu_0\rangle-\langle F,\nu_0\rangle\langle G,\nu_0\rangle,
$$

obtained by the ordinary central limit theorem for finitely many test functions. Full-batch gradient flow adds no fresh training noise; all randomness here originates at initialization. A stochastic-gradient limit would require its own noise term and time/batch scaling, as distinct from assuming one by analogy with other neural-network CLTs [R9].

Because $K$ is linear in $\nu$, the finite-$H$ identity $\sqrt H(K_H-K_\nu)=\langle\Xi,\zeta_H\rangle$ is exact. The approximation enters when replacing $\zeta_H$ by its proposed limiting Gaussian field $\zeta$. Propagate that field's covariance through $DR_S$ to predict leading PR variance and through $u_i^\top\delta Su_i$ for simple spectral modes. This is a testable fluctuation approximation, with a full functional-CLT proof still required before calling it a theorem.

### 5.13 Separate network width, evaluation sampling, and solver error

At initialization, independent heads give exactly

$$
\operatorname{Cov}[K_H(c,c'),K_H(u,u')]
=\frac1H\operatorname{Cov}_{\nu_0}[\Xi_{cc'},\Xi_{uu'}].
$$

After training the heads interact, so using the same iid formula is generally wrong; use the coupled bound or fluctuation dynamics. Evaluate exact context sums initially so evaluation noise does not imitate $H^{-1/2}$ convergence.

For larger contexts, let $X=h(c)-\mathbb E h$ and $X'$ be independent. Then

$$
\mathbb E\|X\|^2=\operatorname{tr}C,
\qquad
\mathbb E[(X^\top X')^2]
=\sum_{i,j}\mathbb E[X_iX_j]\mathbb E[X_i'X_j']=\operatorname{tr}(C^2).
$$

If the population mean is unknown, four independent uncentered activation samples supply mean-free identities. Let $D_{12}=h_1-h_2$ and $D_{34}=h_3-h_4$. Since $\mathbb E[D_{12}D_{12}^\top]=2C$,

$$
\frac12\mathbb E\|D_{12}\|^2=\operatorname{tr}C,
\qquad
\frac14\mathbb E[(D_{12}^\top D_{34})^2]=\operatorname{tr}(C^2).
$$

Disjoint independent pairs/quadruples therefore give unbiased moment estimators, although their PR ratio is not unbiased. Full U-statistics can improve efficiency; overlapping trajectory windows require dependence-aware analysis. A finite sample covariance has rank at most $N_{\mathrm{eval}}-1$.

For a numerical continuum prediction $S_{Q,\Delta\tau}$, decompose the observed error as

$$
S_H-S_{Q,\Delta\tau}
=(S_H-S_\nu)+(S_\nu-S_Q)+(S_Q-S_{Q,\Delta\tau}).
$$

These terms are finite-head error, initialization-law cubature error, and ODE error. Add evaluation-sampling error separately if exact context sums are unavailable. Refine $Q$ and time tolerance until their changes are smaller than the network-size effect being tested. Fixed finite context support implies rank at most $M-1$ regardless of width; it does not test an extensive $\operatorname{PR}\propto H$ limit at fixed $M$.

### 5.14 Add a genuine interaction for dependent generators

For correlated targets, add head parameter $b$, feature $\chi(c;\theta)=\psi_1(c;\theta)\psi_2(c;\theta)$, and logit

$$
f_{12}(c)=\int b\chi(c;\theta)\,d\nu,
\qquad
L_y=y_1f_1+y_2f_2+y_1y_2f_{12}.
$$

Grouping terms with equal versus unequal signs gives

$$
Z=e^{f_{12}}(e^{f_1+f_2}+e^{-f_1-f_2})
+e^{-f_{12}}(e^{f_1-f_2}+e^{-f_1+f_2})
=2e^{f_{12}}\cosh(f_1+f_2)+2e^{-f_{12}}\cosh(f_1-f_2).
$$

The exact conditional loss is $\ell=\log Z-\mu_1f_1-\mu_2f_2-\mu_{12}f_{12}$, with $\mu_{12}=\mathbb E[Y_1Y_2\mid c]$ from full joint HMM filtering. Its gradients are

$$
\begin{aligned}
\partial_{f_1}\log Z&=\frac{2e^{f_{12}}\sinh(f_1+f_2)+2e^{-f_{12}}\sinh(f_1-f_2)}Z,\\
\partial_{f_2}\log Z&=\frac{2e^{f_{12}}\sinh(f_1+f_2)-2e^{-f_{12}}\sinh(f_1-f_2)}Z,\\
\partial_{f_{12}}\log Z&=\frac{2e^{f_{12}}\cosh(f_1+f_2)-2e^{-f_{12}}\cosh(f_1-f_2)}Z.
\end{aligned}
$$

Subtract the corresponding target moments to obtain the three errors. The head functions are now $(a_1\psi_1,a_2\psi_2,b\psi_1\psi_2)$, and

$$
\partial_u\chi=\psi_{1,u}\psi_2+\psi_1\psi_{2,u},\qquad
\dot b=-\sum_cp_ce_{12}(c)\chi(c),
$$

$$
\dot u=-\sum_cp_c\left[\sum_{j=1}^2e_j a_j\psi_{j,u}
+e_{12}b(\psi_{1,u}\psi_2+\psi_1\psi_{2,u})\right].
$$

The same distributional closure and two-dimensional initial-label solver apply if $b(0)=0$. The conditional Hessian of $\log Z$ is the covariance of $(Y_1,Y_2,Y_1Y_2)$ under the model; this follows by differentiating the normalized exponential weights once more. Replace the diagonal $\operatorname{sech}^2f$ factors in sensitivity/fluctuation calculations by that full covariance matrix.

For every strictly positive four-token target distribution, the coefficients of $\log P^*(y\mid c)$ in the Walsh basis give an exact Ising-form logit representation. However, the specific features $b\psi_1\psi_2$ need not approximate every context-dependent interaction function; expressibility of the categorical family does not guarantee realizability by this restricted neural feature class. Measure approximation error against the Bayes predictor before attributing a discrepancy to width theory. The measured $2H$-dimensional residual still has kernel $\int\Xi\,d\nu$; adding a nonlinear feature in the readout does not silently add it to the residual representation.

### 5.15 What the calculation establishes and what the ambitious paper still needs

We now have a closed nonlinear training law, an independently specified numerical continuum solver, an explicit map to representation eigenvalues and PR, a generator-sensitivity equation, an analytic feature-learning onset, and a finite-horizon finite-head error estimate under stated smoothness assumptions. These are stronger than a post-training covariance reconstruction. They do not demonstrate that the model learns Bayes beliefs, that dimensions shrink, or that the predictions match finite networks; those are experiments and numerical calculations still to perform.

The most discriminating first tests are the $\tau^2$ feature-kernel onset, its weak-signal scaling, the unequal-factor spectral response, and the $H^{-1/2}$ RMS approach to the computed continuum. Compare frozen-attention, frozen-feature and unregularized readout-only controls using exactly declared time conventions. Report the full spectrum and kernel shape, not only PR. A finite-network trajectory that changes only overall activation scale does not verify a predicted change in normalized geometry.

To pursue dense cross-head mixing, replace the block-diagonal output/MLP with dense learned matrices and track how the exact Package 4 hidden-output Jacobians couple channels. The present head law then ceases to be autonomous: a feature depends on the joint dense parameter state, not just its own five coordinates and aggregate output. Pehlevan-group transformer DMFT [R4] supplies a different route using forward/backward correlations and response functions; its uncomputed closure cannot be replaced by this five-coordinate transport equation. Deriving and solving that dense closure remains an additional research step.

The ambitious paper target is therefore precise: validate this tractable nonlinear calculation, discover a nontrivial generator-dependent spectral law or failure mechanism, and establish which conclusions survive controlled architectural extensions. Novelty must be assessed against both existing belief-spectrum studies and standard mean-field methodology. No theorem of publication follows from the use of a continuum limit.

## A numerical check of the derived nonlinear onset

The following calculation evaluates the zero-time formulas, not a trained-network spectrum. Use $T=3$, independent binary factors with $(\rho_1,q_1)=(0.7,0.4)$ and $(\rho_2,q_2)=(0.8,0.8)$, exact probabilities for all $4^3=64$ contexts, and the initialization law of Section 5.8. Tensor-product Gauss–Legendre quadrature gives:

| Nodes per initial coordinate | Total characteristic labels | $\operatorname{PR}(0)$ | $\ddot{\operatorname{PR}}(0)$ |
| --- | --- | --- | --- |
| 5 | 25 | 2.044327685523 | -0.008109837825 |
| 9 | 81 | 2.044328397154 | -0.008109883412 |
| 17 | 289 | 2.044328397168 | -0.008109884767 |
| 33 | 1,089 | 2.044328397168 | -0.008109884767 |

The table is deterministic initial-law integration of the derived expressions. Successive-order agreement is a numerical convergence check, not a rigorous cubature error certificate. It predicts, for this declared model and generator,

$$
\operatorname{PR}(\tau)
=2.044328397168-0.004054942384\,\tau^2+O(\tau^3).
$$

The coefficient $-0.004054942384$ is half the predicted acceleration because the Taylor term is $\tau^2\ddot{\operatorname{PR}}(0)/2$. A negative coefficient here does not imply that every generator, initialization or later time exhibits decreasing PR.

The five finite-head parameter gradients were independently compared against central finite differences of the exact population loss on a small random head ensemble; the largest absolute discrepancy was approximately $1.8\times10^{-10}$. The analytic kernel acceleration was also checked by short Runge–Kutta characteristic integration using symmetric time increments around zero. Reducing the increment from $10^{-3}$ to $5\times10^{-4}$ reduced relative acceleration error from about $3.0\times10^{-5}$ to $7.5\times10^{-6}$, consistent with second-order differencing error. Negative time was used only as a local mathematical derivative check. These are algebra and solver checks, not the proposed finite-width validation study.

## What to implement next for the ambitious paper target

The derivation document supplies calculations rather than another list of unspecified machinery. The next implementation should keep their logical dependencies intact:

1. Evaluate the exact generator distribution and target moments. Reproduce the binary posterior recurrence and factored/joint covariance eigenvalues by two independent implementations.
2. Reproduce the balanced linear trajectory with its prescribed initialization and time convention. This checks the covariance and optimizer measurements separately from attention.
3. Integrate the restricted positional-attention ODE and compare predicted attention-statistic spectra. Do not replace the learned filter by its measured final value in the independent prediction.
4. Implement the exact finite nonlinear model and verify its backpropagation formulas, including normalization only when that variant is used. Measure the declared final-position residual.
5. Solve the nonlinear characteristic equations on a converged deterministic initialization quadrature. Predict $S_\nu(\tau)$, its spectrum and PR before running independent random-head networks. Test $H$ scaling with numerical and sampling errors below the expected finite-head effect. The zero-time coefficient table provides a first fixed prediction.

The first quantitative question is whether the nonlinear calculation predicts an entire spectral trajectory and its change under generator anisotropy or dependence. The second is whether the same mechanism survives denser head mixing and richer embeddings. These questions preserve the ambitious objective while separating an explicitly derived tractable model from an unresolved dense-transformer theory. A new paper would need a meaningful result from these tests and a careful novelty comparison; the derivations alone do not establish that result.

## Primary references and attribution boundaries

**[R1]** Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel, Paul M. Riechers. *Transformers Represent Belief State Geometry in their Residual Stream* (NeurIPS 2024). [Paper](https://arxiv.org/abs/2405.15943). Precedent for known generators and linearly represented predictive states.

**[R2]** Mateusz Piotrowski, Paul M. Riechers, Daniel Filan, Adam Shai. *Constrained Belief Updates Explain Geometric Structures in Transformer Representations* (ICML 2025). [Paper](https://arxiv.org/abs/2502.01954). Precedent for architecture-dependent intermediate geometry and concentrated activation variance.

**[R3]** Adam Shai et al. *Transformers learn factored representations* (ICML 2026). [Paper](https://arxiv.org/abs/2602.02385), [code](https://github.com/Astera-org/factored-reps). Closest factorization and activation-spectrum comparison; merely replacing its dimensionality measure with PR would not originate the phenomenon.

**[R4]** Blake Bordelon, Hamza Tahir Chaudhry, Cengiz Pehlevan. *Infinite Limits of Multi-head Transformer Dynamics* (NeurIPS 2024). [Paper](https://arxiv.org/abs/2405.15712). Establishes feature-learning transformer limits with distinct head-dimension and head-count scalings. The full dense theory uses more order parameters than the restricted transport model derived here.

**[R5]** Jiri Hron, Yasaman Bahri, Jascha Sohl-Dickstein, Roman Novak. *Infinite attention: NNGP and NTK for deep attention networks* (ICML 2020). [Paper](https://proceedings.mlr.press/v119/hron20a.html). Initialization and lazy attention-kernel foundation; not a nonlinear feature-learning solution for this generator.

**[R6]** Andrew M. Saxe, James L. McClelland, Surya Ganguli. *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks* (ICLR 2014). [Paper](https://arxiv.org/abs/1312.6120). Background for the controlled mode-aligned linear calculation.

**[R7]** Song Mei, Andrea Montanari, Phan-Minh Nguyen. *A Mean Field View of the Landscape of Two-Layers Neural Networks* (2018). [Paper](https://arxiv.org/abs/1804.06561). Distributional learning dynamics as methodological background. Its theorems are not quoted as a ready-made theorem for the attention architecture here.

**[R8]** Lenaic Chizat, Francis Bach. *On the Global Convergence of Gradient Descent for Over-parameterized Models using Optimal Transport* (NeurIPS 2018). [Paper](https://arxiv.org/abs/1805.09545). Measure-based optimization and transport background. No global-convergence claim from that work is automatically transferred to the present nonlinear heads.

**[R9]** Justin Sirignano, Konstantinos Spiliopoulos. *Mean Field Analysis of Neural Networks: A Central Limit Theorem* (preprint 2018, later journal publication). [Paper](https://arxiv.org/abs/1808.09372). Background for fluctuation analysis. Section 5.12 derives a candidate full-batch fluctuation equation and does not claim to reproduce this paper's stochastic-training theorem.

**[R10]** Blake Bordelon, Cengiz Pehlevan. *Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks* (NeurIPS 2022). [Paper](https://arxiv.org/abs/2205.09653). Background for learned feature kernels, backward correlations and response functions in a denser model.

**[R11]** Dhruva Karkada, James B. Simon, Yasaman Bahri, Michael R. DeWeese. *Closed-Form Training Dynamics Reveal Learned Features and Linear Structure in Word2Vec-like Models* (2025). [Paper](https://arxiv.org/abs/2502.09863). A nearby example of deriving learned representation structure from a simplified language-learning objective.

**[R12]** Greg Yang, Edward J. Hu. *Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks* (ICML 2021). [Paper](https://proceedings.mlr.press/v139/yang21c.html). Background for separating feature-learning parameterizations from lazy limits. The explicit $H$ scaling in Package 5 is derived from its own mean readout, rather than borrowed without its forward convention.
