---
layout: single
title: "From Spectra to Joint Schedules in LLM Pre-training: 3+3(+2) Scaling-Law Regimes"
date: 2026-09-28
permalink: /posts/power-laws-have-a-clock/
excerpt: "We identify when spectra create power-law learning curves, derive how joint schedules transform them, and predict LLM loss across schedules."
tags:
  - scaling laws
  - stochastic optimization
  - language models
author_profile: true
read_time: true
toc: true
toc_sticky: true
classes: wide
header:
  teaser: /images/power-laws-have-a-clock/main-fig2-loss.png
---

*Why training loss follows power laws, how learning rate and batch size change those laws, and how the resulting formulas predict LLM learning curves.*

Power-law fits summarize how training loss falls, but an exponent alone does not explain **why training produces that curve**.

Two processes shape it: training removes initial error, while each sampled batch adds fluctuations that can affect later updates. We call the remaining initial-error contribution **forcing**, and the lasting response to fluctuations **memory**.

The story has three parts:

1. **Where power laws come from.** In a tractable random-feature model, we identify necessary and sufficient spectral conditions for power-law forcing and memory.
2. **How schedules change them.** Changing learning rate and batch size over time can preserve the loss exponent, change it, or stop positive-power decay.
3. **How to use the result.** The formulas guide data allocation. In LLM experiments, fitting one schedule lets us predict another without changing the fitted parameters.

Start with one update: what happens to its batch-sampling error after the update is over?

## Why SGD has memory

Let \\(f_\theta\\) be a model, such as a transformer, with parameters \\(\theta\\) and per-example loss \\(\ell(f_\theta;z)\\). Its training objective is

$$
\mathcal L(\theta)
=\mathbb E_z[\ell(f_\theta;z)].
$$

At update \\(t\\), let \\(\eta_t\\) be the learning rate, \\(B_t\\) the batch size, and \\(\xi_t\\) the difference between the batch gradient and the full gradient. With fresh i.i.d. examples, SGD takes the form

$$
\theta_{t+1}
=\theta_t-\eta_t\nabla\mathcal L(\theta_t)-\eta_t\xi_t,
\qquad
\mathbb E_t[\xi_t]=0,
\qquad
\operatorname{Cov}_t(\xi_t)=\frac{\Sigma(\theta_t)}{B_t}.
$$

Here \\(\mathbb E_t\\) and \\(\operatorname{Cov}_t\\) are conditional on the training history, and \\(\Sigma(\theta_t)\\) is the single-example gradient covariance. Batch averaging reduces the variance by \\(B_t\\), while the learning rate scales the parameter perturbation by \\(\eta_t\\). The injected variance therefore has scale \\(\eta_t^2/B_t\\); at fixed parameters, the typical size of the parameter perturbation is proportional to \\(\eta_t/\sqrt{B_t}\\).

Compare the full-gradient trajectory \\(\bar\theta_t\\) with a run from the same initialization that receives a single noise perturbation at update \\(s\\). Let \\(\delta_t^{(s)}\\) be their parameter difference. Linearizing each subsequent update around \\(\bar\theta_t\\) gives

$$
\delta_{t+1}^{(s)}
\approx
\left[I-\eta_t\nabla^2\mathcal L(\bar\theta_t)\right]\delta_t^{(s)}
=A_t\delta_t^{(s)}.
$$

Starting from \\(\delta_{s+1}^{(s)}=-\eta_s\xi_s\\), the perturbation after \\(n\\) updates is approximately

$$
\delta_n^{(s)}
\approx
-\eta_s\underbrace{A_{n-1}\cdots A_{s+1}}_{\text{propagation through later updates}}\xi_s,
\qquad
A_t=I-\eta_t\nabla^2\mathcal L(\bar\theta_t).
$$

Each update Jacobian \\(A_t\\) can shrink, amplify, or rotate the perturbation. Their product describes how one batch affects the model many updates later. This does not require a globally linear model: the Jacobians change along the training trajectory.

Let \\(G(\theta)\\) be the final validation loss \\(L_{\mathrm{val}}\\) obtained by starting from \\(\theta\\) after update \\(s\\) and following the remaining full-gradient updates. **The propagation matrices enter the curvature of this future loss, and therefore the memory response**, as shown below.

For a small, mean-zero parameter perturbation \\(h\\), a second-order expansion gives

$$
\mathbb E[G(\theta+h)]-G(\theta)
\approx
\frac12\operatorname{Tr}\!\left[
\operatorname{Cov}(h)\nabla^2G(\theta)
\right].
$$

The linear term averages to zero. The quadratic term measures how the perturbation's variance interacts with the curvature of the future loss. For \\(h=-\eta_s\xi_s\\), this gives the factor \\(\eta_s^2/B_s\\).

Define the noise-free baseline \\(F(n)=L_{\mathrm{val}}(\bar\theta_n)\\), and let \\(K(n,s)\\) describe the loss response to noise injected at update \\(s\\), after factoring out \\(\eta_s^2/B_s\\). Summing over updates gives

$$
\mathbb E L_{\mathrm{val}}(\theta_n)
\approx
\underbrace{F(n)}_{\text{baseline learning}}
+\sum_{s<n}
\underbrace{\frac{\eta_s^2}{B_s}}_{\text{injected variance scale}}
\underbrace{K(n,s)}_{\text{effect on the final loss}}.
$$

The coefficient \\(K(n,s)\\) includes both the spread in final parameters and shifts in their average caused by nonlinear updates.

This is **forcing–memory as a coarse-grained response law of SGD**: instead of tracking every parameter, track a baseline loss and how much earlier fluctuations still matter. The same current loss can hide different parameters and training histories, so it need not imply the same response to the next update.

We now have a reason for memory to appear. The next question is why its effect might follow a simple power law.

<details markdown="1">
<summary><strong>Derivation details: from SGD updates to the memory response</strong></summary>

Fix a horizon \\(n\\), an initialization \\(\theta_0\\), and deterministic schedules \\((\eta_t,B_t)\\). Let \\(\theta_t\in\mathbb R^p\\) be the parameters before update \\(t\\). Each update uses fresh i.i.d. examples from the same data distribution. Assume the smoothness and moment bounds required by the expansions below.

**1. The size of the sampling noise.** For the model \\(f_\theta\\) and per-example loss \\(\ell(f_\theta;z)\\), write \\(\mathcal L(\theta)=\mathbb E_z[\ell(f_\theta;z)]\\) and \\(g(\theta;z)=\nabla_\theta\ell(f_\theta;z)\\). At update \\(t\\), let \\(z_{t,i}\\) be the \\(i\\)-th example in the batch. The batch gradient \\(\widehat g_t\\) and its sampling error \\(\xi_t\\) are

$$
\widehat g_t=\frac1{B_t}\sum_{i=1}^{B_t}g(\theta_t;z_{t,i}),
\qquad
\xi_t=\widehat g_t-\nabla\mathcal L(\theta_t).
$$

Let \\(\mathbb E_t\\) denote expectation conditional on the training history, and \\(\mathbb E\\) expectation over all batches. Define the single-example gradient covariance \\(\Sigma(\theta)=\operatorname{Cov}_z(g(\theta;z))\\). Conditional independence of the examples gives

$$
\mathbb E_t[\xi_t]=0,
\qquad
\mathbb E_t[\xi_t\xi_t^{\!\top}]
=\frac1{B_t^2}\sum_{i=1}^{B_t}\Sigma(\theta_t)
=\frac{\Sigma(\theta_t)}{B_t}.
$$

The parameter perturbation is \\(h_t=-\eta_t\xi_t\\), so \\(\mathbb E_t\lVert h_t\rVert^2=\eta_t^2\operatorname{Tr}\Sigma(\theta_t)/B_t\\). This separates the schedule-dependent variance scale, \\(\eta_t^2/B_t\\), from the state-dependent gradient covariance.

**2. How one perturbation travels through later updates.** Define the full-gradient update map \\(D_t\\) and its reference trajectory \\(\bar\theta_t\\):

$$
D_t(\theta)=\theta-\eta_t\nabla\mathcal L(\theta),
\qquad
\bar\theta_{t+1}=D_t(\bar\theta_t),
\qquad
\bar\theta_0=\theta_0.
$$

Inject \\(h_s=-\eta_s\xi_s\\) at a single update \\(s\\), using full-gradient updates otherwise. Let \\(\delta_t^{(s)}\\) be the displacement from \\(\bar\theta_t\\). With a locally Lipschitz Hessian, subtracting the two updates and expanding for \\(t\ge s+1\\) gives

$$
\begin{aligned}
\delta_{t+1}^{(s)}
&=D_t(\bar\theta_t+\delta_t^{(s)})-D_t(\bar\theta_t)\\
&=\left[I-\eta_t\nabla^2\mathcal L(\bar\theta_t)\right]\delta_t^{(s)}
+O\!\left(\eta_t\|\delta_t^{(s)}\|^2\right).
\end{aligned}
$$

Starting from \\(\delta_{s+1}^{(s)}=-\eta_s\xi_s\\) and retaining the linear term gives

$$
\delta_n^{(s)}\approx-\eta_s A_{n-1}\cdots A_{s+1}\xi_s,
\qquad
A_t=I-\eta_t\nabla^2\mathcal L(\bar\theta_t).
$$

Here \\(A_t\\) is the update Jacobian along the reference trajectory. The product describes how later updates reshape the original perturbation; the empty product is \\(I\\).

**3. How the perturbation changes the final loss.** Let \\(L_{\mathrm{val}}\\) be the validation loss. Define \\(G_{n,s}(\theta)\\) as the final validation loss obtained by starting from \\(\theta\\) at update \\(s\\) and following the remaining full-gradient updates:

$$
G_{n,s}(\theta)
=L_{\mathrm{val}}\!\left((D_{n-1}\circ\cdots\circ D_s)(\theta)\right),
\qquad
G_{n,n}(\theta)=L_{\mathrm{val}}(\theta).
$$

By construction, \\(G_{n,s}=G_{n,s+1}\circ D_s\\).

Write \\(\Psi=D_{n-1}\circ\cdots\circ D_{s+1}\\) for the map from parameters after update \\(s\\) to final parameters, with coordinates \\(\Psi_i\\). Then \\(G_{n,s+1}=L_{\mathrm{val}}\circ\Psi\\), and the second-order chain rule gives

$$
\begin{aligned}
\nabla^2G_{n,s+1}(\theta)
={}&\left[\frac{\partial\Psi}{\partial\theta}(\theta)\right]^{\!\top}
\nabla^2L_{\mathrm{val}}(\Psi(\theta))\,\frac{\partial\Psi}{\partial\theta}(\theta)\\
&+\sum_{i=1}^{p}
\partial_i L_{\mathrm{val}}(\Psi(\theta))\,\nabla^2\Psi_i(\theta).
\end{aligned}
$$

At \\(\bar\theta_{s+1}\\), the remaining updates end at \\(\bar\theta_n\\), and \\(\frac{\partial\Psi}{\partial\theta}\\) equals \\(A_{n-1}\cdots A_{s+1}\\). Substituting gives

$$
\begin{aligned}
\nabla^2G_{n,s+1}(\bar\theta_{s+1})
={}&(A_{n-1}\cdots A_{s+1})^{\!\top}
\nabla^2L_{\mathrm{val}}(\bar\theta_n)
(A_{n-1}\cdots A_{s+1})\\
&+\sum_{i=1}^{p}
\partial_i L_{\mathrm{val}}(\bar\theta_n)\,\nabla^2\Psi_i(\bar\theta_{s+1}).
\end{aligned}
$$

The propagation matrices thus enter the curvature of the future loss. The first term measures how loss responds to the spread in final parameters. The second captures the shift in their mean caused by nonlinear updates: a mean-zero perturbation need not remain mean-zero after nonlinear propagation. Both contribute at second order, so both must be retained.

Return to the actual SGD trajectory and define \\(\theta_s^+=D_s(\theta_s)\\), the state after the full-gradient part of update \\(s\\). Since \\(\theta_{s+1}=\theta_s^+ +h_s\\), a conditional Taylor expansion gives

$$
\begin{aligned}
&\mathbb E_s\!\left[G_{n,s+1}(\theta_s^+ +h_s)-G_{n,s+1}(\theta_s^+)\right]\\
&\quad=\frac12\mathbb E_s\!\left[
h_s^{\!\top}\nabla^2G_{n,s+1}(\theta_s^+)h_s\right]
+O\!\left(\eta_s^3\mathbb E_s\|\xi_s\|^3\right)\\
&\quad=\frac{\eta_s^2}{2B_s}\operatorname{Tr}\!\left[
\Sigma(\theta_s)\nabla^2G_{n,s+1}(\theta_s^+)
\right]
+O\!\left(\eta_s^3\mathbb E_s\|\xi_s\|^3\right).
\end{aligned}
$$

The linear term vanishes because \\(\mathbb E_s[h_s]=0\\), and the second equality uses \\(\mathbb E_s[h_sh_s^{\top}]=\eta_s^2\Sigma(\theta_s)/B_s\\). The higher-order remainder is controlled by third derivatives of the future loss along the segment from \\(\theta_s^+\\) to \\(\theta_s^+ +h_s\\).

**4. Adding the effects of all updates.** A telescoping sum cancels the intermediate future-loss values, leaving the difference between the final SGD loss and the noise-free baseline:

$$
\begin{aligned}
L_{\mathrm{val}}(\theta_n)-G_{n,0}(\theta_0)
&=\sum_{s<n}\left[G_{n,s+1}(\theta_{s+1})-G_{n,s}(\theta_s)\right]\\
&=\sum_{s<n}\left[
G_{n,s+1}(\theta_s^+ +h_s)-G_{n,s+1}(\theta_s^+)
\right].
\end{aligned}
$$

Taking expectations, define the noise-free baseline \\(F(n)=G_{n,0}(\theta_0)\\) and the response coefficient

$$
K(n,s)
:=\frac12\mathbb E\!\left[
\operatorname{Tr}\!\left(
\Sigma(\theta_s)\nabla^2G_{n,s+1}(D_s(\theta_s))
\right)\right].
$$

Let \\(\mathcal E_n\\) collect the expected higher-order remainders over all updates. Inserting the expansion from step 3 gives

$$
\mathbb E L_{\mathrm{val}}(\theta_n)
=F(n)
+\sum_{s<n}\frac{\eta_s^2}{B_s}K(n,s)
+\mathcal E_n.
$$

Earlier noise has not been removed from this calculation: \\(\theta_s\\) already depends on all preceding batches. The average defining \\(K(n,s)\\) therefore includes how earlier noise changes the state reached at update \\(s\\).

**5. The error left by the approximation.** Let \\(M_{n,s}\\) uniformly bound \\(\|\nabla^3G_{n,s+1}\|\\) along the Taylor segments, over the sampled histories under consideration. Then

$$
|\mathcal E_n|
\le\frac16\sum_{s<n}
M_{n,s}\eta_s^3\mathbb E\|\xi_s\|^3.
$$

Omitting \\(\mathcal E_n\\) gives the response approximation in the main text. At fixed horizon, scaling the sampling noise by \\(\epsilon\\) makes the retained contribution quadratic in \\(\epsilon\\) to leading order, while the remainder is \\(O(\epsilon^3)\\), under uniform derivative and third-moment bounds.

The calculation tells us what \\(K(n,s)\\) measures: how noise added at update \\(s\\) affects loss at update \\(n\\), including the effects of the states reached during training. Describing that response by a positive power law in the time since the perturbation is a further modeling step. The next sections explain how the spectral model produces such a law and how the LLM experiments test whether it predicts loss under a new schedule.

</details>

## Different directions learn at different speeds

Consider a fixed-feature model \\(f_a(x)=a^{\top}\phi(x)\\), where only \\(a\\) is trained. Under squared loss, its Hessian \\(H\\) is constant. Let \\((\lambda_j,u_j)\\) be its eigenpairs and \\(\rho_{j,t}\\) the component of a parameter perturbation along \\(u_j\\). Full-gradient updates give

$$
Hu_j=\lambda_j u_j,
\qquad
\rho_{j,t+1}=(1-\eta_t\lambda_j)\rho_{j,t}.
$$

These **spectral modes** have different learning speeds. At small learning rates, directions with large positive \\(\lambda_j\\) relax quickly, while those with small \\(\lambda_j\\) retain errors for much longer.

In our random-feature model, summing the mode contributions gives an exact equation for the expected prediction risk \\(R_t\\). Here \\(F_t\\) is the contribution from initialization, \\(\sigma^2\\) is the label-noise variance, and \\(K_{t,s}\\) is the memory kernel:

$$
R_t=F_t+\sum_{s<t}K_{t,s}\bigl(R_s+\sigma^2\bigr).
$$

The factor \\(R_s+\sigma^2\\) is the noise source: it combines remaining prediction error with label noise. The kernel describes how much of that injection survives. The equation closes on the risk history, without tracking individual parameters. Here \\(K_{t,s}\\) includes the schedule factors written separately as \\(\eta_s^2/B_s\\) above.

At constant learning rate \\(\eta\\) and batch size \\(B\\), use **intrinsic time** \\(T=\eta t\\). A slow mode's squared error decays approximately as \\(e^{-2\lambda_j T}\\). Forcing weights these decays by the initial error in each direction. Memory weights them by noise injection and their contribution to loss. The same spectrum can therefore produce different forcing and memory exponents.

**The learning speeds tell us how long errors last. The schedule tells us how much new noise is added. Together, they determine the loss curve.**

## When do these learning speeds produce a power law?

Each mode with a fixed positive eigenvalue decays exponentially. How can their sum produce a power law?

At intrinsic time \\(T\\), modes with \\(\lambda\gg1/T\\) have mostly decayed, while those with \\(\lambda\ll1/T\\) have barely changed. Training progressively removes faster modes, leaving slower ones behind. The remaining loss is therefore governed by the total weight near the bottom of the spectrum.

Let \\(\nu_W^{\mathcal F}((0,x])\\) be the target-weighted spectral mass in \\(0<\lambda\le x\\), and \\(\nu_W^{\mathcal K}((0,x])\\) the sum of squared eigenvalues in that interval. These weight forcing and memory, respectively; \\(W\\) denotes the random-feature realization.

Write \\(F_{W,>0}\\) for forcing with the permanent error floor removed, and \\(K_W\\) for the one-injection memory kernel at constant schedules. Their decay exponents are \\(q_{\mathcal F}\\) and \\(q_{\mathcal K}\\); \\(\ell_{\mathcal F}\\) and \\(\ell_{\mathcal K}\\) are slowly varying factors, allowing corrections such as logarithms.

Our spectral theorem gives necessary and sufficient conditions for these component powers. It considers a joint large-time, large-width limit, so progressively slower modes remain available:

$$
\begin{aligned}
\nu_W^{\mathcal F}((0,x])\asymp x^{q_{\mathcal F}}\ell_{\mathcal F}(1/x)
&\quad\Longleftrightarrow\quad
F_{W,>0}(T)\asymp T^{-q_{\mathcal F}}\ell_{\mathcal F}(T),\\
\nu_W^{\mathcal K}((0,x])\asymp x^{q_{\mathcal K}}\ell_{\mathcal K}(1/x)
&\quad\Longleftrightarrow\quad
\frac{B}{\eta^2}K_W(T)\asymp T^{-q_{\mathcal K}}\ell_{\mathcal K}(T).
\end{aligned}
$$

The **if-and-only-if** result applies separately to forcing and memory: their temporal powers correspond to power laws in the relevant cumulative spectral weights.

Individual eigenvalues need not follow a neat power law; their weighted totals are what matter. Conversely, a power-law spectrum need not give power-law forcing if the target puts too little weight in slow directions. Different weights also let forcing and memory have different exponents.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-forcing.png' | relative_url }}" alt="Target-weighted forcing decaying faster than every inverse power.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-memory.png' | relative_url }}" alt="One-injection memory following a three-quarter power-law tail.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-loss.png' | relative_url }}" alt="Minibatch-SGD risk crossing into the memory-controlled power-law tail.">
  </div>
  <figcaption><strong>Forcing can disappear quickly while memory leaves a power law.</strong> Left: the target puts so little weight in slow directions that forcing falls faster than every inverse power. Middle: the effect of one noise injection falls as \(T^{-3/4}\). Right: the average SGD loss over 20 runs first drops quickly, then follows the slower \(T^{-3/4}\) memory curve.</figcaption>
</figure>

## Three kinds of memory

The two exponents answer different questions: how quickly does the initial error disappear, and how quickly does the effect of a noise injection disappear? They lead to three kinds of behavior:

- **Long memory (LM):** \\(0<q_{\mathcal K}<1\\). The cumulative memory mass diverges: old injections remain important even as the training horizon grows.
- **Integrable memory (IM):** \\(q_{\mathcal K}>1\\). The cumulative memory mass is finite: the total response to equally weighted past injections saturates.
- **Finite bulk (FB):** the size of the memory response remains tied to the number of features. A single positive decay exponent independent of model width does not describe it.

In the power-law random-feature model, comparing the forcing and memory rates gives three LM cases, three IM cases, and two FB cases: the \\(3+3(+2)\\) in the title. These describe the two ingredients of the learning curve; the schedule still determines how they combine.

The left panel uses the response exponents. The right uses the spectrum-decay parameter \\(\alpha\\) and target-decay parameter \\(\beta\\), connecting these response regimes to the underlying model.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:center;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-response-map.png' | relative_url }}" alt="Long-memory, integrable-memory, and finite-bulk regimes in forcing-memory response coordinates.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig3-plrf-map.png' | relative_url }}" alt="The corresponding propagation regimes in power-law random-feature source-capacity coordinates.">
  </div>
  <figcaption><strong>Different learning speeds lead to different kinds of memory.</strong> Left: the diagram is organized by how fast forcing and memory decay. Right: the same classification is shown using the spectrum and target parameters \((\alpha,\beta)\), with the width-dependent FB cases shown separately. The red band marks the LLM fits near \(q_{\mathcal K}=1\), between LM and IM.</figcaption>
</figure>

We will return to the LLM fits that place the red band there.

## How training schedules transform the response

We can now ask what happens when learning rate and batch size change during training. For plain SGD, track two quantities: intrinsic time \\(T_t\\), the sum of learning rates so far, and the ratio \\(r_t\\) of batch size to learning rate:

$$
T_t=\sum_{s<t}\eta_s,
\qquad
r_t=\frac{B_t}{\eta_t}.
$$

One update advances this clock by \\(\Delta T_t=T_{t+1}-T_t=\eta_t\\). Its noise scale is therefore \\(\eta_t^2/B_t=\Delta T_t/r_t\\). In words, \\(1/r_t\\) sets the noise scale per unit of intrinsic time. We write \\(r(T)\\) for this ratio as a function of the new clock. The whole path matters, because noise from earlier updates can still affect the current loss.

Let \\(L(T)\\) be expected loss, \\(L_\infty\\) its floor, and \\(F(T)\\) the baseline excess loss. Approximating the response by a source amplitude \\(J(u)\\) and a lag-dependent kernel \\(k(T-u)\\) gives

$$
L(T)-L_\infty
\approx
F(T)+\int_0^T\frac{J(u)}{r(u)}k(T-u)\,\mathrm du.
$$

Each contribution is the noise injected at time \\(u\\), multiplied by how much of its effect survives to \\(T\\). The integral is the discrete memory sum expressed in intrinsic time.

Suppose \\(k(v)\sim v^{-q_{\mathcal K}}\\) and \\(r(T)\sim T^\vartheta\\), where \\(v\\) is the time since injection and \\(\vartheta\\) is the ratio-growth exponent. Our schedule theorem compares label-noisy SGD with its clean-label counterpart. The label-noise contribution leads to three outcomes:

- **Destroy:** the ratio grows too slowly, and the extra loss does not fall as a positive power of time.
- **Change:** the extra loss falls, but more slowly than the clean loss, so it sets a new, slower exponent.
- **Preserve:** the extra loss falls at least as fast as the clean loss, so the original exponent remains.

There is a limit to what increasing \\(B/\eta\\) can achieve. Even if late updates add very little noise, earlier noise may still be present. Once that old noise sets the decay rate, increasing the ratio faster no longer improves the exponent. This is the **memory ceiling**. It can prevent preservation when memory fades more slowly than the clean loss.

At the marginal boundary \\(q_{\mathcal K}=1\\), cumulative memory grows logarithmically, introducing logarithmic corrections to the power laws.

<figure>
  <img src="{{ '/images/power-laws-have-a-clock/main-fig4-schedule-map.png' | relative_url }}" alt="Phase diagram showing when a schedule preserves, changes, or destroys a clean power law." style="display:block;width:min(100%,760px);margin-inline:auto;">
  <figcaption><strong>Changing the schedule can preserve, change, or destroy the loss power law.</strong> The outcome depends on how quickly \(B/\eta\) grows. The flat part of the black boundary shows where faster growth stops improving the noise-decay exponent: old noise still sets the rate. At \(q_{\mathcal K}=1\), logarithmic factors enter.</figcaption>
</figure>

We can also work backward. Below the long-memory ceiling, the extra loss identifies the long-time shape of \\(B/\eta\\), including slower-varying factors, but not learning rate and batch size separately. At the ceiling, different ratio paths produce the same decay, so this identification is lost.

## From schedule laws to schedule design

The same formula tells us where to spend training data. In the long-memory and integrable-memory cases, fix an ending intrinsic time \\(T\\) and a budget of \\(D\\) training examples. Since one update uses \\(B_t=r_t\Delta T_t\\) examples, the total budget becomes \\(\int_0^T r(u)\,\mathrm du=D\\).

In the theoretical response model, the optimal ratio schedule is

$$
r_T^\star(u)
\propto
\sqrt{k(T-u)\bigl[F(u)+\sigma^2\bigr]}.
$$

Here \\(F(u)\\) is the theoretical baseline risk, including the finite-feature error floor. The product combines the strength of the noise source, \\(F(u)+\sigma^2\\), with its influence on terminal loss, \\(k(T-u)\\). The rule allocates more samples where noise is both strong and persistent, with the normalization fixed by the budget \\(D\\). Learning-rate and batch-size schedules can implement the same ratio path.

We can then optimize training duration and model width. Each phase gives a best loss-versus-data rate and a best rate under compute budgets proportional to width times data. The diagram thus guides resource choices as well as explaining curves.

## From theory to LLM pretraining

The theory poses a practical LLM question: **can a low-dimensional forcing–memory response predict how loss changes under a new schedule?** We test it in three steps.

### Test 1: the ratio path affects loss

From one mature 30M checkpoint, we continued plain-SGD training in eleven ways. All runs ended at the same intrinsic time, but \\(B/\eta\\) grew at different rates. Faster growth lowered validation loss, with diminishing gains. Intrinsic time alone did not determine the loss: the ratio path also mattered.

### Test 2: two ways to produce the same ratio path

For a 300M nanoGPT trained on 6.5B OpenWebText tokens, we shared a 5.2B-token training prefix and branched into 1.3B-token tails from the same checkpoint.

We compared two schedule shapes. In their fixed-batch versions, WSD keeps the learning rate constant for the first 80% of training, then lowers it gradually over the last 20%. The 8-1-1 schedule uses the same first 80%, followed by two lower-rate stages lasting 10% each.

For each schedule, we compared two factorizations: fixed batch size with varying learning rate, and fixed learning rate with varying batch size. We matched data, intrinsic-time increments, and \\(B/\eta\\) over groups of updates. The factorizations used different numbers of optimizer steps, but produced nearly coincident validation-loss curves in intrinsic time.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-step.png' | relative_url }}" alt="Validation risk for matched learning-rate and batch-size schedules plotted against optimizer step.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-intrinsic-time.png' | relative_url }}" alt="The same validation trajectories plotted against intrinsic time, together with the transferred surrogate.">
  </div>
  <figcaption><strong>The same ratio path gives nearly the same loss curve on the right clock.</strong> Left: changing learning rate or changing batch size gives different curves against update number. Right: adding up learning rates to form \(T=\sum_t\eta_t\) brings each pair together. The forcing–memory formula also follows both schedule shapes.</figcaption>
</figure>

We also ran a hybrid-Muon version. A short calibration selected a different clock and ratio for those runs: \\(\widetilde T=\sum_t\eta_t^2\\) and \\(\widetilde r=B/\eta^2\\). Using these quantities brought its paired curves together as well.

### Test 3: fit one schedule, predict another

We approximate the effective response by \\(K(n,s)\approx J(T_s)k(T_n-T_s)\\): the source amplitude depends on when noise enters, while the propagation kernel depends on how long it has been present.

This gives a seven-parameter **surrogate** for validation loss. The parameters are the floor \\(L_\infty\\), forcing amplitude \\(A_{\mathcal F}\\), source amplitudes \\(A_0,A_1\\), memory time-scale parameter \\(c_{\mathcal K}\\), and exponents \\(q_{\mathcal F},q_{\mathcal K}\\):

$$
\widehat L(T)
=L_\infty+A_{\mathcal F}(1+T)^{-q_{\mathcal F}}
+\int_0^T
\frac{A_0+A_1(1+u)^{-q_{\mathcal F}}}{r(u)}
\bigl(1+c_{\mathcal K}(T-u)\bigr)^{-q_{\mathcal K}}
\,\mathrm du.
$$

The first two terms describe baseline loss; the integral adds the noise effects that remain. Once the parameters are fitted, changing the known ratio \\(r(u)\\) predicts a new schedule.

In the main 300M analysis, we fix \\(q_{\mathcal K}=1\\) and fit the other six parameters using only the original, unsmoothed validation-loss measurements from the fixed-batch 8-1-1 run. We then keep all parameters unchanged, insert the WSD schedule, and predict its validation loss. No WSD loss measurements are used to fit those six parameters.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-fit.png' | relative_url }}" alt="Forcing-memory surrogate fitted to the 8-1-1 validation trajectory.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-transfer.png' | relative_url }}" alt="The frozen surrogate predicting the held-out WSD validation trajectory without refitting.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-profile.png' | relative_url }}" alt="Held-out WSD prediction error as a function of the fixed memory exponent.">
  </div>
  <figcaption><strong>Fit on 8-1-1, then predict WSD without fitting again.</strong> Left: fix \(q_{\mathcal K}=1\) and fit the other six parameters to 8-1-1. Middle: keep them unchanged and predict WSD. Right: repeat the procedure for different fixed values of \(q_{\mathcal K}\), always fitting only on 8-1-1. The WSD prediction is most accurate near one.</figcaption>
</figure>

The predicted WSD curve follows the measurements. The fitted response describes more than one curve: it predicts how a new learning-rate history changes the loss.

We also fit all seven parameters freely, including \\(q_{\mathcal K}\\), for several models and datasets:

| Dataset and setting | \\(q_{\mathcal K}\\) | \\(q_{\mathcal F}\\) |
|---|---:|---:|
| OpenWebText, 124M, 2.5B tokens | 1.017 | 0.374 |
| FineWeb sample-10BT subset, 124M, 2.5B tokens | 0.952 | 0.405 |
| peS2o V2 s2orc full text, 124M, 2.5B tokens | 0.995 | 0.365 |
| OpenWebText, 300M, 6.5B tokens | 0.989 | 0.291 |

Across these model and dataset settings, \\(q_{\mathcal K}\approx1\\) and \\(q_{\mathcal F}<1\\). The fitted LLM responses lie near the LM/IM boundary—the red band in the phase diagram.

## What the results change

- **An exponent is not the whole explanation.** Ask which directions still matter, how slowly they learn, and whether the curve is set by initial error or accumulated noise.
- **A learning curve needs a clock and a schedule.** State how time is measured and how learning rate and batch size change before comparing exponents.
- **Predicting a new schedule is a stronger test than fitting one curve.** Keep the fitted parameters fixed and check whether they predict new measurements.
- **A tractable model can reveal a reusable response structure.** The linear theory makes forcing and memory explicit; the LLM experiments test whether the same low-dimensional description predicts nonlinear training.

## The larger lesson

The argument connects four things: the importance of slow learning directions, the resulting forcing and memory, the noise added by a schedule, and the loss we observe:

$$
\text{weight in slow directions}
\longrightarrow
\text{forcing and memory}
\longrightarrow
\text{noise added and retained}
\longrightarrow
\text{loss curve}.
$$

Training removes old error and continually adds new fluctuations. The spectrum tells us how quickly different parts disappear. The schedule controls how much noise enters along the way. A learning curve records the balance of these processes.

The theory tells us when these parts follow power laws and how to allocate training resources. The LLM tests show the idea's predictive value: matching ratio paths aligns curves in intrinsic time, and fitting one schedule predicts another. **Forcing–memory connects a theory we can calculate to learning curves we can measure.**

---

*This post describes joint work by Yichen Wang, Fanghui Liu, and Yudong Chen.*

<!-- Add public paper and code links here when they are available. -->
