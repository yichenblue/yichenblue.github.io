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

*If individual errors decay exponentially, where does a power-law learning curve come from?*

[Paper (arXiv)](https://arxiv.org/abs/2609.40148) · [Code (GitHub)](https://github.com/yichenblue/spectra-to-schedules-in-pretraining)

In a linear model, individual directions can lose error exponentially, yet their combined loss can follow a power law, like those familiar from [language-model scaling](https://arxiv.org/abs/2001.08361). What makes that happen?

**A power law is a phenomenon to explain, not an explanation.** We separate remaining initial error—**forcing**—from the **memory** of training noise. In a random-feature model, we identify necessary and sufficient spectral conditions for each to produce a power law, and show how schedules reshape the resulting curve.

The same idea also predicts LLM loss: fit a forcing–memory formula on an 8-1-1 schedule, freeze its parameters, and predict WSD. To understand why this might work, start with one batch: why does its influence outlast its update?

## Why SGD has memory

A batch changes the parameters from which every later update starts. Two runs can have the same current loss but different histories and respond differently to the next update. Loss alone does not record everything that matters.

For a model \\(f_\theta\\) with training objective \\(\mathcal L\\), let \\(\eta_t\\) be the learning rate, \\(B_t\\) the batch size, and \\(\xi_t\\) the batch-gradient error. With fresh i.i.d. examples, SGD takes the form

$$
\theta_{t+1}
=\theta_t-\eta_t\nabla\mathcal L(\theta_t)-\eta_t\xi_t,
\qquad
\mathbb E_t[\xi_t]=0,
\qquad
\operatorname{Cov}_t(\xi_t)=\frac{\Sigma(\theta_t)}{B_t}.
$$

Here expectations are conditional on the training history, and \\(\Sigma\\) is the single-example gradient covariance. The parameter perturbation \\(-\eta_t\xi_t\\) therefore has variance scale \\(\eta_t^2/B_t\\). Later updates reshape this perturbation rather than simply erasing it.

A small-noise expansion, derived below, adds up these lasting effects. Write \\(F(n)\\) for the noise-free validation loss and \\(K(n,s)\\) for the response to noise injected at update \\(s\\), after removing its variance scale:

$$
\mathbb E L_{\mathrm{val}}(\theta_n)
\approx
\underbrace{F(n)}_{\text{baseline learning}}
+\sum_{s<n}
\underbrace{\frac{\eta_s^2}{B_s}}_{\text{injected variance scale}}
\underbrace{K(n,s)}_{\text{effect on the final loss}}.
$$

This motivates our working hypothesis: **forcing–memory may be an effective response law of SGD, not merely a description of a linear model.** Loss could follow a transferable, coarse-grained law even while parameters and representations change.

The goal is not to reconstruct every change inside the network. It is to describe how those changes affect one observable: loss. Different internal trajectories could share a similar response to training noise. The linear model gives us a setting where we can calculate that response; LLM experiments test whether it travels further.

Memory has a reason to exist. What determines how quickly it fades?

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

An error's lifetime depends on its direction. In a fixed-feature model, the Hessian spectrum gives these learning speeds explicitly, as in [Paquette et al.'s analysis of compute-optimal scaling](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1dccfc3ee01871d05e33457c61037d59-Abstract-Conference.html).

Consider \\(f_a(x)=a^\top\phi(x)\\), with fixed features \\(\phi\\), trainable weights \\(a\\), and squared loss. Let \\(\rho_{j,t}\\) be the parameter error along a Hessian eigenvector with eigenvalue \\(\lambda_j\\). Under small full-gradient steps,

$$
\rho_{j,t}\approx\rho_{j,0}e^{-\lambda_jT_t},
\qquad
T_t=\sum_{s<t}\eta_s.
$$

The clock \\(T_t\\) is **intrinsic time**. Large positive eigenvalues mean fast learning; small ones mean persistent errors, with learning time proportional to \\(1/\lambda_j\\).

Loss combines squared errors across these directions. A slow direction matters only if it carries appreciable weight: an error can persist for a long time without contributing much to total loss. Learning speed and importance are different pieces of information.

Forcing weights the squared decays by initial error; memory weights them by noise injection and their effect on loss. The same learning speeds can therefore produce different forcing and memory curves. But why should either weighted sum be a power law?

<details markdown="1">
<summary><strong>Derivation details: from squared loss to learning time</strong></summary>

**1. From squared loss to the gradient.** Let \\(\phi(x)\\) be the fixed feature vector for input \\(x\\), \\(y\\) its target, and \\(a\\) the trainable weights. Define the population squared loss and its Hessian:

$$
\mathcal L(a)=\frac12\mathbb E\!\left[(a^\top\phi(x)-y)^2\right],
\qquad
H=\mathbb E[\phi(x)\phi(x)^\top].
$$

For a minimizer \\(a_\star\\), the condition \\(\nabla\mathcal L(a_\star)=0\\) gives \\(\mathbb E[\phi(x)y]=Ha_\star\\). Hence

$$
\nabla\mathcal L(a)
=\mathbb E\!\left[\phi(x)(a^\top\phi(x)-y)\right]
=H(a-a_\star).
$$

The gradient is the remaining parameter error transformed by \\(H\\). This identity is exact for squared loss with fixed features.

**2. From the gradient to error dynamics.** First consider full-gradient training, without batch-sampling noise. With learning rate \\(\eta_t\\) and parameter error \\(e_t=a_t-a_\star\\), the update becomes

$$
\begin{aligned}
a_{t+1}&=a_t-\eta_t\nabla\mathcal L(a_t),\\
e_{t+1}&=e_t-\eta_tHe_t=(I-\eta_tH)e_t.
\end{aligned}
$$

Let \\((\lambda_j,u_j)\\) be an orthonormal eigensystem of the symmetric matrix \\(H\\), and define \\(\rho_{j,t}=u_j^\top e_t\\). Projecting the update onto \\(u_j\\) gives

$$
\begin{aligned}
\rho_{j,t+1}
&=u_j^\top(I-\eta_tH)e_t\\
&=(1-\eta_t\lambda_j)\rho_{j,t},
\qquad Hu_j=\lambda_j u_j.
\end{aligned}
$$

Along direction \\(u_j\\), the gradient is \\(\lambda_j\\) times the remaining error. In the small-step regime, one update therefore removes a fraction \\(\eta_t\lambda_j\\) of that error.

**3. From the recurrence to a learning time scale.** Iterating from the initial error \\(\rho_{j,0}\\) gives

$$
\rho_{j,t}
=\rho_{j,0}\prod_{s<t}(1-\eta_s\lambda_j).
$$

For \\(\eta_s\lambda_j\ll1\\), using \\(\log(1-x)\approx-x\\) yields the intrinsic-time approximation

$$
\rho_{j,t}\approx\rho_{j,0}e^{-\lambda_jT_t},
\qquad
T_t=\sum_{s<t}\eta_s.
$$

For \\(\lambda_j>0\\), let \\(T_{\mathrm{learn},j}\\) denote the intrinsic time needed to reduce the error magnitude by a fixed factor, and \\(t_{\mathrm{learn},j}\\) the corresponding number of updates. At constant learning rate \\(\eta\\),

$$
T_{\mathrm{learn},j}\asymp\frac1{\lambda_j},
\qquad
t_{\mathrm{learn},j}\asymp\frac1{\eta\lambda_j}.
$$

Large eigenvalues mean short learning times; small eigenvalues mean persistent errors. At time \\(T\\), modes with \\(\lambda_jT\gg1\\) have largely decayed, while those with \\(\lambda_jT\ll1\\) remain close to their initial values.

**4. From directional errors to loss.** The excess squared loss is

$$
\begin{aligned}
\mathcal L(a_t)-\mathcal L(a_\star)
&=\frac12e_t^\top He_t
=\frac12\sum_j\lambda_j\rho_{j,t}^2\\
&\approx\frac12\sum_j\lambda_j\rho_{j,0}^2e^{-2\lambda_jT_t}.
\end{aligned}
$$

The eigenvalues set the decay speeds, while the initial error in each direction sets its weight in the loss. This is why knowing the eigenvalues alone is not enough: we also need to know which directions matter for the target. Batch sampling adds new errors during SGD; the full-gradient calculation above isolates how existing errors decay.

**5. Connecting the linear and general responses.** In our random-feature model, summing the mode contributions gives an exact equation for expected prediction risk \\(R_t\\):

$$
R_t=F_t+\sum_{s<t}K_{t,s}\bigl(R_s+\sigma^2\bigr).
$$

Here \\(F_t\\) is the contribution from initialization, \\(\sigma^2\\) is label-noise variance, and \\(K_{t,s}\\) is the memory kernel. The source \\(R_s+\sigma^2\\) combines remaining prediction error with label noise; the kernel describes how much survives. Compared with the general response, the source is explicit and \\(K_{t,s}\\) includes \\(\eta_s^2/B_s\\). The terms are regrouped: \\(F_t\\) also absorbs part of the sampling noise into the propagation of initial error, so it is not the pure full-gradient baseline \\(F(n)\\).

</details>

## When do these learning speeds produce a power law?

**Individual eigenvalues need not follow a power law. Their weighted totals near zero are what matter.** At time \\(T\\), fast directions have mostly decayed; directions with \\(\lambda\lesssim1/T\\) still retain error. Their combined weight determines the tail.

Our random-feature theorem makes this an **if-and-only-if** criterion in the large-width, long-time limit. Let \\(\nu_W^{\mathcal F}((0,x])\\) collect target-weighted spectral mass below \\(x\\), and \\(\nu_W^{\mathcal K}((0,x])\\) collect squared eigenvalues there. For a feature realization \\(W\\), write \\(F_{W,>0}\\) for forcing above its floor and \\(K_W\\) for the constant-schedule memory kernel. Schematically,

$$
\begin{aligned}
\nu_W^{\mathcal F}((0,x])\propto x^{q_{\mathcal F}}
&\quad\Longleftrightarrow\quad
F_{W,>0}(T)\propto T^{-q_{\mathcal F}},\\
\nu_W^{\mathcal K}((0,x])\propto x^{q_{\mathcal K}}
&\quad\Longleftrightarrow\quad
\frac{B}{\eta^2}K_W(T)\propto T^{-q_{\mathcal K}}.
\end{aligned}
$$

The positive exponents \\(q_{\mathcal F}\\) and \\(q_{\mathcal K}\\) describe forcing and memory separately. Given the weighted spectrum, we can predict the component's temporal exponent. Conversely, a component power law constrains how its weight accumulates among slow directions. The exponent is linked to a property of the learning problem, rather than introduced only as a fitting parameter.

Even an exact power-law spectrum can fail to produce power-law forcing if the target barely uses its slow directions. In the example below, forcing disappears faster than every inverse power, yet memory leaves a power-law loss tail.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-forcing.png' | relative_url }}" alt="Target-weighted forcing decaying faster than every inverse power.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-memory.png' | relative_url }}" alt="One-injection memory following a three-quarter power-law tail.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-loss.png' | relative_url }}" alt="Minibatch-SGD risk crossing into the memory-controlled power-law tail.">
  </div>
  <figcaption><strong>Fast forcing decay, slow memory tail.</strong> Left: forcing vanishes faster than every inverse power. Middle: one-injection memory decays as \(T^{-3/4}\). Right: SGD loss, averaged over 20 runs, eventually follows that memory tail.</figcaption>
</figure>

**Slow loss decay is not, by itself, evidence of slow signal learning.** The curve can reflect remaining target error or noise that training has not erased. A fitted exponent does not distinguish them.

This is why separating forcing and memory matters: a familiar-looking scaling curve can conceal very different reasons for slow improvement.

## Three kinds of memory

A kernel describes one injection; SGD keeps adding more. For equally weighted injections, does the accumulated response saturate or keep growing? This separates three behaviors:

- **Long memory (LM):** \\(0<q_{\mathcal K}<1\\). Cumulative memory keeps growing with the training horizon.
- **Integrable memory (IM):** \\(q_{\mathcal K}>1\\). Cumulative memory approaches a finite total.
- **Finite bulk (FB):** the response remains tied to model width, rather than a width-independent positive decay exponent.

Comparing forcing and memory rates gives three LM cases, three IM cases, and two FB cases: the \\(3+3(+2)\\) in the title.

These are distinctions about the response to past errors, not three recipes for training. A schedule still determines when and how strongly noise is injected into that response.

The right-hand diagram uses the spectrum and target parameters

$$
\lambda_j=j^{-2\alpha},
\qquad
|\theta_j^\star|=j^{-\beta}.
$$

Here \\(\lambda_j\\) are ordered data-covariance eigenvalues and \\(\theta_j^\star\\) the corresponding target coefficients. Larger \\(\alpha\\) steepens the spectrum; larger \\(\beta\\) reduces target weight in slow directions.

In the LM/IM region with finite target energy, the exponents are

$$
q_{\mathcal F}=\frac{2\alpha+2\beta-1}{2\alpha},
\qquad
q_{\mathcal K}=2-\frac{1}{2\alpha}.
$$

<figure style="display:block;">
  <div style="display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:clamp(12px,2vw,24px);align-items:center;width:100%;max-width:680px;margin:0 auto 1rem;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-response-map.png' | relative_url }}" alt="Long-memory, integrable-memory, and finite-bulk regimes in forcing-memory response coordinates." style="display:block;width:100%;height:auto;margin:0;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig3-plrf-map.png' | relative_url }}" alt="The corresponding propagation regimes in power-law random-feature source-capacity coordinates." style="display:block;width:100%;height:auto;margin:0;">
  </div>
  <figcaption><strong>Memory regimes in two coordinate systems.</strong> Left: response exponents. Right: spectrum and target parameters, with FB shown separately. The red band marks LLM fits near the LM/IM boundary, \(q_{\mathcal K}=1\).</figcaption>
</figure>

These regimes describe how errors persist. What happens when a schedule changes how much new noise enters?

## How training schedules transform the response

**A schedule can change the loss exponent, not just the speed of training.** For plain SGD, the useful coordinates are intrinsic time and the batch-to-learning-rate ratio:

$$
T_t=\sum_{s<t}\eta_s,
\qquad
r_t=\frac{B_t}{\eta_t}.
$$

An update advances the clock by \\(\eta_t\\) and injects variance at scale \\(\eta_t^2/B_t\\). Thus \\(1/r_t\\) is the noise scale per unit of intrinsic time. Write \\(r(T)\\) for the ratio along this clock.

This connects to [increasing batch size instead of decaying learning rate](https://arxiv.org/abs/1711.00489). Memory explains why the whole ratio path matters, not just its current value.

Two runs can reach the same current learning rate and batch size after very different noise histories. Their losses need not agree. To compare schedules, we therefore match the ratio along intrinsic time—not merely the final ratio or the number of optimizer updates.

In the power-law random-feature model's LM and IM regimes,

$$
R(T)\asymp
F(T)+
\int_0^T
\frac{F(u)+\sigma^2}{r(u)}
\,k(T-u)\,\mathrm du.
$$

Here \\(R\\) is prediction risk, \\(F\\) is forcing including its finite-width floor, and \\(\sigma^2\\) is label-noise variance. The integral combines **the noise source**, \\(F(u)+\sigma^2\\); **its strength**, controlled by \\(r(u)\\); and **what survives**, described by \\(k(T-u)\\). The relation holds up to multiplicative constants.

For power-law memory and ratio growth, our theorem compares label-noisy SGD with clean-label SGD:

- **Destroy:** the extra loss does not decay as a positive power.
- **Change:** it decays more slowly than the clean loss and sets a new exponent.
- **Preserve:** it decays at least as fast, leaving the clean exponent unchanged.

The limit is the **memory ceiling**: reducing new noise cannot erase old noise faster than memory decays. Beyond this ceiling, faster growth of \\(B/\eta\\) no longer improves the noise-decay exponent.

This separates two obstacles to lower loss: noise still being added and noise already present. A schedule can suppress the first while the second continues to control the curve.

<figure>
  <img src="{{ '/images/power-laws-have-a-clock/main-fig4-schedule-map.png' | relative_url }}" alt="Phase diagram showing when a schedule preserves, changes, or destroys a clean power law." style="display:block;width:min(100%,760px);margin-inline:auto;">
  <figcaption><strong>Preserve, change, or destroy.</strong> The ratio-growth exponent \(\vartheta\), defined by \(r(T)\sim T^\vartheta\), controls the outcome. The flat boundary marks the memory ceiling; \(q_{\mathcal K}=1\) introduces logarithmic corrections.</figcaption>
</figure>

## From schedule laws to schedule design

With a fixed data budget and ending intrinsic time \\(T\\), where should extra samples go? **Where they reduce noise that would otherwise matter most at the end.** In the LM/IM response model, the optimal ratio is

$$
r_T^\star(u)
\propto
\sqrt{k(T-u)\bigl[F(u)+\sigma^2\bigr]}.
$$

The product measures how much noise is generated and how much survives until \\(T\\); the data budget fixes the normalization. Learning-rate and batch-size schedules can implement the same ratio path. Optimizing duration and width then gives phase-dependent loss-versus-data and loss-versus-compute rates.

Spending samples only where current error is large ignores what happens afterward. Some early noise will fade before training ends; other injections remain influential. The square-root rule balances source strength against this future impact under the shared budget.

[Bordelon and Mori's optimal-control analysis](https://arxiv.org/abs/2602.04774) offers a complementary route to learning-rate design.

## From theory to LLM pretraining

An LLM does not have fixed features. Can its loss still follow this response structure? **A learning-curve model should predict what happens when training changes.** We test that by changing how a schedule is implemented, then predicting a different schedule.

A preliminary 30M SGD experiment confirmed that the clock alone was insufficient: at the same final intrinsic time, faster growth of \\(B/\eta\\) lowered loss, with diminishing gains.

### Two implementations of the same ratio path

Does it matter whether learning rate or batch size changes? For a 300M plain-SGD LLM, we compared fixed-batch and fixed-learning-rate implementations of two schedule shapes: gradual decay (WSD) and two successive drops (8-1-1).

**Matched in intrinsic time and ratio path, the paired validation-loss curves nearly coincide.** Their optimizer-step curves differ because the implementations take different numbers of updates.

The comparison asks whether learning rate and batch size need to be modeled separately, or mainly through the clock and ratio. The alignment supports the latter description for these experiments: changing the implementation preserves the main loss response.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-step.png' | relative_url }}" alt="Validation risk for matched learning-rate and batch-size schedules plotted against optimizer step.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-intrinsic-time.png' | relative_url }}" alt="The same validation trajectories plotted against intrinsic time, together with the transferred surrogate.">
  </div>
  <figcaption><strong>Different step counts, nearly the same intrinsic-time curves.</strong> Paired learning-rate and batch-size implementations separate against optimizer step (left) and align against \(T=\sum_t\eta_t\) (right).</figcaption>
</figure>

### Fit one schedule and predict another

Matching the same path is one test. Predicting a different path is stronger. Building on [Li et al.'s functional-scaling-law approach](https://arxiv.org/abs/2509.19189), we use the seven-parameter surrogate

$$
\widehat L(T)
=L_\infty+A_{\mathcal F}(1+T)^{-q_{\mathcal F}}
+\int_0^T
\frac{A_0+A_1(1+u)^{-q_{\mathcal F}}}{r(u)}
\bigl(1+c_{\mathcal K}(T-u)\bigr)^{-q_{\mathcal K}}
\,\mathrm du.
$$

The first two terms describe baseline loss; the integral adds surviving noise effects. The fitted parameters describe the response, while the known ratio path supplies the schedule.

We fix \\(q_{\mathcal K}=1\\) and fit the other six parameters to raw validation loss from the fixed-batch 8-1-1 run. Keeping all parameters unchanged, we insert WSD's ratio path. **The resulting prediction follows WSD's measured validation loss without refitting.**

The curve changes because its input schedule changes, not because its response parameters are adjusted. Fitting each trajectory separately would not test this distinction. Here the fit must carry enough information from one training history to describe another, including the different timing of its loss decrease.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-fit.png' | relative_url }}" alt="Forcing-memory surrogate fitted to the 8-1-1 validation trajectory.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-transfer.png' | relative_url }}" alt="The frozen surrogate predicting the held-out WSD validation trajectory without refitting.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-profile.png' | relative_url }}" alt="Held-out WSD prediction error as a function of the fixed memory exponent.">
  </div>
  <figcaption><strong>8-1-1 fit (left), zero-refit WSD prediction (middle).</strong> Right: repeat for different fixed \(q_{\mathcal K}\), fitting the other parameters only on 8-1-1. WSD prediction is most accurate near one.</figcaption>
</figure>

The response carries information beyond its fitting curve. That is the standard we advocate: judge a learning-curve model by what it predicts when training changes, not only by how well it fits one run.

<details markdown="1">
<summary><strong>Experiment and surrogate details</strong></summary>

**Training and matching.** The preliminary 30M experiment continued one mature plain-SGD checkpoint under eleven ratio paths, all ending at the same intrinsic time. The main 300M nanoGPT experiment used 6.5B OpenWebText tokens: a shared 5.2B-token prefix followed by 1.3B-token tails. For each schedule, we compared fixed batch size with varying learning rate against fixed learning rate with varying batch size, matching data, intrinsic-time increments, and \\(B/\eta\\) over groups of updates.

In their fixed-batch versions, WSD keeps learning rate constant for the first 80% of training and lowers it gradually over the last 20%. The 8-1-1 schedule shares the first 80%, then uses two lower-rate stages lasting 10% each.

**Surrogate parameters.** The seven parameters are the floor \\(L_\infty\\), forcing amplitude \\(A_{\mathcal F}\\), source amplitudes \\(A_0,A_1\\), memory time-scale parameter \\(c_{\mathcal K}\\), and exponents \\(q_{\mathcal F},q_{\mathcal K}\\). The approximation separates the effective response as \\(K(n,s)\approx J(T_s)k(T_n-T_s)\\), where \\(J\\) is the source amplitude and \\(k\\) the propagation kernel.

With \\(q_{\mathcal K}=1\\), the other six parameters are fitted only to original, unsmoothed 8-1-1 measurements; no WSD loss enters that fit. The profile plot repeats this procedure for each fixed exponent and evaluates the resulting WSD prediction error.

**Hybrid Muon.** A short calibration selected a different clock and ratio for those runs: \\(\widetilde T=\sum_t\eta_t^2\\) and \\(\widetilde r=B/\eta^2\\). Those calibrated quantities brought its paired curves together as well.

</details>

## Why does the memory exponent keep landing near one?

Was near-one memory built into the fit? **The exponent stays close to one even when fitted freely.** Here all seven parameters are free, across different datasets and two model sizes:

| Dataset and setting | \\(q_{\mathcal K}\\) | \\(q_{\mathcal F}\\) |
|---|---:|---:|
| OpenWebText, 124M, 2.5B tokens | 1.017 | 0.374 |
| FineWeb sample-10BT subset, 124M, 2.5B tokens | 0.952 | 0.405 |
| peS2o V2 s2orc full text, 124M, 2.5B tokens | 0.995 | 0.365 |
| OpenWebText, 300M, 6.5B tokens | 0.989 | 0.291 |

These responses sit near the LM/IM boundary: cumulative memory is finite on one side and keeps growing on the other, logarithmically at one.

Why do different tasks land here? A shared effective response, or an effect of finite fitting windows and parameter trade-offs? The curves do not yet decide. **Whether near-one memory survives changes in training and measurement is an open question worth pursuing.**

## The larger lesson

Exponentially learning directions can combine into a power law. The spectrum tells us when; forcing and memory tell us what is decaying; schedules change how those contributions combine. The LLM results suggest that this response structure can remain useful far beyond the model where we can prove it.

**Explain where a learning curve comes from, then test that explanation by predicting how it changes.** Fitting a power law begins the investigation; it does not finish it.

## Further reading

- **Empirical scaling laws:** Kaplan et al. (2020), [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). How language-model loss depends on model size, data, and compute.
- **Spectral models:** Paquette et al. (2024), [4+3 Phases of Compute-Optimal Neural Scaling Laws](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1dccfc3ee01871d05e33457c61037d59-Abstract-Conference.html). A solvable model connecting data and target structure to compute-optimal regimes.
- **Learning rate and batch size:** Smith et al. (2018), [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489). Experiments replacing learning-rate decay with batch-size growth.
- **Predicting full trajectories:** Li et al. (2025), [Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](https://arxiv.org/abs/2509.19189). Intrinsic-time theory and surrogate prediction of LLM learning curves.
- **Optimal schedules:** Bordelon and Mori (2026), [Theory of Optimal Learning Rate Schedules and Scaling Laws for a Random Feature Model](https://arxiv.org/abs/2602.04774). How optimal annealing depends on the spectrum and task.

---

*This post describes joint work by Yichen Wang, Fanghui Liu, and Yudong Chen.*
