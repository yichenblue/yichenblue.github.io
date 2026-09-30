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

*When power laws emerge, how joint schedules preserve, change, or destroy them, and why the same theory predicts LLM learning curves.*

Power-law learning curves are often treated as empirical fingerprints of a model and dataset. We show that they have a precise spectral origin, a sharp transformation law under joint learning-rate and batch-size schedules, and a response structure that predicts LLM training curves across schedules.

We trace the full life cycle of a power law: where it comes from, how a schedule rewrites it, how it guides resource allocation, and how the resulting mechanism carries into LLM pretraining.

The story has three parts:

1. **Spectral origin:** we give necessary-and-sufficient conditions for forcing and memory to become power laws.
2. **Schedule transformation:** we show exactly how learning-rate and batch-size schedules preserve, change, or destroy the power visible in loss.
3. **Design and prediction:** the phase laws yield optimal ratio schedules and resource rates, while a response model fitted on one LLM schedule predicts unseen schedules without refitting.

To see how these pieces connect, start with the training process itself: what happens to a mini-batch fluctuation after the update that created it?

## Why SGD has memory

Let \\(f_\theta\\) be a model with parameters \\(\theta\\), and let \\(\ell(f_\theta;z)\\) measure its error on a training example \\(z\\). The model can be a neural network, including a transformer. Its training objective is

$$
\mathcal L(\theta)
=\mathbb E_z[\ell(f_\theta;z)].
$$

SGD estimates this gradient using a fresh batch of \\(B_t\\) examples. We can separate that estimate into the full gradient and a sampling fluctuation:

$$
\theta_{t+1}
=\theta_t-\eta_t\nabla\mathcal L(\theta_t)-\eta_t\xi_t,
\qquad
\mathbb E_t[\xi_t]=0,
\qquad
\operatorname{Cov}_t(\xi_t)=\frac{\Sigma(\theta_t)}{B_t}.
$$

Here \\(\Sigma(\theta_t)\\) is the covariance of a single-example gradient; the expectations condition on the training history. Averaging independent examples divides the covariance by \\(B_t\\), and multiplying the gradient by \\(\eta_t\\) multiplies that covariance by \\(\eta_t^2\\). The injected parameter noise therefore has covariance \\(\eta_t^2\Sigma(\theta_t)/B_t\\): its variance prefactor is \\(\eta_t^2/B_t\\), while the typical size of the parameter perturbation scales as \\(\eta_t/\sqrt{B_t}\\) at fixed \\(\Sigma\\).

Now follow one injection. Imagine running full-gradient training from the same initialization, producing a reference trajectory \\(\bar\theta_t\\), and adding a small perturbation at update \\(s\\). Let \\(\delta_t^{(s)}\\) be the resulting displacement from the reference path. Subtracting the two gradient updates and Taylor-expanding gives

$$
\delta_{t+1}^{(s)}
\approx
\left[I-\eta_t\nabla^2\mathcal L(\bar\theta_t)\right]\delta_t^{(s)}
=A_t\delta_t^{(s)}.
$$

Starting with \\(\delta_{s+1}^{(s)}=-\eta_s\xi_s\\), successive updates therefore give

$$
\delta_n^{(s)}
\approx
-\eta_s\underbrace{A_{n-1}\cdots A_{s+1}}_{\text{propagation through later updates}}\xi_s,
\qquad
A_t=I-\eta_t\nabla^2\mathcal L(\bar\theta_t).
$$

Each matrix \\(A_t\\) describes how one update changes a nearby perturbation. Their product tells us how training transports, amplifies, or forgets an earlier fluctuation. The model remains nonlinear: these matrices change along its trajectory.

We observe validation loss \\(L_{\mathrm{val}}(\theta_n)\\), a single number summarizing this large parameter state. To connect the perturbation to this observable, let \\(G(\theta)\\) be the final validation loss obtained by running the remaining full-gradient updates from a post-update parameter state \\(\theta\\). For a small zero-mean kick \\(h\\), Taylor expansion gives

$$
\mathbb E[G(\theta+h)]-G(\theta)
\approx
\frac12\operatorname{Tr}\!\left[
\operatorname{Cov}(h)\nabla^2G(\theta)
\right].
$$

The linear term averages to zero; the quadratic term is proportional to the injected covariance. With \\(h=-\eta_s\xi_s\\), its prefactor is \\(\eta_s^2/B_s\\). Accumulating these responses gives

$$
\mathbb E L_{\mathrm{val}}(\theta_n)
\approx
\underbrace{F_{\mathrm{det}}(n)}_{\text{baseline learning}}
+\sum_{s<n}
\underbrace{\frac{\eta_s^2}{B_s}}_{\text{injected variance scale}}
\underbrace{\mathcal K(n,s)}_{\text{effect on the final loss}}.
$$

The baseline \\(F_{\mathrm{det}}(n)=L_{\mathrm{val}}(\bar\theta_n)\\) follows full-gradient training. To obtain \\(\mathcal K(n,s)\\), expand the final loss after all subsequent updates as a function of the injected perturbation. This includes both parameter spread and the shift of the mean trajectory caused by nonlinear updates. The resulting scalar response summarizes the noise directions, their propagation, and the loss that observes them.

This is the intuition behind **forcing–memory as a coarse-grained response law of SGD**. Training continually injects fluctuations; later dynamics determine how long they remain visible in loss. When we compress the parameter trajectory into a scalar learning curve, those accumulated effects appear as memory. Two models with the same current loss can respond differently to the next update because their hidden states retain different training histories.

The general response depends on the evolving trajectory and both times \\((n,s)\\). The next question is what makes it simple enough to calculate—and when it becomes a power law.

<details markdown="1">
<summary><strong>Derivation details: from SGD updates to the memory response</strong></summary>

The argument below uses discrete SGD throughout. Fix an initialization, a deterministic learning-rate and batch-size schedule, and a finite terminal update \\(n\\). Assume fresh independent batches, unbiased stochastic gradients, and enough smoothness and moments for the Taylor expansions and expectations below.

**1. The injected covariance.** Write \\(g(\theta;z)=\nabla_\theta\ell(f_\theta;z)\\) for a single-example gradient. The mini-batch estimate and its sampling error are

$$
\widehat g_t=\frac1{B_t}\sum_{i=1}^{B_t}g(\theta_t;z_{t,i}),
\qquad
\xi_t=\widehat g_t-\nabla\mathcal L(\theta_t).
$$

Conditioning on the history before drawing batch \\(t\\) fixes \\(\theta_t\\). Each centered sample gradient has mean zero and covariance \\(\Sigma(\theta_t)\\). Independence removes the cross-sample terms, so

$$
\mathbb E_t[\xi_t]=0,
\qquad
\mathbb E_t[\xi_t\xi_t^{\!\top}]
=\frac1{B_t^2}\sum_{i=1}^{B_t}\Sigma(\theta_t)
=\frac{\Sigma(\theta_t)}{B_t}.
$$

The parameter kick is \\(h_t=-\eta_t\xi_t\\). Hence \\(\operatorname{Cov}_t(h_t)=\eta_t^2\Sigma(\theta_t)/B_t\\) and \\(\mathbb E_t\lVert h_t\rVert^2=\eta_t^2\operatorname{Tr}\Sigma(\theta_t)/B_t\\). This explains the variance prefactor and the typical size of the parameter perturbation.

**2. Propagating one kick.** Define the full-gradient update map and its reference trajectory by

$$
D_t(\theta)=\theta-\eta_t\nabla\mathcal L(\theta),
\qquad
\bar\theta_{t+1}=D_t(\bar\theta_t),
\qquad
\bar\theta_0=\theta_0.
$$

Inject a kick only at update \\(s\\), then follow full-gradient updates. For \\(t\ge s+1\\), subtracting the reference update yields

$$
\begin{aligned}
\delta_{t+1}^{(s)}
&=D_t(\bar\theta_t+\delta_t^{(s)})-D_t(\bar\theta_t)\\
&=\left[I-\eta_t\nabla^2\mathcal L(\bar\theta_t)\right]\delta_t^{(s)}
+O\!\left(\eta_t\|\delta_t^{(s)}\|^2\right).
\end{aligned}
$$

The remainder estimate follows from a locally Lipschitz Hessian. Iterating the linear term from \\(\delta_{s+1}^{(s)}=-\eta_s\xi_s\\) gives

$$
\delta_n^{(s)}\approx-\eta_s A_{n-1}\cdots A_{s+1}\xi_s,
\qquad
A_t=I-\eta_t\nabla^2\mathcal L(\bar\theta_t).
$$

The empty product is \\(I\\) when \\(n=s+1\\). Every factor is the Jacobian of one update along the reference path; later updates multiply on the left.

**3. Measuring the effect on final loss.** Define a future-loss function

$$
G_{n,s}(\theta)
=L_{\mathrm{val}}\!\left((D_{n-1}\circ\cdots\circ D_s)(\theta)\right),
\qquad
G_{n,n}(\theta)=L_{\mathrm{val}}(\theta).
$$

It takes a parameter state just before update \\(s\\), runs the remaining full-gradient updates, and returns the final validation loss. In particular, \\(G_{n,s}=G_{n,s+1}\circ D_s\\).

For the actual SGD state, let \\(\theta_s^+=D_s(\theta_s)\\) denote the parameter state after the full-gradient step and before the perturbation, so \\(\theta_{s+1}=\theta_s^+ +h_s\\). Conditional Taylor expansion gives

$$
\begin{aligned}
&\mathbb E_s\!\left[G_{n,s+1}(\theta_s^+ +h_s)-G_{n,s+1}(\theta_s^+)\right]\\
&\quad=\frac12\mathbb E_s\!\left[
h_s^{\!\top}\nabla^2G_{n,s+1}(\theta_s^+)h_s\right]\\
&\qquad+O\!\left(\eta_s^3\mathbb E_s\|\xi_s\|^3\right)\\
&\quad=\frac{\eta_s^2}{2B_s}\operatorname{Tr}\!\left[
\Sigma(\theta_s)\nabla^2G_{n,s+1}(\theta_s^+)
\right]\\
&\qquad+O\!\left(\eta_s^3\mathbb E_s\|\xi_s\|^3\right).
\end{aligned}
$$

The first-order term vanishes because \\(\mathbb E_s[h_s]=0\\). The last equality uses \\(\mathbb E[h^{\top}Mh]=\operatorname{Tr}(M\mathbb E[hh^{\top}])\\). The \\(O(\cdot)\\) term is the higher-order Taylor remainder, with its constant controlled by the third derivatives of the future-loss function \\(G_{n,s+1}\\) along the Taylor segment.

The Hessian here belongs to the **entire future-loss function**. To see what this includes, write \\(G_{n,s+1}=L_{\mathrm{val}}\circ\Psi\\), where \\(\Psi=D_{n-1}\circ\cdots\circ D_{s+1}\\). For \\(p\\) parameters, the chain rule gives

$$
\begin{aligned}
\nabla^2G_{n,s+1}(\theta)
={}&[\mathrm D\Psi(\theta)]^{\!\top}
\nabla^2L_{\mathrm{val}}(\Psi(\theta))\,\mathrm D\Psi(\theta)\\
&+\sum_{i=1}^{p}
\partial_i L_{\mathrm{val}}(\Psi(\theta))\,\nabla^2\Psi_i(\theta).
\end{aligned}
$$

Here \\(\mathrm D\Psi\\) is the Jacobian, equal to \\(A_{n-1}\cdots A_{s+1}\\) on the reference path. The first term measures propagated parameter spread. The second accounts for the mean displacement generated by nonlinear subsequent updates. Both contribute at the same order in the injected variance.

**4. Accumulating the history.** The future-loss functions also give an exact telescoping identity along the actual noisy trajectory:

$$
\begin{aligned}
L_{\mathrm{val}}(\theta_n)-G_{n,0}(\theta_0)
&=\sum_{s<n}\left[G_{n,s+1}(\theta_{s+1})-G_{n,s}(\theta_s)\right]\\
&=\sum_{s<n}\left[
G_{n,s+1}(\theta_s^+ +h_s)-G_{n,s+1}(\theta_s^+)
\right].
\end{aligned}
$$

Take expectations and insert the conditional expansion above. With \\(F_{\mathrm{det}}(n)=G_{n,0}(\theta_0)\\), define

$$
\mathcal K(n,s)
:=\frac12\mathbb E\!\left[
\operatorname{Tr}\!\left(
\Sigma(\theta_s)\nabla^2G_{n,s+1}(D_s(\theta_s))
\right)\right].
$$

Then

$$
\mathbb E L_{\mathrm{val}}(\theta_n)
=F_{\mathrm{det}}(n)
+\sum_{s<n}\frac{\eta_s^2}{B_s}\mathcal K(n,s)
+\mathcal R_n.
$$

Here \\(\mathcal R_n\\) collects the expected higher-order remainders over all updates.

This derivation follows the full noisy history: earlier injections affect the distribution of \\(\theta_s\\), and their interactions enter through that distribution. In an expansion around the deterministic path, cross-time quadratic terms also vanish because centered gradient noises are martingale differences.

**5. What the approximation retains.** If \\(\lVert\nabla^3G_{n,s+1}\rVert\le M_{n,s}\\) along the relevant Taylor segments, the remainder satisfies

$$
|\mathcal R_n|
\le\frac16\sum_{s<n}
M_{n,s}\eta_s^3\mathbb E\|\xi_s\|^3.
$$

Dropping this higher-order remainder gives the response formula in the main text. For a fixed horizon, multiplying every noise term by a small amplitude \\(\epsilon\\) makes the remainder \\(O(\epsilon^3)\\) under uniform derivative and moment bounds, while the leading response is \\(O(\epsilon^2)\\).

The kernel \\(\mathcal K(n,s)\\) includes nonlinear propagation and the state distribution induced by the schedule. A positive power-law lag kernel is a further response model: it becomes computable in the spectral theory and is tested by the LLM surrogate's transfer across schedules.

</details>

## Where spectral modes enter

A frozen-feature model makes the propagation explicit. It predicts with \\(f_a(x)=a^{\top}\phi(x)\\), where the feature map \\(\phi\\) is fixed and only the coefficients \\(a\\) are trained. Squared loss gives a fixed curvature matrix \\(H\\). If \\(u_j\\) is an eigenvector and \\(\rho_{j,t}\\) is the perturbation along it, full-gradient training gives

$$
Hu_j=\lambda_j u_j,
\qquad
\rho_{j,t+1}=(1-\eta_t\lambda_j)\rho_{j,t}.
$$

These directions are **spectral modes**. In the stable small-step regime, large \\(\lambda_j\\) means fast relaxation and small \\(\lambda_j\\) means slow relaxation. Diagonalizing the matrix lets us follow each learning timescale separately and then add their contributions to loss.

For our Gaussian random-feature model, averaging the exact squared-error dynamics yields

$$
R_t=F_t+\sum_{s<t}K_{t,s}\bigl(R_s+\sigma^2\bigr).
$$

Here \\(R_t\\) is expected prediction risk, \\(F_t\\) propagates the initial error without stochastic feedback, and \\(K_{t,s}\\) is the response to a variance injection at update \\(s\\). The injection amplitude \\(R_s+\sigma^2\\) combines current prediction error with label-noise variance. This equation closes exactly on the scalar risk: all individual mode coordinates have been summed out.

At constant learning rate \\(\eta\\), a slow mode's squared error survives for intrinsic time \\(T=\eta t\\) with factor approximately \\(e^{-2\lambda_j T}\\). Forcing sums these responses weighted by initial target energy; memory sums them weighted by noise injection and the loss observable. The same set of learning timescales can therefore produce different forcing and memory laws.

**The spectrum determines the response. The schedule drives it. The loss curve is the output.**

## The spectral if-and-only-if criterion

At first sight, exponential spectral modes and power-law learning curves seem incompatible. The resolution is a moving cutoff.

For a constant schedule, intrinsic time is \\(T=\eta t\\). At time \\(T\\), modes with eigenvalue \\(\lambda\gg T^{-1}\\) have mostly relaxed, while modes with \\(\lambda\ll T^{-1}\\) remain largely unresolved. Training continuously sweeps this cutoff toward zero. Long-time behavior is controlled not by any single eigenvalue, but by the cumulative weighted spectral mass below \\(T^{-1}\\).

Two different weights matter:

- target-weighted spectral mass controls forcing;
- squared-spectrum mass controls one-injection memory.

Our main theorem gives the spectral equivalence

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

The equivalence holds separately for forcing and memory. It is an **if and only if** statement: low-spectrum weighted mass produces the temporal power, and the temporal power reveals the corresponding low-spectrum mass law.

The forward direction explains the emergence of the power law; the converse says that a componentwise temporal power cannot appear without the matching low-spectrum mass law.

This overturns the idea that a power-law eigenspectrum alone explains a power-law learning curve. The target can suppress slow modes so strongly that forcing decays faster than every inverse power. Conversely, irregular spectra and targets can still produce a clean temporal power through their cumulative mass. Forcing and memory can also carry different exponents because they see different spectral weights.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-forcing.png' | relative_url }}" alt="Target-weighted forcing decaying faster than every inverse power.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-memory.png' | relative_url }}" alt="One-injection memory following a three-quarter power-law tail.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-loss.png' | relative_url }}" alt="Minibatch-SGD risk crossing into the memory-controlled power-law tail.">
  </div>
  <figcaption><strong>Spectral structure determines whether a power law exists.</strong> Target-weighted forcing and its cumulative spectral mass decay faster than every inverse power (left), while one-injection memory and its spectral mass follow \(T^{-3/4}\) (middle). The minibatch-SGD loss consequently crosses from a fast forcing transient to the \(T^{-3/4}\) memory tail (right; mean over 20 runs).</figcaption>
</figure>

## From component laws to propagation regimes

The response coordinates \\((q_{\mathcal F},q_{\mathcal K})\\) separate long memory, integrable memory, and finite bulk. In the power-law random-feature specialization, these give a \\(3+3(+2)\\) phase map: a complete map of how signal and stochastic error propagate through training.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:center;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-response-map.png' | relative_url }}" alt="Long-memory, integrable-memory, and finite-bulk regimes in forcing-memory response coordinates.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig3-plrf-map.png' | relative_url }}" alt="The corresponding propagation regimes in power-law random-feature source-capacity coordinates.">
  </div>
  <figcaption><strong>One propagation structure, two coordinate systems.</strong> Left: the forcing and memory coordinates \((q_{\mathcal F},q_{\mathcal K})\) separate long-memory, integrable-memory, and finite-bulk responses. Right: the corresponding partition in microscopic source–capacity coordinates \((\alpha,\beta)\). The red band locates the fitted LLM response near the LM/IM boundary.</figcaption>
</figure>

We return below to the LLM fits that locate the red band.

## How training schedules transform the response

Once the componentwise power laws exist, plain-SGD learning-rate and batch-size schedules enter through two natural coordinates:

$$
T_t=\sum_{s<t}\eta_s,
\qquad
r_t=\frac{B_t}{\eta_t}.
$$

Intrinsic time \\(T_t\\) measures accumulated learning rate. Since \\(\Delta T_t=\eta_t\\), the variance scale in one update is \\(\eta_t^2/B_t=\Delta T_t/r_t\\). Thus \\(1/r(T)\\) controls injection per unit intrinsic time. The full ratio path matters because loss remembers earlier injections.

When the propagation is summarized by a lag kernel \\(k\\), the response takes the scaling form

$$
L(T)-L_\infty
\approx
F(T)+\int_0^T\frac{J(u)}{r(u)}k(T-u)\,\mathrm du.
$$

Here \\(L_\infty\\) is the loss floor, \\(F\\) is the remaining baseline decline, and \\(J(u)\\) measures the effective injection amplitude at time \\(u\\). The kernel \\(k(T-u)\\) weights how much of that injection remains visible after lag \\(T-u\\). This is how local SGD updates become a history-dependent learning curve.

If memory has tail \\(k(v)\sim v^{-q_{\mathcal K}}\\) and \\(r(T)\sim T^\vartheta\\), their convolution gives three outcomes. If \\(r\\) grows too slowly, fresh noise cannot be forgotten and positive-power decay is **destroyed**. At intermediate growth, the stochastic gap decays more slowly than clean loss, so the exponent is **changed**. With sufficiently fast growth, stochastic error becomes subleading and the clean exponent is **preserved**.

There is also a **memory ceiling**: reducing late-stage noise cannot erase old injections that are still remembered. Past that ceiling, making \\(B/\eta\\) grow faster no longer improves the decay exponent.

At the marginal boundary \\(q_{\mathcal K}=1\\), cumulative memory grows logarithmically, so the pure-power laws acquire logarithmic corrections.

<figure>
  <img src="{{ '/images/power-laws-have-a-clock/main-fig4-schedule-map.png' | relative_url }}" alt="Phase diagram showing when a schedule preserves, changes, or destroys a clean power law." style="display:block;width:min(100%,760px);margin-inline:auto;">
  <figcaption><strong>Schedules transform an existing response law.</strong> The growth of \(r(T)=B(T)/\eta(T)\) determines whether stochastic memory destroys, changes, or preserves the clean power. The plateau of the black boundary is the memory ceiling; the marginal case \(q_{\mathcal K}=1\) carries a logarithmic correction.</figcaption>
</figure>

The theory also gives a converse below the long-memory ceiling: the complete noisy–clean gap identifies the ratio path \\(B/\eta\\), including its slowly varying factor, but cannot identify learning rate and batch size separately. At the ceiling, even this identification is lost.

## From schedule laws to schedule design

The same response formula turns schedule design into resource allocation. Fix a terminal intrinsic time \\(T\\) and a data budget \\(D\\). Among ratio paths satisfying \\(\int_0^T r(u)\,\mathrm du=D\\), the optimal path is

$$
r_T^\star(u)
\propto
\sqrt{k(T-u)\bigl[F(u)+\sigma^2\bigr]}.
$$

This square-root law has a direct interpretation: allocate more samples where stochastic error is large when injected and likely to survive until the end of training. Learning-rate decay, batch-size growth, and their joint schedules are different implementations of this ratio path.

Optimizing the horizon and model width then produces phase-dependent data and feature-compute rates across the full propagation map. The phase diagram therefore does more than classify learning curves: it determines how training resources should be allocated.

## From theory to LLM pretraining

The general SGD argument tells us why past injections affect present loss. The spectral theory shows how that response becomes a power law in a solvable model. For LLMs, the question is whether a few aggregate response quantities remain stable enough to predict loss as the schedule changes.

We test this in three steps: change the ratio path, change its learning-rate–batch-size implementation, and predict an unseen schedule using a response fitted on another.

### Test 1: the ratio path affects loss

From one mature 30M checkpoint, eleven plain-SGD tails reached the same intrinsic-time horizon with different \\(B/\eta\\) growth rates. Faster growth systematically lowered validation loss, with diminishing gains. The ratio path—not intrinsic time alone—controls the learning curve.

### Test 2: two factorizations of the same schedule

For a 300M nanoGPT trained on 6.5B OpenWebText tokens, a shared 5.2B-token prefix branches into 1.3B-token tails. We implemented both WSD and 8-1-1 ratio paths as either a learning-rate schedule at fixed batch size or a batch-size schedule at fixed learning rate. Each macro step matches intrinsic-time advance, \\(B/\eta\\), and data. The pairs differ against optimizer step and collapse onto the same trajectories in intrinsic time.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-step.png' | relative_url }}" alt="Validation risk for matched learning-rate and batch-size schedules plotted against optimizer step.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-intrinsic-time.png' | relative_url }}" alt="The same validation trajectories plotted against intrinsic time, together with the transferred surrogate.">
  </div>
  <figcaption><strong>Matched schedule factorizations share a response in the intrinsic clock.</strong> Left: learning-rate and batch-size implementations of the same ratio paths differ against optimizer step. Right: in intrinsic time \(T=\sum_t\eta_t\), each pair collapses, and the forcing–memory prediction tracks both schedule shapes.</figcaption>
</figure>

A hybrid-Muon extension reveals the analogous optimizer-specific coordinates \\(\widetilde T=\sum_t\eta_t^2\\) and \\(\widetilde r=B/\eta^2\\), under which its paired trajectories collapse as well.

### Test 3: fit one schedule, predict another

The two-time response \\(\mathcal K(n,s)\\) summarizes how nonlinear training turns each injection into a change in loss. Our surrogate compresses it into an injection amplitude and a lag response, \\(\mathcal K(n,s)\approx J(T_s)k(T_n-T_s)\\). We model the baseline decline and the lag response by shifted powers, giving seven parameters:

$$
\widehat L(T)
=L_\infty+A_{\mathcal F}(1+T)^{-q_{\mathcal F}}
+\int_0^T
\frac{A_0+A_1(1+u)^{-q_{\mathcal F}}}{r(u)}
\bigl(1+c_{\mathcal K}(T-u)\bigr)^{-q_{\mathcal K}}
\,\mathrm du.
$$

The first two terms describe baseline learning. The integral adds the history of noise injections, weighted by how strongly training remembers them. Cross-schedule prediction tests whether these aggregate quantities transfer even while the model's parameters and representations evolve.

In the main 300M analysis, we fix \\(q_{\mathcal K}=1\\) and fit the remaining six parameters only to the raw fixed-batch 8-1-1 trajectory. We then freeze the surrogate and use it to predict the held-out WSD learning-rate trajectory—without refitting.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-fit.png' | relative_url }}" alt="Forcing-memory surrogate fitted to the 8-1-1 validation trajectory.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-transfer.png' | relative_url }}" alt="The frozen surrogate predicting the held-out WSD validation trajectory without refitting.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-profile.png' | relative_url }}" alt="Held-out WSD prediction error as a function of the fixed memory exponent.">
  </div>
  <figcaption><strong>Fit once, then predict a new schedule.</strong> Left: after fixing \(q_{\mathcal K}=1\), the remaining six parameters are fitted only to fixed-batch 8-1-1. Middle: the frozen surrogate predicts held-out WSD without refitting. Right: when \(q_{\mathcal K}\) is scanned and the other parameters are refitted only on 8-1-1, WSD prediction error is minimized near one.</figcaption>
</figure>

The frozen surrogate captures how a new schedule changes the curve. This is the predictive content of the coarse-grained response picture: changing the input history \\(r(T)\\) changes the output loss through the same fitted response.

In complementary unrestricted seven-parameter fits across model scales and datasets, we obtain:

| Dataset and setting | \\(q_{\mathcal K}\\) | \\(q_{\mathcal F}\\) |
|---|---:|---:|
| OpenWebText, 124M, 2.5B tokens | 1.017 | 0.374 |
| FineWeb sample-10BT subset, 124M, 2.5B tokens | 0.952 | 0.405 |
| peS2o V2 s2orc full text, 124M, 2.5B tokens | 0.995 | 0.365 |
| OpenWebText, 300M, 6.5B tokens | 0.989 | 0.291 |

Across web text, scientific text, model scales, and token budgets, the fitted memory coordinate stays pinned near one while the forcing coordinate remains below one. The same forcing–memory geometry repeatedly places the LLM response at the boundary between long and integrable memory.

## What the results change

- **A fitted exponent needs a mechanism:** ask which response dominates and whether the weighted low spectrum supports a stable power.
- **The clock and schedule are part of the observation:** report both with the exponent.
- **Cross-schedule prediction tests the response:** parameters learned from one input history should predict another without refitting.
- **Forcing–memory connects theory to LLMs:** the linear model gives an exact spectral description; LLM experiments test whether a compact version describes nonlinear training.

## The larger lesson

Power-law learning curves have a mechanism:

$$
\text{weighted low-spectrum geometry}
\longrightarrow
\text{forcing and memory laws}
\longrightarrow
\text{schedule-dependent accumulation}
\longrightarrow
\text{observed loss}.
$$

SGD supplies the underlying response structure: learning propagates initial error and continually injects new fluctuations. The spectral theory explains when these responses become power laws. Joint schedules control their accumulation, and their competition determines the observed decay.

The theory identifies when the component powers exist, how joint schedules preserve, change, or destroy them, and how phase structure guides resource allocation. The LLM experiments give this response picture predictive force: the right clock aligns schedule factorizations, and a response learned from one schedule predicts another. **Forcing–memory offers a common language for exact spectral theory and the observed learning curves of nonlinear models.**

---

*This post describes joint work by Yichen Wang, Fanghui Liu, and Yudong Chen.*

<!-- Add public paper and code links here when they are available. -->
