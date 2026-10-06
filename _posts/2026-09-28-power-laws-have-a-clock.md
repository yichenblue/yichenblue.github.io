---
layout: single
title: "From Spectra to Joint Schedules in LLM Pre-training: 3+3(+2) Scaling-Law Regimes"
date: 2026-09-28
permalink: /posts/power-laws-have-a-clock/
excerpt: "A power law tells us how fast loss falls. What does it tell us about learning, and why does changing the training schedule change the curve?"
tags:
  - scaling laws
  - stochastic optimization
  - language models
author_profile: false
read_time: true
reading_guide:
  # Approximate prose-reading times at the site's 160 words/minute.
  # Excludes mathematical source; optional details are counted separately.
  main_minutes: 21
  optional_minutes: 8
  sections:
    - title: Why SGD has memory
      id: why-sgd-has-memory
    - title: Different learning speeds
      id: different-directions-learn-at-different-speeds
    - title: When power laws arise
      id: when-do-these-learning-speeds-produce-a-power-law
    - title: Three kinds of memory
      id: three-kinds-of-memory
    - title: How schedules change loss
      id: how-training-schedules-transform-the-response
    - title: Designing schedules
      id: from-schedule-laws-to-schedule-design
    - title: Testing the theory in LLMs
      id: from-theory-to-llm-pretraining
    - title: Why the exponent is near one
      id: why-does-the-memory-exponent-keep-landing-near-one
    - title: The larger lesson
      id: the-larger-lesson
    - title: Further reading
      id: further-reading
toc: true
toc_sticky: true
classes: wide
header:
  teaser: /images/power-laws-have-a-clock/main-fig2-loss.png
---

*When loss follows a power law, what is the model actually learning?*

[Paper (arXiv)](https://arxiv.org/abs/2609.40148) · [Code (GitHub)](https://github.com/yichenblue/spectra-to-schedules-in-pretraining)

How much better could a language model become with ten times more compute? A hundred times? Scaling laws make these questions quantitative: measurements from smaller experiments can be used to predict the loss of training runs that are still far beyond our budget.

<figure id="scaling-law-overview" style="display:block;max-width:680px;margin:1.4em auto;">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0;background:#fff;padding:6px;border-radius:4px;">
    <!-- Preserve the original unequal panel boundaries and a shared scale when stacking. -->
    <div style="overflow:hidden;min-width:0;width:100%;max-width:260px;margin:0 auto;">
      <img src="{{ '/images/power-laws-have-a-clock/kaplan-2020-scaling-laws.svg' | relative_url }}" alt="Kaplan et al. compute scaling: pale training trajectories and their efficient frontier follow a power-law trend." style="display:block;width:283.201761%;max-width:none;height:auto;margin:0;clip-path:inset(0 64.689485% 0 0);">
    </div>
    <div style="overflow:hidden;min-width:0;width:100%;max-width:260px;margin:0 auto;">
      <img src="{{ '/images/power-laws-have-a-clock/kaplan-2020-scaling-laws.svg' | relative_url }}" alt="Kaplan et al. dataset scaling: test loss decreases regularly as dataset size increases." style="display:block;width:283.201761%;max-width:none;height:auto;margin:0;clip-path:inset(0 31.599757% 0 35.310515%);transform:translateX(-34.200121%);">
    </div>
    <div style="overflow:hidden;min-width:0;width:100%;max-width:260px;margin:0 auto;">
      <img src="{{ '/images/power-laws-have-a-clock/kaplan-2020-scaling-laws.svg' | relative_url }}" alt="Kaplan et al. parameter scaling: test loss follows an approximate power law in non-embedding parameter count." style="display:block;width:283.201761%;max-width:none;height:auto;margin:0;clip-path:inset(0 0 0 68.400243%);transform:translateX(-66.544864%);">
    </div>
  </div>
  <figcaption><strong>Regular scaling across compute, data, and model size.</strong> <a href="https://arxiv.org/html/2001.08361v1#S1.F1">Kaplan et al. (2020), Figure 1</a>. Left: the small-batch-adjusted compute frontier across models, not one training trajectory.</figcaption>
</figure>

Look at how much variation these plots compress. Model size, data, and compute span orders of magnitude, yet the losses arrange themselves along simple trends. The attraction of a scaling law is not just that it fits the points. It turns a collection of completed experiments into a map of experiments we have not run.

That map changes how we spend compute. Should the next run use a larger model, or should a smaller model see more tokens? How far should either be trained? These are the kinds of choices behind [compute-optimal training](https://arxiv.org/abs/2203.15556). A fitted curve can inform a decision long before the full training budget is spent.

But how far can we trust that extrapolation?

Look at the frontier below. Each training configuration offers a different trade-off between cost and performance; the lower envelope traces the best results available at each compute budget. Across the middle of the plot, that envelope follows a remarkably straight path. At both ends, it bends.

<figure id="vision-scaling-frontier" style="display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:20px;max-width:620px;margin:1.4em auto;">
  <a href="{{ '/images/power-laws-have-a-clock/zhai-2022-imagenet-finetune.svg' | relative_url }}" aria-label="View the full-size ImageNet finetuning panel from Zhai Figure 2" style="display:block;flex:0 1 260px;min-width:0;background:#fff;padding:6px;border-radius:4px;">
    <img src="{{ '/images/power-laws-have-a-clock/zhai-2022-imagenet-finetune.svg' | relative_url }}" alt="Zhai Figure 2, left: ImageNet finetuning error versus training compute. The dashed Pareto-frontier fit bends toward saturation outside its middle power-law region." style="display:block;width:100%;height:auto;margin:0;">
  </a>
  <figcaption style="flex:1 1 220px;min-width:0;margin:0;"><strong>A power-law region, not an endless straight line.</strong> ImageNet finetuning error versus training compute; the fitted frontier bends toward saturation at both ends. Left panel from <a href="https://arxiv.org/html/2106.04560v2#S1.F2">Zhai et al. (2022), Figure 2 — view full figure</a>.</figcaption>
</figure>

These [vision-model experiments](https://arxiv.org/html/2106.04560v2#S2.SS2) reveal something that a single fitted exponent can hide. Power-law behavior can describe a broad and useful range without describing the whole curve. At low compute, even a simple predictor performs better than extrapolating the line would suggest. At high compute, further improvements shrink as the error approaches a nonzero floor.

The interesting question is therefore not simply whether scaling laws work. It is why a power-law region appears at all. What determines its slope? What sets its boundaries? And how much of the curve comes from the learning problem itself, rather than the way we train?

To explore that last question, we need to distinguish two different journeys on these plots. One follows a single model as training continues. The other follows the best result available at each compute budget, choosing among models and training configurations. Both can trace power laws. The second is a frontier built from the first; it is not simply one training curve drawn farther to the right.

But even at a fixed model size, which slope should we expect? [Mircea et al.](https://arxiv.org/html/2506.05447v1#A3.SS3) compared separate training runs of the same architecture under two learning-rate schedules: warmup followed by a constant learning rate, and warmup followed by cosine decay. With the other settings held fixed, cosine decay changed not only the final loss but also the fitted power-law exponent of the later training phase.

<figure id="learning-rate-scaling-exponents" style="display:block;max-width:620px;margin:1.4em auto;">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;background:#fff;padding:8px;border-radius:4px;">
    <div style="min-width:0;text-align:center;">
      <p style="margin:0 0 8px;font-size:0.85em;font-weight:600;">Warmup + constant LR</p>
      <a href="https://arxiv.org/html/2506.05447v1#S2.F2" aria-label="View Figure 2 in Mircea et al.">
        <img src="{{ '/images/power-laws-have-a-clock/mircea-2025-constant-lr.svg' | relative_url }}" alt="Original Figure 2: log–log training-loss curves and broken-power-law fits, with constant learning rate after warmup for the 14M–472M models." style="display:block;width:100%;height:330px;object-fit:contain;object-position:50% 0;margin:0;">
      </a>
    </div>
    <div style="min-width:0;text-align:center;">
      <p style="margin:0 0 8px;font-size:0.85em;font-weight:600;">Warmup + cosine decay</p>
      <a href="https://arxiv.org/html/2506.05447v1#A3.SS3" aria-label="View Figure 29 and the learning-rate comparison in Mircea et al.">
        <img src="{{ '/images/power-laws-have-a-clock/mircea-2025-cosine-lr.svg' | relative_url }}" alt="Original Figure 29: log–log training-loss curves and broken-power-law fits with cosine learning-rate decay; the 14M–472M models have steeper fitted late-training slopes." style="display:block;width:100%;height:330px;object-fit:contain;object-position:50% 0;margin:0;">
      </a>
    </div>
  </div>
  <figcaption><strong>Changing the schedule changes the fitted exponent.</strong> <a href="https://arxiv.org/html/2506.05447v1#A3.SS3">Mircea et al. (2025), Figures 2 and 29</a>. Compare 14M–472M; OLMo-1B/7B are external references. Fits use raw training loss over a finite window with zero offset, not floor-subtracted loss.</figcaption>
</figure>

Compare the late-training slopes for the same model across the two panels. For 144M, the fitted exponent rises from **0.023 to 0.036**; for 285M, from **0.025 to 0.040**; for 472M, from **0.035 to 0.045**. The curves do not merely move downward: their fitted slopes become steeper. The model and dataset have not changed, yet the apparent scaling law has. How much of an exponent belongs to the learning problem, and how much belongs to the way we train?

This is where “universality” becomes an interesting question rather than a label. [A mechanism shared across models](https://arxiv.org/abs/2606.25008) would need to explain both the regularity and the changes: how [structure in the data and the task](https://arxiv.org/abs/2210.16859) combines with the training procedure to produce the loss curve we see.

A training schedule gives us a direct way to investigate this question. Instead of only fitting the curve we already have, we can ask what happens when we change the learning rate or batch size. **What if the transferable object is not a fixed exponent, but a rule that predicts how the curve changes?**

To look for that rule, zoom in from the whole curve to one stochastic gradient descent (SGD) update. Its gradient comes from a sampled batch, so it contains a sampling error. That error changes the parameters from which the next update begins, and the next, and the next. A loss measured much later still carries the effects of those earlier updates.

## Why SGD has memory

That history is stored in the parameters, but a loss value hides most of it. Two runs with the same current loss can respond differently to the next update. Describing training through loss alone therefore requires accounting for its history.

For a model \\(f_\theta\\) with training objective \\(\mathcal L\\), let \\(\eta_t\\) be the learning rate, \\(B_t\\) the batch size, and \\(\xi_t\\) the batch-gradient error. With fresh i.i.d. examples, SGD takes the form

$$
\theta_{t+1}
=\theta_t-\eta_t\nabla\mathcal L(\theta_t)-\eta_t\xi_t,
\qquad
\mathbb E_t[\xi_t]=0,
\qquad
\operatorname{Cov}_t(\xi_t)=\frac{\Sigma(\theta_t)}{B_t}.
$$

Here expectations are conditional on the training history, and \\(\Sigma\\) is the single-example gradient covariance. The parameter perturbation \\(-\eta_t\xi_t\\) therefore has variance scale \\(\eta_t^2/B_t\\): a smaller learning rate or a larger batch reduces its size. What matters for a later loss is how subsequent updates propagate that perturbation.

A small-noise expansion lets us add up these effects. Write \\(F(n)\\) for the noise-free validation loss and \\(K(n,s)\\) for the response at update \\(n\\) to noise injected at update \\(s\\), after removing its variance scale:

$$
\mathbb E L_{\mathrm{val}}(\theta_n)
\approx
\underbrace{F(n)}_{\text{baseline learning}}
+\sum_{s<n}
\underbrace{\frac{\eta_s^2}{B_s}}_{\text{injected variance scale}}
\underbrace{K(n,s)}_{\text{effect on the final loss}}.
$$

The first term describes learning without batch noise; the sum collects its lasting effects. This is the starting point of the **forcing–memory** description.

Nothing in this expression yet makes the response a power law. To understand its shape, we need to know how quickly training removes an error.

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

Training removes errors faster in some directions than others. A model with fixed features and trainable output weights lets us calculate those speeds explicitly, as in [Paquette et al.'s analysis of compute-optimal scaling](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1dccfc3ee01871d05e33457c61037d59-Abstract-Conference.html).

Consider \\(f_a(x)=a^\top\phi(x)\\), with fixed features \\(\phi\\), trainable weights \\(a\\), and squared loss. Let \\(\rho_{j,t}\\) be the parameter error along a Hessian eigenvector with eigenvalue \\(\lambda_j\\). Under small full-gradient steps,

$$
\rho_{j,t}\approx\rho_{j,0}e^{-\lambda_jT_t},
\qquad
T_t=\sum_{s<t}\eta_s.
$$

The clock \\(T_t\\) is **intrinsic time**. Large positive eigenvalues mean fast learning; small ones mean persistent errors, with learning time proportional to \\(1/\lambda_j\\).

Loss adds up weighted squared errors across these directions. A slow direction contributes little if there was almost no error there to begin with. Learning speed and importance to the loss are different things.

In this model, forcing follows error inherited from initialization; memory follows error introduced by batch sampling. They involve the same decay speeds but different weights, because the initial error and batch noise need not lie in the same directions.

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

Suppose the target has almost no component along a direction that is slow to learn. There is then little signal to learn in that direction. But batch sampling can still introduce errors there, and those errors take a long time to disappear.

In the example below, the spectrum follows a power law but the target barely uses its slow directions. Forcing disappears faster than every inverse power, yet SGD loss retains a power-law tail. What remains is the memory of batch noise.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-forcing.png' | relative_url }}" alt="Target-weighted forcing decaying faster than every inverse power.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-memory.png' | relative_url }}" alt="One-injection memory following a three-quarter power-law tail.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-loss.png' | relative_url }}" alt="Minibatch-SGD risk crossing into the memory-controlled power-law tail.">
  </div>
  <figcaption><strong>Fast forcing decay, slow memory tail.</strong> Left: forcing vanishes faster than every inverse power. Middle: one-injection memory decays as \(T^{-3/4}\). Right: SGD loss, averaged over 20 runs, eventually follows that memory tail.</figcaption>
</figure>

**Slow loss decay need not mean slow signal learning.** Forcing and memory use the same decay speeds, but assign different weights to them. A fitted loss exponent alone does not tell us which contribution controls the curve.

As training continues, fast directions fade and slower ones take over. At time \\(T\\), the remaining contribution comes mainly from directions with \\(\lambda\lesssim1/T\\). We therefore need to add up their weights separately for forcing and memory.

For a fixed random-feature matrix \\(W\\), let \\(\widehat\lambda_j\\) and \\(\widehat u_j\\) be the eigenvalues and unit eigenvectors of \\(\Lambda^{1/2}WW^\top\Lambda^{1/2}\\), where \\(\Lambda\\) is the input covariance. With target parameter \\(\theta^\star\\), the two cumulative weights are

$$
\begin{aligned}
\nu_W^{\mathcal F}((0,x])
&=\sum_{0<\widehat\lambda_j\le x}
\left|\left\langle\widehat u_j,\Lambda^{1/2}\theta^\star\right\rangle\right|^2,\\
\nu_W^{\mathcal K}((0,x])
&=\sum_{0<\widehat\lambda_j\le x}\widehat\lambda_j^2.
\end{aligned}
$$

Write \\(F_{W,>0}\\) for forcing above its floor and \\(K_W\\) for the constant-schedule memory kernel. In the random-feature model's large-width, long-time limit, these weights and component decay laws determine one another. Schematically,

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

The positive exponents \\(q_{\mathcal F}\\) and \\(q_{\mathcal K}\\) describe forcing and memory separately. **This is a criterion, not just a construction:** each component's power-law decay also constrains its weighted spectrum near zero. Individual eigenvalues need not follow a power law; their weighted totals are what matter.

## Three kinds of memory

If every perturbation fades, can we eventually ignore the distant past? The memory curve above follows one injection, but SGD adds noise at every step. Those responses accumulate.

Write \\(k(v)\\) for the memory response after a delay \\(v\\). With equally weighted injections, the accumulated response can behave in two ways:

- The responses fade fast enough for their total to approach a finite value. This is **integrable memory (IM)**, with \\(q_{\mathcal K}>1\\).
- Each response fades, but too slowly for the total to settle. This is **long memory (LM)**, with \\(0<q_{\mathcal K}<1\\). Cumulative memory keeps growing with the horizon.

At the boundary, \\(q_{\mathcal K}=1\\), cumulative memory grows logarithmically. No individual perturbation has to be permanent for the total to keep growing.

**Finite bulk (FB)** needs separate treatment: the response remains tied to model width, rather than a width-independent positive decay exponent. Comparing memory with forcing gives three LM cases, three IM cases, and two FB cases: the \\(3+3(+2)\\) in the title.

The two diagrams connect these behaviors to the learning problem. The left uses response exponents; the right uses a concrete spectrum and target:

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

So far, we have added equally weighted responses. A training schedule changes the weights: it can make later noise injections weaker than earlier ones. How much can that change the loss curve?

## How training schedules transform the response

Learning-rate decay often produces a striking change in the training curve: loss that was falling slowly begins to drop much more sharply. In [warmup–stable–decay (WSD) training](https://arxiv.org/abs/2404.06395), the learning rate stays constant after warmup until a final decay phase. Follow the gray-green curves in the MiniCPM experiment below: their gradual decline gives way to a sharp drop when decay begins. The same pattern appears at several different stages of training.

<figure style="display:block;max-width:680px;margin:2em auto;">
  <img src="{{ '/images/power-laws-have-a-clock/hu-2024-minicpm-wsd-loss.svg' | relative_url }}" alt="Loss on C4 versus training tokens for a 0.036B model: gray-green WSD curves show sharp drops during decay at several training stages; the orange curve uses a cosine schedule." style="display:block;width:100%;height:auto;margin:0 auto 1rem;">
  <figcaption><strong>A sharp loss drop at different stages of training.</strong> The gray-green curves use WSD with different decay timings and durations; orange shows cosine. Token counts are expressed as multiples of the model's parameter count, \(N\). Figure reproduced unchanged from <a href="https://arxiv.org/html/2404.06395v3#S4.F6.fig1">Hu et al. (2024), MiniCPM, Figure 5</a>, under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</figcaption>
</figure>

These downward turns make it look as though lowering the learning rate has made the model learn faster. But a faster fall in loss need not mean faster learning of the target: reducing the error caused by stochastic updates also lowers loss. Which effect explains the drop? And if annealing helps, why can't a more aggressive schedule keep making loss fall faster?

To separate the two effects, start with plain SGD and return to intrinsic time. Lowering the learning rate both shortens the full-gradient update and reduces its noise; increasing batch size reduces noise without shortening that update. The relevant clock and batch-to-learning-rate ratio are

$$
T_t=\sum_{s<t}\eta_s,
\qquad
r_t=\frac{B_t}{\eta_t}.
$$

An update advances the clock by \\(\eta_t\\) and injects variance at scale \\(\eta_t^2/B_t\\). Thus \\(1/r_t\\) is the noise scale per unit of intrinsic time. Write \\(r(T)\\) for the ratio along this clock.

This connects [increasing batch size with decaying learning rate](https://arxiv.org/abs/1711.00489): both reduce noise per unit of intrinsic time. Since earlier noise can still affect loss, we must match the whole ratio path, not just its final value.

The forcing–memory relation keeps track of what each injection leaves behind. In the power-law random-feature model's LM and IM regimes,

$$
R(T)\asymp
F(T)+
\int_0^T
\frac{F(u)+\sigma^2}{r(u)}
\,k(T-u)\,\mathrm du.
$$

Here \\(R\\) is prediction risk, \\(F\\) is forcing including its finite-width floor, and \\(\sigma^2\\) is label-noise variance. Remaining prediction error generates batch noise even with clean labels; label noise adds to it. The ratio \\(r(u)\\) controls the strength of these injections, and \\(k(T-u)\\) describes how much of their effect survives until \\(T\\). The relation holds up to multiplicative constants.

To see the limit of noise reduction, compare loss with label noise against the clean-label loss. Faster growth of \\(r(T)\\) suppresses recent injections. Once earlier injections dominate this extra loss, its decay is set by memory. **This is the memory ceiling:** beyond it, faster growth of \\(B/\eta\\) no longer improves the noise-decay exponent.

For power-law memory and ratio growth, comparing this extra loss with the clean loss gives three outcomes:

- **Destroy:** the extra loss does not decay as a positive power.
- **Change:** it decays more slowly than the clean loss and sets a new exponent.
- **Preserve:** it decays at least as fast, leaving the clean exponent unchanged.

<figure>
  <img src="{{ '/images/power-laws-have-a-clock/main-fig4-schedule-map.png' | relative_url }}" alt="Phase diagram showing when a schedule preserves, changes, or destroys a clean power law." style="display:block;width:min(100%,570px);height:auto;margin-inline:auto;">
  <figcaption><strong>Preserve, change, or destroy.</strong> The ratio-growth exponent \(\vartheta\), defined by \(r(T)\sim T^\vartheta\), controls the outcome. The flat boundary marks the memory ceiling; \(q_{\mathcal K}=1\) introduces logarithmic corrections.</figcaption>
</figure>

**A schedule can change the loss exponent, not just the speed of training.** But suppressing noise more aggressively eventually stops improving that exponent: the past still has to fade.

## From schedule laws to schedule design

A ceiling on the exponent still leaves a practical choice: with a fixed data budget and ending intrinsic time \\(T\\), where should we spend extra samples? They help most where the noise they suppress would otherwise persist until the end.

In the LM/IM response model, balancing source strength against this future effect gives the optimal ratio

$$
r_T^\star(u)
\propto
\sqrt{k(T-u)\bigl[F(u)+\sigma^2\bigr]}.
$$

The data budget fixes the normalization. The square root balances noise generated now against its future effect: some early perturbations will have faded by the end, while others remain influential. Current loss alone does not capture that difference. [Bordelon and Mori's optimal-control analysis](https://arxiv.org/abs/2602.04774) offers a complementary route to learning-rate design.

Learning-rate and batch-size schedules can implement the same ratio path. Optimizing training duration and model width then gives phase-dependent data and compute scaling laws. This connects the two scales in the opening: individual learning curves determine what can be achieved by choosing across models.

## From theory to LLM pretraining

An LLM keeps changing its representations, unlike the fixed-feature model above. But describing its loss need not require reproducing every change in its parameters.

This leaves room for a broader possibility: **forcing–memory may be a coarse-grained response law of SGD, not merely a description of a linear model.** Different parameter trajectories could produce similar loss responses. Changing the training schedule lets us test whether that response transfers.

### Two implementations of the same ratio path

First, keep the ratio path the same but implement it differently: lower learning rate at fixed batch size in one run, and increase batch size at fixed learning rate in another. Match their clocks and ratio paths.

We did this for a 300M plain-SGD LLM with two schedule shapes: gradual decay (WSD) and two successive drops (8-1-1). If intrinsic time and the ratio path capture the main schedule dependence, each pair should follow nearly the same loss curve.

<figure style="display:block;">
  <div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-end;width:100%;margin-bottom:1rem;">
    <div style="flex:1 1 280px;min-width:0;">
      <img src="{{ '/images/power-laws-have-a-clock/main-fig1-step.png' | relative_url }}" alt="Validation risk for matched learning-rate and batch-size schedules plotted against optimizer step." style="display:block;width:100%;height:auto;margin:0;">
    </div>
    <div style="flex:1.26 1 352.8px;min-width:0;">
      <img src="{{ '/images/power-laws-have-a-clock/main-fig1-intrinsic-time.png' | relative_url }}" alt="The same validation trajectories plotted against intrinsic time, together with the transferred surrogate." style="display:block;width:100%;height:auto;margin:0;">
    </div>
  </div>
  <figcaption><strong>Different step counts, nearly the same intrinsic-time curves.</strong> Paired learning-rate and batch-size implementations separate against optimizer step (left) and align against \(T=\sum_t\eta_t\) (right).</figcaption>
</figure>

That is what we see. The paired curves separate against optimizer step, but nearly coincide against intrinsic time. Changing learning rate or batch size produces much the same loss response when the clock and ratio path are matched.

### Fit one schedule and predict another

The paired runs share a ratio path. A harder test is to change it: can a response learned from the abrupt drops of 8-1-1 predict the gradual decline of WSD?

For that test we need a concrete version of the forcing–memory response. Building on [Li et al.'s functional-scaling-law approach](https://arxiv.org/abs/2509.19189), we use the seven-parameter surrogate

$$
\widehat L(T)
=L_\infty+A_{\mathcal F}(1+T)^{-q_{\mathcal F}}
+\int_0^T
\frac{A_0+A_1(1+u)^{-q_{\mathcal F}}}{r(u)}
\bigl(1+c_{\mathcal K}(T-u)\bigr)^{-q_{\mathcal K}}
\,\mathrm du.
$$

The first two terms describe baseline loss; the integral adds earlier noise weighted by what survives. The parameters describe the response, while \\(r(u)\\) supplies the schedule.

We fix \\(q_{\mathcal K}=1\\) and fit the other six parameters to raw validation loss from the fixed-batch 8-1-1 run. Then we freeze every parameter and replace only the ratio path with WSD's. No WSD loss is used in this fit.

The predicted curve follows WSD's measured validation loss, including its differently timed decline. It changes because the input schedule changes, not because we adjusted the response to fit another run.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-fit.png' | relative_url }}" alt="Forcing-memory surrogate fitted to the 8-1-1 validation trajectory.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-transfer.png' | relative_url }}" alt="The frozen surrogate predicting the held-out WSD validation trajectory without refitting.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-profile.png' | relative_url }}" alt="Held-out WSD prediction error as a function of the fixed memory exponent.">
  </div>
  <figcaption><strong>8-1-1 fit (left), zero-refit WSD prediction (middle).</strong> Right: repeat for different fixed \(q_{\mathcal K}\), fitting the other parameters only on 8-1-1. WSD prediction is most accurate near one.</figcaption>
</figure>

This is a more revealing test than fitting the two curves separately. **A learning-curve model should earn our confidence by predicting what happens when training changes.** Here the same response describes two different histories of noise injection.

<details markdown="1">
<summary><strong>Experiment and surrogate details</strong></summary>

**Training and matching.** The preliminary 30M experiment continued one mature plain-SGD checkpoint under eleven ratio paths, all ending at the same intrinsic time. The main 300M nanoGPT experiment used 6.5B OpenWebText tokens: a shared 5.2B-token prefix followed by 1.3B-token tails. For each schedule, we compared fixed batch size with varying learning rate against fixed learning rate with varying batch size, matching data, intrinsic-time increments, and \\(B/\eta\\) over groups of updates.

In their fixed-batch versions, WSD keeps learning rate constant for the first 80% of training and lowers it gradually over the last 20%. The 8-1-1 schedule shares the first 80%, then uses two lower-rate stages lasting 10% each.

**Surrogate parameters.** The seven parameters are the floor \\(L_\infty\\), forcing amplitude \\(A_{\mathcal F}\\), source amplitudes \\(A_0,A_1\\), memory time-scale parameter \\(c_{\mathcal K}\\), and exponents \\(q_{\mathcal F},q_{\mathcal K}\\). The approximation separates the effective response as \\(K(n,s)\approx J(T_s)k(T_n-T_s)\\), where \\(J\\) is the source amplitude and \\(k\\) the propagation kernel.

With \\(q_{\mathcal K}=1\\), the other six parameters are fitted only to original, unsmoothed 8-1-1 measurements; no WSD loss enters that fit. The profile plot repeats this procedure for each fixed exponent and evaluates the resulting WSD prediction error.

**Hybrid Muon.** A short calibration selected a different clock and ratio for those runs: \\(\widetilde T=\sum_t\eta_t^2\\) and \\(\widetilde r=B/\eta^2\\). Those calibrated quantities brought its paired curves together as well.

</details>

## Why does the memory exponent keep landing near one?

We fixed the memory exponent at one in that test. What happens if we let the fit choose it?

In separate fits, we let all seven parameters vary. Across different datasets and two model sizes, the memory exponent comes back close to one:

| Dataset and setting | \\(q_{\mathcal K}\\) | \\(q_{\mathcal F}\\) |
|---|---:|---:|
| OpenWebText, 124M, 2.5B tokens | 1.017 | 0.374 |
| FineWeb sample-10BT subset, 124M, 2.5B tokens | 0.952 | 0.405 |
| peS2o V2 s2orc full text, 124M, 2.5B tokens | 0.995 | 0.365 |
| OpenWebText, 300M, 6.5B tokens | 0.989 | 0.291 |

One lies at the LM/IM boundary: cumulative memory is finite on one side and keeps growing on the other, logarithmically at one. The fitted responses sit close to that boundary across these different tasks.

Do these tasks share an effective distribution of memory times? Or do finite training windows and trade-offs between fitted parameters pull the estimates toward the same value? The present curves do not distinguish those explanations. Changing the training horizon and measuring the response to deliberate perturbations would help tell them apart.

## The larger lesson

A power-law curve can look deceptively complete: fit the slope, extrapolate, and the story seems finished. Yet the same shape can arise from learning remaining signal or from the persistence of earlier noise. Those explanations matter because they imply different responses when training changes.

That makes a schedule change more than a training choice. It is a way to ask what the curve has been hiding. If an explanation can follow the loss through that change without being fitted again, it tells us something that the original slope alone could not.

## Further reading

- **Empirical scaling laws:** Kaplan et al. (2020), [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). How language-model loss depends on model size, data, and compute.
- **Spectral models:** Paquette et al. (2024), [4+3 Phases of Compute-Optimal Neural Scaling Laws](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1dccfc3ee01871d05e33457c61037d59-Abstract-Conference.html). A solvable model connecting data and target structure to compute-optimal regimes.
- **Learning rate and batch size:** Smith et al. (2018), [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489). Experiments replacing learning-rate decay with batch-size growth.
- **Predicting full trajectories:** Li et al. (2025), [Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](https://arxiv.org/abs/2509.19189). Intrinsic-time theory and surrogate prediction of LLM learning curves.
- **Optimal schedules:** Bordelon and Mori (2026), [Theory of Optimal Learning Rate Schedules and Scaling Laws for a Random Feature Model](https://arxiv.org/abs/2602.04774). How optimal annealing depends on the spectrum and task.

---

*This post describes joint work by Yichen Wang, Fanghui Liu, and Yudong Chen.*
