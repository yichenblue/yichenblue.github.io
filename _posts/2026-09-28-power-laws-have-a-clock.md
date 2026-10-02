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

In a linear model, individual directions can lose error exponentially. Yet a broad collection of learning speeds can combine into a much slower power-law curve, like those familiar from [language-model scaling](https://arxiv.org/abs/2001.08361). Which collections produce this behavior—and which do not?

**A power law is a phenomenon to explain, not an explanation.** Its exponent alone does not tell us whether the curve comes from remaining initial error or the memory of earlier training noise. In a random-feature model, we identify necessary and sufficient spectral conditions for these two contributions—**forcing** and **memory**—to decay as power laws. A power-law tail can persist even when forcing has already faded much faster.

This distinction also changes how we think about training schedules. Learning rate and batch size control the stream of new errors. Changing that stream can change the loss exponent, rather than merely speed up the same curve.

**Can this response structure predict an LLM learning curve it was not fitted to?** We fit a forcing–memory formula to an 8-1-1 learning-rate schedule, freeze its parameters, and use it to predict WSD. The prediction follows the measured validation loss, without using WSD loss measurements to fit the parameters.

Start with one batch: after its update is over, how long does its error remain in the model?

## Why SGD has memory

A batch's influence does not end when its update is over. It changes the parameters from which every later update starts. Two runs can reach the same current loss with different parameters and histories, then respond differently to the next update. To describe training through loss alone, we need to account for how earlier errors still affect it.

We can follow that influence without assuming a linear model. Let \\(f_\theta\\) be a model, such as a transformer, with parameters \\(\theta\\) and per-example loss \\(\ell(f_\theta;z)\\). Its training objective is

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

This motivates a working hypothesis: **forcing–memory may be an effective response law of SGD, not merely a convenient description of a linear model.** Such a coarse-grained law would predict how training history affects loss without first explaining every change inside the network. It is the loss response that needs to transfer—not the parameters, representations, or local curvature.

The calculation gives a reason for memory to appear. Whether that response has a simple, transferable form is a further question. The linear model lets us calculate it explicitly; later, the LLM experiments test its predictive reach.

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

Why can one part of an error disappear quickly while another persists? Training makes different amounts of progress in different directions. In a fixed-feature model, we can calculate these learning speeds from the spectrum, as in [Paquette et al.'s analysis of compute-optimal scaling](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1dccfc3ee01871d05e33457c61037d59-Abstract-Conference.html).

Consider a fixed-feature model \\(f_a(x)=a^{\top}\phi(x)\\), where only \\(a\\) is trained. Under squared loss, its Hessian \\(H\\) is constant. Let \\(a_\star\\) be a minimizer, \\((\lambda_j,u_j)\\) the eigenpairs of \\(H\\) with unit eigenvectors, and \\(\rho_{j,t}=u_j^\top(a_t-a_\star)\\) the error along direction \\(u_j\\). Full-gradient updates give

$$
Hu_j=\lambda_j u_j,
\qquad
\rho_{j,t+1}=(1-\eta_t\lambda_j)\rho_{j,t}.
$$

These **spectral modes** have different learning speeds. At small learning rates, directions with large positive \\(\lambda_j\\) relax quickly, while those with small \\(\lambda_j\\) retain errors for much longer. The derivation at the end of this section connects the squared-loss gradient to this recurrence and the learning time scale \\(1/\lambda_j\\).

In our random-feature model, summing the mode contributions gives an exact equation for the expected prediction risk \\(R_t\\). Here \\(F_t\\) is the contribution from initialization, \\(\sigma^2\\) is the label-noise variance, and \\(K_{t,s}\\) is the memory kernel:

$$
R_t=F_t+\sum_{s<t}K_{t,s}\bigl(R_s+\sigma^2\bigr).
$$

The factor \\(R_s+\sigma^2\\) is the noise source: it combines remaining prediction error with label noise. The kernel describes how much of that injection survives. The equation closes on the risk history, without tracking individual parameters. Compared with the general response above, the state-dependent source is now explicit, and \\(K_{t,s}\\) includes the schedule factor \\(\eta_s^2/B_s\\). The terms are regrouped: \\(F_t\\) also absorbs part of the sampling noise into the propagation of initial error, so it is not the pure full-gradient baseline \\(F(n)\\).

At constant learning rate \\(\eta\\) and batch size \\(B\\), use **intrinsic time** \\(T=\eta t\\). A slow mode's squared error decays approximately as \\(e^{-2\lambda_j T}\\). Forcing weights these decays by the initial error in each direction. Memory weights them by noise injection and their contribution to loss. The same spectrum can therefore produce different forcing and memory exponents.

**The learning speeds tell us how long errors last. The schedule tells us how much new noise is added. Together, they determine the loss curve.**

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

</details>

## When do these learning speeds produce a power law?

**Individual eigenvalues need not follow a power law. Their weighted totals near zero are what matter.** Conversely, even an exact power-law spectrum can fail to produce power-law forcing if the target barely uses its slow directions.

At intrinsic time \\(T\\), modes with \\(\lambda\gg1/T\\) have mostly decayed, while those with \\(\lambda\ll1/T\\) have barely changed. Training progressively removes faster modes, leaving slower ones behind. The remaining loss is therefore governed by the total weight near the bottom of the spectrum.

Let \\(\nu_W^{\mathcal F}((0,x])\\) be the target-weighted spectral mass in \\(0<\lambda\le x\\), and \\(\nu_W^{\mathcal K}((0,x])\\) the sum of squared eigenvalues in that interval. These weight forcing and memory, respectively; \\(W\\) denotes the random-feature realization.

Write \\(F_{W,>0}\\) for forcing with the permanent error floor removed, and \\(K_W\\) for the one-injection memory kernel at constant schedules. Their decay exponents are \\(q_{\mathcal F},q_{\mathcal K}>0\\).

Our spectral theorem gives necessary and sufficient conditions for these component powers. It considers a joint large-time, large-width limit, so progressively slower modes remain available. Informally:

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

The **if-and-only-if** result applies separately to forcing and memory: their temporal powers correspond to power laws in the relevant cumulative spectral weights.

The example below makes the distinction visible: forcing disappears faster than every inverse power, yet memory leaves a power-law tail in the loss.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-forcing.png' | relative_url }}" alt="Target-weighted forcing decaying faster than every inverse power.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-memory.png' | relative_url }}" alt="One-injection memory following a three-quarter power-law tail.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-loss.png' | relative_url }}" alt="Minibatch-SGD risk crossing into the memory-controlled power-law tail.">
  </div>
  <figcaption><strong>Forcing can disappear quickly while memory leaves a power law.</strong> Left: the target puts so little weight in slow directions that forcing falls faster than every inverse power. Middle: the effect of one noise injection falls as \(T^{-3/4}\). Right: the average SGD loss over 20 runs first drops quickly, then follows the slower \(T^{-3/4}\) memory curve.</figcaption>
</figure>

**Slow loss decay is not, by itself, evidence of slow signal learning.** A similar-looking curve can be controlled by error in the target directions or by noise that training has not yet erased. A fitted exponent is therefore a starting point, not a mechanism. How much about learning can we really infer from a scaling curve without separating these contributions?

## Three kinds of memory

Reducing new noise does not erase the noise already present. How much of its effect remains depends on the memory decay—and, in one family, on model width. This distinction gives three kinds of behavior:

- **Long memory (LM):** \\(0<q_{\mathcal K}<1\\). The cumulative memory mass diverges: old injections remain important even as the training horizon grows.
- **Integrable memory (IM):** \\(q_{\mathcal K}>1\\). The cumulative memory mass is finite: the total response to equally weighted past injections saturates.
- **Finite bulk (FB):** the size of the memory response remains tied to the number of features. A single positive decay exponent independent of model width does not describe it.

In the power-law random-feature model, comparing the forcing and memory rates gives three LM cases, three IM cases, and two FB cases: the \\(3+3(+2)\\) in the title. These describe the two ingredients of the learning curve; the schedule still determines how they combine.

To read the right-hand diagram, consider the power-law model

$$
\lambda_j=j^{-2\alpha},
\qquad
|\theta_j^\star|=j^{-\beta}.
$$

Here \\(\lambda_j\\) are the data-covariance eigenvalues, ordered from largest to smallest, and \\(\theta_j^\star\\) is the target coefficient along the corresponding direction. Larger \\(\alpha\\) makes the eigenvalues fall faster. Larger \\(\beta\\) puts less target weight in the small-eigenvalue directions that take longer to learn.

For \\(\alpha>1/4\\) and finite target energy, \\(2\alpha+2\beta>1\\), these model parameters give the response exponents

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
  <figcaption><strong>Different learning speeds lead to different kinds of memory.</strong> Left: the diagram is organized by how fast forcing and memory decay. Right: the same classification is shown using the spectrum and target parameters \((\alpha,\beta)\), with the width-dependent FB cases shown separately. The red band marks the LLM fits near \(q_{\mathcal K}=1\), between LM and IM.</figcaption>
</figure>

The red band raises a question we will return to: why do the fitted LLM responses sit near the boundary between long and integrable memory?

## How training schedules transform the response

Can a schedule change the power law itself, or only how quickly training moves along it? Our theory shows that it can change the exponent: changing learning rate or batch size changes the stream of noise that memory accumulates.

To separate progress from noise injection in plain SGD, track two quantities: intrinsic time \\(T_t\\), the sum of learning rates so far, and the ratio \\(r_t\\) of batch size to learning rate:

$$
T_t=\sum_{s<t}\eta_s,
\qquad
r_t=\frac{B_t}{\eta_t}.
$$

One update advances this clock by \\(\Delta T_t=T_{t+1}-T_t=\eta_t\\). Its noise scale is therefore \\(\eta_t^2/B_t=\Delta T_t/r_t\\). In words, \\(1/r_t\\) sets the noise scale per unit of intrinsic time. We write \\(r(T)\\) for this ratio as a function of the new clock. The whole path matters, because noise from earlier updates can still affect the current loss.

The connection between learning rate and batch size also motivates [Smith et al.'s proposal to increase batch size instead of decaying the learning rate](https://arxiv.org/abs/1711.00489). Here we ask how the full ratio path shapes the accumulation of memory.

In the PLRF model’s LM and IM regimes, the forcing–memory scaling law takes the continuum form

$$
R(T)\asymp
F(T)+
\int_0^T
\frac{F(u)+\sigma^2}{r(u)}
\,k(T-u)\,\mathrm du.
$$

Here \\(R(T)\\) is the prediction risk, and \\(F(T)\\) is the forcing profile, including the finite-width error floor. The relation holds up to multiplicative constants.

The integral combines three effects. The remaining prediction error generates minibatch noise even with clean labels, represented by \\(F(u)\\); label noise adds \\(\sigma^2\\). The ratio \\(r(u)\\) controls the strength of these injections, and \\(k(T-u)\\) describes how much of their effect survives until time \\(T\\).

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

We can also work backward. Below the long-memory ceiling, the extra loss identifies the long-time shape of \\(B/\eta\\), but not learning rate and batch size separately. At the ceiling, different ratio paths produce the same decay, so this identification is lost.

## From schedule laws to schedule design

With a fixed data budget, where should extra samples go? In the LM/IM design problem, they are most valuable where they reduce noise that would otherwise have a large effect on the final loss. The forcing–memory formula tells us where those injections occur.

Fix an ending intrinsic time \\(T\\) and a budget of \\(D\\) training examples. Since one update uses \\(B_t=r_t\Delta T_t\\) examples, the total budget becomes \\(\int_0^T r(u)\,\mathrm du=D\\).

In the theoretical response model, the optimal ratio schedule is

$$
r_T^\star(u)
\propto
\sqrt{k(T-u)\bigl[F(u)+\sigma^2\bigr]}.
$$

Here \\(F(u)\\) is the theoretical baseline risk, including the finite-feature error floor. The product combines the strength of the noise source, \\(F(u)+\sigma^2\\), with its influence on terminal loss, \\(k(T-u)\\). The rule allocates more samples where noise is both strong and persistent, with the normalization fixed by the budget \\(D\\). Learning-rate and batch-size schedules can implement the same ratio path.

We can then optimize training duration and model width. Each phase gives a best loss-versus-data rate and a best rate under compute budgets proportional to width times data. The diagram thus guides resource choices as well as explaining curves.

A complementary approach is [Bordelon and Mori's optimal-control analysis](https://arxiv.org/abs/2602.04774), which derives learning-rate schedules for a power-law random-feature model.

## From theory to LLM pretraining

**A learning-curve model should predict what happens when training changes.** A good fit shows that a formula can describe observations. Holding its parameters fixed and changing the training process asks whether the response it describes can be reused. We think this should be a central test of a learning-curve theory.

For LLMs, this tests our broader hypothesis: the loss may have a transferable response structure even while the network's internal representations evolve. We change the ratio path, change how that path is implemented, and finally predict a schedule whose loss measurements were not used for fitting.

### Test 1: the ratio path affects loss

**The same intrinsic time did not give the same loss.** Increasing \\(B/\eta\\) more quickly lowered validation loss, with diminishing gains. The ratio path mattered as well as the clock.

We tested this by continuing plain-SGD training from one mature 30M checkpoint in eleven ways. All runs ended at the same intrinsic time, but \\(B/\eta\\) grew at different rates.

### Test 2: two ways to produce the same ratio path

**Decaying the learning rate and increasing the batch size produced nearly the same validation-loss curve when matched in intrinsic time and ratio path.** The two implementations used different numbers of optimizer updates, so their curves looked different on the step axis.

For a 300M nanoGPT trained on 6.5B OpenWebText tokens, we shared a 5.2B-token training prefix and branched into 1.3B-token tails from the same checkpoint.

We compared two schedule shapes. In their fixed-batch versions, WSD keeps the learning rate constant for the first 80% of training, then lowers it gradually over the last 20%. The 8-1-1 schedule uses the same first 80%, followed by two lower-rate stages lasting 10% each.

For each schedule, we compared two factorizations: fixed batch size with varying learning rate, and fixed learning rate with varying batch size. We matched data, intrinsic-time increments, and \\(B/\eta\\) over groups of updates.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-step.png' | relative_url }}" alt="Validation risk for matched learning-rate and batch-size schedules plotted against optimizer step.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-intrinsic-time.png' | relative_url }}" alt="The same validation trajectories plotted against intrinsic time, together with the transferred surrogate.">
  </div>
  <figcaption><strong>The same ratio path gives nearly the same loss curve on the right clock.</strong> Left: changing learning rate or changing batch size gives different curves against update number. Right: adding up learning rates to form \(T=\sum_t\eta_t\) brings each pair together. The forcing–memory formula also follows both schedule shapes.</figcaption>
</figure>

We also ran a hybrid-Muon version. A short calibration selected a different clock and ratio for those runs: \\(\widetilde T=\sum_t\eta_t^2\\) and \\(\widetilde r=B/\eta^2\\). Using these quantities brought its paired curves together as well.

### Test 3: fit one schedule, predict another

**The response fitted on 8-1-1 also predicts the WSD validation curve without refitting.** This tests whether the same response formula works under a different learning-rate history.

In the main 300M analysis, we fix \\(q_{\mathcal K}=1\\) and fit the other six parameters using only the original, unsmoothed validation-loss measurements from the fixed-batch 8-1-1 run. We then keep all parameters unchanged, insert the WSD schedule, and predict its validation loss. No WSD loss measurements are used to fit those six parameters.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-fit.png' | relative_url }}" alt="Forcing-memory surrogate fitted to the 8-1-1 validation trajectory.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-transfer.png' | relative_url }}" alt="The frozen surrogate predicting the held-out WSD validation trajectory without refitting.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-profile.png' | relative_url }}" alt="Held-out WSD prediction error as a function of the fixed memory exponent.">
  </div>
  <figcaption><strong>Fit on 8-1-1, then predict WSD without fitting again.</strong> Left: fix \(q_{\mathcal K}=1\) and fit the other six parameters to 8-1-1. Middle: keep them unchanged and predict WSD. Right: repeat the procedure for different fixed values of \(q_{\mathcal K}\), always fitting only on 8-1-1. The WSD prediction is most accurate near one.</figcaption>
</figure>

The result supports a response model that carries information beyond the curve used to fit it. We would like to see learning-curve models compared on this basis: after fitting one run, how much of a changed training run can they predict without adjustment?

What is being transferred between these curves? Building on [Li et al.'s functional-scaling-law approach](https://arxiv.org/abs/2509.19189), we use a forcing–memory surrogate whose fitted parameters describe the response, while the known ratio path supplies the schedule.

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

## Why does the memory exponent keep landing near one?

**The fitted memory exponent stays close to one even when we stop fixing it there.** In the experiments below, all seven parameters are fitted freely, across different datasets and two model sizes:

| Dataset and setting | \\(q_{\mathcal K}\\) | \\(q_{\mathcal F}\\) |
|---|---:|---:|
| OpenWebText, 124M, 2.5B tokens | 1.017 | 0.374 |
| FineWeb sample-10BT subset, 124M, 2.5B tokens | 0.952 | 0.405 |
| peS2o V2 s2orc full text, 124M, 2.5B tokens | 0.995 | 0.365 |
| OpenWebText, 300M, 6.5B tokens | 0.989 | 0.291 |

These fits have \\(q_{\mathcal K}\approx1\\) and \\(q_{\mathcal F}<1\\), placing the LLM responses near the LM/IM boundary—the red band in the phase diagram. One is a meaningful value in the theory: it separates a finite cumulative memory mass from one that keeps growing, with logarithmic growth at the boundary itself.

Why should these different training tasks land near that boundary? Do they share an effective memory response, or can finite training windows and trade-offs between fitted terms pull the exponent toward one? The present curves do not decide between these explanations.

Longer held-out continuations, fits over different time windows, and direct measurements of the loss response to controlled training perturbations would help distinguish them. **The open question is whether near-one memory survives changes in how we train and how we measure it.**

## The larger lesson

We can now answer the opening puzzle. Individual directions can learn exponentially while their weighted sum follows a power law. The spectral criterion identifies when this happens; forcing and memory distinguish the initial error from the effects of past noise:

$$
\text{weight in slow directions}
\longrightarrow
\text{forcing and memory}
\longrightarrow
\text{noise added and retained}
\longrightarrow
\text{loss curve}.
$$

The linear model makes this chain calculable and provable. The LLM results motivate a broader possibility: a few response quantities may predict loss even when the underlying parameter dynamics are far more complicated. Understanding which changes preserve that description—and which break it—is a research question worth pursuing.

That is the shift we want to make: **explain where a learning curve comes from, then test that explanation by predicting how the curve changes.** Fitting a power law begins the investigation; it does not finish it.

## Further reading

- **Empirical scaling laws:** Kaplan et al. (2020), [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). How language-model loss depends on model size, data, and compute.
- **Spectral models:** Paquette et al. (2024), [4+3 Phases of Compute-Optimal Neural Scaling Laws](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1dccfc3ee01871d05e33457c61037d59-Abstract-Conference.html). A solvable model connecting data and target structure to compute-optimal regimes.
- **Learning rate and batch size:** Smith et al. (2018), [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489). Experiments replacing learning-rate decay with batch-size growth.
- **Predicting full trajectories:** Li et al. (2025), [Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](https://arxiv.org/abs/2509.19189). Intrinsic-time theory and surrogate prediction of LLM learning curves.
- **Optimal schedules:** Bordelon and Mori (2026), [Theory of Optimal Learning Rate Schedules and Scaling Laws for a Random Feature Model](https://arxiv.org/abs/2602.04774). How optimal annealing depends on the spectrum and task.

---

*This post describes joint work by Yichen Wang, Fanghui Liu, and Yudong Chen.*
