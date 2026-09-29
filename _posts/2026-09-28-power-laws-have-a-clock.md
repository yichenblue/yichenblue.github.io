---
layout: single
title: "Power Laws Have a Clock: How Training Schedules Reshape Learning Curves"
date: 2026-09-28
permalink: /posts/power-laws-have-a-clock/
excerpt: "How intrinsic time, spectral memory, and joint learning-rate/batch-size schedules shape learning curves."
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
  teaser: /images/power-laws-have-a-clock/main-fig1-intrinsic-time.png
---

*Why cumulative learning rate and \\(B/\eta\\) organize controlled LLM training curves—and what spectra have to do with it*

Power-law learning curves are often treated as fingerprints of a model and its data. Fit an exponent early, the usual story goes, and extrapolate the rest of training.

Consider two 300M nanoGPT runs that start from the same mature checkpoint and consume the same future data. One decays the learning rate at fixed batch size; the other changes batch size at fixed learning rate. The pair is constructed to match the full batch-size-to-learning-rate ratio path.

Against optimizer step, their validation-loss curves look different. Against cumulative learning rate, they nearly coincide.

The same paired schedules tell different stories until they are plotted in the clock used by the dynamics.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-step.png' | relative_url }}" alt="Validation risk for matched learning-rate and batch-size schedules plotted against optimizer step.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-intrinsic-time.png' | relative_url }}" alt="The same validation trajectories plotted against intrinsic time, together with the transferred surrogate.">
  </div>
  <figcaption><strong>The clock reveals the matched response.</strong> Left: learning-rate and batch-size factorizations of the same ratio schedules follow different trajectories against optimizer step. Right: in a 300M plain-SGD nanoGPT run, each pair nearly coincides in intrinsic time \(T=\sum_t\eta_t\), while a forcing–memory surrogate fitted on 8-1-1 transfers to WSD. The collapse is an approximate response-level observation, not an exact identity for end-to-end LLMs.</figcaption>
</figure>

This is more than a plotting trick. In these runs, the schedule changes the validation-loss trajectory, while the clock determines whether the matched structure is visible. The model and dataset alone therefore do not specify the observed learning curve.

Our paper develops a theory around this observation. The short version is:

1. Weighted low-spectrum structure creates two response components: **forcing** and **memory**.
2. For plain SGD, cumulative learning rate measures optimization progress, while the ratio between batch size and learning rate controls stochastic-error injection per unit progress.
3. A schedule accumulates past injections through memory. It can therefore preserve, change, or destroy the power law visible in the total loss.

The theory is derived in a tractable random-feature model. Outside that model, we test two falsifiable consequences: whether these coordinates organize the tested end-to-end plain-SGD nanoGPT curves, and whether a forcing–memory surrogate fitted on one schedule predicts held-out schedules without refitting.

## Two coordinates for a joint training schedule

Learning rate and batch size appear to be two independent knobs. For plain SGD, however, their leading schedule roles combine into two more natural coordinates:

$$
T_t=\sum_{s<t}\eta_s,
\qquad
r_t=\frac{B_t}{\eta_t}.
$$

The first quantity, **intrinsic time** \\(T_t\\), measures accumulated optimization progress. The second, the **noise-control ratio** \\(r_t\\), determines stochastic-error injection per unit intrinsic time: larger \\(B_t/\eta_t\\) means less injected noise.

The relevant object is not only the terminal ratio, but the whole path \\(r(T)\\). Two runs may finish with the same ratio yet expose the dynamics to different streams of earlier noise.

Suppose we prescribe a ratio path \\(r(T)\\). We can realize it by keeping batch size fixed and varying learning rate, or by keeping learning rate fixed and varying batch size. The two implementations may require different numbers of optimizer updates. In the proxy's controlled continuum regime, matching the intrinsic-time increments and the ratio path gives the same leading response; in the tested plain-SGD LLM runs, this invariance is approximate.

This does **not** say that learning rate and batch size are operationally identical. It says that their leading roles in this plain-SGD response are organized by \\(T\\) and \\(B/\eta\\), rather than by either hyperparameter in isolation.

## Two questions hidden in one scaling law

This leaves two separate questions. **Origin:** when do the dynamics produce power-law response components at all? **Transfer:** once those components exist, when does a schedule preserve, change, or destroy the law visible in loss above its irreducible floor? A stable mechanism need not imply a stable observed exponent.

## Two responses: unresolved signal and remembered noise

In our proxy model, a frozen random-feature representation is trained by noisy online SGD. Conditional on that representation, the prediction risk satisfies an exact recurrence with two central objects.

**Forcing** is the target error that remains unresolved. It describes how the initial signal decays when propagated through the frozen dynamics.

**Memory** is the impulse response of stochastic error. It asks: if SGD injects noise now, how much of that injection will still be visible later?

Suppressing finite-width and feedback details, the resulting picture has the schematic form

$$
L(T)-L_\infty
\;\approx\;
F(T)
+\int_0^T
\frac{\text{injection scale at }u}{r(u)}
\,k(T-u)\,\mathrm du.
$$

The first term is unresolved signal. The integral adds the surviving effect of all past stochastic injections. The function \\(k(T-u)\\) tells us how much an injection of age \\(T-u\\) is remembered.

This immediately explains why the observed exponent need not equal a single spectral exponent. The final loss is produced only after clean learning and accumulated stochastic memory have competed with each other.

## How exponential modes add up to a power law

Every fixed spectral mode decays exponentially. A power law emerges because an infinite collection of modes contains many different time scales.

At intrinsic time \\(T\\), modes with eigenvalue \\(\lambda\gg T^{-1}\\) have largely relaxed, while modes with \\(\lambda\ll T^{-1}\\) remain almost untouched. Training therefore sweeps a moving cutoff through the spectrum. Long-time behavior is determined by how much weighted spectral mass remains below \\(T^{-1}\\).

There are two relevant spectral weights:

- target-weighted mass, which determines forcing;
- squared-spectrum mass, which determines one-injection memory.

Our spectral criterion can be summarized as

$$
\text{weighted mass below }x\propto x^q
\quad\Longleftrightarrow\quad
\text{temporal response at }T\propto T^{-q}.
$$

The equivalence holds separately for forcing and memory. This distinction matters: target alignment controls forcing, while memory depends on how stochastic variance is stored across spectral directions.

The criterion also changes how one should interpret “power-law data.” A power-law eigenspectrum alone is insufficient: forcing can decay faster than every inverse power if the target places too little energy in slow directions. Conversely, individual eigenvalues and target coefficients do not need to follow clean power laws. Irregular microscopic structure can still have cumulative weighted mass near zero that produces a componentwise temporal power law.

A controlled construction makes the separation concrete: forcing can disappear faster than every power even while memory—and eventually total loss—develops a power-law tail.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-forcing.png' | relative_url }}" alt="Target-weighted forcing decaying faster than every inverse power.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-memory.png' | relative_url }}" alt="One-injection memory following a three-quarter power-law tail.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-loss.png' | relative_url }}" alt="Minibatch-SGD risk crossing into the memory-controlled power-law tail.">
  </div>
  <figcaption><strong>Forcing and memory need not obey the same law.</strong> In this finite-width construction, target-weighted forcing and its cumulative spectral mass fall faster than every inverse power (left), while one-injection memory and its spectral mass track \(T^{-3/4}\) (middle). The minibatch-SGD loss then crosses from the fast forcing transient to the \(T^{-3/4}\) memory tail (right; mean over 20 runs).</figcaption>
</figure>

For a fixed finite model, this is not the literal \\(T\to\infty\\) law. It describes an infinite-spectrum limit or an extended joint width–time scaling window.

## Preserve, change, or destroy

Suppose one-injection memory has a tail

$$
k(v)\sim v^{-q_{\mathcal K}},
$$

and the ratio path grows approximately as \\(r(T)\sim T^\vartheta\\). The observed stochastic response is a convolution: the schedule decides how much noise is injected at every time, and memory decides how long each injection survives.

Three behaviors follow.

### 1. Destroy

If \\(r(T)\\) grows too slowly, new stochastic error is introduced faster than the dynamics can forget it. The noisy–clean gap then fails to decay with a positive power, so the clean positive-power law is destroyed.

### 2. Change

At intermediate growth rates, the stochastic gap decays, but more slowly than clean loss above its irreducible floor. The schedule therefore changes the exponent visible in total loss above that floor.

### 3. Preserve

If the stochastic response decays faster than clean learning above the floor, the schedule preserves the clean exponent. It may still change the prefactor, but not the leading power.

There is also a **memory ceiling**. Making late-stage noise arbitrarily small cannot erase old injections that are still remembered. Once the response reaches this ceiling, increasing \\(B/\eta\\) more aggressively no longer improves the decay exponent.

These cases are not separate heuristics: for a fixed memory exponent, they form one schedule-response phase diagram.

<figure>
  <img src="{{ '/images/power-laws-have-a-clock/main-fig4-schedule-map.png' | relative_url }}" alt="Phase diagram showing when a schedule preserves, changes, or destroys a clean power law." style="display:block;width:min(100%,760px);margin-inline:auto;">
  <figcaption><strong>How a schedule transforms the clean power law.</strong> The horizontal coordinate \(\vartheta\) controls the growth of \(r(T)=B(T)/\eta(T)\), while \(q_0\) is the clean-loss exponent. The red boundary marks the loss of positive-power decay, the black curve separates changed from preserved scaling, and its plateau is the memory ceiling. This pure-power diagram excludes the marginal case \(q_{\mathcal K}=1\), where logarithmic corrections appear.</figcaption>
</figure>

The marginal case \\(q_{\mathcal K}=1\\) separates long memory from integrable memory. At this boundary, cumulative memory grows logarithmically, so pure-power formulas acquire logarithmic corrections. This turns out to be especially relevant empirically: the LLM fits described below consistently place the effective response close to this boundary.

In the power-law random-feature specialization, these distinctions become three long-memory regimes, three integrable-memory regimes, and two finite-bulk regimes in which memory remains explicitly coupled to width. This \\(3+3(+2)\\) map classifies propagation mechanisms—not eight universal exponents for total loss.

The \\(3+3(+2)\\) structure has two useful views: response coordinates describe what the schedule sees, while power-law random-feature coordinates show one microscopic way those responses can arise.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:center;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-response-map.png' | relative_url }}" alt="Long-memory, integrable-memory, and finite-bulk regimes in forcing-memory response coordinates.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig3-plrf-map.png' | relative_url }}" alt="The corresponding propagation regimes in power-law random-feature source-capacity coordinates.">
  </div>
  <figcaption><strong>One propagation structure, two coordinate systems.</strong> Left: forcing and memory exponents \((q_{\mathcal F},q_{\mathcal K})\) divide long-memory (LM), integrable-memory (IM), and finite-bulk (FB) responses. Right: the corresponding partition in microscopic source–capacity coordinates \((\alpha,\beta)\). The red band marks the fitted finite-window LLM response near the LM/IM boundary; it does not identify a transformer spectrum or an asymptotic random-feature phase. These are propagation regimes, not eight universal total-loss exponents.</figcaption>
</figure>

The theory also gives a sharp converse below the long-memory ceiling: the complete asymptotic noisy–clean gap identifies the ratio path \\(B/\eta\\), including its slowly varying factor, but cannot identify learning rate and batch size separately. At the ceiling, this identification is lost.

## Three tests in language models

The proxy admits an exact conditional risk recurrence; its spectral and schedule laws are asymptotic and assumption-dependent. To test whether the resulting response coordinates remain useful beyond frozen features, we ran three increasingly demanding experiments in nanoGPT.

### Test 1: different ratio paths at the same intrinsic-time horizon

Starting from one mature 30M checkpoint, we forked eleven plain-SGD tails with different growth rates of \\(B/\eta\\). Every run stopped at the same intrinsic-time horizon, but used a different number of optimizer updates and tokens.

The validation trajectories were systematically ordered by the ratio-growth rate: increasing \\(B/\eta\\) produced lower validation loss, with diminishing improvement at the largest growth rates.

This provides qualitative evidence that intrinsic time alone is insufficient and that the ratio path matters as well. The experiment uses one seed, total validation cross-entropy rather than a noisy–clean gap, and no independent estimate of \\(q_{\mathcal K}\\). It therefore does not identify the LM/IM boundary or verify an asymptotic exponent. The runs were also not matched in updates or tokens, so this is not a fixed-compute comparison or a quantitative measurement of the memory ceiling.

### Test 2: the same ratio path through learning rate or batch size

We next trained a 300M nanoGPT model on 6.5B OpenWebText tokens. After a shared 5.2B-token prefix, each schedule used a 1.3B-token tail. We considered two ratio-path shapes—WSD and 8-1-1—and implemented each in two ways:

- a learning-rate schedule at fixed batch size;
- a batch-size schedule at fixed learning rate.

The construction matches the same intrinsic-time advance, the same \\(B/\eta\\) value, and the same contiguous data segment at every completed macro step.

Against optimizer step, the paired trajectories differ because the two implementations use different numbers of updates. Against intrinsic time, each fixed-batch and fixed-learning-rate pair nearly collapses.

A separate hybrid-Muon extension shows an analogous factorization collapse under the empirically calibrated coordinates \\(\widetilde T=\sum_t\eta_t^2\\) and \\(\widetilde r=B/\eta^2\\). This observation is optimizer-specific and is not implied by our plain-SGD theory.

### Test 3: fit one schedule, predict another

Approximate collapse is useful, but prediction is a stronger test. We built a seven-parameter finite-window surrogate inspired by the theoretical forcing–memory decomposition. It contains a clean forcing term and a memory integral whose injection strength depends on the observed ratio path.

In the main 300M analysis shown below, we fix \\(q_{\mathcal K}=1\\) and fit the remaining six parameters only to the raw fixed-batch 8-1-1 trajectory. With those parameters frozen, the surrogate predicts the held-out WSD learning-rate trajectory without refitting. In complementary unrestricted analyses at 124M and 300M, all seven parameters are fitted only on fixed-batch 8-1-1 and then frozen; those fits also track the alternative 8-1-1 factorization and both WSD trajectories without refitting.

The three panels separate the claims that matter: in-schedule fit, zero-refit transfer, and where held-out prediction error is minimized.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-fit.png' | relative_url }}" alt="Forcing-memory surrogate fitted to the 8-1-1 validation trajectory.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-transfer.png' | relative_url }}" alt="The frozen surrogate predicting the held-out WSD validation trajectory without refitting.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-profile.png' | relative_url }}" alt="Held-out WSD prediction error as a function of the fixed memory exponent.">
  </div>
  <figcaption><strong>Fit once, then transfer without refitting.</strong> Left: after fixing \(q_{\mathcal K}=1\), the remaining six parameters are fitted only to the fixed-batch 8-1-1 trajectory. Middle: those frozen parameters predict the held-out WSD learning-rate trajectory. Right: when \(q_{\mathcal K}\) is scanned and the other parameters are refitted only on 8-1-1, normalized WSD prediction error is minimized near one. This is a finite-window response diagnostic, not an independent measurement of an asymptotic LLM memory exponent.</figcaption>
</figure>

The frozen surrogate tracks the long-horizon loss decay and the schedule-induced change. In other words, the test is not merely “can a flexible curve fit one trajectory?” It is “can parameters inferred from one schedule predict a different schedule?”

We repeated the analysis across datasets and model scales:

| Dataset and setting | Effective \\(q_{\mathcal K}\\) | Effective \\(q_{\mathcal F}\\) |
|---|---:|---:|
| OpenWebText, 124M, 2.5B tokens | 1.017 | 0.374 |
| FineWeb sample-10BT subset, 124M, 2.5B tokens | 0.952 | 0.405 |
| peS2o V2 s2orc full text, 124M, 2.5B tokens | 0.995 | 0.365 |
| OpenWebText, 300M, 6.5B tokens | 0.989 | 0.291 |

Across the tested web and scientific-text corpora, the fitted memory coordinate stays close to one, while the forcing coordinate remains below one. This places the finite-window response near the boundary between long and integrable memory.

The number near one should be interpreted as an **effective finite-window response coordinate**, not as a directly measured asymptotic exponent. Separately from the displayed fit, we scan fixed candidate values of \\(q_{\mathcal K}\\); at each value, we refit the other six parameters only on 8-1-1 and evaluate post-fork WSD prediction error without refitting. That transfer error is minimized near one. This sensitivity profile is not an independent estimate of the LLM memory exponent. An unrestricted seven-parameter fit gives \\(q_{\mathcal K}=0.989\\). Together, these results support a near-one finite-window description, but identify neither an asymptotic memory exponent nor an LLM spectrum.

## What changes in practice?

Three consequences seem most immediate.

**Report the clock and schedule with the exponent.** A learning-curve exponent is incomplete without saying whether the horizontal axis is optimizer step, tokens, compute, or intrinsic time—and without specifying the ratio path that produced the curve.

**Design learning rate and batch size jointly.** Within the plain-SGD theory—and approximately in the tested nanoGPT runs—the target object is a path \\(r(T)\\), not a unique pair of schedules. This suggests choosing a hardware-compatible factorization while checking that the expected response remains intact, rather than assuming that learning-rate and batch-size schedules are uniquely determined. In the LM/IM proxy at fixed terminal time and data budget, the optimal ratio follows a square-root principle: allocate more samples where an injection is both large when created and likely to survive.

**Prefer transferable response models to isolated curve fits.** A power law fitted independently to each schedule describes what happened. A forcing–memory surrogate aims to predict what happens when the schedule changes.

## Scope

The theorems concern noisy online SGD with frozen linear random features, together with stability and scaling assumptions. The end-to-end LLM experiments are external-validity checks of response-level consequences—organization by intrinsic time and \\(B/\eta\\), approximate factorization collapse, and cross-schedule prediction—not theorem-facing tests that a transformer has the proxy model's spectrum or occupies an asymptotic random-feature phase. The fitted \\(q_{\mathcal K}\approx1\\) is a finite-window coordinate, where logarithmic corrections are difficult to distinguish from a pure power. Muon's \\(\sum_t\eta_t^2\\) and \\(B/\eta^2\\) coordinates are empirically calibrated rather than derived by the theory; momentum, AdamW, representation drift, and production-scale nonstationarity remain outside its present scope.

## The larger lesson

The point is not that every learning curve has a new universal exponent. The point is that an observed exponent is the endpoint of a chain:

$$
\text{low-spectrum geometry}
\longrightarrow
\text{forcing and memory}
\longrightarrow
\text{schedule-induced noise}
\longrightarrow
\text{observed loss}.
$$

The spectrum and target determine what can decay. Memory determines what stochastic error survives. The schedule determines how those responses accumulate. And the clock determines how we see the result.

A learning curve is therefore incomplete without its schedule and its clock.

---

*This post describes joint work by Yichen Wang, Fanghui Liu, and Yudong Chen.*

<!-- Add public paper and code links here when they are available. -->
