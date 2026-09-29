---
layout: single
title: "Power-Law Learning Curves: Spectral Origins, Schedule Transformations, and Predictive Tests in LLMs"
date: 2026-09-28
permalink: /posts/power-laws-have-a-clock/
excerpt: "A spectral if-and-only-if criterion for the forcing and memory laws behind learning curves, sharp schedule transformations, and cross-schedule prediction in LLM pretraining."
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

*When power-law responses exist, how training schedules transform them, and whether the resulting mechanism predicts LLM loss*

Why should a learning curve follow a power law when every fixed spectral mode decays exponentially?

Our answer is that power laws are not primitive empirical laws. They are collective dynamical responses. For each of the two components that drive noisy learning—unresolved target signal and remembered stochastic error—a temporal power law appears precisely when the corresponding cumulative weighted spectrum has matching scale-free mass near zero.

That is the paper's central result. It separates three questions that are often conflated:

1. **Existence:** when do power-law response components emerge at all?
2. **Transformation:** once they exist, how do learning-rate and batch-size schedules preserve, change, or destroy the power visible in total loss?
3. **External validity:** can the same response coordinates predict controlled LLM learning curves beyond the tractable model?

The theory gives necessary-and-sufficient spectral conditions for the first question and sharp schedule laws for the second. In controlled LLM pretraining, a forcing–memory surrogate fitted on one schedule predicts a held-out schedule without refitting, while matched learning-rate and batch-size implementations nearly collapse in the intrinsic clock.

## Two dynamical responses behind a learning curve

We study noisy online SGD in a frozen random-feature model, where the prediction risk admits an exact recurrence. Conditional on the representation, two objects govern the dynamics.

**Forcing** is unresolved target error. It describes how the part of the signal not yet learned propagates through the dynamics.

**Memory** is the response to one stochastic injection. It asks how much of noise injected now remains visible later.

Suppressing finite-width and feedback details, total loss above its irreducible floor has the schematic form

$$
L(T)-L_\infty
\;\approx\;
F(T)
+\int_0^T
a(u)\,k(T-u)\,\mathrm du.
$$

The first term is unresolved signal. The integral sums the surviving effects of the noise-injection history \\(a(u)\\). The observed loss is therefore not governed by one exponent in isolation: forcing, memory, and the history of noise injection compete.

## The spectral if-and-only-if criterion

At first sight, exponential spectral modes and power-law learning curves seem incompatible. The resolution is a moving cutoff.

For a constant schedule, intrinsic time is \\(T=\eta t\\). At time \\(T\\), modes with eigenvalue \\(\lambda\gg T^{-1}\\) have mostly relaxed, while modes with \\(\lambda\ll T^{-1}\\) remain largely unresolved. Training continuously sweeps this cutoff toward zero. Long-time behavior is controlled not by any single eigenvalue, but by the cumulative weighted spectral mass below \\(T^{-1}\\).

Two different weights matter:

- target-weighted spectral mass controls forcing;
- squared-spectrum mass controls one-injection memory.

Under the paper's stability and uniform spectral-window conditions, the main result is

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

The equivalence holds separately for forcing and memory. It is an **if and only if** statement: the low-spectrum weighted mass produces the temporal power, and observing that componentwise temporal power constrains the corresponding low-spectrum mass.

This changes the usual interpretation of “power-law data.” A coordinatewise power-law eigenspectrum alone is not sufficient, and coordinatewise power laws are not necessary:

- a power-law eigenspectrum can fail to produce power-law forcing if the target places too little mass in slow directions;
- irregular eigenvalues and targets can still produce a clean temporal power law when their cumulative weighted mass has the right behavior near zero;
- forcing and memory can have different exponents because they weight the same spectrum differently.

The following construction makes the last point explicit.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-forcing.png' | relative_url }}" alt="Target-weighted forcing decaying faster than every inverse power.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-memory.png' | relative_url }}" alt="One-injection memory following a three-quarter power-law tail.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig2-loss.png' | relative_url }}" alt="Minibatch-SGD risk crossing into the memory-controlled power-law tail.">
  </div>
  <figcaption><strong>Spectral structure determines whether a power law exists.</strong> Target-weighted forcing and its cumulative spectral mass decay faster than every inverse power (left), while one-injection memory and its spectral mass follow \(T^{-3/4}\) (middle). The minibatch-SGD loss consequently crosses from a fast forcing transient to the \(T^{-3/4}\) memory tail (right; mean over 20 runs).</figcaption>
</figure>

For a fixed finite model, the literal asymptotic decay is eventually exponential. The power law describes an infinite-spectrum limit or an extended joint width–time scaling window. The theorem is also componentwise: it characterizes forcing and memory, not total observed loss without further analysis of their competition.

## From component laws to propagation regimes

The forcing and memory exponents provide response coordinates \\((q_{\mathcal F},q_{\mathcal K})\\). They separate three qualitatively different propagation mechanisms:

- **long memory (LM):** old stochastic injections remain influential over long horizons;
- **integrable memory (IM):** the total memory mass is finite;
- **finite bulk (FB):** width-dependent bulk effects remain coupled to the dynamics.

In the power-law random-feature specialization, these become three LM regimes, three IM regimes, and two FB regimes—the paper's \\(3+3(+2)\\) phase map. This is a classification of propagation mechanisms, not eight universal exponents for total loss.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:center;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-response-map.png' | relative_url }}" alt="Long-memory, integrable-memory, and finite-bulk regimes in forcing-memory response coordinates.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig3-plrf-map.png' | relative_url }}" alt="The corresponding propagation regimes in power-law random-feature source-capacity coordinates.">
  </div>
  <figcaption><strong>One propagation structure, two coordinate systems.</strong> Left: the forcing and memory coordinates \((q_{\mathcal F},q_{\mathcal K})\) separate long-memory, integrable-memory, and finite-bulk responses. Right: the corresponding partition in microscopic source–capacity coordinates \((\alpha,\beta)\). The red band marks the fitted finite-window LLM response near the LM/IM boundary; it does not identify a transformer spectrum or an asymptotic random-feature phase.</figcaption>
</figure>

We return below to the LLM fits that locate the red band.

## How training schedules transform the response

Only after establishing when the componentwise power laws exist do we ask how a training schedule changes what is observed.

For plain SGD, learning rate and batch size enter through two natural coordinates:

$$
T_t=\sum_{s<t}\eta_s,
\qquad
r_t=\frac{B_t}{\eta_t}.
$$

Intrinsic time \\(T_t\\) measures accumulated optimization progress. The ratio path \\(r(T)\\) controls stochastic-error injection per unit intrinsic time: larger \\(B/\eta\\) means less injected noise. The full path matters because loss remembers earlier injections. In the response formula above, the leading schedule-dependent scale of \\(a(u)\\) is \\(1/r(u)\\).

Suppose one-injection memory has tail \\(k(v)\sim v^{-q_{\mathcal K}}\\) and \\(r(T)\sim T^\vartheta\\). Their convolution gives three possibilities.

### Destroy

If \\(r(T)\\) grows too slowly, fresh stochastic error is injected faster than it can be forgotten. The noisy–clean gap then fails to decay with a positive power.

### Change

At intermediate growth rates, the stochastic gap decays, but more slowly than clean loss above the floor. The schedule changes the exponent visible in total loss.

### Preserve

If the stochastic response decays faster than clean learning, the schedule preserves the clean exponent, though it may change the prefactor.

There is also a **memory ceiling**: reducing late-stage noise cannot erase old injections that are still remembered. Past that ceiling, making \\(B/\eta\\) grow faster no longer improves the decay exponent.

<figure>
  <img src="{{ '/images/power-laws-have-a-clock/main-fig4-schedule-map.png' | relative_url }}" alt="Phase diagram showing when a schedule preserves, changes, or destroys a clean power law." style="display:block;width:min(100%,760px);margin-inline:auto;">
  <figcaption><strong>Schedules transform an existing response law.</strong> The growth of \(r(T)=B(T)/\eta(T)\) determines whether stochastic memory destroys, changes, or preserves the clean power. The plateau of the black boundary is the memory ceiling. The pure-power diagram excludes the marginal case \(q_{\mathcal K}=1\), where logarithmic corrections appear.</figcaption>
</figure>

The theory also gives a converse below the long-memory ceiling: the complete noisy–clean gap identifies the ratio path \\(B/\eta\\), including its slowly varying factor, but cannot identify learning rate and batch size separately. At the ceiling, even this identification is lost.

## Testing the response picture in LLM pretraining

The theorems concern frozen random features, not transformers. We therefore treat the LLM experiments as external-validity tests of response-level predictions. They ask whether the same coordinates organize end-to-end loss and whether a response model fitted on one schedule predicts another.

### Test 1: the ratio path affects loss

Starting from one mature 30M checkpoint, we forked eleven plain-SGD tails with different growth rates of \\(B/\eta\\). Every run stopped at the same intrinsic-time horizon but used a different number of optimizer updates and tokens.

The validation trajectories were systematically ordered by ratio growth: increasing \\(B/\eta\\) lowered validation loss, with diminishing improvements at the largest values. This shows that intrinsic time alone is insufficient and that the injection path matters. Because this was a one-seed, finite-window, unequal-compute comparison, it does not identify an asymptotic memory exponent or quantitatively verify the ceiling.

### Test 2: two factorizations of the same schedule

We next trained a 300M nanoGPT model on 6.5B OpenWebText tokens. After a shared 5.2B-token prefix, each schedule used a 1.3B-token tail. We considered WSD and 8-1-1 ratio paths, each implemented in two ways:

- a learning-rate schedule at fixed batch size;
- a batch-size schedule at fixed learning rate.

The construction matches the intrinsic-time advance, the \\(B/\eta\\) value, and the contiguous data segment at every completed macro step. Against optimizer step, the paired trajectories differ because the factorizations use different numbers of updates. Against intrinsic time, each pair nearly collapses.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-step.png' | relative_url }}" alt="Validation risk for matched learning-rate and batch-size schedules plotted against optimizer step.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig1-intrinsic-time.png' | relative_url }}" alt="The same validation trajectories plotted against intrinsic time, together with the transferred surrogate.">
  </div>
  <figcaption><strong>Matched schedule factorizations share a response in the intrinsic clock.</strong> Left: learning-rate and batch-size implementations of the same ratio paths differ against optimizer step. Right: in intrinsic time \(T=\sum_t\eta_t\), each pair nearly coincides, and the forcing–memory prediction tracks both schedule shapes. This is an approximate finite-window observation, not an exact identity for end-to-end LLMs.</figcaption>
</figure>

A separate hybrid-Muon extension shows analogous collapse under empirically calibrated coordinates \\(\widetilde T=\sum_t\eta_t^2\\) and \\(\widetilde r=B/\eta^2\\). Those coordinates are optimizer-specific and are not derived by the plain-SGD theory.

### Test 3: fit one schedule, predict another

Approximate collapse is useful, but cross-schedule prediction is a stronger test. We use a seven-parameter finite-window surrogate motivated by the forcing–memory decomposition:

$$
\widehat L(T)
=L_\infty+A_{\mathcal F}(1+T)^{-q_{\mathcal F}}
+\int_0^T
\frac{A_0+A_1(1+u)^{-q_{\mathcal F}}}{r(u)}
\bigl(1+c_{\mathcal K}(T-u)\bigr)^{-q_{\mathcal K}}
\,\mathrm du.
$$

In the main 300M analysis, we fix \\(q_{\mathcal K}=1\\) and fit the remaining six parameters only to the raw fixed-batch 8-1-1 trajectory. We then freeze the surrogate and use it to predict the held-out WSD learning-rate trajectory—without refitting.

<figure>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;align-items:end;">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-fit.png' | relative_url }}" alt="Forcing-memory surrogate fitted to the 8-1-1 validation trajectory.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-transfer.png' | relative_url }}" alt="The frozen surrogate predicting the held-out WSD validation trajectory without refitting.">
    <img src="{{ '/images/power-laws-have-a-clock/main-fig5-profile.png' | relative_url }}" alt="Held-out WSD prediction error as a function of the fixed memory exponent.">
  </div>
  <figcaption><strong>Fit once, then predict a new schedule.</strong> Left: after fixing \(q_{\mathcal K}=1\), the remaining six parameters are fitted only to fixed-batch 8-1-1. Middle: the frozen surrogate predicts held-out WSD without refitting. Right: when \(q_{\mathcal K}\) is scanned and the other parameters are refitted only on 8-1-1, WSD prediction error is minimized near one. This is a finite-window response diagnostic, not a direct measurement of a transformer spectrum.</figcaption>
</figure>

The important test is not whether a flexible curve fits one trajectory. It is whether the fitted response model predicts what happens when the schedule changes.

In complementary unrestricted seven-parameter fits across model scales and datasets, we obtain:

| Dataset and setting | Effective \\(q_{\mathcal K}\\) | Effective \\(q_{\mathcal F}\\) |
|---|---:|---:|
| OpenWebText, 124M, 2.5B tokens | 1.017 | 0.374 |
| FineWeb sample-10BT subset, 124M, 2.5B tokens | 0.952 | 0.405 |
| peS2o V2 s2orc full text, 124M, 2.5B tokens | 0.995 | 0.365 |
| OpenWebText, 300M, 6.5B tokens | 0.989 | 0.291 |

Across the tested web and scientific-text corpora, the fitted memory coordinate stays near one while the forcing coordinate remains below one. This places the finite-window response close to the LM/IM boundary. The near-one value is an effective response coordinate: it neither identifies a transformer spectrum nor establishes an asymptotic LLM memory exponent.

## What the results change

Three practical consequences follow from the full picture.

**A fitted exponent needs a mechanism.** Before extrapolating a learning curve, ask which response dominates and whether the relevant weighted low-spectrum mass actually supports a stable power law.

**The clock and schedule are part of the observation.** Once forcing and memory laws exist, the horizontal axis and noise-injection path determine how they combine. An exponent should therefore be reported with its clock and schedule.

**Cross-schedule prediction is stronger than an isolated fit.** A power law fitted separately to each run describes what happened. A forcing–memory model that transfers without refitting tests whether the proposed mechanism survives a controlled intervention.

## Scope

The spectral and schedule theorems concern noisy online SGD with frozen linear random features, under stability and scaling assumptions. The LLM experiments test organization by intrinsic time and \\(B/\eta\\), approximate factorization collapse, and cross-schedule prediction. They do not prove that a transformer has the proxy model's spectrum or occupies an asymptotic random-feature phase. Momentum, AdamW, representation drift, and production-scale nonstationarity remain outside the present theory.

## The larger lesson

The main message is not merely that schedules matter. An observed power-law learning curve is the endpoint of a mechanism:

$$
\text{weighted low-spectrum geometry}
\longrightarrow
\text{forcing and memory laws}
\longrightarrow
\text{schedule-dependent accumulation}
\longrightarrow
\text{observed loss}.
$$

Low-spectrum geometry determines whether the componentwise power laws exist. The training schedule determines how those responses appear in total loss. Successful zero-refit prediction in controlled LLM experiments tests whether this description remains useful beyond the tractable proxy.

---

*This post describes joint work by Yichen Wang, Fanghui Liu, and Yudong Chen.*

<!-- Add public paper and code links here when they are available. -->
