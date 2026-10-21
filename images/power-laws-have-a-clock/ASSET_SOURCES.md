# Opening figures

## Kaplan scaling-law panels

- File: `kaplan-2020-scaling-laws.svg`
- Work: Jared Kaplan et al., *Scaling Laws for Neural Language Models* (2020), Figure 1.
- Paper: https://arxiv.org/html/2001.08361v1#S1.F1
- Original asset: https://arxiv.org/html/2001.08361v1/SimplePowerLaws.svg
- Downloaded: 2026-10-05.
- SHA-256: `59a02c0c1352acd20b6f98f48dd971a205d20f92e138cc1ddf3a8585dfa83e26`.
- The source SVG is unmodified. The blog displays its three panels side by side or stacked depending on screen width.
- Attribution appears in the figure caption. Copyright remains with the original rights holders; this asset is not covered by the repository's software license. The paper lists the arXiv non-exclusive distribution license, not a Creative Commons license. No downstream reuse license is asserted here.
- The compute panel includes the original paper's small-batch adjustment and represents a frontier across models, not a fixed-model training trajectory.

## Non-power-law frontier in synthetic regression

- File: `qiu-2024-no-power-law.png`.
- Work: Shikai Qiu, Atish Agarwala, Jeffrey Pennington, and Lechao Xiao, *Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks*, OPT 2024 workshop version, Figure 3(c), PDF page 6.
- Source PDF: https://www.opt-ml.org/papers/2024/paper78.pdf
- This is the OPT 2024 experiment, not a figure from the later ICML 2025 version.
- Source retrieved: 2026-10-08. Added to the blog: 2026-10-09.
- Source PDF SHA-256: `193c052442430c404c7d81895b6eca9eb1100f7142355052c2d40bcfbc954f27`.
- Asset SHA-256: `8f995f27caec4f1950e173530a25a4e359899929b139acc202d709c29a1ace70`.
- Extraction: Poppler `pdftoppm`, page 6, `-scale-to 4800 -x 1854 -y 543 -W 636 -H 615`. Original panel, curves, axes, fit legend, aspect ratio, and panel label retained; no data were redrawn.
- Scope: MLPs trained on a synthetic regression target with exponentially distributed Fourier frequencies and a constant learning rate. This is not an LLM experiment. Figure 3(a) also uses a different learning-rate schedule, so the blog does not frame the two panels as a controlled change of data distribution alone.
- Attribution appears in the caption. Copyright remains with the original rights holders; this asset is not covered by the repository's software license. No broader reuse license is asserted here.

## Training recipe and power-law extrapolation

- File: `porian-2024-training-recipes.png`.
- Work: Tomer Porian, Mitchell Wortsman, Jenia Jitsev, Ludwig Schmidt, and Yair Carmon, *Resolving Discrepancies in Compute-Optimal Scaling of Language Models* (2024), arXiv version 4, Figure 19, PDF page 28.
- Source PDF: https://arxiv.org/pdf/2406.19146v4
- Explanation: https://arxiv.org/html/2406.19146v4#A9
- Selected panels: the lower-row `Optimizer tuning (no decay)` and `Cosine decay` panels on RefinedWeb.
- Source retrieved: 2026-10-08. Added to the blog: 2026-10-09.
- Source PDF SHA-256: `d47e3792ae67a30f4b53ada4d89d22c36185afafedeceaeefecee1c9d4520ac0`.
- Asset SHA-256: `d0c5d639a634d1a53ecc46e435a73877b85fb6bedc88216e4c36e65c9828c3f4`.
- Extraction: Poppler `pdftoppm`, page 28, `-scale-to 3600 -x 1120 -y 1828 -W 1160 -H 598`. Original titles, axes, fit formulas, curves, and aspect ratios retained; no data were redrawn. The shared vertical-axis title and legend lie outside this crop, so their meaning is supplied in the blog caption.
- Scope: empirical optimal loss over model-size/token allocations within each training recipe, fitted by a power law plus a loss floor. The recipes differ in hyperparameter tuning as well as learning-rate decay; this is not a controlled test of cosine decay alone, nor a demonstration that a globally optimized frontier cannot be a power law.
- The source article is licensed under Creative Commons Attribution 4.0, as linked from https://arxiv.org/abs/2406.19146v4: https://creativecommons.org/licenses/by/4.0/. The caption credits the source, links the license, and identifies the crop. Copyright remains with the original rights holders; the repository's software license does not replace the source license.

## Combined compute-frontier triptych

- Arranged on 2026-10-13 using the unchanged Qiu and Porian PNGs above. No data, curves, axes, or fit formulas were redrawn.
- Left: Qiu viewport `0 26 636 479`, removing surrounding whitespace and the original `(c) No power law` footer. Middle and right: Porian viewports `0 44 580 544` and `580 44 580 544`, separating the two panels and replacing their colored headings with matching HTML headings.
- New `(a)`, `(b)`, and `(c)` labels refer only to the blog triptych; original source figure numbers are retained in the shared caption. Axes, tick labels, fit legends, and aspect ratios are preserved. The slightly wider left column aligns the different original plotting-area shapes without stretching either source.
- Each panel links to its unchanged, full-size source crop. Narrow screens retain a single row with horizontal scrolling rather than shrinking all three plots to unreadable sizes. Source credits and the Porian CC BY 4.0 link remain in the shared caption.

## Vision-transformer compute frontier (archived)

- Files: `zhai-2022-imagenet-finetune.svg` and `zhai-2022-scaling-frontier.svg`.
- Work: Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer, *Scaling Vision Transformers* (CVPR 2022), Figure 2, arXiv version 2.
- Figure: https://arxiv.org/html/2106.04560v2#S1.F2
- Explanation of the double-saturating fit: https://arxiv.org/html/2106.04560v2#S2.SS2
- Original left-panel asset: https://arxiv.org/html/2106.04560v2/imagenet_finetune.svg
- Original center/right-panel asset: https://arxiv.org/html/2106.04560v2/scaling_laws_teaser_saturating2.svg
- Downloaded: 2026-10-05. Both source SVGs are unmodified. The opening previously displayed the original left (ImageNet finetuning) panel. It was replaced by the Qiu and Porian figures on 2026-10-09; both Zhai source assets remain archived here and are no longer displayed in the opening.
- SHA-256, left panel: `887d659f443be05d637340508d01e04579266a219f42426d9eca201a8bfb8a8c`.
- SHA-256, center/right panels: `3a692f391af55e5a686f95e889e28e29d1f67c53ad55ab1ad8e62df45d50523f`.
- The two main panels measure ImageNet finetuning and linear 10-shot transfer error against training compute in TPUv3 core-days. They are not LLM pretraining-loss plots. Section 2.2 describes a middle power-law region and low/high-compute saturation using E = a(C + d)^(-b) + c; the high-compute error floor alone does not refute a floor-subtracted asymptotic power law.
- Attribution appears in the caption. Copyright remains with the original rights holders; these assets are not covered by this repository's software license. The arXiv article lists a perpetual non-exclusive distribution license, not a Creative Commons license. No downstream reuse license is asserted here.

## Learning-rate schedule and fitted loss exponents

- Files: `mircea-2025-constant-lr.svg` and `mircea-2025-cosine-lr.svg`.
- Work: Andrei Mircea, Supriyo Chakraborty, Nima Chitsazan, Irina Rish, and Ekaterina Lobacheva, *Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning* (2025), Figures 2 and 29, version 1.
- Main results: https://arxiv.org/html/2506.05447v1#S2
- Controlled LR comparison: https://arxiv.org/html/2506.05447v1#A3.SS3
- Fitting method: https://arxiv.org/html/2506.05447v1#A1.SS2
- Original Figure 2 asset: https://arxiv.org/html/2506.05447v1/02-bnsl_fit.svg
- Original Figure 29 asset: https://arxiv.org/html/2506.05447v1/bnsl_fit_cosine.svg
- Downloaded: 2026-10-05. Both source SVGs are unmodified. The blog uses a shared 300 × 456 display canvas and aligns each plotting area to (60, 24, 216, 360). Figure 2 is translated; Figure 29 is scaled by 4/3 horizontally and 5/4 vertically and then translated, correcting its different original plotting-area aspect ratio. These display-only transforms retain all labels, curves, and legends without cropping; the underlying data and source files are unchanged. Panel labels sit outside the images, and the panels stack on narrow screens.
- SHA-256, Figure 2: `087716b8006ed6b1d1229be4f88f98d82dc85658b44aecfd47d88b573ead40b0`.
- SHA-256, Figure 29: `aee9ed44947753fae6a13a4abc743afcc132617bca10b95568dcfe30c0767ed6`.
- Exact reported late-training exponent pairs (constant LR to cosine decay), from Tables 1 and 7: 144M, 0.023 to 0.036; 285M, 0.025 to 0.040; 472M, 0.035 to 0.045.
- The controlled comparison concerns the authors' 14M–472M runs, with 2,000-step warmup and otherwise unchanged training settings. OLMo-1B/7B are external reference runs, not a controlled constant-versus-cosine pair.
- Appendix A.2 fits smoothed per-step minibatch training losses with a broken power law and fixes the loss offset to zero. These finite-window exponents are not identified with asymptotic exponents of excess loss above a fitted floor, or with this blog's memory-kernel exponent.
- Attribution appears in the figure caption, and each panel links to the source. Copyright remains with the original rights holders; the assets are not covered by this repository's software license. The article uses the arXiv non-exclusive distribution license, not a Creative Commons license. Its code/checkpoint/log license does not establish a reuse license for these article figures.

# Blog-specific prediction label

- File: main-fig5-transfer.png.
- Source: the authors' iclr2027/figures/llm_maintext_wsd_zero_refit_preview.pdf, generated from the original 300M validation observations and frozen 8-1-1 fit.
- On 2026-10-08, the legend label changed from "Fitted surrogate" to "Frozen prediction". No data, fitted parameters, curves, axes, or image dimensions changed.
- Reproduction: scripts/render_blog_wsd_prediction.py uses the original plotting function and verifies pixel-identical rendering outside the legend. The paper's figure and plotting source remain unchanged.

# Opening fit/prediction pair

- Files: `llm-global-811-fit.png` and `llm-global-wsd-prediction.png`. Added on 2026-10-21 from the author's supplied `llm_global_811_surrogate_transfer_preview.png` and `llm_global_wsd_zero_refit_preview.png`; copied without pixel edits, cropping, or resampling. The supplied PDFs are alternate source exports and are not required by the webpage.
- Both PNGs are byte-identical to the corresponding files in `iclr2027/figures`. Renderer: `iclr2027/experiments/experiments/nanogpt_local/plot_llm_global_surrogate_preview_v001.py`. It extends the observed validation cross-entropy curves to intrinsic time zero without performing new training or fitting.
- This pair uses the fixed-memory-exponent fit, q_K = 1, from the 8-1-1 learning-rate arm. WSD uses the same frozen parameters with a different schedule. The runs share the prefix before the dotted branch line; that shared prefix is not an independent held-out WSD prediction. Surrogate curves start at their fitted support, T = 664.64, not initialization.
- The insets enlarge late training, including a short interval before the branch. Original colors, legends, axes, and insets are retained. HTML headings identify fitting versus zero-refit prediction. Desktop shows two columns within the text width; narrow screens stack them, and each panel links to its full-resolution PNG.
- 8-1-1 image: 1584 × 1072; SHA-256 `4c454adcd5b7c03cfd8d659e82c2107eb1dd4f6c98de967882f82bd83b78f19f`.
- WSD image: 1609 × 1072; SHA-256 `9fcd374b161b2fcbb412ae9e98708f7d5a8edec71e30237c598a1cfd3f1da6e4`.

# Training-clock animation

- Files: `main-fig1-clock-animation.gif` and `main-fig1-clock-animation-still.png`. Added on 2026-10-14 to replace the displayed step/intrinsic-time pair; the original `main-fig1-step.png` and `main-fig1-intrinsic-time.png` remain archived in this directory.
- Source: the authors' 300M plain-SGD OpenWebText Figure 1 data, loaded with `iclr2027/experiments/experiments/nanogpt_local/plot_e2e_sgd_b16_free_q_figure1_panels_300m_v001.py`. The animation uses the original paper's blue/orange schedule palette and gray shared prefix.
- Sequence: optimizer step; horizontal remapping to intrinsic time; 8-1-1 surrogate fit; frozen WSD prediction; unfolding into surrogate, learning-rate-schedule, and batch-size-schedule layers. No bottom explanatory footer is included.
- The four observed trajectories and their original smoothing are preserved (9-point Hann for observations, 21-point Hann for predictions, with the shared prefix and branch tails treated separately). Display-point simplification does not add smoothing or force paired trajectories to coincide. The full step-panel time range is retained, including the WSD tail beyond the original static intrinsic-time panel's cutoff.
- Fit: `eight-one-one-free-q-matched124m-raw-fit.json`, SHA-256 `41d6f3d891ede015e796182f0c40f093b681779e0aea3637ccc0ecb96294a871`. Parameters, including the memory exponent, are fitted to the 8-1-1 fixed-batch/learning-rate arm and reused without refitting for WSD. This is distinct from the fixed-memory-exponent fit in the following blog section.
- GIF: 1100 × 740, 271 frames, 20.05-second loop; SHA-256 `cac14ee19d4f0eb37db40418469f6c83ae3195b6ac3351dc8eda9de41b7087bc`.
- Static fallback: the final three-layer frame, 1100 × 740; SHA-256 `872e167bcff13a31983dfabedeebef149cd13d5b23a573f49d414a24cce1c743`. A `picture` source displays this frame when the reader requests reduced motion.

# Source bias–variance and forcing–memory comparison

- File: `exponential-teacher-two-decompositions-T10000.png`. Added on 2026-10-21, copied unchanged from the authors' reviewed experiment figure.
- Local source: `output/linear-sgd-decomposition-comparison-review/exponential-teacher-two-decompositions-T10000.png`; renderer: `render_comparison.py` in the same review directory. Data: `output/linear-sgd-teacher-comparison-T1000000/exponential_curves.npz`, restricted to intrinsic time at most 10,000.
- Gaussian linear model with covariance eigenvalues j^(-0.8), squared teacher coordinates proportional to exp(-j), dimension 524,288, zero initialization, and initial prediction risk normalized to 1. Constant learning rate 0.05, batch size 16, and independent mean-zero label-noise variance 0 or 0.01.
- The observable is expected last-iterate squared prediction error against the noiseless target, excluding irreducible test-label noise and without a factor of one half. Intrinsic time is 0.05 times the update count.
- Expectations use full-spectrum calculations through intrinsic time 1,000 and a checked positive grouped-moment approximation thereafter. These are expected-risk calculations, not newly sampled SGD trajectories to time 10,000.
- Left column: forcing plus accumulated memory. Right column: source bias (the clean-label risk, including random minibatch effects) plus label-noise variance. Top row is clean; bottom row is noisy. Total-risk arrays are identical within each row. This figure does not plot the effective bound terms from Zou, Wu et al. (2021).
- The forcing term retains its finite-batch second-moment correction. Memory here is the accumulated contribution, not the one-injection response kernel in the following triptych. Values below the plot limits are not replaced with artificial floors.
- Image: 2560 × 1760; SHA-256 `5f961da4625899fced7669c4ad19212b737de3ac7781a41d37117ca43acba225`.
- Source data SHA-256: `260f58c83b1a70095f8ef3740716f10ef499144acd74cdaf7bdd7889225c743d`. The review validation reports a maximum component-identity residual of 8.67e-19 and confirms that the source data were unchanged.
