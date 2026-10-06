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

## Vision-transformer compute frontier

- Files: `zhai-2022-imagenet-finetune.svg` and `zhai-2022-scaling-frontier.svg`.
- Work: Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer, *Scaling Vision Transformers* (CVPR 2022), Figure 2, arXiv version 2.
- Figure: https://arxiv.org/html/2106.04560v2#S1.F2
- Explanation of the double-saturating fit: https://arxiv.org/html/2106.04560v2#S2.SS2
- Original left-panel asset: https://arxiv.org/html/2106.04560v2/imagenet_finetune.svg
- Original center/right-panel asset: https://arxiv.org/html/2106.04560v2/scaling_laws_teaser_saturating2.svg
- Downloaded: 2026-10-05. Both source SVGs are unmodified. The blog displays only the original left (ImageNet finetuning) panel, with a compact caption alongside it on wide screens and below it on narrow screens. The panel opens at full size when clicked; the caption links to the complete original Figure 2. The center/right asset remains archived here but is not displayed in the opening.
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
