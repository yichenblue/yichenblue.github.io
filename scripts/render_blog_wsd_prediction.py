"""Reproduce the blog's WSD panel with an unambiguous frozen-prediction legend.

Run from the enclosing research workspace with its ./.venv/bin/python.
The paper's plotting module, frozen q_K=1 fit, and hash-verified observations
are read without modification. No fitting or training is performed.

The paper PDF rendered by pdftoppm at 220 dpi reproduces the original blog
PNG exactly. Before writing, this script checks that an unchanged rendering
still matches that original pixel for pixel, and that the new rendering only
differs within the legend (whose frame widens to fit the longer label).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch


SITE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
PAPER_PDF = WORKSPACE_ROOT / "iclr2027/figures/llm_maintext_wsd_zero_refit_preview.pdf"
DEFAULT_OUTPUT = SITE_ROOT / "images/power-laws-have-a-clock/main-fig5-transfer.png"


def rasterize(pdf: Path, png: Path) -> None:
    subprocess.run(
        ["pdftoppm", "-r", "220", "-png", "-singlefile", str(pdf), str(png.with_suffix(""))],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()

    sys.dont_write_bytecode = True
    experiments_root = WORKSPACE_ROOT / "iclr2027/experiments"
    sys.path.insert(0, str(experiments_root))
    with tempfile.TemporaryDirectory(prefix="blog-wsd-prediction-") as directory:
        temporary = Path(directory)
        os.environ.setdefault("MPLCONFIGDIR", str(temporary / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(temporary / "cache"))

        import matplotlib
        from matplotlib.figure import Figure
        import numpy as np
        from PIL import Image
        from experiments.nanogpt_local import plot_llm_maintext_surrogate_preview_v001 as original

        source = WORKSPACE_ROOT / original.analysis.SOURCE_RELATIVE
        curves, _, _ = original.profile_analysis._matched_124m_curves(source)
        parameters = original._read_fixed_q_fits(
            experiments_root / "experiments/results/nanogpt300m-e2e-sgd-qk-schedule-profiles-v001/eight_one_one_qk_profile.csv"
        )
        curve = curves["wsd_exp_80_20__fixed_batch_lr"]
        predictions = {
            q_k: original.analysis._predict(curve, values)
            for q_k, values in parameters.items()
        }
        figures = []

        def capture(figure: Figure, *args: object, **kwargs: object) -> None:
            if not figures:
                figures.append(figure)
            elif figures[0] is not figure:
                raise RuntimeError("Original renderer unexpectedly produced multiple figures")

        # Reuse the exact rendering function without letting it write paper files.
        with patch.object(Figure, "savefig", capture):
            original._render_panel(
                curve_batch=curve,
                predictions=predictions,
                output=temporary,
                stem="captured",
                validation_color="#2878b5",
                surrogate_colors={1.0: "#0f4f82"},
                view_end=float(curves["eight_one_one__fixed_batch_lr"].intrinsic_time[-1]),
                validation_smoothing_window=original.FIGURE1_VALIDATION_SMOOTHING_WINDOW,
                surrogate_smoothing_window=original.FIGURE1_SURROGATE_SMOOTHING_WINDOW,
            )
        figure = figures[0]
        baseline_pdf = temporary / "baseline.pdf"
        baseline_png = temporary / "baseline.png"
        reference_png = temporary / "paper-reference.png"
        figure.savefig(baseline_pdf, bbox_inches="tight")
        rasterize(baseline_pdf, baseline_png)
        rasterize(PAPER_PDF, reference_png)
        baseline = np.asarray(Image.open(baseline_png).convert("RGB"))
        reference = np.asarray(Image.open(reference_png).convert("RGB"))
        if not np.array_equal(baseline, reference):
            raise RuntimeError("Original plot was not reproduced exactly; output was not changed")

        axis = figure.axes[0]
        labels = axis.get_legend().get_texts()
        if [label.get_text() for label in labels] != ["LR schedule", "Fitted surrogate"]:
            raise RuntimeError("Unexpected original legend; output was not changed")
        labels[1].set_text("Frozen prediction")
        prediction_pdf = temporary / "frozen-prediction.pdf"
        prediction_png = temporary / "frozen-prediction.png"
        figure.savefig(prediction_pdf, bbox_inches="tight")
        rasterize(prediction_pdf, prediction_png)
        revised = np.asarray(Image.open(prediction_png).convert("RGB"))
        if revised.shape != baseline.shape:
            raise RuntimeError("Image dimensions changed; output was not changed")
        different = np.any(revised != baseline, axis=2)
        rows, columns = np.nonzero(different)
        if not rows.size:
            raise RuntimeError("Legend did not change; output was not changed")
        # The complete original and revised legends occupy this fixed pixel box.
        # Every curve, tick, gridline, axis, and outside pixel must remain exact.
        permitted = np.zeros(different.shape, dtype=bool)
        permitted[520:705, 305:980] = True
        if np.any(different & ~permitted):
            raise RuntimeError("Pixels outside the legend changed; output was not changed")
        if output.exists():
            existing = np.asarray(Image.open(output).convert("RGB"))
            if not (np.array_equal(existing, baseline) or np.array_equal(existing, revised)):
                raise RuntimeError("Output differs from the known original and revised assets")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prediction_png, output)
        print(f"Saved: {output}")
        print(f"Original reproduced exactly; Matplotlib {matplotlib.__version__}")
        print(f"Dimensions: {revised.shape[1]} x {revised.shape[0]}")
        changed_box = tuple(int(value) for value in (
            columns.min(), rows.min(), columns.max() + 1, rows.max() + 1
        ))
        print(f"Changed pixels: {rows.size}; bbox: {changed_box}")
        print("Every pixel outside the legend is unchanged")
        print(f"Shared-prefix fork: {original.analysis.PREFIX_UPDATES * original.analysis.BASE_LR}")


if __name__ == "__main__":
    main()
