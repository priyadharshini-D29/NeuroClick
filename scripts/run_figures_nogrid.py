import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

_orig_savefig = matplotlib.figure.Figure.savefig

def _nogrid_savefig(self, *args, **kwargs):
    for ax in self.get_axes():
        try:
            ax.grid(False)
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                gl.set_visible(False)
        except Exception:
            pass
    return _orig_savefig(self, *args, **kwargs)

matplotlib.figure.Figure.savefig = _nogrid_savefig

import runpy
import sys

SCRIPTS = [
    "scripts/06_generate_manuscript_figures.py",
    "scripts/08_generate_ensemble_figure.py",
]

for script in SCRIPTS:
    print(f"=== running {script} with gridlines disabled ===")
    sys.argv = [script]
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        pass
    except Exception as exc:
        print(f"!! {script} failed: {exc}")
        print("   If it needs command-line arguments, paste this error into the chat.")
