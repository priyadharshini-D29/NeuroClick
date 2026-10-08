import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
_orig = matplotlib.figure.Figure.savefig
def _nogrid(self, *a, **k):
    for ax in self.get_axes():
        try:
            ax.grid(False)
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                gl.set_visible(False)
        except Exception:
            pass
    return _orig(self, *a, **k)
matplotlib.figure.Figure.savefig = _nogrid
import runpy, sys
sys.argv = ["scripts/06_generate_manuscript_figures.py", "--output", "outputs/figures_verify_v1"]
runpy.run_path("scripts/06_generate_manuscript_figures.py", run_name="__main__")
