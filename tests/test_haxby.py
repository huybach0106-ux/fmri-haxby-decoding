import subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import haxby as H
from synthetic import make_haxby_like


def test_select_lag_and_rest():
    y = np.array(["rest", "a", "a", "b", "rest", "rest"])
    X = np.arange(6)[:, None].astype(float)
    runs = np.zeros(6, int)
    Xs, ys, _ = H.select(X, y, runs, lag=1)
    assert list(ys) == ["a", "a", "b"] and list(Xs[:, 0]) == [2, 3, 4]   # volume t mang nhãn của t-1
    Xs, ys, _ = H.select(X, y, runs, lag=0)
    assert list(ys) == ["a", "a", "b"] and list(Xs[:, 0]) == [1, 2, 3]


def test_leakage_and_lag_on_synthetic():
    X0, y0, r0 = make_haxby_like(1)
    X, y, r = H.select(X0, y0, r0, lag=2)
    cv = H.exp_cv(X, y, r)
    assert cv["random_kfold"] > cv["leave_one_run_out"] > 0.3          # chia ngẫu nhiên bị thổi phồng
    lag = H.exp_lag(X0, y0, r0, lags=[0, 2, 5])
    assert lag.set_index("lag_tr").accuracy.idxmax() == 2               # bắt đúng độ trễ đã gài
    pm = H.exp_permutation(X, y, r, n_permutations=20)
    assert abs(pm["null_mean"] - 0.125) < 0.05


def test_run_script_synthetic(tmp_path):
    r = subprocess.run([sys.executable, "run.py", "--synthetic", "--subjects", "1", "--n-permutations", "10",
                        "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "results" / "SUMMARY.md").exists() and (tmp_path / "figures" / "fig5_confusion.png").exists()
