"""Chạy toàn bộ thí nghiệm và vẽ hình.

    python run.py                      # 6 subject, mọi thí nghiệm
    python run.py --subjects 1 2       # thử nhanh 2 subject
    python run.py --synthetic          # chạy thử không cần tải dữ liệu

Kết quả: results/*.csv, results/SUMMARY.md, figures/*.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import haxby as H

p = argparse.ArgumentParser()
p.add_argument("--subjects", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
p.add_argument("--n-permutations", type=int, default=100)
p.add_argument("--lag", type=int, default=0,
               help="dịch nhãn bao nhiêu TR; 0 = dùng nhãn đúng như dataset cung cấp (như tutorial nilearn)")
p.add_argument("--synthetic", action="store_true", help="dùng dữ liệu giả lập (kiểm tra code)")
p.add_argument("--out", default=".")
a = p.parse_args()
res, fig = Path(a.out) / "results", Path(a.out) / "figures"
res.mkdir(parents=True, exist_ok=True)
fig.mkdir(parents=True, exist_ok=True)


def get(subject, mask):
    if a.synthetic:
        from synthetic import make_haxby_like
        return make_haxby_like(subject, mask)
    return H.load_subject(subject, mask)


cv_rows, perm_rows, budget, lag, roi_rows, conf = [], [], [], [], [], []
for s in a.subjects:
    print(f"== subject {s}", flush=True)
    X0, y0, r0 = get(s, "vt")
    X, y, runs = H.select(X0, y0, r0, lag=a.lag)
    print(f"   {X.shape[0]} volume có nhãn, {X.shape[1]} voxel trong vùng VT, {len(np.unique(runs))} run")

    cv = H.exp_cv(X, y, runs)
    cv_rows.append({"subject": s, **{k: v for k, v in cv.items() if k != "per_run"}})
    print(f"   leave-one-run-out {100 * cv['leave_one_run_out']:.1f}%  random k-fold {100 * cv['random_kfold']:.1f}%"
          f"  (may rủi {100 * cv['chance']:.1f}%)")

    pm = H.exp_permutation(X, y, runs, a.n_permutations)
    perm_rows.append({"subject": s, **{k: v for k, v in pm.items() if k != "null"}})
    print(f"   permutation: null trung bình {100 * pm['null_mean']:.1f}%, p = {pm['p_value']:.3f}")

    b = H.exp_budget(X, y, runs)
    b["subject"] = s
    budget.append(b)

    lg = H.exp_lag(X0, y0, r0)
    lg["subject"] = s
    lag.append(lg)

    for m in H.MASKS:
        Xm, ym, rm = H.select(*get(s, m), lag=a.lag)
        acc = H.exp_cv(Xm, ym, rm)["leave_one_run_out"]
        roi_rows.append({"subject": s, "mask": m, "n_voxels": Xm.shape[1], "accuracy": acc})
    conf.append(H.exp_confusion(X, y, runs))

cv_df, perm_df = pd.DataFrame(cv_rows), pd.DataFrame(perm_rows)
budget, lag, roi = pd.concat(budget), pd.concat(lag), pd.DataFrame(roi_rows)
for name, df in [("cv", cv_df), ("permutation", perm_df), ("budget", budget), ("lag", lag), ("roi", roi)]:
    df.to_csv(res / f"{name}.csv", index=False)
C = np.mean(conf, axis=0)
labels = [c for c in H.CATEGORIES if c in set(y)]
pd.DataFrame(C, index=labels, columns=labels).to_csv(res / "confusion.csv")


def sem(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


BLUE, GRAY, ORANGE = "#2b6cb0", "#8a8a8a", "#dd6b20"
chance = 100 / len(labels)

# Hình 1: cách chia dữ liệu và kiểm định hoán vị
f, ax = plt.subplots(figsize=(6, 3.6))
vals = [100 * cv_df.leave_one_run_out.mean(), 100 * cv_df.random_kfold.mean(), 100 * perm_df.null_mean.mean()]
errs = [100 * sem(cv_df.leave_one_run_out), 100 * sem(cv_df.random_kfold), 100 * sem(perm_df.null_mean)]
ax.bar(["Leave-one-run-out\n(đúng)", "Random k-fold\n(rò rỉ)", "Nhãn xáo trộn\n(null)"], vals, yerr=errs,
       color=[BLUE, ORANGE, GRAY], capsize=3)
ax.axhline(chance, color="k", ls="--", lw=1, label=f"may rủi {chance:.1f}%")
ax.set_ylabel("Độ chính xác 8 lớp (%)")
ax.set_title(f"Decoding loại vật thể từ vùng VT ({len(a.subjects)} subject)")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
f.savefig(fig / "fig1_cv_and_permutation.png", dpi=200, bbox_inches="tight")

# Hình 2: số run train
g = budget.groupby(["n_train_runs", "subject"]).accuracy.mean().reset_index()
xs = sorted(g.n_train_runs.unique())
f, ax = plt.subplots(figsize=(5.5, 3.6))
ax.errorbar(xs, [100 * g[g.n_train_runs == x].accuracy.mean() for x in xs],
            yerr=[100 * sem(g[g.n_train_runs == x].accuracy) for x in xs], marker="o", color=BLUE, capsize=3)
ax.axhline(chance, color=GRAY, ls="--", lw=1)
ax.set_xticks(xs)
ax.set_xlabel("Số run dùng để train (mỗi run ~3,6 phút)")
ax.set_ylabel("Độ chính xác (%)")
ax.set_title("Cần bao nhiêu dữ liệu?")
ax.spines[["top", "right"]].set_visible(False)
f.savefig(fig / "fig2_data_budget.png", dpi=200, bbox_inches="tight")

# Hình 3: độ trễ huyết động
g = lag.groupby(["lag_s", "subject"]).accuracy.mean().reset_index()
xs = sorted(g.lag_s.unique())
f, ax = plt.subplots(figsize=(5.5, 3.6))
ax.errorbar(xs, [100 * g[g.lag_s == x].accuracy.mean() for x in xs],
            yerr=[100 * sem(g[g.lag_s == x].accuracy) for x in xs], marker="o", color=BLUE, capsize=3)
ax.axhline(chance, color=GRAY, ls="--", lw=1)
ax.set_xlabel("Độ trễ gán nhãn (giây)")
ax.set_ylabel("Độ chính xác (%)")
ax.set_title("Tín hiệu BOLD đến chậm vài giây")
ax.spines[["top", "right"]].set_visible(False)
f.savefig(fig / "fig3_hemodynamic_lag.png", dpi=200, bbox_inches="tight")

# Hình 4: vùng não
g = roi.groupby(["mask", "subject"]).accuracy.mean().reset_index()
f, ax = plt.subplots(figsize=(5, 3.6))
names = {"vt": "Ventral temporal", "face": "Vùng chọn lọc mặt", "house": "Vùng chọn lọc nhà"}
ax.bar([names[m] for m in H.MASKS], [100 * g[g["mask"] == m].accuracy.mean() for m in H.MASKS],
       yerr=[100 * sem(g[g["mask"] == m].accuracy) for m in H.MASKS], color=BLUE, capsize=3)
ax.axhline(chance, color=GRAY, ls="--", lw=1)
ax.set_ylabel("Độ chính xác 8 lớp (%)")
ax.set_title("Thông tin nằm ở đâu?")
ax.spines[["top", "right"]].set_visible(False)
f.savefig(fig / "fig4_roi.png", dpi=200, bbox_inches="tight")

# Hình 5: ma trận nhầm lẫn
f, ax = plt.subplots(figsize=(5.5, 4.8))
im = ax.imshow(100 * C, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{100 * C[i, j]:.0f}", ha="center", va="center", fontsize=7,
                color="white" if C[i, j] > 0.5 else "black")
ax.set_xlabel("Dự đoán")
ax.set_ylabel("Nhãn đúng")
ax.set_title("Nhầm lẫn giữa các loại (%)")
f.colorbar(im, ax=ax, fraction=0.046)
f.savefig(fig / "fig5_confusion.png", dpi=200, bbox_inches="tight")

best_lag = lag.groupby("lag_s").accuracy.mean()
off = C - np.diag(np.diag(C))
i, j = np.unravel_index(off.argmax(), off.shape)
summary = f"""# Tóm tắt ({len(a.subjects)} subject{', DỮ LIỆU GIẢ LẬP' if a.synthetic else ''}, độ trễ của phân tích chính: {a.lag} TR)

| Thí nghiệm | Kết quả |
| --- | --- |
| Leave-one-run-out (vùng VT) | {100 * cv_df.leave_one_run_out.mean():.1f}% ± {100 * sem(cv_df.leave_one_run_out):.1f} (may rủi {chance:.1f}%) |
| Random k-fold (rò rỉ) | {100 * cv_df.random_kfold.mean():.1f}% |
| Null hoán vị | {100 * perm_df.null_mean.mean():.1f}%, p lớn nhất qua subject = {perm_df.p_value.max():.3f} |
| Độ trễ tốt nhất | {best_lag.idxmax():.1f} s ({100 * best_lag.max():.1f}%) |
| 1 run train / 11 run train | {100 * budget[budget.n_train_runs == 1].accuracy.mean():.1f}% / {100 * budget[budget.n_train_runs == budget.n_train_runs.max()].accuracy.mean():.1f}% |
| Nhầm lẫn nhiều nhất | {labels[i]} → {labels[j]} ({100 * C[i, j]:.0f}%) |
"""
(res / "SUMMARY.md").write_text(summary)
print(summary)
