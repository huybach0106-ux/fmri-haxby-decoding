"""Decoding loại vật thể từ fMRI (Haxby et al., 2001) và kiểm tra xem độ chính xác có đáng tin không.

Dữ liệu: 6 subject, 12 run, mỗi run có 8 block (face, cat, house, chair, scissors, shoe, bottle,
scrambledpix), TR = 2,5 s. nilearn tải tự động (~300 MB mỗi subject).

Mọi hàm thí nghiệm nhận mảng numpy:
    X    (n_volume, n_voxel)  đáp ứng đã z-score theo run
    y    (n_volume,)          nhãn, gồm cả 'rest'
    runs (n_volume,)          chỉ số run
nên có thể test bằng dữ liệu giả lập mà không cần tải gì.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CATEGORIES = ["face", "cat", "house", "chair", "scissors", "shoe", "bottle", "scrambledpix"]
TR = 2.5
MASKS = ["vt", "face", "house"]


# ---------------------------------------------------------------- dữ liệu

def load_subject(subject: int, mask: str = "vt", data_dir="data", cache_dir="cache"):
    """Tải (lần đầu) và trích đáp ứng trong một mask. Kết quả được cache thành .npz."""
    cache = Path(cache_dir) / f"sub-{subject}_{mask}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["runs"]
    from nilearn import datasets
    from nilearn.maskers import NiftiMasker

    h = datasets.fetch_haxby(data_dir=str(data_dir), subjects=[subject], verbose=0)
    lab = pd.read_csv(h.session_target[0], sep=" ")
    mask_img = {"vt": h.mask_vt, "face": h.mask_face, "house": h.mask_house}[mask][0]
    try:
        masker = NiftiMasker(mask_img=mask_img, standardize="zscore_sample", detrend=True,
                             runs=lab["chunks"].to_numpy(), t_r=TR)
    except TypeError:  # nilearn cũ
        masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True,
                             runs=lab["chunks"].to_numpy(), t_r=TR)
    X = masker.fit_transform(h.func[0]).astype(np.float32)
    y, runs = lab["labels"].to_numpy().astype(str), lab["chunks"].to_numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, runs=runs)
    return X, y, runs


def select(X, y, runs, lag: int = 0):
    """Dịch nhãn đi `lag` volume trong từng run (bù độ trễ huyết động), rồi bỏ volume 'rest'.

    lag = 2 nghĩa là volume t được gán nhãn của kích thích lúc t - 2 TR (5 giây trước)."""
    Xs, ys, rs = [], [], []
    for r in np.unique(runs):
        m = np.where(runs == r)[0]
        if lag > 0:
            xi, yi = m[lag:], m[:-lag]
        else:
            xi, yi = m, m
        Xs.append(X[xi]); ys.append(y[yi]); rs.append(runs[xi])
    X, y, runs = np.concatenate(Xs), np.concatenate(ys), np.concatenate(rs)
    keep = y != "rest"
    return X[keep], y[keep], runs[keep]


def classifier(alpha: float = 1000.0):
    """Ridge classifier tuyến tính trên dữ liệu đã chuẩn hoá: nghiệm dạng đóng nên rất nhanh (cần cho
    kiểm định hoán vị), độ chính xác tương đương SVM tuyến tính. alpha lớn = regularization mạnh,
    hợp với fMRI có nhiều voxel và ít mẫu. Không chọn alpha bằng CV ngẫu nhiên trong tập train, vì
    các volume liền nhau tương quan với nhau (xem exp_cv)."""
    return make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))


# ---------------------------------------------------------------- thí nghiệm

def exp_cv(X, y, runs, seed=0) -> dict:
    """Độ chính xác với hai cách chia train/test:
    - leave-one-run-out: train và test ở các run khác nhau (đúng);
    - random 5-fold: trộn volume ngẫu nhiên. Các volume liền nhau trong cùng block rất giống nhau
      (tín hiệu BOLD chậm), nên volume ở train và test gần như trùng -> độ chính xác bị thổi phồng."""
    logo = cross_val_score(classifier(), X, y, groups=runs, cv=LeaveOneGroupOut())
    rnd = cross_val_score(classifier(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return {"leave_one_run_out": float(logo.mean()), "random_kfold": float(rnd.mean()),
            "chance": 1 / len(np.unique(y)), "per_run": logo.tolist()}


def exp_permutation(X, y, runs, n_permutations=100, seed=0) -> dict:
    """Kiểm định hoán vị: xáo nhãn trong từng run rồi decode lại. Phân phối null phải quanh may rủi."""
    score, null, p = permutation_test_score(classifier(), X, y, groups=runs, cv=LeaveOneGroupOut(),
                                            n_permutations=n_permutations, random_state=seed)
    return {"score": float(score), "null_mean": float(null.mean()), "null_95": float(np.quantile(null, 0.95)),
            "p_value": float(p), "null": null.tolist()}


def exp_budget(X, y, runs, n_train_list=(1, 2, 4, 8, 11), n_draws=5, seed=0) -> pd.DataFrame:
    """Cần bao nhiêu run để train? Test trên một run, train trên n run khác chọn ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(runs)
    rows = []
    for test_run in uniq:
        others = uniq[uniq != test_run]
        te = runs == test_run
        for n in n_train_list:
            if n > len(others):
                continue
            for d in range(n_draws if n < len(others) else 1):
                tr = np.isin(runs, rng.choice(others, n, replace=False))
                if len(np.unique(y[tr])) < 2:
                    continue
                acc = classifier().fit(X[tr], y[tr]).score(X[te], y[te])
                rows.append({"test_run": int(test_run), "n_train_runs": n, "draw": d, "accuracy": acc})
    return pd.DataFrame(rows)


def exp_lag(X_all, y_all, runs_all, lags=range(0, 6)) -> pd.DataFrame:
    """Độ chính xác (leave-one-run-out) theo độ trễ gán nhãn. Đỉnh nên ở khoảng 2 TR (~5 s),
    đúng với độ trễ của đáp ứng huyết động."""
    rows = []
    for lag in lags:
        X, y, runs = select(X_all, y_all, runs_all, lag)
        acc = cross_val_score(classifier(), X, y, groups=runs, cv=LeaveOneGroupOut()).mean()
        rows.append({"lag_tr": lag, "lag_s": lag * TR, "accuracy": float(acc)})
    return pd.DataFrame(rows)


def exp_confusion(X, y, runs, labels=CATEGORIES) -> np.ndarray:
    """Ma trận nhầm lẫn (hàng = nhãn đúng, đã chuẩn hoá theo hàng), cộng dồn qua các fold."""
    labels = [l for l in labels if l in set(y)]
    M = np.zeros((len(labels), len(labels)))
    for tr, te in LeaveOneGroupOut().split(X, y, runs):
        pred = classifier().fit(X[tr], y[tr]).predict(X[te])
        M += confusion_matrix(y[te], pred, labels=labels)
    return M / M.sum(1, keepdims=True)
