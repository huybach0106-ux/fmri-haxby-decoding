"""Dữ liệu giả lập cùng cấu trúc với Haxby (12 run x 8 block x 9 volume, có rest xen giữa), dùng để
kiểm tra code. Tín hiệu có độ trễ 2 TR và tự tương quan theo thời gian, nên chạy đúng thì phải thấy:
random k-fold cao hơn leave-one-run-out, độ trễ tốt nhất ~2 TR, null hoán vị quanh 12,5%."""
import numpy as np

from haxby import CATEGORIES


def make_haxby_like(subject=1, mask="vt", n_runs=12, block=9, rest=6, snr=0.35, delay=2):
    n_vox = {"vt": 400, "face": 150, "house": 150}[mask]
    rng = np.random.default_rng(1000 * subject + {"vt": 0, "face": 1, "house": 2}[mask])
    patterns = rng.standard_normal((len(CATEGORIES), n_vox))
    if mask != "vt":
        patterns *= 0.6
    Xs, ys, rs = [], [], []
    for r in range(n_runs):
        order = rng.permutation(len(CATEGORIES))
        labels = ["rest"] * rest
        for c in order:
            labels += [CATEGORIES[c]] * block + ["rest"] * rest
        labels = np.array(labels)
        sig = np.array([patterns[CATEGORIES.index(l)] if l != "rest" else np.zeros(n_vox) for l in labels])
        sig = np.vstack([np.zeros((delay, n_vox)), sig[:-delay]])           # trễ huyết động
        noise = np.zeros((len(labels), n_vox))
        e = rng.standard_normal((len(labels), n_vox))
        for t in range(len(labels)):                                         # nhiễu tự tương quan AR(1)
            noise[t] = (0.7 * noise[t - 1] if t else 0) + e[t]
        block_offset = np.repeat(rng.standard_normal((len(labels) // block + 1, n_vox)) * 0.8, block, 0)[:len(labels)]
        X = snr * sig + noise + block_offset
        X = (X - X.mean(0)) / X.std(0)
        Xs.append(X); ys.append(labels); rs.append(np.full(len(labels), r))
    return np.vstack(Xs).astype(np.float32), np.concatenate(ys), np.concatenate(rs)
