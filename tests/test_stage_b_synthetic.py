"""Stage B synthetic harness (SKILL §7) — written BEFORE the estimators are
trusted; no real bundle is touched until this is green.

Generators plant: static scene + rigid mover on a known SE(3) path, piecewise
scale gauge with configurable boundaries and jump sizes (defaults = the
measured 16-19% at [35,47,59]), injected b_t, heavy-tailed (Student-t) noise,
an outlier fraction, and a frame-dominant variant for the rho/fallback path.

Run:  python tests/test_stage_b_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.config import load_config                      # noqa: E402
from src.preprocess.stage_b import follow_cam, static_fallback     # noqa: E402
from src.preprocess.stage_b.clustering import cluster_tracks       # noqa: E402
from src.preprocess.stage_b.drift import fit_drift, two_pass       # noqa: E402
from src.preprocess.stage_b.scoring import eta_weights, pair_weights  # noqa: E402
from src.preprocess.stage_b.select import score_clusters           # noqa: E402

HW = 256
K = np.array([[200.0, 0, 128], [0, 200, 128], [0, 0, 1]])

TH = {"rho": 0.4, "tau_b": 0.02, "tau_z": 2.5, "tau_r": 0.1, "tau_d": 0.5,
      "tau_x": 3.0, "f_min": 20, "n_min": 10}
LAMBDAS = (1.0, 1.0, 1.0)


def gen_bundle(seed=0, T=81, boundaries=(35, 47, 59), jumps=(0.18, 0.10, 0.16),
               drift_rate=0.003, n_static=260, n_dyn=60, noise=0.001,
               outlier_frac=0.05, b_inject=None, invalid_frac=0.08,
               noise_seg_growth=None, late_spawn_frac=0.0, turning=False,
               speed=None, fake_blob_n=0):
    rng = np.random.default_rng(seed)
    # static points spread to the image borders (so the fallback has borders)
    ang = rng.uniform(0, 2 * np.pi, n_static)
    rad = rng.uniform(0.3, 1.0, n_static) ** 0.5
    Zs = rng.uniform(4, 12, n_static)
    Xs = np.stack([rad * np.cos(ang) * 0.6 * Zs, rad * np.sin(ang) * 0.6 * Zs, Zs], 1)
    # rigid mover: compact blob, translating + slowly rotating about centroid
    c0 = np.array([0.5, 0.2, 7.0])
    blob = c0 + rng.normal(0, 0.35, (n_dyn, 3))
    omega = 0.01
    if speed is None:
        speed = np.linalg.norm([0.08, 0.0, 0.02])
    unit_v = np.array([0.08, 0.0, 0.02]) / np.linalg.norm([0.08, 0.0, 0.02])
    X_true = np.zeros((n_static + n_dyn, T, 3))
    X_true[:n_static] = Xs[:, None, :]
    c_t = c0.copy()
    for t in range(T):
        th_r = omega * t
        R = np.array([[np.cos(th_r), 0, np.sin(th_r)], [0, 1, 0],
                      [-np.sin(th_r), 0, np.cos(th_r)]])
        if turning:
            # drifting-car geometry: velocity DIRECTION sweeps a full circle,
            # so the whole-track median motion cancels EXACTLY — its direction
            # is then pure noise (the measured real-clip failure) while
            # per-segment directions remain well-defined
            phi = 6.28 * t / T
            v_t = speed * np.array([np.cos(phi), 0.0, np.sin(phi)])
        else:
            v_t = speed * unit_v
        if t > 0:
            c_t = c_t + v_t
        X_true[n_static:, t] = (blob - c0) @ R.T + c_t
    y_true = np.zeros(n_static + n_dyn, bool)
    y_true[n_static:] = True

    # piecewise gauge: within-segment drift + jumps at boundaries
    g = np.ones(T)
    for t in range(1, T):
        step = 1.0 + drift_rate
        if t in boundaries:
            step *= 1.0 + jumps[list(boundaries).index(t)]
        g[t] = g[t - 1] * step
    X0 = X_true * g[None, :, None]
    if b_inject is not None:
        X0 = X0 + np.cumsum(np.broadcast_to(b_inject, (T, 3)), axis=0)[None]

    N = n_static + n_dyn
    tnoise = rng.standard_t(3, (N, T, 3)) * noise * X_true[..., 2:3]
    if noise_seg_growth is not None:
        # per-segment noise multipliers — the measured instrument behavior:
        # X0 noise steps up at window boundaries (late windows are noisier)
        seg_of_t = np.zeros(T, int)
        for bd in boundaries:
            seg_of_t[int(bd):] += 1
        tnoise *= np.asarray(noise_seg_growth)[seg_of_t][None, :, None]
    out_mask = rng.random((N, T)) < outlier_frac
    tnoise[out_mask] *= 20
    X0_obs = X0 + tnoise

    o = np.stack([0.02 * np.arange(T), np.zeros(T), 0.05 * np.arange(T)], 1)
    G = np.tile(np.eye(4), (T, 1, 1))
    G[:, :3, 3] = o
    X_local = X0 - o[None] + rng.normal(0, 0.5 * noise * 7, (N, T, 3))

    if fake_blob_n > 0:
        # correlated-noise blob (the "sky cluster" mechanism, 2026-08-09b):
        # far static-truth tracks whose X0 shares a common random walk that
        # X_local does NOT follow -> pairwise-rigid, direction-coherent, and
        # low-eta by construction (X0 disagrees with G·X_local)
        blob_xy = rng.uniform(-2, 2, (fake_blob_n, 2))
        blob_z = rng.uniform(10, 12, fake_blob_n)
        Xb = np.concatenate([blob_xy, blob_z[:, None]], 1)
        walk = np.cumsum(rng.normal(0, 0.02, (T, 3)), axis=0)
        Xb_true = np.repeat(Xb[:, None, :], T, axis=1)
        Xb0 = Xb_true * g[None, :, None] + walk[None, :, :] \
            + rng.standard_t(3, (fake_blob_n, T, 3)) * noise * Xb_true[..., 2:3]
        Xb_local = Xb_true * g[None, :, None] - o[None] \
            + rng.normal(0, 0.5 * noise * 7, (fake_blob_n, T, 3))
        X0_obs = np.concatenate([X0_obs, Xb0])
        X_local = np.concatenate([X_local, Xb_local])
        X_true = np.concatenate([X_true, Xb_true])
        y_true = np.concatenate([y_true, np.zeros(fake_blob_n, bool)])
        out_mask = np.concatenate([out_mask,
                                   np.zeros((fake_blob_n, T), bool)])
        N = N + fake_blob_n  # Q and anchors are built below with updated N

    Q = rng.random((N, T)) > invalid_frac
    # late-spawned static tracks (mid-entry anchors, s_i > 0): their whole
    # life sits in the noisy late windows — the real clips' FP population
    s_i = np.zeros(N, int)
    if late_spawn_frac > 0 and boundaries:
        n_late = int(late_spawn_frac * n_static)
        pick = rng.choice(n_static, n_late, replace=False)
        s_i[pick] = rng.choice(list(boundaries), n_late)
        for i in pick:
            Q[i, :s_i[i]] = False
    spawn_t = np.clip(s_i, 0, T - 1)
    pr = X_local[np.arange(N), spawn_t] @ K.T
    uv = np.clip(pr[:, :2] / np.maximum(pr[:, 2:3], 1e-6), 0, HW - 1)
    anchors = np.column_stack([uv, s_i.astype(float)])
    a_true = np.full(T, np.nan)
    a_true[1:] = g[1:] / g[:-1] - 1.0
    return {"X0": X0_obs, "X_local": X_local, "G": G, "Q": Q,
            "anchors": anchors, "y_true": y_true, "a_true": a_true,
            "boundaries": list(boundaries), "g": g, "out_mask": out_mask}


def prep(bundle, cfg):
    eta = eta_weights(bundle["X0"], bundle["X_local"], bundle["G"], bundle["Q"])
    w = pair_weights(eta, bundle["Q"])
    return eta, w


def test_drift_recovery(cfg):
    b = gen_bundle(seed=1, n_dyn=0, outlier_frac=0.03)
    _, w = prep(b, cfg)
    a, bt, r = fit_drift(b["X0"], b["Q"], w, np.ones(len(b["Q"]), bool),
                         b["boundaries"], cfg)
    inseg = np.ones(81, bool)
    inseg[0] = False
    for t in b["boundaries"]:
        inseg[t] = False
    err = np.abs(a[inseg] - b["a_true"][inseg])
    assert np.nanmedian(err) < 5e-4, f"in-segment a err {np.nanmedian(err)}"
    assert np.nanmedian(np.linalg.norm(bt[inseg], axis=1)) < 5e-3
    for t, jump in zip(b["boundaries"], (0.18, 0.10, 0.16)):
        assert abs(a[t] - b["a_true"][t]) < 0.1 * b["a_true"][t], \
            f"jump at {t}: fitted {a[t]:.4f} vs true {b['a_true'][t]:.4f}"
    # jumps localized: large |a| ONLY at boundaries
    big = np.flatnonzero(np.nan_to_num(np.abs(a)) > 0.05)
    assert set(big) == set(b["boundaries"]), f"spurious jumps at {big}"
    print(f"PASS drift recovery (in-seg med err {np.nanmedian(err):.2e}; "
          f"jumps localized to {b['boundaries']})")


def test_b_injection(cfg):
    b = gen_bundle(seed=7, n_dyn=0, b_inject=np.array([0.01, -0.005, 0.0]),
                   drift_rate=0.0, jumps=(0, 0, 0))
    _, w = prep(b, cfg)
    a, bt, _ = fit_drift(b["X0"], b["Q"], w, np.ones(len(b["Q"]), bool),
                         b["boundaries"], cfg)
    ok = np.isfinite(bt[:, 0])
    med_b = np.median(bt[ok], axis=0)
    assert np.linalg.norm(med_b - [0.01, -0.005, 0.0]) < 2e-3, med_b
    print(f"PASS b_t injection recovered ({med_b.round(4)})")


def test_labels_and_clustering(cfg):
    b = gen_bundle(seed=2)
    eta, w = prep(b, cfg)
    ctx = {"anchors": b["anchors"], "process_hw": (HW, HW)}
    res = two_pass(b["X0"], b["X_local"], b["Q"], w, b["boundaries"], cfg, TH,
                   static_fallback.select, ctx)
    # fallback must NOT fire here; rho must be small
    assert res["rho_pass1"] < 0.4 and not res["used_fallback"], res["rho_pass1"]
    y, yt, lab = res["y_dyn"], b["y_true"], res["labeled"]
    tp = (y & yt & lab).sum()
    prec = tp / max((y & lab).sum(), 1)
    rec = tp / max((yt & lab).sum(), 1)
    assert prec >= 0.95 and rec >= 0.95, (prec, rec)

    cid, diag = cluster_tracks(b["X0"], res["r"], b["Q"], b["anchors"],
                               res["y_dyn"], b["boundaries"], cfg, TH,
                               a=res["a"])
    in_main = cid[b["y_true"] & res["labeled"]]
    main = np.bincount(in_main[in_main >= 0]).argmax()
    frac = (in_main == main).mean()
    assert frac >= 0.9, f"cluster capture {frac}"
    o_src, rows = score_clusters(cid, res["m"], b["Q"], LAMBDAS, TH["n_min"])
    assert o_src == main
    print(f"PASS labels (prec {prec:.3f} rec {rec:.3f}) + clustering "
          f"(capture {frac:.2f}, {diag['n_clusters']} cluster(s), O_src={o_src})")


def test_fallback_path(cfg):
    b = gen_bundle(seed=3, n_static=90, n_dyn=240)   # frame-dominant mover
    eta, w = prep(b, cfg)
    ctx = {"anchors": b["anchors"], "process_hw": (HW, HW)}
    res = two_pass(b["X0"], b["X_local"], b["Q"], w, b["boundaries"], cfg, TH,
                   static_fallback.select, ctx)
    # HARNESS FINDING: at 73% contamination the fit fully INVERTS — rho stays
    # small (labels flip), and the G3 arm (b_med_pass1 >> tau_b) must fire.
    assert res["used_fallback"], \
        (res["rho_pass1"], res["b_med_pass1"], "breakdown not detected")
    assert res["b_med_pass1"] > TH["tau_b"], res["b_med_pass1"]
    y, yt, lab = res["y_dyn"], b["y_true"], res["labeled"]
    rec = (y & yt & lab).sum() / max((yt & lab).sum(), 1)
    assert rec >= 0.9, f"post-fallback recall {rec}"
    print(f"PASS fallback path (rho {res['rho_pass1']:.2f} -> fallback, "
          f"recall {rec:.3f})")


def test_eta_downweights_outliers(cfg):
    b = gen_bundle(seed=4, n_dyn=0, outlier_frac=0.06)
    eta = eta_weights(b["X0"], b["X_local"], b["G"], b["Q"])
    med_out = np.median(eta[b["out_mask"]])
    med_in = np.median(eta[~b["out_mask"]])
    assert med_out < 0.5 * med_in, (med_out, med_in)
    print(f"PASS eta downweights outliers ({med_out:.3f} vs {med_in:.3f})")


def test_time_heteroscedastic(cfg):
    """Measured failure (real follow clip, 2026-08-08): late-window X0 noise
    made a GLOBAL null floor label quiet late-window background as dynamic.
    Plant per-segment noise growth; global floor must fail, segment floor
    must fix — precision is the discriminator, recall must survive both."""
    import copy
    b = gen_bundle(seed=5, noise_seg_growth=(1.0, 2.5, 4.0, 6.0),
                   late_spawn_frac=0.4)
    _, w = prep(b, cfg)
    ctx = {"anchors": b["anchors"], "process_hw": (HW, HW)}
    out = {}
    for mode in ("none", "segment"):
        c = copy.deepcopy(cfg)
        c["scoring"]["time_bands"] = mode
        res = two_pass(b["X0"], b["X_local"], b["Q"], w, b["boundaries"], c,
                       TH, static_fallback.select, ctx)
        y, yt, lab = res["y_dyn"], b["y_true"], res["labeled"]
        tp = (y & yt & lab).sum()
        out[mode] = (tp / max((y & lab).sum(), 1), tp / max((yt & lab).sum(), 1))
    (p0, r0), (p1, r1) = out["none"], out["segment"]
    assert p0 < 0.9, f"global floor should mislabel late statics (prec {p0:.2f})"
    assert p1 >= 0.95 and r1 >= 0.9, (p1, r1)
    print(f"PASS time-heteroscedastic (global floor prec {p0:.2f} -> "
          f"segment floor prec {p1:.2f}, rec {r1:.2f})")


def test_turning_mover(cfg):
    """Drifting-car case (measured 2026-08-08): a mover whose velocity
    direction rotates. The norm-of-median-vector floor disqualified it (real
    O_src fell to 5 noise tracks); the median-SPEED floor must keep it."""
    b = gen_bundle(seed=6, turning=True)
    _, w = prep(b, cfg)
    ctx = {"anchors": b["anchors"], "process_hw": (HW, HW)}
    res = two_pass(b["X0"], b["X_local"], b["Q"], w, b["boundaries"], cfg, TH,
                   static_fallback.select, ctx)
    cid, diag = cluster_tracks(b["X0"], res["r"], b["Q"], b["anchors"],
                               res["y_dyn"], b["boundaries"], cfg, TH,
                               a=res["a"])
    in_main = cid[b["y_true"] & res["labeled"]]
    assert (in_main >= 0).any(), "turning mover lost every cluster edge"
    main_c = np.bincount(in_main[in_main >= 0]).argmax()
    frac = (in_main == main_c).mean()
    o_src, _ = score_clusters(cid, res["m"], b["Q"], LAMBDAS, TH["n_min"])
    assert frac >= 0.85 and o_src == main_c, (frac, o_src, main_c)
    # contrast: the appendix-literal whole-track r_bar mode must do WORSE on
    # a full-circle mover (its median motion cancels; direction = noise)
    import copy
    c2 = copy.deepcopy(cfg)
    c2["clustering"]["direction_mode"] = "r_bar"
    cid2, _ = cluster_tracks(b["X0"], res["r"], b["Q"], b["anchors"],
                             res["y_dyn"], b["boundaries"], c2, TH,
                             a=res["a"])
    in2 = cid2[b["y_true"] & res["labeled"]]
    frac2 = (in2 == (np.bincount(in2[in2 >= 0]).argmax() if (in2 >= 0).any()
                     else -2)).mean()
    # NOTE: r_bar mode may also pass here — 60 clean points percolate even at
    # random-direction edge rates. The real clip stacks direction noise on
    # tau_r cuts + window incoherence; the edge autopsy in run_stage_b is the
    # instrument that shows the per-criterion rates on real data.
    print(f"PASS turning mover (segment capture {frac:.2f}, O_src={o_src}; "
          f"r_bar-mode reference capture {frac2:.2f})")


def test_low_snr(cfg):
    """The measured regime: object motion ~1.5x the pairwise noise floor
    (real clips measured 1.3-1.8x). Uses the REAL provisional-threshold
    formulas (noise-floor-anchored tau_r, de-gauged) end to end."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_stage_b import provisional_cluster_th
    b = gen_bundle(seed=8, speed=0.05, drift_rate=0.004)
    _, w = prep(b, cfg)
    ctx = {"anchors": b["anchors"], "process_hw": (HW, HW)}
    res = two_pass(b["X0"], b["X_local"], b["Q"], w, b["boundaries"], cfg, TH,
                   static_fallback.select, ctx)
    res["labeled"] = res["labeled"]
    y, yt, lab = res["y_dyn"], b["y_true"], res["labeled"]
    rec = (y & yt & lab).sum() / max((yt & lab).sum(), 1)
    assert rec >= 0.8, f"low-SNR labeling recall {rec:.2f}"
    import yaml as _y
    cfg_b = _y.safe_load(Path("configs/stage_b.yaml").read_text(encoding="utf-8"))
    th = provisional_cluster_th(
        {"X0": b["X0"], "Q": b["Q"], "X_local": b["X_local"]}, res,
        {"rho": TH["rho"], "tau_b": TH["tau_b"], "tau_z": TH["tau_z"],
         "n_min": TH["n_min"]}, cfg_b)
    cid, diag = cluster_tracks(b["X0"], res["r"], b["Q"], b["anchors"],
                               y, b["boundaries"], cfg, th, a=res["a"])
    in_main = cid[yt & lab & y]
    assert (in_main >= 0).any(), \
        f"low-SNR mover lost all edges (th={ {k: round(float(v), 4) for k, v in th.items()} })"
    main_c = np.bincount(in_main[in_main >= 0]).argmax()
    frac = (in_main == main_c).mean()
    o_src, _ = score_clusters(cid, res["m"], b["Q"], LAMBDAS, th["n_min"])
    assert frac >= 0.8 and o_src == main_c, (frac, o_src, main_c)
    print(f"PASS low-SNR (recall {rec:.2f}, capture {frac:.2f}, "
          f"tau_r={th['tau_r']:.4f} floor-anchored, O_src={o_src})")


def test_eta_referee(cfg):
    """The sky-cluster mechanism (measured 2026-08-09b): a correlated-noise
    blob is rigid and direction-coherent by construction — no motion
    statistic rejects it — but its X0 disagrees with G·X_local, so eta does.
    The gate must exclude it and O_src must be the true mover."""
    NB = 90
    b = gen_bundle(seed=9, fake_blob_n=NB)
    eta, w = prep(b, cfg)
    N = len(b["y_true"])
    blob = np.arange(N - NB, N)
    other = np.arange(N - NB)
    et = np.where(b["Q"], eta, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        et = np.nanmean(et, axis=1)
    assert np.nanmedian(et[blob]) < 0.3 * np.nanmedian(et[other]), \
        "blob not low-eta — generator broken"
    ctx = {"anchors": b["anchors"], "process_hw": (HW, HW)}
    res = two_pass(b["X0"], b["X_local"], b["Q"], w, b["boundaries"], cfg, TH,
                   static_fallback.select, ctx)
    cid, diag = cluster_tracks(b["X0"], res["r"], b["Q"], b["anchors"],
                               res["y_dyn"], b["boundaries"], cfg, TH,
                               a=res["a"], eta=eta)
    o_src, rows = score_clusters(cid, res["m"], b["Q"], LAMBDAS, TH["n_min"],
                                 eta=eta)
    assert o_src is not None, "no cluster selected at all"
    o_members = np.flatnonzero(cid == o_src)
    mover_frac = float(b["y_true"][o_members].mean())
    blob_frac = float(np.isin(o_members, blob).mean())
    assert mover_frac >= 0.8 and blob_frac <= 0.1, (mover_frac, blob_frac)
    print(f"PASS eta referee (O_src purity {mover_frac:.2f}, blob in O_src "
          f"{blob_frac:.2f}, eta-gated {diag['edge_diag']['n_eta_gated']} tracks)")


def gen_followcam_bundle(seed=0, T=81, boundaries=(35, 47, 59), n_static=260,
                         n_follow=60, n_halo=40, noise=0.001, wobble_amp=0.02):
    """Follow-cam scenario, replicating the MEASURED real-clip pathology:
    followers ride the camera exactly (camera-frame quiet); the cross-frame
    head FAILS on them (X0 shows only the relative wobble, no world travel);
    halo tracks are static-truth points whose measurements are glued to the
    subject's motion field (tracker bleed) — indistinguishable from the
    followers by construction."""
    rng = np.random.default_rng(seed)
    o = np.stack([0.02 * np.arange(T), np.zeros(T), 0.05 * np.arange(T)], 1)
    ang = rng.uniform(0, 2 * np.pi, n_static)
    rad = rng.uniform(0.3, 1.0, n_static) ** 0.5
    Zs = rng.uniform(4, 12, n_static)
    Xs = np.stack([rad * np.cos(ang) * 0.6 * Zs, rad * np.sin(ang) * 0.6 * Zs,
                   Zs], 1)
    X0_s = np.repeat(Xs[:, None, :], T, 1) \
        + rng.standard_t(3, (n_static, T, 3)) * noise * Xs[:, None, 2:3]
    Xl_s = Xs[:, None, :] - o[None] \
        + rng.normal(0, 0.5 * noise * 7, (n_static, T, 3))

    # followed blob + bleed halo: identical construction, wider spread
    w = wobble_amp * np.stack([np.sin(2 * np.pi * np.arange(T) / 27),
                               np.zeros(T), np.zeros(T)], 1)
    vf = np.array([0.3, 0.2, 6.0]) + rng.normal(0, 0.35, (n_follow, 3))
    vh = np.array([0.3, 0.2, 6.0]) + rng.normal(0, 0.9, (n_halo, 3))
    v0 = np.concatenate([vf, vh])
    n_b = n_follow + n_halo
    Xl_b = v0[:, None, :] + w[None] \
        + rng.normal(0, 0.5 * noise * 7, (n_b, T, 3))
    X0_b = v0[:, None, :] + w[None] \
        + rng.standard_t(3, (n_b, T, 3)) * noise * v0[:, None, 2:3]

    X0 = np.concatenate([X0_s, X0_b])
    Xl = np.concatenate([Xl_s, Xl_b])
    N = n_static + n_b
    G = np.tile(np.eye(4), (T, 1, 1))
    G[:, :3, 3] = o
    Q = rng.random((N, T)) > 0.08
    pr = Xl[:, 0] @ K.T
    uv = np.clip(pr[:, :2] / np.maximum(pr[:, 2:3], 1e-6), 0, HW - 1)
    anchors = np.column_stack([uv, np.zeros(N)])
    P = np.full((N, T, 2), np.nan, np.float32)
    for t in range(T):
        prt = Xl[:, t] @ K.T
        P[:, t] = np.clip(prt[:, :2] / np.maximum(prt[:, 2:3], 1e-6),
                          0, HW - 1)
    y_blob = np.zeros(N, bool)
    y_blob[n_static:] = True
    return {"X0": X0, "X_local": Xl, "G": G, "Q": Q, "anchors": anchors,
            "P": P, "boundaries": list(boundaries), "y_blob": y_blob,
            "wobble": w}


def test_follow_cam_detect(cfg):
    b = gen_followcam_bundle(seed=4)
    fc = follow_cam.detect(b["X_local"], b["X0"], b["G"], b["Q"], b["P"],
                           b["anchors"], b["boundaries"], cfg)
    assert fc["fired"], fc
    blob = np.zeros(len(b["y_blob"]), bool)
    blob[fc["blob_idx"]] = True
    recall = float(blob[b["y_blob"]].mean())
    leak = float(blob[~b["y_blob"]].mean())
    assert recall >= 0.9, recall
    assert leak <= 0.05, leak
    # provisional relative rigid motion recovers the planted wobble
    ref = fc["ref_frame"]
    ok = fc["rel_valid"]
    err = np.linalg.norm(fc["tau_rel"][ok]
                         - (b["wobble"][ok] - b["wobble"][ref]), axis=1)
    assert np.median(err) < 0.005, np.median(err)
    # seed bbox covers the blob's true image footprint
    u = b["P"][b["y_blob"], fc["seed_frame"], 0]
    x0, y0, x1, y1 = fc["seed_bbox"]
    inside = float(((u >= x0 - 5) & (u <= x1 + 5)).mean())
    assert inside >= 0.85, inside
    print(f"PASS follow-cam detect (recall {recall:.2f}, static leak "
          f"{leak:.2f}, wobble err {np.median(err):.4f}, "
          f"step/floor {fc['step_over_floor']})")


def test_follow_cam_negative(cfg):
    # lateral mover, clean gauge: streams in camera frame -> the branch must
    # NOT fire (the X0 path owns this case, and it works)
    b = gen_bundle(seed=2, drift_rate=0.0, jumps=(0, 0, 0))
    fc = follow_cam.detect(b["X_local"], b["X0"], b["G"], b["Q"], None,
                           b["anchors"], b["boundaries"], cfg)
    assert not fc["fired"], fc
    # static camera: quiet population may be huge, but zero travel blocks it
    b2 = gen_followcam_bundle(seed=5)
    b2["G"][:, :3, 3] = 0.0
    b2["X_local"] = b2["X0"] + 0.0     # static cam: local == world frame
    fc2 = follow_cam.detect(b2["X_local"], b2["X0"], b2["G"], b2["Q"],
                            b2["P"], b2["anchors"], b2["boundaries"], cfg)
    assert not fc2["fired"], fc2
    print(f"PASS follow-cam negatives (lateral n_blob {fc['n_blob']}, "
          f"static-cam step/floor {fc2['step_over_floor']})")


def test_starvation_predicate(cfg):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_stage_b import starved, winner_blob_overlap
    N = 100
    cid = np.full(N, -1)
    cid[:20] = 0                              # winner = tracks 0..19
    rows = [{"cluster": 0, "size": 20}]
    res = {"y_dyn": np.ones(N, bool)}
    fc_hit = {"fired": True, "blob_idx": np.arange(10, 40),
              "blob_core_idx": np.arange(10, 25)}   # core meets the winner
    fc_miss = {"fired": True, "blob_idx": np.arange(50, 90),
               "blob_core_idx": np.arange(50, 65)}  # disjoint
    # v3 (2026-08-21): halo-defeat — winner touches the BLOB heavily but
    # not its core (the 299.1.006 signature: whole-blob 1.0, core 0.0)
    fc_halo = {"fired": True, "blob_idx": np.arange(0, 60),
               "blob_core_idx": np.arange(40, 60)}
    assert winner_blob_overlap(cid, 0, fc_hit) == 0.5
    assert winner_blob_overlap(cid, 0, fc_hit, core=True) == 0.5
    assert not starved(0, rows, res, cfg, cid, fc_hit), \
        "winner covering the subject core must keep the X0 path"
    assert starved(0, rows, res, cfg, cid, fc_miss), \
        "winner disjoint from the subject must yield to the branch"
    assert winner_blob_overlap(cid, 0, fc_halo) == 1.0
    assert starved(0, rows, res, cfg, cid, fc_halo), \
        "halo-only overlap must yield to the branch (v3)"
    # legacy fc record without blob_core_idx -> whole-blob path still works
    fc_legacy = {"fired": True, "blob_idx": np.arange(10, 40)}
    assert not starved(0, rows, res, cfg, cid, fc_legacy)
    assert starved(None, rows, res, cfg, cid, fc_hit), "no winner -> starved"
    # detector silent -> v1 size arm only
    assert not starved(0, rows, res, cfg, cid, {"fired": False}), \
        "20% winner with silent detector is healthy"
    res_big = {"y_dyn": np.ones(1000, bool)}
    assert starved(0, rows, res_big, cfg, cid, None), \
        "2% winner with silent detector is starved (size arm)"
    print("PASS starvation predicate v2 (overlap arm + silent-detector size arm)")


def test_write_outputs_keys(cfg):
    """Regression (2026-08-10): the fc block in write_outputs REASSIGNED
    `extra`, silently discarding the post-hoc instrumentation arrays on every
    clip where the detector fired. Assert all output groups coexist."""
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_stage_b import write_outputs
    N, F = 40, 81
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        cfg_fake = {"paths": {"stage_b_cache": td},
                    "_meta": {"config_hash": "testhash"}}
        b = {"provenance": {"config_hash": "uphash"}}
        res = {"y_dyn": rng.random(N) < 0.3, "m": rng.random(N),
               "a": np.full(F, 0.001), "b": np.zeros((F, 3)),
               "rho_pass1": 0.1, "b_med_pass1": 0.001, "b_med_pass2": 0.001,
               "used_fallback": False, "labeled": np.ones(N, bool),
               "segment_scales": []}
        cid = np.where(np.arange(N) < 10, 0, -1)
        cdiag = {"edge_diag": {"joint": 0.1},
                 "posthoc_mask": np.zeros(N, bool),
                 "posthoc_votes": np.zeros(N, np.int32)}
        fc = {"fired": True, "blob_idx": np.arange(5),
              "blob_core_idx": np.arange(3),
              "R_rel": np.zeros((F, 3, 3), np.float32),
              "tau_rel": np.zeros((F, 3), np.float32),
              "rel_valid": np.ones(F, bool),
              "seed_points": np.zeros((4, 2), np.float32),
              "n_quiet": 5, "n_blob": 5, "seed_frame": 0,
              "seed_bbox": [0, 0, 1, 1]}
        write_outputs(cfg_fake, cfg, "testclip", b, res, cid, 0,
                      [{"cluster": 0, "size": 10}], [35, 47, 59],
                      {"tau_z": 1.5}, cdiag=cdiag, fc=fc, branch_active=True)
        with np.load(Path(td) / "testclip.npz") as z:
            keys = set(z.files)
        need = {"y_dyn", "m", "cluster_id", "o_src_members", "a", "b",
                "posthoc_mask", "posthoc_votes", "fc_blob_idx",
                "fc_blob_core_idx", "fc_R_rel",
                "fc_tau_rel", "fc_rel_valid", "fc_seed_points"}
        assert need <= keys, f"missing from npz: {need - keys}"
        assert "posthoc_mask" in cdiag, "caller's cdiag must not be mutated"
    print(f"PASS write_outputs keys ({len(need)} required arrays coexist)")


def main():
    cfg = load_config("configs/stage_a.yaml")
    import yaml
    cfg_b = yaml.safe_load(Path("configs/stage_b.yaml").read_text(encoding="utf-8"))
    cfg = {**cfg_b, "eps": cfg_b["eps"]}
    test_drift_recovery(cfg)
    test_b_injection(cfg)
    test_eta_downweights_outliers(cfg)
    test_labels_and_clustering(cfg)
    test_fallback_path(cfg)
    test_time_heteroscedastic(cfg)
    test_turning_mover(cfg)
    test_low_snr(cfg)
    test_eta_referee(cfg)
    test_follow_cam_detect(cfg)
    test_follow_cam_negative(cfg)
    test_starvation_predicate(cfg)
    test_write_outputs_keys(cfg)
    print("\nall Stage B synthetic tests passed")


if __name__ == "__main__":
    main()
