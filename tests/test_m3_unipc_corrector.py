"""Self-test (rule 13) of the mechanism the sampler relies on since 2026-10-14: diffusers' UniPCMultistepScheduler.step(k) runs its corrector
BEFORE the predictor, and the corrector REBUILDS the sample from the sample of step k-1 and the model outputs; the sample passed into step(k)
enters only through its x0 conversion (x0 = sample - sigma * v). So a z-form A3 correction applied at step k and passed into step(k) is
ATTENUATED there unless the corrector is off at that step (disable_corrector holds k-1 for step k). Known answer on the real scheduler (flow
sigmas, the lock's shift, synthetic model outputs held FIXED so the sample's only route is the scheduler's algebra, every comparison against
the unperturbed trajectory of the SAME scheduler configuration):
  corrector on  -> a perturbation delta of the sample passed into step(k) survives at c * delta with c well below 1 (measured 0.09 at sigma 0.92,
                   0.15 at 0.82, 0.44 at 0.56): the z-form correction would act at a fraction of its gamma;
  corrector off -> c = 1.0-1.15, as under the FlowMatchEuler scheduler the gamma probe was run with (coefficient exactly 1 there), and the
                   corrector of the NEXT step starts from the corrected sample, so the correction carries forward.
The map is affine in the sample for fixed outputs (proportionality residual < 1e-4), so the coefficients are scale-free.
"""
from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

STEPS, SHIFT = 10, 3.0
SHAPE = (1, 4, 2, 6, 6)


def make(disable):
    sch = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=SHIFT, solver_order=2, disable_corrector=list(disable))
    sch.set_timesteps(STEPS)
    return sch


def run(sch, perturb_at, delta, model_outputs):
    z = torch.randn(*SHAPE, generator=torch.Generator().manual_seed(1)); zs = []
    for k, t in enumerate(sch.timesteps):
        if k == perturb_at:
            z = z + delta
        z = sch.step(model_outputs[k], t, z, return_dict=False)[0]
        zs.append(z.clone())
    return zs


def coefficient(o, base, delta, j):
    d = o[j] - base[j]; c = float((d * delta).sum() / (delta * delta).sum())
    return c, float((d - c * delta).norm() / (d.norm() + 1e-12))


def main():
    gen = torch.Generator().manual_seed(2)
    outs = [torch.randn(*SHAPE, generator=gen) for _ in range(STEPS)]                   # fixed model outputs: the sample's only route is the scheduler
    delta = 0.5 * torch.randn(*SHAPE, generator=gen)
    sig = make([]).sigmas
    eu = FlowMatchEulerDiscreteScheduler(shift=SHIFT); eu.set_timesteps(STEPS)
    # the two grids are NOT the same (UniPC spaces the timesteps, FlowMatchEuler the sigmas, before the shift): at 50 steps they agree to 0.02
    # until the tail, and the A3 window [0.85, 0.60] holds 16 steps (19..34) under UniPC vs 15 (19..33) under Euler; the sampler reads the live grid
    d = float((eu.sigmas[:STEPS] - sig[:STEPS]).abs().max()); assert 1e-3 < d < 0.3, d
    print(f"sigma grids differ (max |UniPC - Euler| = {d:.3f} at {STEPS} steps): the sampler selects the window on the live grid")
    print(f"{'K':>2} {'sigma':>6} | corrector on: c at K, K+1 | corrector off at K: c at K, K+1 | (c = how much of a z-perturbation at K survives)")
    for K in (2, 4, 7):
        row = {}
        for lab, dis in (("on", []), ("off", [K - 1])):
            base = run(make(dis), None, delta, outs); o = run(make(dis), K, delta, outs); o_small = run(make(dis), K, 0.1 * delta, outs)
            c0, r0 = coefficient(o, base, delta, K); c1, r1 = coefficient(o, base, delta, K + 1)
            cs, _ = coefficient(o_small, base, 0.1 * delta, K)
            assert max(r0, r1) < 1e-4 and abs(cs - c0) < 1e-3, (r0, r1, cs, c0)                 # affine in the sample: the coefficient is scale-free
            row[lab] = (c0, c1)
        print(f"{K:>2} {float(sig[K]):6.3f} | {row['on'][0]:+.3f} {row['on'][1]:+.3f}            | {row['off'][0]:+.3f} {row['off'][1]:+.3f}")
        assert row["on"][0] < 0.5 and row["on"][1] < 0.2, row["on"]                                 # attenuated with the corrector on
        assert 0.95 < row["off"][0] < 1.2 and 0.95 < row["off"][1] < 1.2, row["off"]                # survives, and carries into the next (corrected) step
    # Euler, the scheduler of the gamma probe: a perturbation of the sample survives with coefficient exactly 1 (z_t = z_s + (sigma_t - sigma_s) v)
    def run_eu(perturb_at):
        e = FlowMatchEulerDiscreteScheduler(shift=SHIFT); e.set_timesteps(STEPS)
        z = torch.randn(*SHAPE, generator=torch.Generator().manual_seed(1)); zs = []
        for k, t in enumerate(e.timesteps):
            if k == perturb_at:
                z = z + delta
            z = e.step(outs[k], t, z, return_dict=False)[0]; zs.append(z.clone())
        return zs
    b, o = run_eu(None), run_eu(4)
    c, r = coefficient(o, b, delta, 4); assert abs(c - 1) < 1e-5 and r < 1e-5, (c, r)
    print(f"FlowMatchEuler: c = {c:.5f} (exactly 1: the gamma of record was probed under this)")
    print("ok")


if __name__ == "__main__":
    main()
