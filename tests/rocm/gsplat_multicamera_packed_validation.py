#!/usr/bin/env python3
"""Validate gsplat packed/unpacked public APIs with multiple cameras.

This targeted ROCm/Wave32 test covers the cases that a one-camera/all-visible
scene cannot prove:

* two cameras with different backgrounds;
* Gaussians visible in both cameras and in only one camera;
* public per-Gaussian colors [N, 3];
* public per-view colors [C, N, 3];
* 3DGS and 2DGS forward/backward paths;
* complete output, gradient, radii and visibility comparisons.

It validates internal ROCm path consistency, not CUDA/ROCm parity.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import gsplat
from gsplat.rendering import rasterization, rasterization_2dgs


TensorMap = Dict[str, torch.Tensor]
LEAF_NAMES = ("means", "quats", "scales", "opacities", "colors")


@dataclass
class Snapshot:
    outputs: TensorMap
    grads: TensorMap
    radii: torch.Tensor
    visible: torch.Tensor
    loss: float


class Validator:
    def __init__(self, output_atol: float, output_rel_l2: float,
                 grad_atol: float, grad_rel_l2: float) -> None:
        self.output_atol = output_atol
        self.output_rel_l2 = output_rel_l2
        self.grad_atol = grad_atol
        self.grad_rel_l2 = grad_rel_l2
        self.failures: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"    FAIL: {message}")

    @staticmethod
    def metrics(actual: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
        actual64 = actual.detach().cpu().to(torch.float64)
        reference64 = reference.detach().cpu().to(torch.float64)
        difference = actual64 - reference64
        max_abs = difference.abs().max().item() if difference.numel() else 0.0
        denominator = torch.linalg.vector_norm(reference64).clamp_min(1e-12)
        rel_l2 = (torch.linalg.vector_norm(difference) / denominator).item()
        return max_abs, rel_l2

    def compare_float(self, label: str, actual: torch.Tensor,
                      reference: torch.Tensor, *, gradient: bool = False) -> None:
        if actual.shape != reference.shape:
            self.fail(f"{label}: shape {tuple(actual.shape)} != {tuple(reference.shape)}")
            return
        max_abs, rel_l2 = self.metrics(actual, reference)
        max_limit = self.grad_atol if gradient else self.output_atol
        rel_limit = self.grad_rel_l2 if gradient else self.output_rel_l2
        passed = max_abs <= max_limit and rel_l2 <= rel_limit
        print(
            f"    {label:<48} max_abs={max_abs:.3e}  "
            f"rel_l2={rel_l2:.3e}  {'OK' if passed else 'FAIL'}"
        )
        if not passed:
            self.fail(
                f"{label}: limits exceeded "
                f"(max_abs<={max_limit:.1e}, rel_l2<={rel_limit:.1e})"
            )

    def compare_exact(self, label: str, actual: torch.Tensor,
                      reference: torch.Tensor) -> None:
        passed = actual.shape == reference.shape and torch.equal(actual, reference)
        print(f"    {label:<48} {'exact' if passed else 'DIFFERENT'}")
        if not passed:
            self.fail(f"{label}: integer/bool tensors differ")

    def compare_snapshots(self, label: str, packed: Snapshot,
                          unpacked: Snapshot) -> None:
        print(f"\n  Volltensor-Vergleich: {label}")
        if packed.outputs.keys() != unpacked.outputs.keys():
            self.fail(f"{label}: output keys differ")
        for name in sorted(packed.outputs.keys() & unpacked.outputs.keys()):
            self.compare_float(
                f"{label}/output/{name}", packed.outputs[name], unpacked.outputs[name]
            )
        if packed.grads.keys() != unpacked.grads.keys():
            self.fail(f"{label}: gradient keys differ")
        for name in sorted(packed.grads.keys() & unpacked.grads.keys()):
            self.compare_float(
                f"{label}/grad/{name}", packed.grads[name], unpacked.grads[name],
                gradient=True,
            )
        self.compare_exact(f"{label}/radii", packed.radii, unpacked.radii)
        self.compare_exact(f"{label}/visible", packed.visible, unpacked.visible)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussians", type=int, default=384)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-atol", type=float, default=2e-5)
    parser.add_argument("--output-rel-l2", type=float, default=2e-5)
    parser.add_argument("--grad-atol", type=float, default=5e-4)
    parser.add_argument("--grad-rel-l2", type=float, default=5e-3)
    return parser.parse_args()


def make_scene(n: int, res: int, seed: int, device: torch.device) -> TensorMap:
    if n < 12 or n % 3:
        raise ValueError("--gaussians must be divisible by 3 and at least 12")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    group = n // 3

    means = torch.empty((n, 3), dtype=torch.float32)
    # Camera positions are x=-1 and x=+1.  Left/right groups are visible only
    # from their nearby camera; the center group is visible from both.
    means[:group, 0] = torch.rand(group, generator=generator) * 0.30 - 1.85
    means[group:2 * group, 0] = torch.rand(group, generator=generator) * 0.60 - 0.30
    means[2 * group:, 0] = torch.rand(group, generator=generator) * 0.30 + 1.55
    means[:, 1] = torch.rand(n, generator=generator) * 0.90 - 0.45
    means[:, 2] = torch.rand(n, generator=generator) * 1.00 + 2.80

    quats = torch.randn((n, 4), generator=generator)
    quats[:, 0] += 2.0
    quats = torch.nn.functional.normalize(quats, dim=-1)
    scales = torch.rand((n, 3), generator=generator)
    scales = scales * torch.tensor([0.080, 0.055, 0.035]) + torch.tensor(
        [0.055, 0.040, 0.030]
    )
    opacities = torch.rand(n, generator=generator) * 0.55 + 0.25
    colors = torch.rand((n, 3), generator=generator)
    # Make the per-view variant genuinely camera-dependent.
    colors_per_view = torch.stack((colors, 0.15 + 0.75 * (1.0 - colors)), dim=0)

    viewmats = torch.eye(4).repeat(2, 1, 1)
    viewmats[0, 0, 3] = 1.0   # camera center x=-1
    viewmats[1, 0, 3] = -1.0  # camera center x=+1
    focal = 0.86 * res
    K = torch.tensor(
        [[focal, 0.0, res / 2], [0.0, focal, res / 2], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    Ks = K.unsqueeze(0).repeat(2, 1, 1)
    backgrounds = torch.tensor(
        [[0.010, 0.020, 0.035], [0.080, 0.025, 0.010]], dtype=torch.float32
    )

    return {
        "means": means.to(device),
        "quats": quats.to(device),
        "scales": scales.to(device),
        "opacities": opacities.to(device),
        "colors_gaussian": colors.to(device),
        "colors_view": colors_per_view.to(device),
        "viewmats": viewmats.to(device),
        "Ks": Ks.to(device),
        "backgrounds": backgrounds.to(device),
    }


def fresh_leaves(scene: TensorMap, color_layout: str) -> TensorMap:
    colors = scene["colors_gaussian" if color_layout == "gaussian" else "colors_view"]
    return {
        "means": scene["means"].detach().clone().requires_grad_(True),
        "quats": scene["quats"].detach().clone().requires_grad_(True),
        "scales": scene["scales"].detach().clone().requires_grad_(True),
        "opacities": scene["opacities"].detach().clone().requires_grad_(True),
        "colors": colors.detach().clone().requires_grad_(True),
        "viewmats": scene["viewmats"],
        "Ks": scene["Ks"],
        "backgrounds": scene["backgrounds"],
    }


def objective(tensor: torch.Tensor, coefficient: float) -> torch.Tensor:
    return coefficient * (tensor.square().mean() + 0.17 * tensor.mean())


def collect_grads(validator: Validator, label: str, leaves: TensorMap) -> TensorMap:
    result: TensorMap = {}
    for name in LEAF_NAMES:
        grad = leaves[name].grad
        if grad is None:
            validator.fail(f"{label}/grad/{name}: missing")
            continue
        if not torch.isfinite(grad).all().item():
            validator.fail(f"{label}/grad/{name}: NaN or Inf")
        if grad.abs().max().item() <= 1e-12:
            validator.fail(f"{label}/grad/{name}: entirely zero")
        result[name] = grad.detach().cpu()
    return result


def dense_radii(meta: dict, cameras: int, n: int) -> torch.Tensor:
    radii = meta["radii"].detach()
    camera_ids: Optional[torch.Tensor] = meta.get("camera_ids")
    gaussian_ids: Optional[torch.Tensor] = meta.get("gaussian_ids")
    if camera_ids is None or gaussian_ids is None:
        return radii.cpu()
    result = torch.zeros(
        (cameras, n, radii.shape[-1]), dtype=radii.dtype, device=radii.device
    )
    result[camera_ids.long(), gaussian_ids.long()] = radii
    return result.cpu()


def check_visibility(validator: Validator, label: str, visible: torch.Tensor) -> None:
    if visible.shape[0] != 2:
        validator.fail(f"{label}: expected two cameras, got {tuple(visible.shape)}")
        return
    camera0, camera1 = visible[0], visible[1]
    counts = {
        "camera0": camera0.sum().item(),
        "camera1": camera1.sum().item(),
        "both": (camera0 & camera1).sum().item(),
        "only0": (camera0 & ~camera1).sum().item(),
        "only1": (~camera0 & camera1).sum().item(),
        "neither": (~camera0 & ~camera1).sum().item(),
    }
    print(
        f"  {label:<24} cam0={counts['camera0']}/{camera0.numel()}  "
        f"cam1={counts['camera1']}/{camera1.numel()}  both={counts['both']}  "
        f"only0={counts['only0']}  only1={counts['only1']}  neither={counts['neither']}"
    )
    for key in ("both", "only0", "only1"):
        if counts[key] == 0:
            validator.fail(f"{label}: visibility class {key} is empty")


def run_3d(validator: Validator, scene: TensorMap, *, label: str,
           color_layout: str, packed: bool, res: int) -> Snapshot:
    leaves = fresh_leaves(scene, color_layout)
    render, alpha, meta = rasterization(
        leaves["means"], leaves["quats"], leaves["scales"],
        leaves["opacities"], leaves["colors"], leaves["viewmats"], leaves["Ks"],
        res, res, packed=packed, tile_size=8, backgrounds=leaves["backgrounds"],
        render_mode="RGB+D",
    )
    loss = objective(render, 1.0) + objective(alpha, 0.31)
    loss.backward()
    torch.cuda.synchronize()
    for name, tensor in (("render", render), ("alpha", alpha)):
        if not torch.isfinite(tensor).all().item():
            validator.fail(f"{label}/{name}: NaN or Inf")
    radii = dense_radii(meta, 2, scene["means"].shape[0])
    visible = (radii > 0).all(dim=-1)
    return Snapshot(
        outputs={"render": render.detach().cpu(), "alpha": alpha.detach().cpu()},
        grads=collect_grads(validator, label, leaves), radii=radii, visible=visible,
        loss=loss.detach().item(),
    )


def run_2d(validator: Validator, scene: TensorMap, *, label: str,
           color_layout: str, packed: bool, res: int) -> Snapshot:
    leaves = fresh_leaves(scene, color_layout)
    render, alpha, normals, normals_from_depth, distort, median, meta = rasterization_2dgs(
        leaves["means"], leaves["quats"], leaves["scales"],
        leaves["opacities"], leaves["colors"], leaves["viewmats"], leaves["Ks"],
        res, res, packed=packed, tile_size=8, backgrounds=leaves["backgrounds"],
        render_mode="RGB+D", distloss=True, depth_mode="expected",
    )
    if normals_from_depth is None:
        validator.fail(f"{label}/normals_from_depth: unexpectedly None")
        normals_from_depth = torch.zeros_like(normals)
    outputs = {
        "render": render,
        "alpha": alpha,
        "normals": normals,
        "normals_from_depth": normals_from_depth,
        "distort": distort,
        "median": median,
    }
    loss = sum(
        objective(tensor, coefficient)
        for tensor, coefficient in zip(
            outputs.values(), (1.0, 0.31, 0.17, 0.13, 0.07, 0.05)
        )
    )
    loss.backward()
    torch.cuda.synchronize()
    for name, tensor in outputs.items():
        if not torch.isfinite(tensor).all().item():
            validator.fail(f"{label}/{name}: NaN or Inf")
    radii = dense_radii(meta, 2, scene["means"].shape[0])
    visible = (radii > 0).all(dim=-1)
    return Snapshot(
        outputs={name: tensor.detach().cpu() for name, tensor in outputs.items()},
        grads=collect_grads(validator, label, leaves), radii=radii, visible=visible,
        loss=loss.detach().item(),
    )


def main() -> int:
    args = parse_args()
    validator = Validator(
        args.output_atol, args.output_rel_l2, args.grad_atol, args.grad_rel_l2
    )
    print("=" * 104)
    print("GSPLAT MULTI-CAMERA PACKED VALIDATION - ROCm / gfx1201 / Wave32")
    print("=" * 104)
    print(f"gsplat:  {Path(gsplat.__file__).resolve()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"HIP:     {torch.version.hip}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 2
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Scene:   C=2, N={args.gaussians}, {args.res}x{args.res}, seed={args.seed}")

    scene = make_scene(args.gaussians, args.res, args.seed, torch.device("cuda"))
    run_pairs: list[tuple[str, Snapshot, Snapshot]] = []
    for renderer_name, runner in (("3DGS", run_3d), ("2DGS", run_2d)):
        for color_layout in ("gaussian", "view"):
            base_label = f"{renderer_name}/{color_layout}-colors"
            unpacked = runner(
                validator, scene, label=f"{base_label}/unpacked",
                color_layout=color_layout, packed=False, res=args.res,
            )
            packed = runner(
                validator, scene, label=f"{base_label}/packed",
                color_layout=color_layout, packed=True, res=args.res,
            )
            check_visibility(validator, base_label, unpacked.visible)
            run_pairs.append((base_label, packed, unpacked))

    for label, packed, unpacked in run_pairs:
        validator.compare_snapshots(label, packed, unpacked)

    print("\n" + "=" * 104)
    if validator.failures:
        print(f"GSPLAT_MULTICAMERA_PACKED_VALIDATION: FAIL ({len(validator.failures)} Punkt(e))")
        for index, failure in enumerate(validator.failures, 1):
            print(f"  {index:02d}. {failure}")
        return 1
    print("GSPLAT_MULTICAMERA_PACKED_VALIDATION: PASS")
    print("8/8 Forward/Backward-Laeufe; public [N,3] and [C,N,3] APIs validated.")
    print("Hinweis: interne ROCm-Pfadkonsistenz, keine CUDA/ROCm-Paritaetsmessung.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
