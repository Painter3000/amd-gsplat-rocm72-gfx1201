#!/usr/bin/env python3
"""Validate the remaining gsplat ROCm/Wave32 rasterization paths.

Targets AMD-Ecosystem/gsplat release/1.5.3b2 after the gfx1201 Wave32 port.
The script compares complete tensors (not samples) for:

* 3DGS: tile sizes 8/16, packed/unpacked, repeated baseline
* 3DGS: RGB, D, ED, RGB+D and RGB+ED render modes
* 2DGS: tile sizes 8/16, packed/unpacked, repeated baseline
* 2DGS: expected and median depth-normal paths

It checks outputs, reconstructed radii, all expected leaf gradients, maximum
absolute deviation and relative L2 deviation.  It does not establish CUDA/ROCm
parity; it establishes internal consistency between ROCm code paths.
"""

from __future__ import annotations

import argparse
import math
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
    def __init__(
        self,
        output_atol: float,
        output_rel_l2: float,
        grad_atol: float,
        grad_rel_l2: float,
    ) -> None:
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
        actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
        reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
        difference = actual64 - reference64
        max_abs = difference.abs().max().item() if difference.numel() else 0.0
        denominator = torch.linalg.vector_norm(reference64).clamp_min(1e-12)
        rel_l2 = (torch.linalg.vector_norm(difference) / denominator).item()
        return max_abs, rel_l2

    def compare_float(
        self,
        label: str,
        actual: torch.Tensor,
        reference: torch.Tensor,
        *,
        gradient: bool = False,
    ) -> None:
        if actual.shape != reference.shape:
            self.fail(
                f"{label}: shape {tuple(actual.shape)} != {tuple(reference.shape)}"
            )
            return
        max_abs, rel_l2 = self.metrics(actual, reference)
        atol = self.grad_atol if gradient else self.output_atol
        rel_limit = self.grad_rel_l2 if gradient else self.output_rel_l2
        status = "OK" if max_abs <= atol and rel_l2 <= rel_limit else "FAIL"
        print(
            f"    {label:<42} max_abs={max_abs:.3e}  "
            f"rel_l2={rel_l2:.3e}  {status}"
        )
        if status == "FAIL":
            self.fail(
                f"{label}: limits exceeded "
                f"(max_abs<={atol:.1e}, rel_l2<={rel_limit:.1e})"
            )

    def compare_exact(
        self, label: str, actual: torch.Tensor, reference: torch.Tensor
    ) -> None:
        same = actual.shape == reference.shape and torch.equal(actual, reference)
        print(f"    {label:<42} {'exact' if same else 'DIFFERENT'}")
        if not same:
            self.fail(f"{label}: integer/bool tensors differ")

    def compare_snapshots(
        self, label: str, actual: Snapshot, reference: Snapshot
    ) -> None:
        print(f"\n  Vergleich: {label}")
        if actual.outputs.keys() != reference.outputs.keys():
            self.fail(
                f"{label}: output keys {sorted(actual.outputs)} != "
                f"{sorted(reference.outputs)}"
            )
        for name in sorted(actual.outputs.keys() & reference.outputs.keys()):
            self.compare_float(
                f"{label}/output/{name}", actual.outputs[name], reference.outputs[name]
            )
        if actual.grads.keys() != reference.grads.keys():
            self.fail(
                f"{label}: gradient keys {sorted(actual.grads)} != "
                f"{sorted(reference.grads)}"
            )
        for name in sorted(actual.grads.keys() & reference.grads.keys()):
            self.compare_float(
                f"{label}/grad/{name}",
                actual.grads[name],
                reference.grads[name],
                gradient=True,
            )
        self.compare_exact(f"{label}/radii", actual.radii, reference.radii)
        self.compare_exact(f"{label}/visible", actual.visible, reference.visible)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussians", type=int, default=256)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output-atol", type=float, default=2e-5)
    parser.add_argument("--output-rel-l2", type=float, default=2e-5)
    parser.add_argument("--grad-atol", type=float, default=5e-4)
    parser.add_argument("--grad-rel-l2", type=float, default=5e-3)
    return parser.parse_args()


def make_scene(n: int, res: int, seed: int, device: torch.device) -> TensorMap:
    if n < 8:
        raise ValueError("--gaussians must be at least 8")
    if res < 32:
        raise ValueError("--res must be at least 32")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty((n, 3), dtype=torch.float32)
    means[:, :2] = torch.rand((n, 2), generator=generator) * 1.30 - 0.65
    means[:, 2] = torch.rand((n,), generator=generator) * 1.25 + 2.25

    quats = torch.randn((n, 4), generator=generator, dtype=torch.float32)
    quats[:, 0] += 2.0
    quats = torch.nn.functional.normalize(quats, dim=-1)

    # Anisotropic splats ensure that quaternion gradients carry a real signal.
    scales = torch.rand((n, 3), generator=generator, dtype=torch.float32)
    scales = scales * torch.tensor([0.085, 0.060, 0.040]) + torch.tensor(
        [0.055, 0.040, 0.030]
    )
    opacities = torch.rand((n,), generator=generator) * 0.55 + 0.25
    colors = torch.rand((n, 3), generator=generator)

    focal = 0.86 * res
    viewmats = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    Ks = torch.tensor(
        [[[focal, 0.0, res / 2], [0.0, focal, res / 2], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    backgrounds = torch.tensor([[0.015, 0.025, 0.035]], dtype=torch.float32)

    return {
        "means": means.to(device),
        "quats": quats.to(device),
        "scales": scales.to(device),
        "opacities": opacities.to(device),
        "colors": colors.to(device),
        "viewmats": viewmats.to(device),
        "Ks": Ks.to(device),
        "backgrounds": backgrounds.to(device),
    }


def fresh_leaves(scene: TensorMap) -> TensorMap:
    leaves = {
        name: scene[name].detach().clone().requires_grad_(True) for name in LEAF_NAMES
    }
    leaves["viewmats"] = scene["viewmats"]
    leaves["Ks"] = scene["Ks"]
    leaves["backgrounds"] = scene["backgrounds"]
    return leaves


def objective(tensor: torch.Tensor, coefficient: float) -> torch.Tensor:
    # Mean and square terms exercise signs and magnitudes without creating a huge loss.
    return coefficient * (tensor.square().mean() + 0.17 * tensor.mean())


def assert_finite(
    validator: Validator, label: str, tensors: Iterable[tuple[str, torch.Tensor]]
) -> None:
    for name, tensor in tensors:
        if not torch.isfinite(tensor).all().item():
            validator.fail(f"{label}/{name}: NaN or Inf")


def collect_grads(
    validator: Validator,
    label: str,
    leaves: TensorMap,
    expected: Iterable[str],
) -> TensorMap:
    expected_set = set(expected)
    result: TensorMap = {}
    for name in LEAF_NAMES:
        grad = leaves[name].grad
        if name in expected_set:
            if grad is None:
                validator.fail(f"{label}/grad/{name}: missing")
                continue
            if not torch.isfinite(grad).all().item():
                validator.fail(f"{label}/grad/{name}: NaN or Inf")
            if grad.abs().max().item() <= 1e-12:
                validator.fail(f"{label}/grad/{name}: entirely zero")
            result[name] = grad.detach().cpu()
        elif grad is not None and grad.abs().max().item() > 1e-12:
            validator.fail(f"{label}/grad/{name}: unexpected nonzero gradient")
    return result


def dense_radii(meta: dict, n: int, cameras: int = 1) -> torch.Tensor:
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


def run_3d(
    validator: Validator,
    scene: TensorMap,
    *,
    label: str,
    packed: bool,
    tile_size: int,
    render_mode: str,
    res: int,
) -> Snapshot:
    leaves = fresh_leaves(scene)
    render, alpha, meta = rasterization(
        leaves["means"],
        leaves["quats"],
        leaves["scales"],
        leaves["opacities"],
        leaves["colors"],
        leaves["viewmats"],
        leaves["Ks"],
        res,
        res,
        packed=packed,
        tile_size=tile_size,
        backgrounds=leaves["backgrounds"],
        render_mode=render_mode,
    )
    loss = objective(render, 1.0) + objective(alpha, 0.31)
    loss.backward()
    torch.cuda.synchronize()

    assert_finite(validator, label, (("render", render), ("alpha", alpha)))
    expected = LEAF_NAMES if "RGB" in render_mode else LEAF_NAMES[:-1]
    grads = collect_grads(validator, label, leaves, expected)
    radii = dense_radii(meta, scene["means"].shape[0])
    visible = (radii > 0).all(dim=-1)
    print(
        f"  {label:<27} loss={loss.detach().item():.8f}  "
        f"visible={visible.sum().item()}/{visible.numel()}  "
        f"render={tuple(render.shape)}"
    )
    return Snapshot(
        outputs={"render": render.detach().cpu(), "alpha": alpha.detach().cpu()},
        grads=grads,
        radii=radii,
        visible=visible,
        loss=loss.detach().item(),
    )


def run_2d(
    validator: Validator,
    scene: TensorMap,
    *,
    label: str,
    packed: bool,
    tile_size: int,
    depth_mode: str,
    res: int,
) -> Snapshot:
    leaves = fresh_leaves(scene)
    (
        render,
        alpha,
        normals,
        normals_from_depth,
        distort,
        median,
        meta,
    ) = rasterization_2dgs(
        leaves["means"],
        leaves["quats"],
        leaves["scales"],
        leaves["opacities"],
        leaves["colors"],
        leaves["viewmats"],
        leaves["Ks"],
        res,
        res,
        packed=packed,
        tile_size=tile_size,
        backgrounds=leaves["backgrounds"],
        render_mode="RGB+D",
        distloss=True,
        depth_mode=depth_mode,
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

    assert_finite(validator, label, outputs.items())
    grads = collect_grads(validator, label, leaves, LEAF_NAMES)
    radii = dense_radii(meta, scene["means"].shape[0])
    visible = (radii > 0).all(dim=-1)
    print(
        f"  {label:<27} loss={loss.detach().item():.8f}  "
        f"visible={visible.sum().item()}/{visible.numel()}  "
        f"render={tuple(render.shape)}"
    )
    return Snapshot(
        outputs={name: tensor.detach().cpu() for name, tensor in outputs.items()},
        grads=grads,
        radii=radii,
        visible=visible,
        loss=loss.detach().item(),
    )


def compare_3d_depth_relations(
    validator: Validator, runs: Dict[str, Snapshot]
) -> None:
    print("\n  Volltensor-Beziehungen der 3DGS-Render-Modi")
    rgb = runs["rgb"].outputs["render"]
    d = runs["d"].outputs["render"]
    ed = runs["ed"].outputs["render"]
    rgb_d = runs["rgb_d"].outputs["render"]
    rgb_ed = runs["rgb_ed"].outputs["render"]
    validator.compare_float("RGB+D/RGB == RGB", rgb_d[..., :3], rgb)
    validator.compare_float("RGB+D/D == D", rgb_d[..., 3:], d)
    validator.compare_float("RGB+ED/RGB == RGB", rgb_ed[..., :3], rgb)
    validator.compare_float("RGB+ED/ED == ED", rgb_ed[..., 3:], ed)
    for name in ("d", "ed", "rgb_d", "rgb_ed"):
        validator.compare_float(
            f"{name}/alpha == RGB/alpha",
            runs[name].outputs["alpha"],
            runs["rgb"].outputs["alpha"],
        )
        validator.compare_exact(
            f"{name}/radii == RGB/radii", runs[name].radii, runs["rgb"].radii
        )


def main() -> int:
    args = parse_args()
    validator = Validator(
        args.output_atol,
        args.output_rel_l2,
        args.grad_atol,
        args.grad_rel_l2,
    )

    print("=" * 100)
    print("GSPLAT REMAINING PATHS VALIDATION - ROCm / gfx1201 / Wave32")
    print("=" * 100)
    print(f"gsplat:  {Path(gsplat.__file__).resolve()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"HIP:     {torch.version.hip}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 2
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(
        f"Scene:   N={args.gaussians}, {args.res}x{args.res}, seed={args.seed}\n"
        f"Limits:  output max={args.output_atol:.1e}, output relL2={args.output_rel_l2:.1e}, "
        f"grad max={args.grad_atol:.1e}, grad relL2={args.grad_rel_l2:.1e}"
    )

    device = torch.device("cuda")
    scene = make_scene(args.gaussians, args.res, args.seed, device)

    print("\n[1/3] 3DGS: Wiederholung, Tile 8/16 und packed/unpacked")
    runs3: Dict[str, Snapshot] = {}
    configs3 = (
        ("rgb", False, 8, "RGB"),
        ("rgb_repeat", False, 8, "RGB"),
        ("rgb_tile16", False, 16, "RGB"),
        ("rgb_packed8", True, 8, "RGB"),
        ("rgb_packed16", True, 16, "RGB"),
    )
    for label, packed, tile, mode in configs3:
        runs3[label] = run_3d(
            validator,
            scene,
            label=label,
            packed=packed,
            tile_size=tile,
            render_mode=mode,
            res=args.res,
        )
    for label in ("rgb_repeat", "rgb_tile16", "rgb_packed8", "rgb_packed16"):
        validator.compare_snapshots(label, runs3[label], runs3["rgb"])

    print("\n[2/3] 3DGS: D, ED, RGB+D und RGB+ED")
    mode_configs = (
        ("d", "D"),
        ("ed", "ED"),
        ("rgb_d", "RGB+D"),
        ("rgb_ed", "RGB+ED"),
    )
    for label, mode in mode_configs:
        runs3[label] = run_3d(
            validator,
            scene,
            label=label,
            packed=False,
            tile_size=8,
            render_mode=mode,
            res=args.res,
        )
    compare_3d_depth_relations(validator, runs3)

    print("\n[3/3] 2DGS: Wiederholung, Tile, packed und depth_mode")
    runs2: Dict[str, Snapshot] = {}
    configs2 = (
        ("2d_expected", False, 8, "expected"),
        ("2d_repeat", False, 8, "expected"),
        ("2d_tile16", False, 16, "expected"),
        ("2d_packed8", True, 8, "expected"),
        ("2d_packed16", True, 16, "expected"),
        ("2d_median", False, 8, "median"),
    )
    for label, packed, tile, depth_mode in configs2:
        runs2[label] = run_2d(
            validator,
            scene,
            label=label,
            packed=packed,
            tile_size=tile,
            depth_mode=depth_mode,
            res=args.res,
        )
    for label in ("2d_repeat", "2d_tile16", "2d_packed8", "2d_packed16"):
        validator.compare_snapshots(label, runs2[label], runs2["2d_expected"])

    # depth_mode only changes the normal reconstructed from depth. All other
    # forward tensors and all projection radii should be identical.
    print("\n  2DGS expected/median: unveränderte Ausgaben")
    for name in ("render", "alpha", "normals", "distort", "median"):
        validator.compare_float(
            f"2d_median/{name}",
            runs2["2d_median"].outputs[name],
            runs2["2d_expected"].outputs[name],
        )
    validator.compare_exact(
        "2d_median/radii", runs2["2d_median"].radii, runs2["2d_expected"].radii
    )

    print("\n" + "=" * 100)
    if validator.failures:
        print(f"GSPLAT_REMAINING_PATHS_VALIDATION: FAIL ({len(validator.failures)} Punkt(e))")
        for number, failure in enumerate(validator.failures, 1):
            print(f"  {number:02d}. {failure}")
        return 1

    print("GSPLAT_REMAINING_PATHS_VALIDATION: PASS")
    print("15/15 Forward/Backward-Laeufe ohne NaN/Inf/fehlende Gradienten.")
    print("Hinweis: Dies belegt interne ROCm-Pfadkonsistenz, nicht CUDA/ROCm-Paritaet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
