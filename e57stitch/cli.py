"""Command line interface: `e57stitch serve` or headless `e57stitch register`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .session import SESSION


def _wait(job_id: str, label: str) -> dict:
    last = -1.0
    while True:
        job = SESSION.job(job_id)
        if job is None:
            raise SystemExit(f"unknown job {job_id}")
        pct = float(job.get("progress") or 0.0)
        if pct - last >= 0.02 or job["status"] in ("done", "error"):
            msg = job.get("message") or ""
            print(f"\r  {label:10s} {pct * 100:5.1f}%  {msg[:70]:<70}", end="", flush=True)
            last = pct
        if job["status"] == "done":
            print()
            return job.get("result") or {}
        if job["status"] == "error":
            print()
            raise SystemExit(f"{label} failed: {job.get('error')}")
        time.sleep(0.25)


def cmd_info(args: argparse.Namespace) -> int:
    from .e57io import probe_file

    for path in args.files:
        refs = probe_file(path)
        print(f"{path}  ({len(refs)} scan{'s' if len(refs) != 1 else ''})")
        for r in refs:
            print(
                f"  #{r.index}  {r.point_count:>12,} pts   pose={'yes' if r.has_pose else 'no'}"
                f"   fields={','.join(r.fields)}"
            )
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    paths = [str(Path(p).expanduser()) for p in args.inputs]
    added = SESSION.add_files(paths)
    if not SESSION.scans:
        print("No E57 scans found.")
        return 1
    SESSION.params.update(
        {
            "voxel": args.voxel,
            "preview_points": args.preview_points,
            "pair_workers": args.workers,
            "accept_score": args.accept,
            "use_pose": not args.ignore_pose,
            "detect_features": not args.no_features,
            "cleanup": args.cleanup,
            "max_attempts": args.attempts,
        }
    )
    print(f"Found {len(SESSION.scans)} scan(s) in {len(added['added'])} file(s).")

    _wait(SESSION.load(), "load")
    if args.resume:
        import json as _json

        SESSION.restore_session(_json.load(open(args.resume)))
        print(f"  resumed session from {args.resume}")
    if args.resume:
        _wait(SESSION.solve(), "solve")
    else:
        # floor-plan candidates + global selection (+ heavy matchers with --deep)
        _wait(
            SESSION.register_scans(keep_improving=args.deep, hours=args.hours, voxel=args.voxel),
            "register",
        )
    if args.refine > 0:
        _wait(SESSION.refine(voxel=args.refine_voxel, rounds=args.refine), "refine")

    summary = SESSION.summary()
    print(
        f"\n  pairs: {summary['pairs']}   accepted: {summary['accepted']}   "
        f"groups: {len(summary['groups'])}"
    )
    for g in summary["groups"]:
        names = ", ".join(SESSION.scans[m].ref.label for m in g["members"])
        tag = "UNCONNECTED" if g["unconnected"] else f"{g['edges']} edges"
        print(f"    group {g['id']}: {len(g['members'])} scans ({tag})")
        print(f"      {names}")

    # save the solved transforms next to the output so the alignment can be
    # inspected or re-applied later
    session_path = Path(args.out).expanduser().with_suffix(".session.json") if args.out else None
    if session_path is not None:
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(json.dumps(SESSION.session_data(), indent=1))
        print(f"\n  session -> {session_path}")

    if args.out:
        out = Path(args.out).expanduser()
        fmt = args.format or out.suffix.lstrip(".").lower() or "e57"
        scope = "all" if args.scope == "all" else "active"
        print(f"\n  exporting {scope} group(s) -> {out} ({fmt})")
        _wait(
            SESSION.export(
                {
                    "path": str(out),
                    "format": fmt,
                    "mode": args.mode,
                    "scope": scope,
                    "voxel": args.export_voxel,
                    "precision": args.precision,
                    "colors": not args.no_colors,
                    "density_cap": args.density_cap,
                    "density_voxel": args.density_voxel,
                    "filter_noise": not args.no_noise_filter,
                    "name": out.stem,
                }
            ),
            "export",
        )
        size = out.stat().st_size / 1e6 if out.exists() else 0
        print(f"  wrote {out}  ({size:.1f} MB)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve
    from . import workers

    workers.START_METHOD = "spawn"

    if args.files:
        SESSION.add_files([str(Path(p).expanduser()) for p in args.files])
    serve(host=args.host, port=args.port, open_browser=not args.no_open)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e57stitch",
        description="Register multiple E57 scans into one aligned point cloud.",
    )
    parser.add_argument("--version", action="version", version=f"e57stitch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the local web UI")
    serve.add_argument("files", nargs="*", help="optional E57 files/folders to preload")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-open", action="store_true", help="do not open a browser")
    serve.set_defaults(func=cmd_serve)

    info = sub.add_parser("info", help="print scan metadata of E57 files")
    info.add_argument("files", nargs="+")
    info.set_defaults(func=cmd_info)

    reg = sub.add_parser("register", help="headless registration + export")
    reg.add_argument("inputs", nargs="+", help="E57 files or folders")
    reg.add_argument("-o", "--out", help="output file (default: no export)")
    reg.add_argument("--format", choices=["e57", "ply", "pts", "xyz"], help="output format")
    reg.add_argument("--mode", choices=["merged", "separate"], default="merged")
    reg.add_argument("--scope", choices=["active", "all"], default="active")
    reg.add_argument("--voxel", type=float, default=0.05, help="registration voxel size (m)")
    reg.add_argument("--export-voxel", type=float, default=0.0, help="uniform export voxel (0 = full resolution)")
    reg.add_argument("--density-cap", type=int, default=0, help="adaptive: max points per density voxel (0 = off)")
    reg.add_argument("--density-voxel", type=float, default=0.02, help="density voxel for --density-cap")
    reg.add_argument("--no-noise-filter", action="store_true", help="do not transfer the load-time noise filter to the export")
    reg.add_argument("--accept", type=float, default=0.15, help="structure overlap threshold")
    reg.add_argument("--workers", type=int, default=8)
    reg.add_argument("--attempts", type=int, default=2, help="global matching attempts per pair")
    reg.add_argument("--refine", type=int, default=8, help="max rounds of the final whole-building refine (0 = skip)")
    reg.add_argument("--refine-voxel", type=float, default=0.02, help="resolution of the final refine in m (0.02 or 0.01)")
    reg.add_argument("--deep", action="store_true", help="after the fast pass, run every heavy matcher where links are missing or unsure")
    reg.add_argument("--hours", type=float, default=10.0, help="time budget for --deep")
    reg.add_argument("--resume", type=str, help="resume from a saved session.json (skips the pairwise stage)")
    reg.add_argument("--preview-points", type=int, default=120_000)
    reg.add_argument("--ignore-pose", action="store_true", help="ignore poses stored in E57")
    reg.add_argument("--no-features", action="store_true", help="skip plane/edge detection")
    reg.add_argument("--cleanup", choices=["off", "light", "strong"], default="light", help="reflection/glass fragment cleanup at load")
    reg.add_argument("--no-colors", action="store_true", help="drop colours on export")
    reg.add_argument("--precision", choices=["double", "single"], default="double")
    reg.set_defaults(func=cmd_register)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
