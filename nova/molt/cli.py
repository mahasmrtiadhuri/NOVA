"""Standalone CLI for the MOLT research module."""

import argparse
import json
from pathlib import Path

from .benchmark import DemoConfig, run_demo
from .store import load_skill


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA MOLT: neural-to-symbolic skill experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Train a tiny teacher and attempt skill compilation")
    demo.add_argument("--operator", choices=("add_mod", "xor", "max", "affine_mod"), default="max")
    demo.add_argument("--modulus", type=int, default=8)
    demo.add_argument("--steps", type=int, default=300)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--requests", type=int, default=256)
    demo.add_argument("--output", default="runs/molt")
    query = subparsers.add_parser("query", help="Query a locally stored certified skill")
    query.add_argument("--skill", type=Path, required=True)
    query.add_argument("--left", type=int, required=True)
    query.add_argument("--right", type=int, required=True)
    args = parser.parse_args()
    if args.command == "query":
        skill = load_skill(args.skill)
        print(f"{skill.name}({args.left}, {args.right}) = {skill.predict(args.left, args.right)}")
        return
    report = run_demo(DemoConfig(
        operator=args.operator, modulus=args.modulus, steps=args.steps,
        seed=args.seed, benchmark_requests=args.requests, output_dir=args.output,
    ))
    print("MOLT compiler:", report["compiler"]["status"])
    print("Teacher full-domain accuracy:", report["training"]["teacher_full_domain_accuracy"])
    for name, result in report["results"].items():
        print(f"{name:19s}: accuracy={result['accuracy']:.3f}, "
              f"median latency={result['median_ms']:.5f} ms")
    print("Results written to:", Path(args.output) / "report.json")


if __name__ == "__main__":
    main()
