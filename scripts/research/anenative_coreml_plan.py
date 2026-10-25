"""Inspect planned Core ML placement; no claim that CPU_AND_NE forces ANE."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def placement(plan):
    operations = []

    def visit(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            device = (
                type(usage.preferred_compute_device).__name__
                if usage
                else "unassigned_or_constant"
            )
            operations.append(
                {"operator": op.operator_name, "preferred_device": device}
            )
            for nested in op.blocks:
                visit(nested)

    program = plan.model_structure.program
    if program is None:
        raise ValueError("expected ML Program structure")
    for function in program.functions.values():
        visit(function.block)
    return {
        "preferred_counts": dict(Counter(r["preferred_device"] for r in operations)),
        "operations": operations,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if any(not path.is_dir() for path in a.packages):
        raise FileNotFoundError("missing Core ML package")
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    records = []
    for path in a.packages:
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        plan = MLComputePlan.load_from_path(
            model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        record = {"package": str(path), **placement(plan), "planned_only": True}
        print(json.dumps(record), flush=True)
        records.append(record)
        del plan, model
    a.out.write_text(json.dumps({"models": records, "complete": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
