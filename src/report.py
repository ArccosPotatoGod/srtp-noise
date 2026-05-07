"""Report generation — table formatting and text-based summaries."""

from typing import Dict, List


def format_metrics_table(metrics: Dict) -> str:
    """Format a single metrics dict as an aligned text table row."""
    lines = []
    lines.append("-" * 60)
    lines.append(f"{'Metric':<25} {'Value':>15}")
    lines.append("-" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key:<25} {value:15.4f}")
        else:
            lines.append(f"{key:<25} {str(value):>15}")
    lines.append("-" * 60)
    return "\n".join(lines)


def format_summary(results: List[Dict]) -> str:
    """Produce a summary table from a list of per-experiment metric dicts."""
    if not results:
        return "No results."
    keys = list(results[0].keys())
    header = " | ".join(f"{k:>12}" for k in keys)
    sep = "-+-".join("-" * 12 for _ in keys)
    rows = [header, sep]
    for r in results:
        row = " | ".join(f"{r[k]:12.4f}" if isinstance(r[k], float) else f"{str(r[k]):>12}"
                         for k in keys)
        rows.append(row)
    return "\n".join(rows)
