from utils import save_text
from config import SUMMARIES_FOLDER


def build_final_summary(descriptions, analysis_results, shared_columns):
    lines = []
    lines.append("# Final Data Summary\n")
    lines.append("## Combined Understanding\n")
    lines.append(
        "The dataset collection was first profiled structurally, then analyzed for patterns and trends."
    )

    lines.append("\n## Main Findings\n")
    lines.append(f"- Total datasets analyzed: {len(descriptions)}")
    lines.append(f"- Dataset relationships identified: {len(shared_columns)}")

    lines.append("\n## Development Readiness\n")
    lines.append(
        "This dataset now has a foundational description, preliminary trend analysis, and a summary of its main characteristics. The next stage can focus on building deeper analytics, predictive models, dashboards, or application features."
    )

    return "\n".join(lines)


def run_summary(descriptions, analysis_results, shared_columns):
    summary = build_final_summary(descriptions, analysis_results, shared_columns)
    save_text(summary, f"{SUMMARIES_FOLDER}/final_summary.md")
    return summary