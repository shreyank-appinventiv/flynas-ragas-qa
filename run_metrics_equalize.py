"""
Equalize RAGAS Metrics Runner
Runs metrics until all have equal number of runs.
"""

import asyncio
import os
import sys
import pandas as pd
import glob
import time

# Import from the main runner
from run_ragas_metrics import (
    METRICS, RESULTS_BASE_DIR, FAQ_FILE,
    load_questions_from_faq, collect_responses, run_metric
)


def get_run_counts() -> dict:
    """Get current run counts for each metric"""

    summary_files = {
        'answer_relevancy': 'answer_relevancy_summary.xlsx',
        'context_precision': 'context_precision_metric_summary.xlsx',
        'context_relevancy': 'context_relevancy_summary.xlsx',
        'faithfulness': 'faithfulness_summary.xlsx'
    }

    run_counts = {}
    for metric_key, summary_file in summary_files.items():
        summary_path = os.path.join(RESULTS_BASE_DIR, metric_key, 'summary', summary_file)
        if os.path.exists(summary_path):
            df = pd.read_excel(summary_path, sheet_name='Summary')
            runs = len(df[df['run'] != 'TOTALS'])
            run_counts[metric_key] = runs
        else:
            run_counts[metric_key] = 0

    return run_counts


def get_metrics_to_run(run_counts: dict, target_runs: int) -> list:
    """Get list of metric keys that need more runs"""
    metrics_needed = []
    for metric_key, current_runs in run_counts.items():
        if current_runs < target_runs:
            metrics_needed.append(metric_key)
    return metrics_needed


async def main():
    """Main entry point"""

    # Check for API key
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    if not OPENAI_API_KEY:
        print("\n[ERROR] OPENAI_API_KEY environment variable is required.")
        return

    # Get current run counts
    print("\n" + "=" * 60)
    print("RAGAS METRICS EQUALIZER")
    print("=" * 60)

    run_counts = get_run_counts()

    print("\nCurrent run counts:")
    print("-" * 40)
    for metric, count in run_counts.items():
        print(f"  {metric}: {count} runs")

    max_runs = max(run_counts.values())
    print(f"\nTarget: {max_runs} runs for all metrics")

    # Calculate runs needed
    runs_needed = {}
    total_collection_runs = 0

    for metric, count in run_counts.items():
        needed = max_runs - count
        if needed > 0:
            runs_needed[metric] = needed
            total_collection_runs = max(total_collection_runs, needed)

    if not runs_needed:
        print("\nAll metrics already have equal runs!")
        return

    print("\nRuns needed:")
    print("-" * 40)
    for metric, needed in runs_needed.items():
        print(f"  {metric}: {needed} more runs")

    print(f"\nTotal collection runs required: {total_collection_runs}")

    # Confirm (auto-confirm if --auto flag is passed)
    if '--auto' not in sys.argv:
        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    else:
        print("\n[AUTO] Proceeding automatically...")

    # Load questions
    print(f"\nLoading questions from FAQ file...")
    questions = load_questions_from_faq(FAQ_FILE)
    print(f"Loaded {len(questions)} questions")

    # Map metric keys to METRICS config keys
    key_mapping = {
        'answer_relevancy': '1',
        'context_precision': '2',
        'context_relevancy': '3',
        'faithfulness': '4'
    }

    overall_start = time.time()

    # Run collection loops
    for run_num in range(1, total_collection_runs + 1):
        print(f"\n{'#' * 60}")
        print(f"# COLLECTION RUN {run_num} of {total_collection_runs}")
        print(f"{'#' * 60}")

        # Determine which metrics to run this iteration
        current_counts = get_run_counts()
        metrics_this_run = []

        for metric_key in runs_needed.keys():
            if current_counts[metric_key] < max_runs:
                metrics_this_run.append(metric_key)

        if not metrics_this_run:
            print("All metrics equalized!")
            break

        print(f"\nMetrics to evaluate this run: {', '.join(metrics_this_run)}")

        # Collect responses
        print("\nCollecting chatbot responses...")
        start_time = time.time()
        responses = await collect_responses(questions)
        collection_time = time.time() - start_time
        print(f"\nResponse collection completed in {collection_time:.1f}s")

        # Run each metric
        for metric_key in metrics_this_run:
            config_key = key_mapping[metric_key]
            config = METRICS[config_key]
            await run_metric(config, responses, OPENAI_API_KEY, questions)

        # Show progress
        updated_counts = get_run_counts()
        print(f"\n--- Progress after run {run_num} ---")
        for metric, count in updated_counts.items():
            status = "DONE" if count >= max_runs else f"{max_runs - count} left"
            print(f"  {metric}: {count}/{max_runs} ({status})")

    # Generate consolidated report
    print("\n" + "=" * 60)
    print("Generating Consolidated Report...")
    print("=" * 60)

    from generate_consolidated_report import ConsolidatedReportGenerator
    generator = ConsolidatedReportGenerator(os.path.dirname(RESULTS_BASE_DIR))
    report_file = generator.generate_report()

    total_time = time.time() - overall_start

    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    final_counts = get_run_counts()
    for metric, count in final_counts.items():
        print(f"  {metric}: {count} runs")
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
