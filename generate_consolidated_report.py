"""
Consolidated RAGAS Metrics Report Generator
Reads all metric summaries and generates a comprehensive consolidated report.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import glob


class ConsolidatedReportGenerator:
    """Generates consolidated reports from all RAGAS metric summaries"""

    def __init__(self, category_path: str):
        self.category_path = category_path
        self.results_path = os.path.join(category_path, "results")
        self.metrics_config = {
            "answer_relevancy": {
                "name": "Answer Relevancy",
                "score_column": "avg_relevancy_score",
                "summary_file": "answer_relevancy_summary.xlsx",
                "description": "Measures if the answer addresses the question",
                "needs_contexts": False
            },
            "context_precision": {
                "name": "Context Precision",
                "score_column": "avg_precision_score",
                "summary_file": "context_precision_metric_summary.xlsx",
                "description": "Measures if relevant contexts are ranked higher",
                "needs_contexts": True
            },
            "context_relevancy": {
                "name": "Context Relevancy",
                "score_column": "avg_relevancy_score",
                "summary_file": "context_relevancy_summary.xlsx",
                "description": "Measures if retrieved contexts are relevant to the question",
                "needs_contexts": True
            },
            "faithfulness": {
                "name": "Faithfulness",
                "score_column": "avg_faithfulness_score",
                "summary_file": "faithfulness_summary.xlsx",
                "description": "Measures if the answer is grounded in the retrieved contexts",
                "needs_contexts": True
            }
        }
        self.summaries: Dict[str, pd.DataFrame] = {}
        self.question_legends: Dict[str, pd.DataFrame] = {}

    def load_summaries(self) -> Dict[str, dict]:
        """Load all available metric summaries"""
        loaded_metrics = {}

        for metric_key, config in self.metrics_config.items():
            metric_path = os.path.join(self.results_path, metric_key, "summary")

            # Try to find the summary file
            summary_files = glob.glob(os.path.join(metric_path, "*summary*.xlsx"))
            if not summary_files:
                print(f"  [SKIP] {config['name']}: No summary file found")
                continue

            # Use the configured file or the first found
            summary_file = os.path.join(metric_path, config["summary_file"])
            if not os.path.exists(summary_file):
                summary_file = summary_files[0]

            try:
                # Load Summary sheet
                df = pd.read_excel(summary_file, sheet_name='Summary')
                self.summaries[metric_key] = df

                # Try to load Question Legend sheet
                try:
                    legend_df = pd.read_excel(summary_file, sheet_name='Question Legend')
                    self.question_legends[metric_key] = legend_df
                except:
                    pass

                # Extract totals row
                totals_row = df[df['run'] == 'TOTALS'].iloc[0] if 'TOTALS' in df['run'].values else None
                runs_df = df[df['run'] != 'TOTALS']

                loaded_metrics[metric_key] = {
                    "name": config["name"],
                    "description": config["description"],
                    "total_runs": len(runs_df),
                    "avg_score": totals_row[config["score_column"]] if totals_row is not None and config["score_column"] in df.columns else runs_df[config["score_column"]].mean() if config["score_column"] in runs_df.columns else None,
                    "pass_rate": totals_row['pass_rate'] if totals_row is not None else None,
                    "total_na": totals_row['na_not_evaluated'] if totals_row is not None else runs_df['na_not_evaluated'].sum(),
                    "runs_df": runs_df,
                    "score_column": config["score_column"],
                    "needs_contexts": config.get("needs_contexts", True)
                }

                print(f"  [OK] {config['name']}: {len(runs_df)} runs loaded")

            except Exception as e:
                print(f"  [ERROR] {config['name']}: {str(e)}")

        return loaded_metrics

    def get_question_legend(self) -> pd.DataFrame:
        """Get the question legend from any available metric"""
        for metric_key, legend_df in self.question_legends.items():
            if legend_df is not None and not legend_df.empty:
                return legend_df
        return pd.DataFrame()

    def generate_overview_sheet(self, metrics: Dict[str, dict]) -> pd.DataFrame:
        """Generate the metrics overview sheet"""
        rows = []
        for metric_key, data in metrics.items():
            avg_score = data.get("avg_score")
            if avg_score is not None and not pd.isna(avg_score):
                if avg_score >= 0.8:
                    rating = "HIGH"
                elif avg_score >= 0.5:
                    rating = "MEDIUM"
                else:
                    rating = "LOW"
            else:
                rating = "N/A"

            # For metrics that don't need contexts, N/A is not applicable
            needs_contexts = data.get("needs_contexts", True)
            if needs_contexts:
                na_value = int(data["total_na"]) if data["total_na"] and not pd.isna(data["total_na"]) else 0
            else:
                na_value = "-"  # N/A not applicable for this metric

            rows.append({
                "Metric": data["name"],
                "Description": data["description"],
                "Total Runs": data["total_runs"],
                "Average Score": round(avg_score, 4) if avg_score and not pd.isna(avg_score) else "N/A",
                "Pass Rate": data["pass_rate"] if data["pass_rate"] else "N/A",
                "Rating": rating,
                "Total N/A Evaluations": na_value
            })

        return pd.DataFrame(rows)

    def generate_runs_comparison_sheet(self, metrics: Dict[str, dict]) -> pd.DataFrame:
        """Generate a sheet comparing runs across all metrics"""
        # Find the maximum number of runs across all metrics
        max_runs = max(data["total_runs"] for data in metrics.values()) if metrics else 0

        rows = []
        for run_num in range(1, max_runs + 1):
            row = {"Run": run_num}

            for metric_key, data in metrics.items():
                runs_df = data["runs_df"]
                score_col = data["score_column"]
                metric_name = data["name"].replace(" ", "_")

                # Find this run in the dataframe
                run_data = runs_df[runs_df['run'] == run_num]

                if not run_data.empty:
                    run_row = run_data.iloc[0]
                    row[f"{metric_name}_Score"] = round(run_row[score_col], 4) if score_col in run_row and not pd.isna(run_row[score_col]) else "N/A"
                    row[f"{metric_name}_PassRate"] = run_row.get('pass_rate', 'N/A')
                    row[f"{metric_name}_Failed"] = run_row.get('failed', 0)
                    row[f"{metric_name}_NA"] = run_row.get('na_not_evaluated', 0)
                else:
                    row[f"{metric_name}_Score"] = "-"
                    row[f"{metric_name}_PassRate"] = "-"
                    row[f"{metric_name}_Failed"] = "-"
                    row[f"{metric_name}_NA"] = "-"

            rows.append(row)

        return pd.DataFrame(rows)

    def generate_question_analysis_sheet(self, metrics: Dict[str, dict]) -> pd.DataFrame:
        """Generate question-level failure analysis across all metrics"""
        legend_df = self.get_question_legend()
        if legend_df.empty:
            return pd.DataFrame()

        rows = []
        for _, legend_row in legend_df.iterrows():
            letter = legend_row['letter']
            question = legend_row['question']

            row = {
                "Letter": letter,
                "Question": question[:80] + "..." if len(str(question)) > 80 else question
            }

            total_failures = 0
            total_na = 0
            total_runs = 0

            for metric_key, data in metrics.items():
                runs_df = data["runs_df"]
                metric_name = data["name"].replace(" ", "_")

                fail_count = 0
                na_count = 0
                num_runs = len(runs_df)

                for _, run_row in runs_df.iterrows():
                    failed_str = str(run_row.get('failed_questions', ''))
                    na_str = str(run_row.get('na_questions', ''))

                    if letter in failed_str.replace(' ', '').split(','):
                        fail_count += 1
                    if letter in na_str.replace(' ', '').split(','):
                        na_count += 1

                row[f"{metric_name}_Failures"] = fail_count
                row[f"{metric_name}_NA"] = na_count
                row[f"{metric_name}_FailRate"] = f"{(fail_count / num_runs * 100):.1f}%" if num_runs > 0 else "0%"

                total_failures += fail_count
                total_na += na_count
                total_runs = max(total_runs, num_runs)

            row["Total_Failures"] = total_failures
            row["Total_NA"] = total_na

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by total failures descending
        if 'Total_Failures' in df.columns:
            df = df.sort_values('Total_Failures', ascending=False)

        return df

    def _get_metric_short_name(self, metric_name: str) -> str:
        """Get short form of metric name"""
        short_names = {
            "Answer Relevancy": "AR",
            "Context Precision": "CP",
            "Context Relevancy": "CR",
            "Faithfulness": "F"
        }
        return short_names.get(metric_name, metric_name[:2].upper())

    def _get_question_status_data(self, metrics: Dict[str, dict]) -> List[dict]:
        """Get detailed question status data for all metrics"""
        legend_df = self.get_question_legend()
        if legend_df.empty:
            return []

        questions_data = []
        for _, legend_row in legend_df.iterrows():
            letter = legend_row['letter']
            question = legend_row['question']

            question_info = {
                "letter": letter,
                "question": question,
                "metrics_status": {}
            }

            for metric_key, data in metrics.items():
                runs_df = data["runs_df"]
                metric_name = data["name"]
                num_runs = len(runs_df)
                needs_contexts = data.get("needs_contexts", True)

                fail_count = 0
                na_count = 0
                pass_count = 0

                for _, run_row in runs_df.iterrows():
                    failed_str = str(run_row.get('failed_questions', ''))
                    na_str = str(run_row.get('na_questions', ''))

                    if letter in failed_str.replace(' ', '').split(','):
                        fail_count += 1
                    elif letter in na_str.replace(' ', '').split(','):
                        na_count += 1
                    else:
                        pass_count += 1

                question_info["metrics_status"][metric_name] = {
                    "passed": pass_count,
                    "failed": fail_count,
                    "na": na_count,
                    "total_runs": num_runs,
                    "pass_rate": (pass_count / num_runs * 100) if num_runs > 0 else 0,
                    "fail_rate": (fail_count / num_runs * 100) if num_runs > 0 else 0,
                    "na_rate": (na_count / num_runs * 100) if num_runs > 0 else 0,
                    "needs_contexts": needs_contexts
                }

            questions_data.append(question_info)

        return questions_data

    def generate_failed_questions_sheet(self, metrics: Dict[str, dict]) -> pd.DataFrame:
        """Generate sheet for questions that have failures across metrics"""
        questions_data = self._get_question_status_data(metrics)
        if not questions_data:
            return pd.DataFrame()

        rows = []
        for q_data in questions_data:
            # Check if this question has any failures
            has_failures = any(
                status["failed"] > 0
                for status in q_data["metrics_status"].values()
            )

            if not has_failures:
                continue

            row = {
                "Letter": q_data["letter"],
                "Question": q_data["question"][:100] + "..." if len(str(q_data["question"])) > 100 else q_data["question"]
            }

            total_failures = 0
            total_runs = 0

            for metric_name, status in q_data["metrics_status"].items():
                short_name = self._get_metric_short_name(metric_name)
                # Combined format: failed/total
                row[short_name] = f"{status['failed']}/{status['total_runs']}"

                total_failures += status["failed"]
                total_runs += status["total_runs"]

            row["Total"] = f"{total_failures}/{total_runs}"

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty and 'Total' in df.columns:
            # Sort by extracting numerator from Total
            df['_sort'] = df['Total'].apply(lambda x: int(x.split('/')[0]))
            df = df.sort_values('_sort', ascending=False).drop('_sort', axis=1)

        return df

    def generate_na_questions_sheet(self, metrics: Dict[str, dict]) -> pd.DataFrame:
        """Generate sheet for questions that have N/A (not evaluated) across context-dependent metrics"""
        questions_data = self._get_question_status_data(metrics)
        if not questions_data:
            return pd.DataFrame()

        rows = []
        for q_data in questions_data:
            # Check if this question has any N/A (only for context-dependent metrics)
            has_na = any(
                status["na"] > 0 and status.get("needs_contexts", True)
                for status in q_data["metrics_status"].values()
            )

            if not has_na:
                continue

            row = {
                "Letter": q_data["letter"],
                "Question": q_data["question"][:100] + "..." if len(str(q_data["question"])) > 100 else q_data["question"]
            }

            total_na = 0
            total_runs = 0

            for metric_name, status in q_data["metrics_status"].items():
                short_name = self._get_metric_short_name(metric_name)
                needs_contexts = status.get("needs_contexts", True)

                if needs_contexts:
                    # Combined format: na/total
                    row[short_name] = f"{status['na']}/{status['total_runs']}"
                    total_na += status["na"]
                    total_runs += status["total_runs"]
                else:
                    # N/A not applicable for this metric
                    row[short_name] = "-"

            row["Total"] = f"{total_na}/{total_runs}"

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty and 'Total' in df.columns:
            # Sort by extracting numerator from Total
            df['_sort'] = df['Total'].apply(lambda x: int(x.split('/')[0]))
            df = df.sort_values('_sort', ascending=False).drop('_sort', axis=1)

        return df

    def generate_successful_questions_sheet(self, metrics: Dict[str, dict]) -> pd.DataFrame:
        """Generate sheet for questions that passed consistently across all metrics"""
        questions_data = self._get_question_status_data(metrics)
        if not questions_data:
            return pd.DataFrame()

        rows = []
        for q_data in questions_data:
            # Check if this question passed in at least one run for each metric
            has_passes = any(
                status["passed"] > 0
                for status in q_data["metrics_status"].values()
            )

            if not has_passes:
                continue

            row = {
                "Letter": q_data["letter"],
                "Question": q_data["question"][:100] + "..." if len(str(q_data["question"])) > 100 else q_data["question"]
            }

            total_passes = 0
            total_runs = 0
            all_metrics_passed = True

            for metric_name, status in q_data["metrics_status"].items():
                short_name = self._get_metric_short_name(metric_name)
                needs_contexts = status.get("needs_contexts", True)

                # Combined format: passed/total
                row[short_name] = f"{status['passed']}/{status['total_runs']}"

                total_passes += status["passed"]
                total_runs += status["total_runs"]

                # Check if this metric has any failures
                # N/A only counts against "perfect" for context-dependent metrics
                if status["failed"] > 0:
                    all_metrics_passed = False
                elif needs_contexts and status["na"] > 0:
                    all_metrics_passed = False

            row["Total"] = f"{total_passes}/{total_runs}"
            row["Perfect"] = "Yes" if all_metrics_passed else "No"

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            # Sort by Perfect first, then by total passes
            df['_sort'] = df['Total'].apply(lambda x: int(x.split('/')[0]))
            df = df.sort_values(['Perfect', '_sort'], ascending=[False, False]).drop('_sort', axis=1)

        return df

    def generate_summary_statistics(self, metrics: Dict[str, dict]) -> pd.DataFrame:
        """Generate summary statistics"""
        rows = []

        # Overall statistics
        all_scores = []
        all_pass_rates = []
        total_runs = 0
        total_na = 0
        context_dependent_metrics = 0

        for metric_key, data in metrics.items():
            if data["avg_score"] and not pd.isna(data["avg_score"]):
                all_scores.append(data["avg_score"])
            if data["pass_rate"]:
                try:
                    rate = float(str(data["pass_rate"]).replace('%', ''))
                    all_pass_rates.append(rate)
                except:
                    pass
            total_runs += data["total_runs"]
            # Only count N/A for metrics that depend on contexts
            if data.get("needs_contexts", True):
                context_dependent_metrics += 1
                if data["total_na"] and not pd.isna(data["total_na"]):
                    total_na += int(data["total_na"])

        rows.append({
            "Statistic": "Number of Metrics Evaluated",
            "Value": len(metrics)
        })
        rows.append({
            "Statistic": "Total Test Runs (across all metrics)",
            "Value": total_runs
        })
        rows.append({
            "Statistic": "Overall Average Score",
            "Value": f"{sum(all_scores) / len(all_scores):.4f}" if all_scores else "N/A"
        })
        rows.append({
            "Statistic": "Overall Average Pass Rate",
            "Value": f"{sum(all_pass_rates) / len(all_pass_rates):.1f}%" if all_pass_rates else "N/A"
        })
        rows.append({
            "Statistic": "Total N/A Evaluations (context-dependent metrics only)",
            "Value": total_na
        })

        # Best and worst performing metrics
        if all_scores:
            best_metric = max(metrics.items(), key=lambda x: x[1]["avg_score"] if x[1]["avg_score"] and not pd.isna(x[1]["avg_score"]) else 0)
            worst_metric = min(metrics.items(), key=lambda x: x[1]["avg_score"] if x[1]["avg_score"] and not pd.isna(x[1]["avg_score"]) else 1)

            rows.append({
                "Statistic": "Best Performing Metric",
                "Value": f"{best_metric[1]['name']} ({best_metric[1]['avg_score']:.4f})"
            })
            rows.append({
                "Statistic": "Needs Improvement",
                "Value": f"{worst_metric[1]['name']} ({worst_metric[1]['avg_score']:.4f})"
            })

        # Rating distribution
        high_count = sum(1 for data in metrics.values() if data["avg_score"] and not pd.isna(data["avg_score"]) and data["avg_score"] >= 0.8)
        medium_count = sum(1 for data in metrics.values() if data["avg_score"] and not pd.isna(data["avg_score"]) and 0.5 <= data["avg_score"] < 0.8)
        low_count = sum(1 for data in metrics.values() if data["avg_score"] and not pd.isna(data["avg_score"]) and data["avg_score"] < 0.5)

        rows.append({
            "Statistic": "Metrics with HIGH rating (>=0.8)",
            "Value": high_count
        })
        rows.append({
            "Statistic": "Metrics with MEDIUM rating (0.5-0.79)",
            "Value": medium_count
        })
        rows.append({
            "Statistic": "Metrics with LOW rating (<0.5)",
            "Value": low_count
        })

        rows.append({
            "Statistic": "Report Generated At",
            "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        return pd.DataFrame(rows)

    def generate_recommendations(self, metrics: Dict[str, dict]) -> pd.DataFrame:
        """Generate recommendations based on metrics analysis"""
        recommendations = []

        for metric_key, data in metrics.items():
            avg_score = data.get("avg_score")
            if avg_score is None or pd.isna(avg_score):
                continue

            metric_name = data["name"]

            if avg_score < 0.5:
                if metric_key == "faithfulness":
                    recommendations.append({
                        "Priority": "HIGH",
                        "Metric": metric_name,
                        "Issue": f"Low score ({avg_score:.2f})",
                        "Recommendation": "Answers contain information not grounded in retrieved contexts. Review answer generation to ensure responses are based on retrieved content."
                    })
                elif metric_key == "context_relevancy":
                    recommendations.append({
                        "Priority": "HIGH",
                        "Metric": metric_name,
                        "Issue": f"Low score ({avg_score:.2f})",
                        "Recommendation": "Retrieved contexts are not relevant to questions. Improve retrieval system, embeddings, or chunk strategies."
                    })
                elif metric_key == "context_precision":
                    recommendations.append({
                        "Priority": "HIGH",
                        "Metric": metric_name,
                        "Issue": f"Low score ({avg_score:.2f})",
                        "Recommendation": "Relevant contexts are not ranked highly. Consider improving the ranking/reranking mechanism."
                    })
                elif metric_key == "answer_relevancy":
                    recommendations.append({
                        "Priority": "HIGH",
                        "Metric": metric_name,
                        "Issue": f"Low score ({avg_score:.2f})",
                        "Recommendation": "Answers do not address the questions well. Review prompt engineering and answer generation logic."
                    })
            elif avg_score < 0.8:
                recommendations.append({
                    "Priority": "MEDIUM",
                    "Metric": metric_name,
                    "Issue": f"Moderate score ({avg_score:.2f})",
                    "Recommendation": f"Consider optimizing {metric_name.lower()} for better performance."
                })

            # Check for high N/A rate (only for context-dependent metrics)
            if data.get("needs_contexts", True) and data["total_na"] and data["total_runs"]:
                na_rate = data["total_na"] / (data["total_runs"] * 35) * 100  # Assuming 35 questions
                if na_rate > 10:
                    recommendations.append({
                        "Priority": "MEDIUM",
                        "Metric": metric_name,
                        "Issue": f"High N/A rate ({na_rate:.1f}%)",
                        "Recommendation": "Many evaluations failed. Check if contexts are being retrieved properly for all questions."
                    })

        if not recommendations:
            recommendations.append({
                "Priority": "INFO",
                "Metric": "Overall",
                "Issue": "All metrics performing well",
                "Recommendation": "Continue monitoring metrics and consider expanding test coverage."
            })

        return pd.DataFrame(recommendations)

    def generate_questions_summary(self, metrics: Dict[str, dict], failed_df: pd.DataFrame, na_df: pd.DataFrame, successful_df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for question performance"""
        rows = []

        # Get total questions from legend
        legend_df = self.get_question_legend()
        total_questions = len(legend_df) if not legend_df.empty else 0

        rows.append({"Statistic": "Total Questions", "Value": total_questions})
        rows.append({"Statistic": "Questions with Failures", "Value": len(failed_df)})
        rows.append({"Statistic": "Questions with N/A (context-dependent metrics)", "Value": len(na_df)})
        rows.append({"Statistic": "Questions with Passes", "Value": len(successful_df)})

        # Count perfect questions
        perfect_count = len(successful_df[successful_df['Perfect'] == 'Yes']) if not successful_df.empty and 'Perfect' in successful_df.columns else 0
        rows.append({"Statistic": "Perfect Questions (no failures/N/A)", "Value": perfect_count})

        rows.append({"Statistic": "", "Value": ""})
        rows.append({"Statistic": "METRICS EVALUATED", "Value": ""})

        for metric_key, data in metrics.items():
            short_name = self._get_metric_short_name(data["name"])
            needs_ctx = "Yes" if data.get("needs_contexts", True) else "No"
            rows.append({
                "Statistic": f"  {short_name} - {data['name']}",
                "Value": f"{data['total_runs']} runs (Context-dependent: {needs_ctx})"
            })

        rows.append({"Statistic": "", "Value": ""})
        rows.append({"Statistic": "Report Generated At", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

        return pd.DataFrame(rows)

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate the consolidated report"""
        print("\n" + "=" * 60)
        print("CONSOLIDATED RAGAS METRICS REPORT GENERATOR")
        print("=" * 60)

        print("\n[1] Loading metric summaries...")
        metrics = self.load_summaries()

        if not metrics:
            print("\n[ERROR] No metric summaries found!")
            return ""

        print(f"\n[2] Generating consolidated report for {len(metrics)} metrics...")

        # Generate question status sheets
        failed_questions_df = self.generate_failed_questions_sheet(metrics)
        na_questions_df = self.generate_na_questions_sheet(metrics)
        successful_questions_df = self.generate_successful_questions_sheet(metrics)
        legend_df = self.get_question_legend()

        # Generate summary
        summary_df = self.generate_questions_summary(metrics, failed_questions_df, na_questions_df, successful_questions_df)

        # Determine output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.results_path, "consolidated_reports")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"consolidated_report_{timestamp}.xlsx")

        print(f"\n[3] Saving report to {output_file}...")

        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            if not failed_questions_df.empty:
                failed_questions_df.to_excel(writer, sheet_name='Failed Questions', index=False)
            if not na_questions_df.empty:
                na_questions_df.to_excel(writer, sheet_name='NA Questions', index=False)
            if not successful_questions_df.empty:
                successful_questions_df.to_excel(writer, sheet_name='Successful Questions', index=False)
            if not legend_df.empty:
                legend_df.to_excel(writer, sheet_name='Question Legend', index=False)

        print(f"\n[4] Report generated successfully!")
        print(f"    - Failed Questions: {len(failed_questions_df)} questions with failures")
        print(f"    - NA Questions: {len(na_questions_df)} questions with N/A evaluations")
        print(f"    - Successful Questions: {len(successful_questions_df)} questions with passes")

        # Print summary to console
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))

        print("\n" + "=" * 60)
        print(f"Full report saved to: {output_file}")
        print("=" * 60)

        return output_file


def main():
    # Configuration
    CATEGORY_PATH = "/home/shreyank06/Desktop/projects/testing_bulls_pvt_ltd/category/cabin_baggage"

    # Generate report
    generator = ConsolidatedReportGenerator(CATEGORY_PATH)
    output_file = generator.generate_report()

    if output_file:
        print(f"\nReport generation complete!")
        print(f"Open the file to view detailed analysis: {output_file}")


if __name__ == "__main__":
    main()
