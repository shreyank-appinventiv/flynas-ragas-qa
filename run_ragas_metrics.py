"""
Unified RAGAS Metrics Test Runner for Flynas Chatbot
Run one or multiple RAGAS metrics with a single script.
Supports: Answer Relevancy, Context Precision, Context Relevancy, Faithfulness
"""

import asyncio
import os
import glob
import re
import json
import time
import random
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field

# Playwright for browser automation
from playwright.async_api import async_playwright, Page
from playwright_stealth.stealth import Stealth

# RAGAS imports
from ragas.metrics import (
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
    ContextRelevance,
    Faithfulness
)
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================
CHATBOT_URL = "https://flynaschb-e0eyd2atbvh8g8ha.a02.azurefd.net/"
FAQ_FILE = "/home/shreyank06/Desktop/projects/testing_bulls_pvt_ltd/category/cabin_baggage/flynas_cabin_baggage_faq.xlsx"
RESULTS_BASE_DIR = "/home/shreyank06/Desktop/projects/testing_bulls_pvt_ltd/category/cabin_baggage/results"
PASS_THRESHOLD = 0.5


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class ChatbotResponse:
    """Represents a response from the chatbot with retrieved contexts"""
    question: str
    answer: str
    retrieved_contexts: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None


@dataclass
class MetricResult:
    """Represents evaluation result for a single question"""
    question: str
    answer: str
    contexts: List[str]
    score: Optional[float]
    success: bool
    error: Optional[str] = None


# =============================================================================
# CHATBOT BROWSER CLASS
# =============================================================================
class FlynasChatbotWithContextCapture:
    """Interacts with Flynas chatbot and captures retrieved contexts from network"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.page: Optional[Page] = None
        self.browser = None
        self.playwright = None

    async def initialize(self):
        """Initialize the browser and navigate to chatbot"""
        print("    Launching browser...")
        self.playwright = await async_playwright().start()

        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-dev-shm-usage'
            ]
        )

        context = await self.browser.new_context(
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        self.page = await context.new_page()

        stealth_config = Stealth()
        await stealth_config.apply_stealth_async(self.page)

        print(f"    Navigating to {self.base_url}...")
        await self.page.goto(self.base_url, wait_until="networkidle", timeout=60000)

        await asyncio.sleep(random.uniform(3, 5))

        print("    Simulating page scroll...")
        await self.page.mouse.wheel(0, random.randint(200, 400))
        await asyncio.sleep(random.uniform(0.5, 1))
        await self.page.mouse.wheel(0, random.randint(-100, -50))
        await asyncio.sleep(random.uniform(0.5, 1))

        await self.page.mouse.move(random.randint(100, 300), random.randint(100, 300))
        await asyncio.sleep(random.uniform(0.5, 1))

        print("    Opening chatbot...")
        floating_btn = await self.page.query_selector(".floatingButton")
        if floating_btn:
            box = await floating_btn.bounding_box()
            if box:
                await self.page.mouse.move(
                    box['x'] + box['width'] / 2 + random.randint(-5, 5),
                    box['y'] + box['height'] / 2 + random.randint(-5, 5),
                    steps=random.randint(10, 20)
                )
                await asyncio.sleep(random.uniform(0.3, 0.6))
            await floating_btn.click()
            await asyncio.sleep(random.uniform(4, 6))

        print("    Chatbot ready.")

    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    def _extract_contexts_from_response(self, body: dict) -> tuple[str, List[str]]:
        """Extract answer and contexts from API response body"""
        answer = ""
        contexts = []

        response_data = body.get('response', {})
        if isinstance(response_data, dict):
            answer = response_data.get('message', '')

        search_result = body.get('search_result', {})
        if search_result:
            references = search_result.get('references', [])
            if references:
                for ref in references:
                    content = ref.get('content', '')
                    if content and content not in contexts:
                        contexts.append(content)

        debug_data = body.get('debug_data')
        if debug_data and isinstance(debug_data, dict) and 'steps' in debug_data:
            for step in debug_data['steps']:
                step_data = step.get('data', {})
                if step_data:
                    faq_trace = step_data.get('faq_trace', {})
                    if faq_trace:
                        step_search_result = faq_trace.get('search_result', {})
                        references = step_search_result.get('references', [])
                        if references:
                            for ref in references:
                                content = ref.get('content', '')
                                if content and content not in contexts:
                                    contexts.append(content)

        return answer, contexts

    async def send_message(self, message: str) -> ChatbotResponse:
        """Send a message to the chatbot and capture response with contexts"""
        if not self.page:
            return ChatbotResponse(
                question=message, answer="", success=False,
                error="Browser not initialized"
            )

        try:
            chat_area = await self.page.query_selector(".chat-messages, .messages-container, .chat-body")
            if chat_area:
                await chat_area.hover()
                await self.page.mouse.wheel(0, random.randint(50, 100))
                await asyncio.sleep(random.uniform(0.3, 0.5))

            input_element = await self.page.query_selector("div.editable-input")
            if not input_element:
                input_element = await self.page.query_selector("div[contenteditable='true']")

            if not input_element:
                return ChatbotResponse(
                    question=message, answer="", success=False,
                    error="Could not find chat input field"
                )

            box = await input_element.bounding_box()
            if box:
                await self.page.mouse.move(
                    box['x'] + box['width'] / 2 + random.randint(-10, 10),
                    box['y'] + box['height'] / 2 + random.randint(-3, 3),
                    steps=random.randint(8, 15)
                )
                await asyncio.sleep(random.uniform(0.2, 0.4))

            await input_element.click()
            await asyncio.sleep(random.uniform(0.3, 0.6))
            await input_element.fill("")

            for char in message:
                await self.page.keyboard.type(char, delay=0)
                delay = random.uniform(40, 130) / 1000
                if char in ' .,?!':
                    delay = random.uniform(100, 250) / 1000
                await asyncio.sleep(delay)

            await asyncio.sleep(random.uniform(0.4, 0.9))

            send_btn = await self.page.query_selector(".send-mic-btn")

            async with self.page.expect_response(
                lambda r: '/chat' in r.url and r.request.method == 'POST',
                timeout=60000
            ) as response_info:
                if send_btn:
                    box = await send_btn.bounding_box()
                    if box:
                        await self.page.mouse.move(
                            box['x'] + box['width'] / 2 + random.randint(-3, 3),
                            box['y'] + box['height'] / 2 + random.randint(-3, 3),
                            steps=random.randint(5, 10)
                        )
                        await asyncio.sleep(random.uniform(0.2, 0.4))
                    await send_btn.click()
                else:
                    await self.page.keyboard.press("Enter")

            response = await response_info.value
            await asyncio.sleep(0.5)

            try:
                body = await response.json()
            except Exception as e:
                try:
                    text = await response.text()
                    body = json.loads(text)
                except:
                    return ChatbotResponse(
                        question=message, answer="", success=False,
                        error=f"Could not parse response JSON: {str(e)}"
                    )

            answer, contexts = self._extract_contexts_from_response(body)
            await asyncio.sleep(random.uniform(2, 4))

            if answer:
                return ChatbotResponse(
                    question=message, answer=answer,
                    retrieved_contexts=contexts, success=True
                )
            else:
                return ChatbotResponse(
                    question=message, answer="", success=False,
                    error="No answer in response"
                )

        except asyncio.TimeoutError:
            return ChatbotResponse(
                question=message, answer="", success=False,
                error="Timeout waiting for /chat response (60s)"
            )
        except Exception as e:
            return ChatbotResponse(
                question=message, answer="", success=False,
                error=str(e)
            )


# =============================================================================
# METRIC EVALUATORS
# =============================================================================
class BaseEvaluator:
    """Base class for metric evaluators"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = LangchainLLMWrapper(ChatOpenAI(model=model))
        self.scorer = None
        self.metric_name = "base"
        self.score_column = "score"

    async def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> MetricResult:
        raise NotImplementedError


class AnswerRelevancyEvaluator(BaseEvaluator):
    """Evaluates Answer Relevancy metric"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        super().__init__(openai_api_key, model)
        self.embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        self.scorer = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        self.metric_name = "answer_relevancy"
        self.score_column = "relevancy_score"

    async def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> MetricResult:
        try:
            sample = SingleTurnSample(user_input=question, response=answer)
            score = await self.scorer.single_turn_ascore(sample)
            return MetricResult(question=question, answer=answer, contexts=contexts,
                              score=score, success=True)
        except Exception as e:
            return MetricResult(question=question, answer=answer, contexts=contexts,
                              score=None, success=False, error=str(e))


class ContextPrecisionEvaluator(BaseEvaluator):
    """Evaluates Context Precision metric"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        super().__init__(openai_api_key, model)
        self.scorer = LLMContextPrecisionWithoutReference(llm=self.llm)
        self.metric_name = "context_precision"
        self.score_column = "precision_score"

    async def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> MetricResult:
        try:
            if not contexts:
                return MetricResult(question=question, answer=answer, contexts=contexts,
                                  score=None, success=False, error="No contexts retrieved")

            sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
            score = await self.scorer.single_turn_ascore(sample)
            return MetricResult(question=question, answer=answer, contexts=contexts,
                              score=score, success=True)
        except Exception as e:
            return MetricResult(question=question, answer=answer, contexts=contexts,
                              score=None, success=False, error=str(e))


class ContextRelevancyEvaluator(BaseEvaluator):
    """Evaluates Context Relevancy metric"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        super().__init__(openai_api_key, model)
        self.scorer = ContextRelevance(llm=self.llm)
        self.metric_name = "context_relevancy"
        self.score_column = "relevancy_score"

    async def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> MetricResult:
        try:
            if not contexts:
                return MetricResult(question=question, answer=answer, contexts=contexts,
                                  score=None, success=False, error="No contexts retrieved")

            sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
            score = await self.scorer.single_turn_ascore(sample)
            return MetricResult(question=question, answer=answer, contexts=contexts,
                              score=score, success=True)
        except Exception as e:
            return MetricResult(question=question, answer=answer, contexts=contexts,
                              score=None, success=False, error=str(e))


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluates Faithfulness metric"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        super().__init__(openai_api_key, model)
        self.scorer = Faithfulness(llm=self.llm)
        self.metric_name = "faithfulness"
        self.score_column = "faithfulness_score"

    async def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> MetricResult:
        try:
            if not contexts:
                return MetricResult(question=question, answer=answer, contexts=contexts,
                                  score=None, success=False, error="No contexts retrieved")

            sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
            score = await self.scorer.single_turn_ascore(sample)
            return MetricResult(question=question, answer=answer, contexts=contexts,
                              score=score, success=True)
        except Exception as e:
            return MetricResult(question=question, answer=answer, contexts=contexts,
                              score=None, success=False, error=str(e))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_questions_from_faq(faq_file: str) -> list[str]:
    """Load questions from FAQ Excel/CSV file"""
    try:
        df = pd.read_csv(faq_file)
    except:
        try:
            df = pd.read_excel(faq_file, engine='openpyxl')
        except:
            df = pd.read_excel(faq_file)

    question_col = None
    for col in df.columns:
        if 'question' in col.lower():
            question_col = col
            break

    if question_col is None:
        for col in df.columns:
            if 'id' not in col.lower() and 'category' not in col.lower():
                question_col = col
                break

    if question_col is None:
        question_col = df.columns[0]

    questions = df[question_col].dropna().tolist()
    return [str(q) for q in questions]


def get_next_serial(results_dir: str, prefix: str) -> int:
    """Get the next serial number for result files"""
    existing_files = glob.glob(os.path.join(results_dir, f"{prefix}_results_*.xlsx"))
    max_serial = 0
    for f in existing_files:
        match = re.search(rf'{prefix}_results_(\d{{1,4}})\.xlsx$', f)
        if match:
            num = int(match.group(1))
            if num > max_serial:
                max_serial = num
    return max_serial + 1


def save_results(results: List[MetricResult], metric_name: str, score_column: str,
                results_dir: str, questions: List[str], execution_time: float):
    """Save results to Excel and update summary"""

    os.makedirs(results_dir, exist_ok=True)
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    next_serial = get_next_serial(results_dir, metric_name)
    xlsx_filename = os.path.join(results_dir, f"{metric_name}_results_{next_serial}.xlsx")

    # Build rows
    rows = []
    for result in results:
        score = result.score
        rating = "N/A"
        pass_fail = "N/A"

        if result.success and score is not None:
            if score >= 0.8:
                rating = "HIGH"
            elif score >= 0.5:
                rating = "MEDIUM"
            else:
                rating = "LOW"
            pass_fail = "PASS" if score >= PASS_THRESHOLD else "FAIL"

        rows.append({
            'question': result.question,
            'answer': result.answer[:500] if result.answer else '',
            'num_contexts': len(result.contexts),
            'contexts': ' ||| '.join(result.contexts)[:2000],
            score_column: round(score, 4) if score is not None else None,
            'rating': rating,
            'pass_fail': pass_fail,
            'evaluation_status': "OK" if result.success else "ERROR",
            'error': result.error or ''
        })

    df = pd.DataFrame(rows)
    df.to_excel(xlsx_filename, index=False, engine='openpyxl')
    print(f"    Results saved to {xlsx_filename}")

    # Update summary
    summary_file = os.path.join(summary_dir, f"{metric_name}_summary.xlsx")

    passed_count = sum(1 for r in rows if r['pass_fail'] == 'PASS')
    failed_count = sum(1 for r in rows if r['pass_fail'] == 'FAIL')
    na_count = sum(1 for r in rows if r['pass_fail'] == 'N/A')
    total_tests = len(rows)
    pass_rate = f"{(passed_count / total_tests * 100):.1f}%" if total_tests > 0 else "0%"

    successful_scores = [r[score_column] for r in rows if r[score_column] is not None]
    avg_score = sum(successful_scores) / len(successful_scores) if successful_scores else 0

    failed_letters = []
    na_letters = []
    for i, r in enumerate(rows):
        letter = chr(65 + i) if i < 26 else f"Q{i+1}"
        if r['pass_fail'] == 'FAIL':
            failed_letters.append(letter)
        elif r['pass_fail'] == 'N/A':
            na_letters.append(letter)

    new_run = {
        'run': next_serial,
        'total_tests': total_tests,
        'passed': passed_count,
        'failed': failed_count,
        'na_not_evaluated': na_count,
        'pass_rate': pass_rate,
        f'avg_{score_column}': round(avg_score, 4) if successful_scores else None,
        'failed_questions': ', '.join(failed_letters) if failed_letters else '-',
        'na_questions': ', '.join(na_letters) if na_letters else '-',
        'execution_time_seconds': round(execution_time, 2)
    }

    try:
        summary_df = pd.read_excel(summary_file, sheet_name='Summary')
        summary_df = summary_df[summary_df['run'] != 'TOTALS']
    except FileNotFoundError:
        summary_df = pd.DataFrame()

    summary_df = pd.concat([summary_df, pd.DataFrame([new_run])], ignore_index=True)

    # Calculate totals
    num_runs = len(summary_df)
    total_na = int(summary_df['na_not_evaluated'].sum())
    avg_pass_rate = summary_df['passed'].sum() / summary_df['total_tests'].sum() * 100 if summary_df['total_tests'].sum() > 0 else 0
    score_col_name = f'avg_{score_column}'
    overall_avg_score = summary_df[score_col_name].mean() if score_col_name in summary_df.columns else 0
    avg_execution_time = summary_df['execution_time_seconds'].mean()

    totals_row = {
        'run': 'TOTALS',
        'total_tests': f"{num_runs} runs",
        'passed': None,
        'failed': None,
        'na_not_evaluated': total_na,
        'pass_rate': f"{avg_pass_rate:.1f}%",
        score_col_name: round(overall_avg_score, 4) if not pd.isna(overall_avg_score) else None,
        'failed_questions': None,
        'na_questions': None,
        'execution_time_seconds': round(avg_execution_time, 2) if not pd.isna(avg_execution_time) else None
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([totals_row])], ignore_index=True)

    # Question legend
    question_legend = []
    for i, q in enumerate(questions):
        letter = chr(65 + i) if i < 26 else f"Q{i+1}"
        question_legend.append({'letter': letter, 'question': q})
    legend_df = pd.DataFrame(question_legend)

    # Question failure stats
    runs_df = summary_df[summary_df['run'] != 'TOTALS'].copy()
    num_runs_for_stats = len(runs_df)

    question_failure_stats = []
    for i, q in enumerate(questions):
        letter = chr(65 + i) if i < 26 else f"Q{i+1}"
        fail_count = 0
        for failed_str in runs_df['failed_questions'].fillna(''):
            if letter in str(failed_str).replace(' ', '').split(','):
                fail_count += 1
        na_count_q = 0
        for na_str in runs_df['na_questions'].fillna(''):
            if letter in str(na_str).replace(' ', '').split(','):
                na_count_q += 1
        pass_count = num_runs_for_stats - fail_count - na_count_q
        fail_pct = (fail_count / num_runs_for_stats * 100) if num_runs_for_stats > 0 else 0

        question_failure_stats.append({
            'letter': letter,
            'question': q[:80] + '...' if len(q) > 80 else q,
            'total_runs': num_runs_for_stats,
            'passed': pass_count,
            'failed': fail_count,
            'na': na_count_q,
            'failure_rate': f"{fail_pct:.1f}%"
        })

    failure_stats_df = pd.DataFrame(question_failure_stats)

    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        failure_stats_df.to_excel(writer, sheet_name='Question Failure Rates', index=False)
        legend_df.to_excel(writer, sheet_name='Question Legend', index=False)

    print(f"    Summary updated: {summary_file}")


# =============================================================================
# METRIC CONFIGURATION
# =============================================================================
METRICS = {
    "1": {
        "name": "Answer Relevancy",
        "key": "answer_relevancy",
        "evaluator_class": AnswerRelevancyEvaluator,
        "needs_contexts": False,
        "description": "Measures if the answer addresses the question"
    },
    "2": {
        "name": "Context Precision",
        "key": "context_precision",
        "evaluator_class": ContextPrecisionEvaluator,
        "needs_contexts": True,
        "description": "Measures if relevant contexts are ranked higher"
    },
    "3": {
        "name": "Context Relevancy",
        "key": "context_relevancy",
        "evaluator_class": ContextRelevancyEvaluator,
        "needs_contexts": True,
        "description": "Measures if retrieved contexts are relevant to the question"
    },
    "4": {
        "name": "Faithfulness",
        "key": "faithfulness",
        "evaluator_class": FaithfulnessEvaluator,
        "needs_contexts": True,
        "description": "Measures if the answer is grounded in the retrieved contexts"
    }
}


# =============================================================================
# MAIN RUNNER
# =============================================================================
async def run_metric(metric_config: dict, responses: List[ChatbotResponse],
                    openai_api_key: str, questions: List[str]) -> float:
    """Run a single metric evaluation"""

    metric_name = metric_config["name"]
    metric_key = metric_config["key"]

    print(f"\n{'='*60}")
    print(f"Running {metric_name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Initialize evaluator
    evaluator = metric_config["evaluator_class"](openai_api_key)

    # Evaluate all responses
    results = []
    for i, response in enumerate(responses, 1):
        if response.success:
            print(f"    Evaluating {i}/{len(responses)}...")
            result = await evaluator.evaluate_single(
                response.question,
                response.answer,
                response.retrieved_contexts
            )
            results.append(result)
        else:
            results.append(MetricResult(
                question=response.question,
                answer=response.answer,
                contexts=[],
                score=None,
                success=False,
                error=response.error
            ))

    execution_time = time.time() - start_time

    # Print results
    total_score = 0
    successful_evals = 0

    for i, result in enumerate(results, 1):
        if result.success and result.score is not None:
            total_score += result.score
            successful_evals += 1

    avg_score = total_score / successful_evals if successful_evals > 0 else 0

    print(f"\n    {metric_name} Results:")
    print(f"    - Successful evaluations: {successful_evals}/{len(results)}")
    print(f"    - Average score: {avg_score:.4f}")
    print(f"    - Execution time: {execution_time:.1f}s")

    # Save results
    results_dir = os.path.join(RESULTS_BASE_DIR, metric_key)
    save_results(results, metric_key, evaluator.score_column, results_dir, questions, execution_time)

    return avg_score


async def collect_responses(questions: List[str]) -> List[ChatbotResponse]:
    """Collect responses from chatbot for all questions"""

    print(f"\n{'='*60}")
    print("Collecting Chatbot Responses")
    print(f"{'='*60}")

    all_responses = []

    for i, question in enumerate(questions, 1):
        print(f"\n    [{i}/{len(questions)}] Asking: {question[:70]}{'...' if len(question) > 70 else ''}")

        chatbot = FlynasChatbotWithContextCapture(CHATBOT_URL)
        await chatbot.initialize()

        response = await chatbot.send_message(question)
        await chatbot.close()

        await asyncio.sleep(random.uniform(3, 5))

        if response.success:
            print(f"         Response: {response.answer[:80]}...")
            print(f"         Contexts captured: {len(response.retrieved_contexts)}")
        else:
            print(f"         [FAILED] {response.error}")

        all_responses.append(response)

    return all_responses


def show_menu() -> List[str]:
    """Display menu and get user selection"""

    print("\n" + "=" * 60)
    print("RAGAS METRICS TEST RUNNER - Flynas Chatbot")
    print("=" * 60)
    print("\nAvailable Metrics:")
    print("-" * 40)

    for key, config in METRICS.items():
        ctx_indicator = "[needs contexts]" if config["needs_contexts"] else "[no contexts needed]"
        print(f"  [{key}] {config['name']}")
        print(f"      {config['description']}")
        print(f"      {ctx_indicator}")

    print("\n  [5] Run ALL metrics")
    print("  [0] Exit")
    print("-" * 40)

    while True:
        choice = input("\nEnter your choice (comma-separated for multiple, e.g., 1,3): ").strip()

        if choice == "0":
            return []

        if choice == "5":
            return list(METRICS.keys())

        selected = [c.strip() for c in choice.split(",")]

        valid = all(s in METRICS for s in selected)
        if valid and selected:
            return selected

        print("Invalid choice. Please enter valid metric numbers.")


async def main():
    """Main entry point"""

    # Check for API key
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    if not OPENAI_API_KEY:
        print("\n[ERROR] OPENAI_API_KEY environment variable is required.")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        return

    # Show menu and get selection
    selected_metrics = show_menu()

    if not selected_metrics:
        print("\nExiting...")
        return

    # Load questions
    print(f"\n[1] Loading questions from FAQ file...")
    questions = load_questions_from_faq(FAQ_FILE)
    print(f"    Loaded {len(questions)} questions")

    # Show selected metrics
    print(f"\n[2] Selected metrics to run:")
    for key in selected_metrics:
        print(f"    - {METRICS[key]['name']}")

    # Confirm
    confirm = input(f"\nProceed with {len(questions)} questions? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    # Collect responses (only once, reuse for all metrics)
    print(f"\n[3] Collecting chatbot responses...")
    start_time = time.time()
    responses = await collect_responses(questions)
    collection_time = time.time() - start_time
    print(f"\n    Response collection completed in {collection_time:.1f}s")

    # Run selected metrics
    print(f"\n[4] Running {len(selected_metrics)} metric(s)...")

    results_summary = {}
    for key in selected_metrics:
        config = METRICS[key]
        avg_score = await run_metric(config, responses, OPENAI_API_KEY, questions)
        results_summary[config["name"]] = avg_score

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for metric_name, score in results_summary.items():
        rating = "HIGH" if score >= 0.8 else "MEDIUM" if score >= 0.5 else "LOW"
        print(f"  {metric_name}: {score:.4f} ({rating})")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")

    # Automatically generate consolidated report
    print("\n" + "=" * 60)
    print("Generating Consolidated Report...")
    print("=" * 60)

    from generate_consolidated_report import ConsolidatedReportGenerator
    generator = ConsolidatedReportGenerator(os.path.dirname(RESULTS_BASE_DIR))
    report_file = generator.generate_report()

    if report_file:
        print(f"\nConsolidated report saved to: {report_file}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
