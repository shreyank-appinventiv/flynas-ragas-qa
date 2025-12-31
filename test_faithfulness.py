"""
RAGAS Faithfulness Test for Flynas Chatbot
Tests the chatbot's faithfulness using the RAGAS framework with browser automation.
Uses Direct Response Waiting (V3 pattern) - No Race Condition.

Faithfulness measures whether the answer is grounded in the retrieved contexts.
Does NOT require ground truth/reference.
"""

import asyncio
import os
import glob
import re
import json
import time
import random
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field

# Playwright for browser automation
from playwright.async_api import async_playwright, Page
from playwright_stealth.stealth import Stealth

# RAGAS imports
from ragas.metrics import Faithfulness
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import pandas as pd


@dataclass
class ChatbotResponse:
    """Represents a response from the chatbot with retrieved contexts"""
    question: str
    answer: str
    retrieved_contexts: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None


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

        # Launch browser with anti-detection args
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

        # Apply playwright-stealth to mask automation fingerprints
        stealth_config = Stealth()
        await stealth_config.apply_stealth_async(self.page)

        print(f"    Navigating to {self.base_url}...")
        await self.page.goto(self.base_url, wait_until="networkidle", timeout=60000)

        # Human-like delay with randomness
        await asyncio.sleep(random.uniform(3, 5))

        # Scroll behavior - scroll down and up to simulate reading
        print("    Simulating page scroll...")
        await self.page.mouse.wheel(0, random.randint(200, 400))
        await asyncio.sleep(random.uniform(0.5, 1))
        await self.page.mouse.wheel(0, random.randint(-100, -50))
        await asyncio.sleep(random.uniform(0.5, 1))

        # Move mouse naturally before clicking
        await self.page.mouse.move(random.randint(100, 300), random.randint(100, 300))
        await asyncio.sleep(random.uniform(0.5, 1))

        # Click floating button to open chatbot
        print("    Opening chatbot...")
        floating_btn = await self.page.query_selector(".floatingButton")
        if floating_btn:
            # Get button position and move mouse there naturally
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

        # Extract answer
        response_data = body.get('response', {})
        if isinstance(response_data, dict):
            answer = response_data.get('message', '')

        # Method 1: Extract contexts from top-level search_result
        search_result = body.get('search_result', {})
        if search_result:
            references = search_result.get('references', [])
            if references:
                for ref in references:
                    content = ref.get('content', '')
                    if content and content not in contexts:
                        contexts.append(content)

        # Method 2: Extract contexts from debug_data.steps
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
                question=message,
                answer="",
                success=False,
                error="Browser not initialized"
            )

        try:
            # Scroll chat area slightly before typing
            chat_area = await self.page.query_selector(".chat-messages, .messages-container, .chat-body")
            if chat_area:
                await chat_area.hover()
                await self.page.mouse.wheel(0, random.randint(50, 100))
                await asyncio.sleep(random.uniform(0.3, 0.5))

            # Find the editable input div
            input_element = await self.page.query_selector("div.editable-input")
            if not input_element:
                input_element = await self.page.query_selector("div[contenteditable='true']")

            if not input_element:
                return ChatbotResponse(
                    question=message,
                    answer="",
                    success=False,
                    error="Could not find chat input field"
                )

            # Move mouse to input field naturally
            box = await input_element.bounding_box()
            if box:
                await self.page.mouse.move(
                    box['x'] + box['width'] / 2 + random.randint(-10, 10),
                    box['y'] + box['height'] / 2 + random.randint(-3, 3),
                    steps=random.randint(8, 15)
                )
                await asyncio.sleep(random.uniform(0.2, 0.4))

            # Click and type the message with human-like delays
            await input_element.click()
            await asyncio.sleep(random.uniform(0.3, 0.6))
            await input_element.fill("")

            # Type with variable delays to simulate human typing
            for char in message:
                await self.page.keyboard.type(char, delay=0)
                delay = random.uniform(40, 130) / 1000
                if char in ' .,?!':
                    delay = random.uniform(100, 250) / 1000
                await asyncio.sleep(delay)

            await asyncio.sleep(random.uniform(0.4, 0.9))

            # Find send button
            send_btn = await self.page.query_selector(".send-mic-btn")

            # Use expect_response to wait for /chat API response directly
            async with self.page.expect_response(
                lambda r: '/chat' in r.url and r.request.method == 'POST',
                timeout=60000
            ) as response_info:
                # Click send (or press Enter)
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

            # Get the response directly
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
                        question=message,
                        answer="",
                        success=False,
                        error=f"Could not parse response JSON: {str(e)}"
                    )

            # Extract answer and contexts
            answer, contexts = self._extract_contexts_from_response(body)

            await asyncio.sleep(random.uniform(2, 4))

            if answer:
                return ChatbotResponse(
                    question=message,
                    answer=answer,
                    retrieved_contexts=contexts,
                    success=True
                )
            else:
                return ChatbotResponse(
                    question=message,
                    answer="",
                    success=False,
                    error="No answer in response"
                )

        except asyncio.TimeoutError:
            return ChatbotResponse(
                question=message,
                answer="",
                success=False,
                error="Timeout waiting for /chat response (60s)"
            )
        except Exception as e:
            return ChatbotResponse(
                question=message,
                answer="",
                success=False,
                error=str(e)
            )


class FaithfulnessEvaluator:
    """Evaluates chatbot responses using RAGAS Faithfulness metric"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = LangchainLLMWrapper(ChatOpenAI(model=model))
        self.scorer = Faithfulness(llm=self.llm)

    async def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> dict:
        """Evaluate faithfulness for a single question"""
        try:
            if not contexts:
                return {
                    "question": question,
                    "answer": answer,
                    "retrieved_contexts": contexts,
                    "faithfulness_score": None,
                    "success": False,
                    "error": "No contexts retrieved"
                }

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts
            )

            score = await self.scorer.single_turn_ascore(sample)

            return {
                "question": question,
                "answer": answer,
                "retrieved_contexts": contexts,
                "faithfulness_score": score,
                "success": True
            }
        except Exception as e:
            return {
                "question": question,
                "answer": answer,
                "retrieved_contexts": contexts,
                "faithfulness_score": None,
                "success": False,
                "error": str(e)
            }


def load_questions_from_faq(faq_file: str) -> list[str]:
    """Load questions from FAQ Excel/CSV file"""
    try:
        df = pd.read_csv(faq_file)
    except:
        try:
            df = pd.read_excel(faq_file, engine='openpyxl')
        except:
            df = pd.read_excel(faq_file)

    # Look for question column
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


async def main():
    print("=" * 60)
    print("RAGAS Faithfulness Test - Flynas Chatbot")
    print("(Direct Response Waiting - No Race Condition)")
    print("=" * 60)

    # Configuration
    CHATBOT_URL = "https://flynaschb-e0eyd2atbvh8g8ha.a02.azurefd.net/"
    FAQ_FILE = "/home/shreyank06/Desktop/projects/testing_bulls_pvt_ltd/category/cabin_baggage/flynas_cabin_baggage_faq.xlsx"
    RESULTS_DIR = "/home/shreyank06/Desktop/projects/testing_bulls_pvt_ltd/category/cabin_baggage/results/faithfulness"
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

    if not OPENAI_API_KEY:
        print("\n[ERROR] OPENAI_API_KEY environment variable is required.")
        return

    # Load questions
    print("\n[1] Loading questions from FAQ file...")
    TEST_QUESTIONS = load_questions_from_faq(FAQ_FILE)
    print(f"    Loaded {len(TEST_QUESTIONS)} questions")

    # Setup output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_dir = os.path.join(RESULTS_DIR, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    print(f"\n[2] Starting test with {len(TEST_QUESTIONS)} questions...")

    # Start timing
    start_time = time.time()

    try:
        # Collect responses from chatbot
        print("\n[3] Sending test questions to chatbot and capturing contexts...")
        all_responses = []

        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n    [{i}/{len(TEST_QUESTIONS)}] Asking: {question[:70]}{'...' if len(question) > 70 else ''}")

            # Fresh chatbot session for each question
            chatbot = FlynasChatbotWithContextCapture(CHATBOT_URL)
            await chatbot.initialize()

            response = await chatbot.send_message(question)
            await chatbot.close()

            # Wait between sessions
            await asyncio.sleep(random.uniform(3, 5))

            if response.success:
                print(f"         Response: {response.answer[:80]}...")
                print(f"         Contexts captured: {len(response.retrieved_contexts)}")
                all_responses.append(response)
            else:
                print(f"         [FAILED] {response.error}")
                all_responses.append(response)

        # RAGAS Evaluation
        print(f"\n[4] Running RAGAS Faithfulness evaluation...")
        evaluator = FaithfulnessEvaluator(OPENAI_API_KEY)

        results = []
        for i, response in enumerate(all_responses, 1):
            if response.success:
                print(f"    Evaluating {i}/{len(all_responses)}...")
                result = await evaluator.evaluate_single(
                    response.question,
                    response.answer,
                    response.retrieved_contexts
                )
                results.append(result)
            else:
                results.append({
                    "question": response.question,
                    "answer": response.answer,
                    "retrieved_contexts": [],
                    "faithfulness_score": None,
                    "success": False,
                    "error": response.error
                })

        print("\n" + "=" * 60)
        print("RAGAS FAITHFULNESS RESULTS")
        print("=" * 60)

        total_score = 0
        successful_evals = 0

        for i, result in enumerate(results, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Question: {result['question'][:80]}{'...' if len(result['question']) > 80 else ''}")

            if result['success']:
                score = result['faithfulness_score']
                print(f"Faithfulness Score: {score:.4f}")
                print(f"Contexts Used: {len(result['retrieved_contexts'])}")

                if score >= 0.8:
                    print("Rating: HIGH - PASS")
                elif score >= 0.5:
                    print("Rating: MEDIUM - PASS")
                else:
                    print("Rating: LOW - FAIL")

                total_score += score
                successful_evals += 1
            else:
                print(f"Evaluation Error: {result.get('error', 'Unknown error')}")

        avg_score = 0
        if successful_evals > 0:
            avg_score = total_score / successful_evals
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Total test cases: {len(results)}")
            print(f"Successful evaluations: {successful_evals}")
            print(f"Average Faithfulness Score: {avg_score:.4f}")

        # Find the latest serial number
        existing_files = glob.glob(os.path.join(RESULTS_DIR, "faithfulness_results_*.xlsx"))
        max_serial = 0
        for f in existing_files:
            match = re.search(r'faithfulness_results_(\d{1,4})\.xlsx$', f)
            if match:
                num = int(match.group(1))
                if num > max_serial:
                    max_serial = num

        next_serial = max_serial + 1
        xlsx_filename = os.path.join(RESULTS_DIR, f"faithfulness_results_{next_serial}.xlsx")

        print(f"\n[5] Saving results to {xlsx_filename}...")

        PASS_THRESHOLD = 0.5

        # Build data for DataFrame
        rows = []
        for result in results:
            score = result.get('faithfulness_score')
            rating = "N/A"
            pass_fail = "N/A"

            if result.get('success') and score is not None:
                if score >= 0.8:
                    rating = "HIGH"
                elif score >= 0.5:
                    rating = "MEDIUM"
                else:
                    rating = "LOW"
                pass_fail = "PASS" if score >= PASS_THRESHOLD else "FAIL"

            rows.append({
                'question': result['question'],
                'answer': result.get('answer', '')[:500],
                'num_contexts': len(result.get('retrieved_contexts', [])),
                'contexts': ' ||| '.join(result.get('retrieved_contexts', []))[:2000],
                'faithfulness_score': round(score, 4) if score is not None else None,
                'rating': rating,
                'pass_fail': pass_fail,
                'evaluation_status': "OK" if result.get('success') else "ERROR",
                'error': result.get('error', '')
            })

        # Save to Excel
        df = pd.DataFrame(rows)
        df.to_excel(xlsx_filename, index=False, engine='openpyxl')

        print(f"    Results saved to {xlsx_filename}")

        # Update summary file
        summary_file = os.path.join(summary_dir, "faithfulness_summary.xlsx")
        print(f"\n[6] Updating summary file {summary_file}...")

        # Calculate summary statistics
        passed_count = sum(1 for r in rows if r['pass_fail'] == 'PASS')
        failed_count = sum(1 for r in rows if r['pass_fail'] == 'FAIL')
        na_count = sum(1 for r in rows if r['pass_fail'] == 'N/A')
        total_tests = len(rows)
        pass_rate = f"{(passed_count / total_tests * 100):.1f}%" if total_tests > 0 else "0%"

        # Get question letters for failed and N/A questions
        failed_letters = []
        na_letters = []
        for i, r in enumerate(rows):
            letter = chr(65 + i) if i < 26 else f"Q{i+1}"
            if r['pass_fail'] == 'FAIL':
                failed_letters.append(letter)
            elif r['pass_fail'] == 'N/A':
                na_letters.append(letter)

        failed_questions_str = ', '.join(failed_letters) if failed_letters else '-'
        na_questions_str = ', '.join(na_letters) if na_letters else '-'

        # Calculate execution time
        end_time = time.time()
        execution_time_seconds = round(end_time - start_time, 2)
        execution_time_formatted = f"{int(execution_time_seconds // 60)}m {int(execution_time_seconds % 60)}s"

        print(f"\n[7] Test execution time: {execution_time_formatted} ({execution_time_seconds} seconds)")

        # Create new run row
        new_run = {
            'run': next_serial,
            'total_tests': total_tests,
            'passed': passed_count,
            'failed': failed_count,
            'na_not_evaluated': na_count,
            'pass_rate': pass_rate,
            'avg_faithfulness_score': round(avg_score, 4) if successful_evals > 0 else None,
            'failed_questions': failed_questions_str,
            'na_questions': na_questions_str,
            'execution_time_seconds': execution_time_seconds
        }

        # Load existing summary or create new one
        try:
            summary_df = pd.read_excel(summary_file, sheet_name='Summary')
            summary_df = summary_df[summary_df['run'] != 'TOTALS']
        except FileNotFoundError:
            summary_df = pd.DataFrame(columns=[
                'run', 'total_tests', 'passed', 'failed', 'na_not_evaluated',
                'pass_rate', 'avg_faithfulness_score', 'failed_questions', 'na_questions',
                'execution_time_seconds'
            ])

        # Append new run
        summary_df = pd.concat([summary_df, pd.DataFrame([new_run])], ignore_index=True)

        # Calculate totals
        num_runs = len(summary_df)
        total_na = int(summary_df['na_not_evaluated'].sum())
        avg_pass_rate = summary_df['passed'].sum() / summary_df['total_tests'].sum() * 100 if summary_df['total_tests'].sum() > 0 else 0
        overall_avg_score = summary_df['avg_faithfulness_score'].mean()
        avg_execution_time = summary_df['execution_time_seconds'].mean() if 'execution_time_seconds' in summary_df.columns else None

        totals_row = {
            'run': 'TOTALS',
            'total_tests': f"{num_runs} runs",
            'passed': None,
            'failed': None,
            'na_not_evaluated': total_na,
            'pass_rate': f"{avg_pass_rate:.1f}%",
            'avg_faithfulness_score': round(overall_avg_score, 4) if not pd.isna(overall_avg_score) else None,
            'failed_questions': None,
            'na_questions': None,
            'execution_time_seconds': round(avg_execution_time, 2) if avg_execution_time and not pd.isna(avg_execution_time) else None
        }

        summary_df = pd.concat([summary_df, pd.DataFrame([totals_row])], ignore_index=True)

        # Create question legend sheet
        question_legend = []
        for i, q in enumerate(TEST_QUESTIONS):
            letter = chr(65 + i) if i < 26 else f"Q{i+1}"
            question_legend.append({'letter': letter, 'question': q})
        legend_df = pd.DataFrame(question_legend)

        # Calculate failure percentage per question across all runs
        runs_df = summary_df[summary_df['run'] != 'TOTALS'].copy()
        num_runs_for_stats = len(runs_df)

        question_failure_stats = []
        for i, q in enumerate(TEST_QUESTIONS):
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

        # Save all sheets to Excel
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            failure_stats_df.to_excel(writer, sheet_name='Question Failure Rates', index=False)
            legend_df.to_excel(writer, sheet_name='Question Legend', index=False)

        print(f"    Summary updated with run {next_serial}")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
