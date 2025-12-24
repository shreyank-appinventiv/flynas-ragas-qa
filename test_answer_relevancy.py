"""
RAGAS Answer Relevancy Test for Flynas Chatbot
Tests the chatbot's answer relevancy using the RAGAS framework with browser automation.
"""

import asyncio
import os
import glob
import re
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field

# Playwright for browser automation
from playwright.async_api import async_playwright, Page

# RAGAS imports
from ragas.metrics import AnswerRelevancy
from ragas.metrics._answer_relevance import ResponseRelevanceInput, ResponseRelevancePrompt
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd


@dataclass
class ChatbotResponse:
    """Represents a response from the chatbot"""
    question: str
    answer: str
    success: bool
    error: Optional[str] = None
    followup_questions: List[str] = field(default_factory=list)


class FlynasChatbotBrowser:
    """Interacts with Flynas chatbot using browser automation"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.page: Optional[Page] = None
        self.browser = None
        self.playwright = None

    async def initialize(self):
        """Initialize the browser and navigate to chatbot"""
        print("    Launching browser...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="en-US"
        )
        self.page = await context.new_page()

        print(f"    Navigating to {self.base_url}...")
        await self.page.goto(self.base_url, wait_until="networkidle", timeout=60000)
        await asyncio.sleep(3)

        # Click floating button to open chatbot
        print("    Opening chatbot...")
        floating_btn = await self.page.query_selector(".floatingButton")
        if floating_btn:
            await floating_btn.click()
            await asyncio.sleep(3)

        print("    Chatbot ready.")

    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def send_message(self, message: str) -> ChatbotResponse:
        """Send a message to the chatbot and get a response"""
        if not self.page:
            return ChatbotResponse(
                question=message,
                answer="",
                success=False,
                error="Browser not initialized"
            )

        try:
            # Find the editable input div
            input_element = await self.page.query_selector("div.editable-input")

            if not input_element:
                # Fallback selectors
                input_element = await self.page.query_selector("div[contenteditable='true']")

            if not input_element:
                return ChatbotResponse(
                    question=message,
                    answer="",
                    success=False,
                    error="Could not find chat input field"
                )

            # Count bot messages before sending
            bot_messages_before = await self.page.query_selector_all(".bot-message, .assistant-message, [class*='bot'], [class*='assistant']")
            count_before = len(bot_messages_before)

            # Click and type the message
            await input_element.click()
            await input_element.fill("")
            await self.page.keyboard.type(message, delay=50)

            # Click send button
            send_btn = await self.page.query_selector(".send-mic-btn, button[aria-label='Send message'], button[aria-label*='Send']")
            if send_btn:
                await send_btn.click()
            else:
                await self.page.keyboard.press("Enter")

            # Wait for response
            await asyncio.sleep(10)

            # Get chatbot container text
            chatbot = await self.page.query_selector("[class*='chatbot']")
            if not chatbot:
                return ChatbotResponse(question=message, answer="", success=False, error="Chatbot not found")

            all_text = await chatbot.inner_text()

            # Split by timestamps (format: X:XX PM or X:XX AM)
            parts = re.split(r'\d{1,2}:\d{2}\s*[AP]M', all_text)

            # Filter out empty parts, user message, and welcome messages
            response_text = ""
            skip_phrases = ["welcome aboard", "hi there", "flynas ai", message.lower()]

            for part in reversed(parts):
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                part_lower = part.lower()
                if any(skip in part_lower for skip in skip_phrases):
                    continue
                if "analysing" in part_lower:
                    continue
                response_text = part
                break

            # Extract follow-up questions from the response
            followup_questions = []
            # Look for questions in the response (sentences ending with ?)
            question_pattern = r'([^.!?\n]*\?)'
            found_questions = re.findall(question_pattern, response_text)
            for q in found_questions:
                q = q.strip()
                # Filter out very short questions or the original question
                if len(q) > 10 and q.lower() != message.lower():
                    followup_questions.append(q)

            if response_text and response_text.strip():
                return ChatbotResponse(
                    question=message,
                    answer=response_text.strip(),
                    success=True,
                    followup_questions=followup_questions
                )
            else:
                return ChatbotResponse(
                    question=message,
                    answer="",
                    success=False,
                    error="Could not extract response from chatbot"
                )

        except Exception as e:
            return ChatbotResponse(
                question=message,
                answer="",
                success=False,
                error=str(e)
            )


class RAGASEvaluator:
    """Evaluates chatbot responses using RAGAS Answer Relevancy metric"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = LangchainLLMWrapper(ChatOpenAI(model=model))
        self.embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        self.scorer = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        self.question_generator = ResponseRelevancePrompt()

    async def generate_questions_from_answer(self, answer: str, n: int = 3) -> List[str]:
        """Generate questions from answer using RAGAS prompt (same as internal RAGAS logic)"""
        try:
            prompt_input = ResponseRelevanceInput(response=answer)
            responses = await self.question_generator.generate_multiple(
                data=prompt_input, llm=self.llm, callbacks=None, n=n
            )
            return [r.question for r in responses]
        except Exception as e:
            print(f"    Warning: Could not generate questions: {e}")
            return []

    async def evaluate_single(self, question: str, answer: str) -> dict:
        """Evaluate a single question-answer pair"""
        try:
            from ragas import SingleTurnSample

            sample = SingleTurnSample(
                user_input=question,
                response=answer
            )

            # Get the score from RAGAS
            score = await self.scorer.single_turn_ascore(sample)

            # Generate questions separately to capture them (same logic RAGAS uses internally)
            generated_questions = await self.generate_questions_from_answer(answer)

            return {
                "question": question,
                "answer": answer,
                "relevancy_score": score,
                "generated_questions": generated_questions,
                "success": True
            }
        except Exception as e:
            return {
                "question": question,
                "answer": answer,
                "relevancy_score": None,
                "generated_questions": [],
                "success": False,
                "error": str(e)
            }

    async def evaluate_batch(self, qa_pairs: list[tuple[str, str]]) -> list[dict]:
        """Evaluate multiple question-answer pairs"""
        tasks = [self.evaluate_single(q, a) for q, a in qa_pairs]
        return await asyncio.gather(*tasks)


def get_available_categories(category_base_path: str) -> list[dict]:
    """Scan category folder and return available categories with their FAQ files"""
    categories = []

    if not os.path.exists(category_base_path):
        return categories

    for item in os.listdir(category_base_path):
        item_path = os.path.join(category_base_path, item)
        if os.path.isdir(item_path):
            # Look for FAQ file (pattern: flynas_*_faq.xlsx or similar)
            faq_files = glob.glob(os.path.join(item_path, "flynas_*_faq.xlsx"))
            if not faq_files:
                faq_files = glob.glob(os.path.join(item_path, "*_faq.xlsx"))
            if not faq_files:
                faq_files = glob.glob(os.path.join(item_path, "*.xlsx"))

            if faq_files:
                categories.append({
                    "name": item,
                    "path": item_path,
                    "faq_file": faq_files[0]
                })

    return categories


def load_questions_from_faq(faq_file: str) -> list[str]:
    """Load questions from FAQ Excel/CSV file"""
    try:
        # Try reading as CSV first (in case it's misnamed)
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
        # Use first column that's not 'id' or 'category'
        for col in df.columns:
            if 'id' not in col.lower() and 'category' not in col.lower():
                question_col = col
                break

    if question_col is None:
        question_col = df.columns[0]

    questions = df[question_col].dropna().tolist()
    return [str(q) for q in questions]


def select_category(categories: list[dict]) -> dict:
    """Display categories and let user select one"""
    print("\n" + "=" * 60)
    print("AVAILABLE CATEGORIES")
    print("=" * 60)

    for i, cat in enumerate(categories, 1):
        # Count questions in FAQ file
        try:
            questions = load_questions_from_faq(cat["faq_file"])
            q_count = len(questions)
        except:
            q_count = "?"

        print(f"  [{i}] {cat['name']} ({q_count} questions)")
        print(f"      FAQ: {os.path.basename(cat['faq_file'])}")

    print("\n" + "-" * 60)

    while True:
        try:
            choice = input("Select category number to test: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(categories):
                return categories[idx]
            else:
                print(f"Please enter a number between 1 and {len(categories)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)


async def main():
    print("=" * 60)
    print("RAGAS Answer Relevancy Test - Flynas Chatbot")
    print("(Using Browser Automation)")
    print("=" * 60)

    # Configuration
    CHATBOT_URL = "https://flynaschb-frontendapp-qa.blackflower-c82fc685.westeurope.azurecontainerapps.io"
    CATEGORY_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "category")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

    if not OPENAI_API_KEY:
        print("\n[ERROR] OPENAI_API_KEY environment variable is required.")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        return

    # Parse available categories
    print("\n[1] Scanning for available categories...")
    categories = get_available_categories(CATEGORY_BASE_PATH)

    if not categories:
        print(f"\n[ERROR] No categories found in {CATEGORY_BASE_PATH}")
        print("Please create category folders with FAQ files (e.g., flynas_*_faq.xlsx)")
        return

    print(f"    Found {len(categories)} category(ies)")

    # Let user select category
    selected_category = select_category(categories)
    category_name = selected_category["name"]
    category_path = selected_category["path"]
    faq_file = selected_category["faq_file"]

    print(f"\n[2] Selected category: {category_name}")
    print(f"    Loading questions from: {os.path.basename(faq_file)}")

    # Load questions
    TEST_QUESTIONS = load_questions_from_faq(faq_file)
    print(f"    Loaded {len(TEST_QUESTIONS)} questions")

    # Setup output directories
    results_dir = os.path.join(category_path, "results")
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    print(f"\n[3] Starting test with {len(TEST_QUESTIONS)} questions...")
    print("    (Fresh browser session for each question)")

    try:
        # Collect responses from chatbot
        print("\n[4] Sending test questions to chatbot via browser...")
        qa_pairs = []
        all_responses = []  # Store full response objects for CSV

        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n    [{i}/{len(TEST_QUESTIONS)}] Asking: {question[:70]}{'...' if len(question) > 70 else ''}")

            # Fresh chatbot session for each question
            chatbot = FlynasChatbotBrowser(CHATBOT_URL)
            await chatbot.initialize()

            response = await chatbot.send_message(question)
            await chatbot.close()

            if response.success:
                print(f"         Response: {response.answer[:100]}...")
                qa_pairs.append((question, response.answer))
                all_responses.append(response)
            else:
                print(f"         [FAILED] {response.error}")
                all_responses.append(response)

        if not qa_pairs:
            print("\n[ERROR] Could not get any responses from the chatbot.")
            print("The chatbot UI may have changed or requires different interaction.")
            return

        # RAGAS Evaluation
        print(f"\n[5] Running RAGAS Answer Relevancy evaluation on {len(qa_pairs)} responses...")
        evaluator = RAGASEvaluator(OPENAI_API_KEY)

        results = await evaluator.evaluate_batch(qa_pairs)

        print("\n" + "=" * 60)
        print("RAGAS ANSWER RELEVANCY RESULTS")
        print("=" * 60)

        total_score = 0
        successful_evals = 0

        for i, result in enumerate(results, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Question: {result['question'][:80]}{'...' if len(result['question']) > 80 else ''}")
            print(f"Answer: {result['answer'][:200]}...")

            if result['success']:
                score = result['relevancy_score']
                print(f"Relevancy Score: {score:.4f}")

                # Show the generated questions from answer
                gen_qs = result.get('generated_questions', [])
                if gen_qs:
                    print(f"Generated Questions from Answer:")
                    for j, gq in enumerate(gen_qs, 1):
                        print(f"  {j}. {gq}")

                if score >= 0.8:
                    print("Rating: HIGH (Good answer relevancy) - PASS")
                elif score >= 0.5:
                    print("Rating: MEDIUM (Moderate answer relevancy) - PASS")
                else:
                    print("Rating: LOW (Poor answer relevancy) - FAIL")

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
            print(f"Average Relevancy Score: {avg_score:.4f}")

            if avg_score >= 0.8:
                print("Overall Rating: HIGH - Chatbot provides highly relevant answers")
            elif avg_score >= 0.5:
                print("Overall Rating: MEDIUM - Chatbot provides moderately relevant answers")
            else:
                print("Overall Rating: LOW - Chatbot needs improvement in answer relevancy")

        # Find the latest serial number from existing files
        existing_files = glob.glob(os.path.join(results_dir, "answer_relevancy_results_*.xlsx"))
        max_serial = 0
        for f in existing_files:
            match = re.search(r'answer_relevancy_results_(\d{1,4})\.xlsx$', f)
            if match:
                num = int(match.group(1))
                if num > max_serial:
                    max_serial = num

        next_serial = max_serial + 1
        xlsx_filename = os.path.join(results_dir, f"answer_relevancy_results_{next_serial}.xlsx")

        print(f"\n[6] Saving results to {xlsx_filename}...")

        PASS_THRESHOLD = 0.5  # Score >= 0.5 is PASS

        # Build data for DataFrame
        rows = []
        result_idx = 0
        for i, response in enumerate(all_responses):
            score = None
            rating = "N/A"
            pass_fail = "N/A"
            generated_questions = []

            if response.success and result_idx < len(results):
                result = results[result_idx]
                result_idx += 1
                if result.get('success'):
                    score = result.get('relevancy_score')
                    generated_questions = result.get('generated_questions', [])
                    if score is not None:
                        if score >= 0.8:
                            rating = "HIGH"
                        elif score >= 0.5:
                            rating = "MEDIUM"
                        else:
                            rating = "LOW"
                        pass_fail = "PASS" if score >= PASS_THRESHOLD else "FAIL"

            rows.append({
                'question': response.question,
                'answer': response.answer,
                'generated_questions': ' | '.join(generated_questions) if generated_questions else '',
                'relevancy_score': round(score, 4) if score is not None else None,
                'rating': rating,
                'pass_fail': pass_fail,
                'evaluation_status': "OK" if response.success else "ERROR",
                'error': response.error or ''
            })

        # Save to Excel
        df = pd.DataFrame(rows)
        df.to_excel(xlsx_filename, index=False, engine='openpyxl')

        print(f"    Results saved to {xlsx_filename}")

        # Update summary file
        summary_file = os.path.join(summary_dir, "answer_relevancy_summary.xlsx")
        print(f"\n[7] Updating summary file {summary_file}...")

        # Calculate summary statistics
        passed_count = sum(1 for r in rows if r['pass_fail'] == 'PASS')
        failed_count = sum(1 for r in rows if r['pass_fail'] == 'FAIL')
        na_count = sum(1 for r in rows if r['pass_fail'] == 'N/A')
        total_tests = len(rows)
        pass_rate = f"{(passed_count / total_tests * 100):.1f}%" if total_tests > 0 else "0%"

        # Get question letters for failed and N/A questions (A=Q1, B=Q2, etc.)
        failed_letters = []
        na_letters = []
        for i, r in enumerate(rows):
            letter = chr(65 + i) if i < 26 else f"Q{i+1}"  # A-Z, then Q27, Q28, etc.
            if r['pass_fail'] == 'FAIL':
                failed_letters.append(letter)
            elif r['pass_fail'] == 'N/A':
                na_letters.append(letter)

        failed_questions_str = ', '.join(failed_letters) if failed_letters else '-'
        na_questions_str = ', '.join(na_letters) if na_letters else '-'

        # Create new run row
        new_run = {
            'run': next_serial,
            'total_tests': total_tests,
            'passed': passed_count,
            'failed': failed_count,
            'na_not_evaluated': na_count,
            'pass_rate': pass_rate,
            'avg_relevancy_score': round(avg_score, 4) if successful_evals > 0 else None,
            'failed_questions': failed_questions_str,
            'na_questions': na_questions_str
        }

        # Load existing summary or create new one
        try:
            summary_df = pd.read_excel(summary_file, sheet_name='Summary')
            # Remove the TOTALS row if it exists
            summary_df = summary_df[summary_df['run'] != 'TOTALS']
        except FileNotFoundError:
            summary_df = pd.DataFrame(columns=[
                'run', 'total_tests', 'passed', 'failed', 'na_not_evaluated',
                'pass_rate', 'avg_relevancy_score', 'failed_questions', 'na_questions'
            ])

        # Append new run
        summary_df = pd.concat([summary_df, pd.DataFrame([new_run])], ignore_index=True)

        # Calculate totals
        num_runs = len(summary_df)
        total_na = int(summary_df['na_not_evaluated'].sum())
        avg_pass_rate = summary_df['passed'].sum() / summary_df['total_tests'].sum() * 100 if summary_df['total_tests'].sum() > 0 else 0
        overall_avg_score = summary_df['avg_relevancy_score'].mean()

        totals_row = {
            'run': 'TOTALS',
            'total_tests': f"{num_runs} runs",
            'passed': None,
            'failed': None,
            'na_not_evaluated': total_na,
            'pass_rate': f"{avg_pass_rate:.1f}%",
            'avg_relevancy_score': round(overall_avg_score, 4) if not pd.isna(overall_avg_score) else None,
            'failed_questions': None,
            'na_questions': None
        }

        # Append totals row
        summary_df = pd.concat([summary_df, pd.DataFrame([totals_row])], ignore_index=True)

        # Create question legend sheet (letter -> question mapping)
        question_legend = []
        for i, q in enumerate(TEST_QUESTIONS):
            letter = chr(65 + i) if i < 26 else f"Q{i+1}"
            question_legend.append({
                'letter': letter,
                'question': q
            })
        legend_df = pd.DataFrame(question_legend)

        # Calculate failure percentage per question across all runs
        runs_df = summary_df[summary_df['run'] != 'TOTALS'].copy()
        num_runs_for_stats = len(runs_df)

        question_failure_stats = []
        for i, q in enumerate(TEST_QUESTIONS):
            letter = chr(65 + i) if i < 26 else f"Q{i+1}"
            # Count how many times this letter appears in failed_questions
            fail_count = 0
            for failed_str in runs_df['failed_questions'].fillna(''):
                if letter in str(failed_str).replace(' ', '').split(','):
                    fail_count += 1

            # Count N/A occurrences
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
