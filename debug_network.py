"""
Debug script to capture network traffic from Flynas chatbot
to identify the API calls and retrieved context structure.
"""

import asyncio
import json
from playwright.async_api import async_playwright


async def main():
    CHATBOT_URL = "https://flynaschb-frontendapp-qa.blackflower-c82fc685.westeurope.azurecontainerapps.io"
    TEST_QUESTION = "Can i carry my dog onboard?"

    print("=" * 60)
    print("Network Debug - Capturing API calls")
    print("=" * 60)

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(viewport={"width": 1280, "height": 800})
    page = await context.new_page()

    # Store captured requests/responses
    captured_data = []

    # Listen to all network responses
    async def handle_response(response):
        url = response.url
        # Filter for API calls (skip static assets)
        if any(ext in url for ext in ['.js', '.css', '.png', '.jpg', '.svg', '.ico', '.woff']):
            return

        try:
            content_type = response.headers.get('content-type', '')
            if 'json' in content_type or 'text' in content_type:
                try:
                    body = await response.text()
                    # Try to parse as JSON
                    try:
                        json_body = json.loads(body)
                        captured_data.append({
                            'url': url,
                            'status': response.status,
                            'method': response.request.method,
                            'body': json_body
                        })
                        print(f"\n[API] {response.request.method} {url[:80]}")
                        print(f"      Status: {response.status}")
                        # Pretty print JSON (truncated)
                        json_str = json.dumps(json_body, indent=2)
                        if len(json_str) > 500:
                            print(f"      Body: {json_str[:500]}...")
                        else:
                            print(f"      Body: {json_str}")
                    except json.JSONDecodeError:
                        if len(body) < 200:
                            print(f"\n[TEXT] {url[:60]} -> {body[:100]}")
                except:
                    pass
        except Exception as e:
            pass

    page.on("response", handle_response)

    print(f"\n[1] Navigating to {CHATBOT_URL}...")
    await page.goto(CHATBOT_URL, wait_until="networkidle", timeout=60000)
    await asyncio.sleep(3)

    print("\n[2] Opening chatbot...")
    floating_btn = await page.query_selector(".floatingButton")
    if floating_btn:
        await floating_btn.click()
        await asyncio.sleep(3)

    print(f"\n[3] Sending question: {TEST_QUESTION}")
    input_element = await page.query_selector("div.editable-input")
    if not input_element:
        input_element = await page.query_selector("div[contenteditable='true']")

    if input_element:
        await input_element.click()
        await input_element.fill("")
        await page.keyboard.type(TEST_QUESTION, delay=50)

        send_btn = await page.query_selector(".send-mic-btn")
        if send_btn:
            await send_btn.click()
        else:
            await page.keyboard.press("Enter")

        print("\n[4] Waiting for response (15 seconds)...")
        await asyncio.sleep(15)

    await browser.close()
    await playwright.stop()

    # Print summary of captured API calls
    print("\n" + "=" * 60)
    print("CAPTURED API CALLS SUMMARY")
    print("=" * 60)

    # Extract retrieved contexts from the chat API response
    retrieved_contexts = []
    for data in captured_data:
        if '/chat' in data['url'] and data['method'] == 'POST':
            body = data['body']
            debug_data = body.get('debug_data')
            if debug_data and 'steps' in debug_data:
                for step in debug_data['steps']:
                    step_data = step.get('data', {})
                    faq_trace = step_data.get('faq_trace', {})
                    search_result = faq_trace.get('search_result', {})
                    references = search_result.get('references', [])
                    if references:
                        for ref in references:
                            retrieved_contexts.append({
                                'content': ref.get('content', ''),
                                'source_url': ref.get('source_url', ''),
                                'category': ref.get('category', '')
                            })

    print(f"\n[RETRIEVED CONTEXTS] Found {len(retrieved_contexts)} context chunks:")
    print("-" * 60)
    for i, ctx in enumerate(retrieved_contexts, 1):
        print(f"\n--- Context {i} ---")
        print(f"Category: {ctx['category']}")
        print(f"Source: {ctx['source_url']}")
        print(f"Content (first 300 chars):\n{ctx['content'][:300]}...")

    # Save full captured data to file
    with open('/home/shreyank06/Desktop/projects/testing_bulls_pvt_ltd/dog_onoard_network_debug.json', 'w') as f:
        json.dump(captured_data, f, indent=2, default=str)

    print(f"\n[5] Full data saved to network_debug.json")


if __name__ == "__main__":
    asyncio.run(main())
