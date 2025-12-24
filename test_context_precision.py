import asyncio
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

async def test():
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    scorer = LLMContextPrecisionWithoutReference(llm=llm)

    sample = SingleTurnSample(
        user_input="What is baggage allowance?",
        response="Economy allows 23kg checked baggage.",
        retrieved_contexts=[
            "Economy class allows 23kg checked baggage.",  # relevant
            "Flynas serves meals on flights.",              # irrelevant
        ]
    )

    score = await scorer.single_turn_ascore(sample)
    print(f"Context Precision Score: {score}")

if __name__ == "__main__":
    asyncio.run(test())
