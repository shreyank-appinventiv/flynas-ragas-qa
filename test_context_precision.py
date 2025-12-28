import asyncio
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

async def test():
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    scorer = LLMContextPrecisionWithoutReference(llm=llm)

    sample = SingleTurnSample(
        user_input="What are the maximum dimensions allowed for checked baggage?",
        response="The maximum size allowed for checked baggage on flynas is 75cm in height, 50cm in width, and 33cm in depth. Any bag exceeding these dimensions will not be accepted, and no single checked bag should weigh more than 32kg. For special or oversized items, additional fees and conditions may apply.",
        retrieved_contexts=[
            "specific allowance before traveling. • Maximum dimensions for checked baggage are 75cm H, 50cm W, 33cm D. Any baggage exceeding these dimensions will not be accepted. • No single item of checked baggage should exceed 32 kg. • Codeshare and interline flights may have different baggage policies. #### Important Notes: •Fragile, valuable, or perishable items should not be packed in checked baggage. The airline is not responsible for damage to improperly packed items. •Dangerous goods, including flammable materials, explosives, and compressed gases, are strictly prohibited in checked baggage. For a full list of restricted items, please visit [flynas prohibited items](https://www.flynas.com/en/baggage/prohibited-items). •Acceptance of excess baggage is at the airline's discretion and subject to flynas' baggage allowance policy. #### EXCESS BAGGAGE - If you plan to travel",
            "2X23 kg Economy Class Economy Class Economy Class Premium Class - Group 1 :(Abu Dhabi, Almaty, Amman, Antalya, Baghdad, Bahrain, Baku, Batumi, Bishkek, Bodrum, Brussels, Casablanca, Damascus, Dhaka, Doha, Dubai, Erbil, Geneva, Hurghadah, Istanbul, Kraków, Kuwait, Marseille, Milan, Moscow, Namangan, Najaf, North Coast, Osh, Prague, Pristina, Rize, Salalah, Salzburg, Sarajevo, Sharm El Sheikh, Sharjah, Tirana, Tbilisi, Trabzon, Tashkent, Vienna) - Group 2 : (Algiers, Cairo, Constantine.) - Group 3 : (Addis Ababa, Asmara, Calicut, Hyderabad, Islamabad, Karachi, Lahore, Lucknow, Mumbai, New Delhi, Nairobi, Sohag) Infant passengers are entitled to take 10 kg of hold baggage. #### Checked Baggage • Baggage allowances vary depending on the fare type and route. Guests are encouraged to check their specific allowance before traveling. • Maximum dimensions for checked baggage are 75cm H, 50cm W, 33cm D. Any baggage exceeding these dimensions will not be accepted. • No single item of checked",
            "required for a particular flight may change at any time until the booking is confirmed, and the applicable Miles are applied as payment for the redeemed flight. ## 1. How much baggage can I take into the cabin? Each guest may carry 1 bag of max 7kg, maximum dimensions of 56(H)X36(W)X23(D)cm for no additional charge. If any cabin baggage is heavier or larger than the permitted weight/dimensions it must be checked in as hold baggage and relevant excess baggage charges must be paid.. ## 2. How much baggage can I check-in? Your hold baggage allowance will depend on the fare you have booked.. ## 3. Can I check-in larger baggage, such as sports equipment? Non-standard/oversized items of baggage such as Golf Clubs and Skis may be accepted for carriage as Hold Baggage. If accepted, any item over our 158CM (H+W+D) will be subject to a handling fee of SAR100 per piece. We retain the right to refuse to carry as baggage any item due to its size, shape, weight, nature or character. We retain"
        ]
    )

    score = await scorer.single_turn_ascore(sample)
    print(f"Context Precision Score: {score}")

if __name__ == "__main__":
    asyncio.run(test())
