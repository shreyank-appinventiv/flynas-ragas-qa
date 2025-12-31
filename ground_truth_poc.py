import os
from datetime import datetime

import pytest
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy, FactualCorrectness, Faithfulness, ContextPrecision, ContextRecall

from utils import get_llm_response, load_test_data


@pytest.mark.parametrize("get_data", load_test_data("TestFile4_OpenRouterWithFramework.json"), indirect=True)
@pytest.mark.asyncio
async def test_relevancy_factual(llm_wrapper_openai,  get_data): # Here we have used emp_wrapper extra function because our openrouter llm model does not support embedding. openAI model supports embedding directly
    metrics = [ResponseRelevancy(llm=llm_wrapper_openai),
                FactualCorrectness(llm=llm_wrapper_openai),
                Faithfulness(llm=llm_wrapper_openai),
                ContextPrecision(llm=llm_wrapper_openai),
                ContextRecall(llm=llm_wrapper_openai)
               ]

    eval_dataset = EvaluationDataset([get_data])   #It will be used when we have multiple metrics in single test
                                                    #It takes the single turn object and convert it into ragas dataset. It is taking list as input because we may have multiple test data in one files
    results  = evaluate(dataset = eval_dataset, metrics = metrics) #Used to evaluate the metrics result, It takes two inputs, one is data and which metric u want to evaluate
    # results = evaluate(dataset=eval_dataset) #It is used to call default metrics rather than explicitly give the metrics for evaluation but it will work only with OpenAI key
    df = results.to_pandas()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"TestFile4_Output_{timestamp}.csv"
    output_path = os.path.join(os.getcwd(), file_name)
    df.to_csv(output_path, index=False)


    print(results)
    print(results["answer_relevancy"])
    print(results["factual_correctness"])
    #results.upload()   #It is used for uploading the result on app.ragas.io


@pytest.fixture
def get_data(request):
    test_data = request.param
    response_dict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=response_dict["answer"],
        retrieved_contexts=[
            doc["page_content"] for doc in response_dict.get("retrieved_docs", [])],
        reference=test_data["reference"]

    )
    return sample
