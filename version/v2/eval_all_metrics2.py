{
    "version" : "rag version 2",
    "goal" : ("V2 is for checking k = 2 (when performing `similarity_search_operation` , number of retrieved paragraph is 2)")
}

import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "../../")))
from groq_evaluator import GroqEvaluator
model = GroqEvaluator()

BASE_DIR = os.path.dirname(__file__)

from deepeval.metrics import AnswerRelevancyMetric , FaithfulnessMetric , ContextualRelevancyMetric , GEval
from deepeval.evaluate import CacheConfig
from deepeval.test_case import LLMTestCase , SingleTurnParams
from deepeval import evaluate


with open(os.path.join(BASE_DIR, "results2.jsonl") , "r" , encoding="utf-8") as file:
    results = [ json.loads(line) for line in file]
    

questions_tests = []

for i , result in enumerate(results , start=1):
    print(f"\nEvaluating test {i}/{len(results)}")
    
    context = result["context"] or ""
    
    if isinstance(context , str):
        retrieval_context = [context]
    elif isinstance(context , list) :
        retrieval_context = context
    else:
        retrieval_context = []
    
    test_case = LLMTestCase(
        input=result["question"],
        expected_output=result["expected_answer"],
        actual_output=result["actual_answer"],
        retrieval_context=retrieval_context
    )
    
    questions_tests.append(test_case)

#metrics
answer_relevancy = AnswerRelevancyMetric(threshold=0.7 , model=model)

faithfulness = FaithfulnessMetric(threshold=0.7 , model=model)

context_relevancy = ContextualRelevancyMetric(threshold=0.7 , model=model)

correctness = GEval(name="correctness" , 
                    model=model ,
                    criteria=(
                        "Evaluate whether the actual output is factually correct "
                        "and semantically consistent with the expected output. "
                        "Accept equivalent wording and equivalent meanings. "
                        "Penalize factual errors, contradictions, and missing "
                        "essential information."
                    ),
                    evaluation_params=[
                        SingleTurnParams.ACTUAL_OUTPUT,
                        SingleTurnParams.EXPECTED_OUTPUT
                    ], 
                    threshold=0.7)

evaluation_results = evaluate(
    test_cases=questions_tests,
    metrics=[
        answer_relevancy,
        faithfulness,
        context_relevancy,
        correctness
    ],
    cache_config=CacheConfig(
        use_cache=False,
        write_cache=False
    )
)


final_results = []

for i , (original_result , test_result) in enumerate( 
        zip(results , evaluation_results.test_results) , start=1):
    
    print(f"\nCollecting scores for test {i}/{len(results)}")
    
    answer_relevancy_score = None
    faithfulness_score = None
    context_relevancy_score = None
    correctness_score = None
    
    for metric in test_result.metrics_data or [] :
        
        metric_name = metric.name.strip().lower()
        
        if metric_name == "answer relevancy":
            answer_relevancy_score = metric.score
        elif metric_name == "faithfulness":
            faithfulness_score = metric.score
        elif metric_name == "contextual relevancy":
            context_relevancy_score = metric.score
        elif metric_name.startswith("correctness"):
            correctness_score = metric.score
    
    eval_result = {
        **original_result,
        "answer_relevancy" : answer_relevancy_score,
        "faithfulness" : faithfulness_score,
        "context_relevancy" : context_relevancy_score,
        "correctness" : correctness_score,
    }
    final_results.append(eval_result)
    
    

with open("version/v2/final_results2.jsonl" , "w" , encoding="utf-8") as file :
    for result in final_results :
        file.write(json.dumps(result,ensure_ascii=False) + "\n" )

print("\nDone! Results saved to final_results2.jsonl")