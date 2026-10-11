import json
from RAG1.agent import ask

with open("./eval_dataset.jsonl" , "r" , encoding="utf-8") as file:
    test_cases = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

    results = []
    
    for index , test_case in enumerate(test_cases , start=1) :
        
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        
        print(f"Processing question {index}/{len(test_cases)}")
        
        response = ask(question=question)
        
        result = {
            "question":question,
            "expected_answer":expected_answer,
            "actual_answer":response["answer"],
            "context":response["context"],
        }
        
        results.append(result)
    
    
with open("results1.jsonl", "w" , encoding="utf-8") as file:
    for result in results:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

print("\nDone! Results1 save to results1.jsonl")    
