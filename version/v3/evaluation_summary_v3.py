# import os
# import sys
# import json
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "../../")))
import json
from statistics import mean

with open(r"version\\v3\\final_results3.jsonl" , mode="r" , encoding="utf-8") as file :
    results = [ json.loads(line) for line in file]
    

metrics = ["answer_relevancy", "faithfulness" , "context_relevancy" , "correctness"]

threshold = 0.7

for metric in metrics :
    
    scores = [result[metric] for result in results]
    
    avg_score = mean(scores)
    
    pass_count = sum( threshold<=score for score in scores)
    
    fail_count = len(scores) - pass_count
    
    passing_rate = (pass_count / len(scores)) * 100
    
    print(metric.replace("_", " ").title())
    
    print(f"Average Score: {avg_score:.2f}")
    print(f"Passed       : {pass_count}/{len(scores)}")
    print(f"Failed       : {fail_count}/{len(scores)}")
    print(f"Pass RAte    : {passing_rate:.1f}%")
    print()