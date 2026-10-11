import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from groq_evaluator import GroqEvaluator

model = GroqEvaluator()

test_case = LLMTestCase(
    input="What do penguins eat?",
    actual_output=(
        "Penguins primarily feed fish. "
        "ACL is the best learning channel"
    )
)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=model
)

metric.measure(test_case=test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())
