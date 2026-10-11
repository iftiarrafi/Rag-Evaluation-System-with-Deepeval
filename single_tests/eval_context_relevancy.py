import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from groq_evaluator import GroqEvaluator
model = GroqEvaluator()

test_case = LLMTestCase(
    input= "Where do penguins live?",
    actual_output= "Penguins live in the Southern Hemisphere, ranging from the icy coasts of Antarctica to the cool shores of South America and southern Africa, and even the Galapagos Islands near the equator.",
    retrieval_context=["Penguins are flightless seabirds that are specially adapted for life in the Southern Hemisphere. \nAlthough people often associate them only with Antarctica, penguins live in a wide range of \nenvironments, including the icy coasts of Antarctica, the cool shores of South America and \nsouthern Africa, and even the Galapagos Islands near the equator. Their upright posture, \ncompact bodies, and distinctive black-and-white coloring make them among the most \nrecognizable birds in the world.\n\nPenguin diets vary by species and location, but most feed on marine animals. Small species may \nrely heavily on krill and tiny fish, while larger penguins can pursue bigger fish and squid. They \nhunt by sight and can dive repeatedly during a feeding trip. Some penguins stay relatively close \nto shore, whereas others travel many kilometers in search of productive feeding areas."]
)

metric = ContextualRelevancyMetric(
    threshold=0.7,
    model=model
)

metric.measure(test_case=test_case)

print("Score: ", metric.score)
print("Reason: ", metric.reason)
print("Passed: ", metric.is_successful())