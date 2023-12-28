# https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain
from langchain.evaluation import load_evaluator
from langchain.evaluation import EvaluatorType
from langchain.evaluation import Criteria
# list(Criteria)
evaluator = load_evaluator(EvaluatorType.CRITERIA, criteria="helpfulness")
eval_result = evaluator.evaluate_strings(
    prediction="What's 2+2? That's an elementary question. The answer you're looking for is that two and two is four.",
    input="What's 2+2?",
)
print(eval_result)