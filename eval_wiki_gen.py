# https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain

'''
main automated metrics:
- (1) readability scores 
        sh 
        ```pip install py-readability-metrics
        python -m nltk.downloader punkt```
- (2) custom criterion
- (3) search API
'''
from data_helper import get_output
from readability import Readability
import language_tool_python
import matplotlib.pyplot as plt
from collections import Counter


company_symbol = "JNJ"
text = get_output(company_symbol, "ESG_approach","txt", True)
r = Readability(text)

# ARI
ari = r.ari()
print(ari.score)
print(ari.grade_levels)
print(ari.ages)


# language tool to check grammar and spelling mistakes
tool = language_tool_python.LanguageTool('en-US') 
matches = tool.check(text)
print(len(matches))
# print(matches)

# Extract 'category' for each data point
categories = [match['category'] for match in matches]

# Count the occurrences of each category
category_counts = Counter(categories)

# Create a bar plot
categories, counts = zip(*category_counts.items())
plt.bar(categories, counts)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Distribution of Categories')
plt.show()


# from langchain.evaluation import load_evaluator
# from langchain.evaluation import EvaluatorType
# from langchain.evaluation import Criteria
# # list(Criteria)
# evaluator = load_evaluator(EvaluatorType.CRITERIA, criteria="helpfulness")
# eval_result = evaluator.evaluate_strings(
#     prediction="What's 2+2? That's an elementary question. The answer you're looking for is that two and two is four.",
#     input="What's 2+2?",
# )
# print(eval_result)















#====================
'''
diff readability metrics
but i feel like its more for a non technical content understanding
'''
# # Using Flesch Reading Ease - gd is 60+ level that can be understood by grade 8-9 US school kids
# f = r.flesch()
# print(f.score)
# print(f.ease)
# print(f.grade_levels)

# # Flesch-Kincaid Grade Level - text shld not be higher than grade 9
# fk = r.flesch_kincaid()
# print(fk.score)
# print(fk.grade_level)

# # esp for technical documents
# cl = r.coleman_liau()
# print(cl.score)
# print(cl.grade_level)

# # Linsear Write is a readability metric for English text, purportedly developed for the United States Air Force to help them calculate the readability of their technical manuals.
# lw = r.linsear_write()
# print(lw.score)
# print(lw.grade_level)
