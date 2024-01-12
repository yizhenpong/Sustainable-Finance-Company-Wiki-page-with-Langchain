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
import data_helper as dh
from readability import Readability
import language_tool_python
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

def get_eval(company_symbol, section = ["ESG_approach","ESG_overview"],ToCStatus=False):
    text = dh.get_output(company_symbol, section,"txt", ToC=ToCStatus)
    num_words = len(text)
    r = Readability(text)
    # ARI
    ari = r.ari()
    # print(ari.score)
    # print(ari.grade_levels)
    # print(ari.ages)
    # language tool to check grammar and spelling mistakes
    tool = language_tool_python.LanguageTool('en-US') 
    matches = tool.check(text)
    # print(len(matches)) # number of lang errors
    # Extract 'category' for each data point
    categories = [match.category for match in matches]
    # Count the occurrences of each category
    category_counts = Counter(categories)
    # errors
    df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
    # print(df)
    overview_eval = {"num_words":num_words, "ari_score": ari.score, "ari_grade_levels": ari.grade_levels,
            "ari_ages": ari.ages, "num_lang_errors":len(matches)}
    return overview_eval, df

get_eval("JNJ", "ESG_approach",ToCStatus=True)


def eval_company(company_symbol):
    for section in ["ESG_approach","ESG_overview"]:
        dh.write_output(company_symbol,text_output=f"----- section: {section}-----", eval=True)
        # print(f"----- section: {section}-----")
        x = get_eval("JNJ", section,ToCStatus=True)
        dh.write_output(company_symbol,text_output=x[0], eval=True)
        # print(x[0])
        if x[0]['num_lang_errors'] != 0:
            dh.write_output(company_symbol,text_output=x[1], eval=True)
            # print(x[1])
            # x[1].to_json(orient='table')
            # print(x[1])


if __name__ == '__main__':
    print("============= evaluation for all companies started!! =============")
    for symbol in dh.get_all_companies()['Symbol']:
        dh.write_output(symbol,text_output=f"============= evaluation for {symbol} =============", eval=True)
        # print(f"============= evaluation for {symbol} =============")
        eval_company(symbol)

        dh.write_output(symbol,text_output=f"\n \n \n", eval=True)

    print("============= evaluation for all companies completed!! =============")


##############################################################################################################################
# company_symbol = "JNJ"
# text = get_output(company_symbol, "ESG_approach","txt", True)

# #readibility indices
# r = Readability(text)
# # ARI
# ari = r.ari()
# print(ari.score)
# print(ari.grade_levels)
# print(ari.ages)

# # language tool to check grammar and spelling mistakes
# tool = language_tool_python.LanguageTool('en-US') 
# matches = tool.check(text)
# print(len(matches))
# # print(matches)

# # Extract 'category' for each data point
# categories = [match.category for match in matches]

# # Count the occurrences of each category
# category_counts = Counter(categories)

# # Create a bar plot
# categories, counts = zip(*category_counts.items())
# plt.bar(categories, counts)
# plt.xlabel('Category')
# plt.ylabel('Count')
# plt.title('Distribution of Categories')
# plt.show()
# m0 = matches[0]
# print(m0)
# print(m0.category)

'''
output: for `JNJ ESG Approach` : realise that the grammar mistake is actually just the last name. 
Not significant. 

Offset 131, length 4, Rule ID: MORFOLOGIK_RULE_EN_US
Message: Possible spelling mistake found.
Suggestion: Wait; Hit; Hair; Hat; Habit; Halt; Hart; Hail; Haiti; Bait; Gait; Heist; AIT; Haft; Hast; HAI; HIT; Hie; Hied; Hies; Ha it; HIIT
... under the leadership of Dr. William N. Hait, Executive Vice President, Chief Extern...
                                           ^^^^
TYPOS
'''


##############################################################################################################################

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
