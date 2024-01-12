import json

# Load the JSON file
with open('eval/section_A_eval.json', 'r') as file:
    data = json.load(file)

# Iterate through each key
for company, value in data.items():
    score = 0
    fields_wrong = []
    for field, field_eval in value.items(): 
        score += field_eval["Eval"]
        if field_eval["Eval"] == 0:
            fields_wrong.append(field)
    print(company, score, fields_wrong)
