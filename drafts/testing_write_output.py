from data_helper import write_output
outline = ['1','2','3 some kind of long outline in list form']
write_output("company_symbol",outline,"ESG_approach_outline", list_type=True)
point = 'some kind of point here'
write_output("company_symbol",point,"ESG_approach",header=True) 
article = "some article here"
write_output("company_symbol",article,"ESG_approach")
