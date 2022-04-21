from profanity_filter import ProfanityFilter
import spacy  #needed for language profanity filtering


# def post_profanity_removal(df):
#     #import spacy
#     #nlp = spacy.load('en')
#     #pf = ProfanityFilter(nlps={'en': nlp}) # set the filter
#     pf = ProfanityFilter() # set the filter
    
#     def filter_profanity_func(text):
#         return pf.censor(text)
    
#     df = df.copy()

#     df2 = df.copy()[['post_id','post_text']].drop_duplicates()
#     df2['new_col'] = df2['post_text'].apply(filter_profanity_func) 

#     output_df = df.copy()
#     output_df = output_df.merge(df2, how='left',on=['post_id','post_text']).rename({'post_text':'delete_col'},axis=1).drop(['delete_col'],axis=1).rename({'new_col':'post_text'},axis=1) #merge and rearrange
#     cols = list(df.columns)
#     output_df = output_df.copy()[cols] #puts columns back in order
#     return output_df

t = "fuck the world, butthole, and then take a shit"
#nlp = spacy.load("en_core_web_sm")
# import en_core_web_sm
# nlp = en_core_web_sm.load()
nlp = spacy.load('en_core_web_sm')
print(type(nlp))
print(nlp)
#spacy.load('en_core_web_sm')
pf = ProfanityFilter(nlps={'en': nlp}) # set the filter
nlp.add_pipe(pf.spacy_component, last=True)
#print(nlp(t))
#print(type(nlp(t)))

print(pf.censor(t))


# def filter_profanity_func(text):
#     return pf.censor(text)

# print(
# filter_profanity_func(nlp(t))
# )

# import pandas as pd
# df = pd.DataFrame({'text': ['fuck','shit','ass','unicorn']})
# print(df)

