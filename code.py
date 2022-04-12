import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv('[name file]') #input your name file

cols = ['ticket_id','menu_name','authorized'] #input the name column
data = data[cols]
data

data = data.dropna()
data.info() #describe your data

data = data[data['void'] == 0]
data.info()

basket = data.groupby(['ticket_id','menu_name'])['authorized'].sum().unstack().reset_index().fillna(0).set_index('ticket_id') #pivot your data
basket

def encode(x):
  if x <= 0:
    return 0
  if x>= 1:
    return 1

basket_encode = basket.applymap(encode)
basket_encode

frequent_itemsets = apriori(basket_encode, min_support=0.05, use_colnames=True).sort_values('support', ascending = False).reset_index(drop=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1).sort_values('lift', ascending = False).reset_index(drop=True)
rules
