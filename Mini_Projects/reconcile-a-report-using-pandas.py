# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Code starts here
df=pd.read_csv(path)
df['state']=df['state'].str.lower()
df['total']=df['Jan']+df['Feb']+df['Mar']
sum_row=df[["Jan", "Feb", "Mar", "total"]].sum()
print(sum_row)
df_final=df.append(sum_row, ignore_index=True)
print(df_final)
# Code ends here


# --------------
import requests
# Code starts here
url='https://en.wikipedia.org/wiki/List_of_U.S._state_abbreviations'
response=requests.get(url)
df1=pd.read_html(response.content)[0]
df1=df1.drop(df1.index[:11])
# df1=df1.reset_index()
new_header = df1.iloc[0] 
df1 = df1[1:] 
df1.columns = new_header 
# Lets see the 5 first rows of the new dataset
df1['United States of America'] = df1['United States of America'].str.replace(' ', '')
print(df1.head())
# Code ends here


# --------------
df1['United States of America'] = df1['United States of America'].astype(str).apply(lambda x: x.lower())
df1['US'] = df1['US'].astype(str)
# Code starts here
mapping={"Country":"Abbreviation"}
df_final['abbr']=map(df_final['state'],mapping)
print(df_final.head())
# Code ends here


# --------------
# Code stars here
df_final.set_value(6,"abbr","MS")
df_final.set_value(10,"abbr","TN")
print(df_final.head(15))
# Code ends here


# --------------
# Code starts here
# print(df_final.head(15))
df_sub=df_final.groupby("abbr")["Jan","Feb","Mar","total"].sum()
print(df_sub.head())
formatted_df=df_sub.applymap(lambda x: "${:,.0f}".format(x))
print(formatted_df)
# Code ends here



# --------------
# Code starts here
sum_row=df_final[["Jan","Feb","Mar","total"]].sum()
df_sub_sum=sum_row.transpose()
# df_sub_sum=df_sub_sum.applymap(lambda x: "${:,.0f}".format(x))
# df_sub_sum=df_sub_sum.map(( lambda x: 'a' + x))
df_sub_sum = '$' + df_sub_sum.astype(str)
print(df_sub_sum)
final_table=formatted_df.append(df_sub_sum,ignore_index=True)
print(final_table)
# final_table=final_table.rename(index={'0': 'Total'})
# print(final_table)
# Code ends here


# --------------
# Code starts here
a=final_table.iloc[-1].str[1:]
# b=df_sub['total'].astype(float).sum()
# print(b)
df_sub['total']=a
plt.pie(a)
plt.show()
# df_sub_sum = '$' + df_sub_sum.astype(str)
# d['Report Number'].str[1:]  
# print(final_table.iloc[-1].sum())
# df=df.sum()
# print(df)
# print(df_sub)‬
# plt.pie(df_sub['total'])
# plt.show()
# Code ends here


