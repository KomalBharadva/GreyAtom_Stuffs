# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# code starts here
df=pd.read_csv(path)
total=len(df)
p_a=((df['fico']>700).sum())/total
print(p_a)
p_b=((df['purpose']=='debt_consolidation').sum())/total
print(p_b)
df1=df[df['purpose']=='debt_consolidation']
p_a_b=((df1['fico']>700).sum())/total
print(p_a_b)
if (p_a_b == p_a):
    result=True
    print(result)
else:
    result=False
    print(result)
# code ends here


# --------------
# code starts here
total=len(df)
prob_lp=((df['paid.back.loan']== 'Yes').sum())/total
print(prob_lp)
prob_cs=((df['credit.policy']== 'Yes').sum())/total
print(prob_cs)
new_df=df[df['paid.back.loan']=='Yes']
# Calculate the P(A|B)
# p_a_b = df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]
prob_pd_cs=new_df[new_df['credit.policy']=='Yes'].shape[0]/new_df.shape[0]
# prob_pd_cs=((new_df['credit.policy']=='Yes').sum())/total
print(prob_pd_cs)
bayes=(prob_pd_cs*prob_lp)/prob_cs
print(bayes)
# code ends here


# --------------
# code starts here
plt.bar(df.purpose,df.index)
df1=df[df['paid.back.loan']== 'No']
print(df1.head())
# plt.plot(df1.purpose,df1.index)
plt.show()
# code ends here


# --------------
# code starts here
inst_median=df.installment.median()
inst_mean=df.installment.mean()
plt.hist(inst_median)
plt.hist(inst_mean)
plt.show()
# code ends here


