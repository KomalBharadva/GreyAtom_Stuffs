# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Path of the file
data=pd.read_csv(path)
data=data.rename(columns={"Total":"Total_Medals"})
print(data.head(10))
#Code starts here


# --------------
#Code starts here




data['Better_Event']=np.where(data['Total_Summer']>data['Total_Winter'],'Summer',(np.where(data['Total_Summer']<data['Total_Winter'], 'Winter','Both')))
sum_med=data['Better_Event'].value_counts()
print(sum_med)
if sum_med[0]>sum_med[1]:
    better_event='Summer'
elif sum_med[0]<sum_med[1]:
    better_event='Winter'
else:
    better_event='both'
print(better_event)


# --------------
#Code starts here
top_countries=data[['Country_Name','Total_Summer', 'Total_Winter','Total_Medals']]
# drop last n rows
top_countries.drop(top_countries.tail(1).index,inplace=True) 

def top_ten(df,col_name):
    country_list=[]
    top=df.nlargest(10,col_name)
    country_list=list(top['Country_Name'])
    return country_list

top_10_summer=top_ten(top_countries,'Total_Summer')
top_10_winter=top_ten(top_countries,'Total_Winter')
top_10=top_ten(top_countries,'Total_Medals')

common=list(set(top_10_summer) & set(top_10_winter) & set(top_10))
print(common)




# --------------
#Code starts here
summer_df=data[data['Country_Name'].isin(top_10_summer)]
winter_df=data[data['Country_Name'].isin(top_10_winter)]
top_df=data[data['Country_Name'].isin(top_10)]

fig, axs = plt.subplots(3, figsize=(20,10))
sum=summer_df.groupby(['Country_Name', 'Total_Summer']).size().unstack()
sum.plot(kind='bar', stacked=True, ax=axs[0])

win=winter_df.groupby(['Country_Name', 'Total_Winter']).size().unstack()
win.plot(kind='bar', stacked=True, ax=axs[1])

top=top_df.groupby(['Country_Name', 'Total_Medals']).size().unstack()
top.plot(kind='bar', stacked=True, ax=axs[2])

plt.show()





# --------------
#Code starts here
summer_df['Golden_Ratio'] =  data['Gold_Summer'] / data['Total_Summer']
max1=summer_df.loc[summer_df['Golden_Ratio'].idxmax()]
summer_max_ratio=max1['Golden_Ratio']
summer_country_gold=max1['Country_Name']
print(summer_max_ratio)
print(summer_country_gold)

winter_df['Golden_Ratio'] = data['Gold_Winter'] / data['Total_Winter'] 
max2=winter_df.loc[winter_df['Golden_Ratio'].idxmax()]
winter_max_ratio=max2['Golden_Ratio']
winter_country_gold=max2['Country_Name']
print(winter_max_ratio)
print(winter_country_gold)

top_df['Golden_Ratio'] = data['Gold_Total'] / data['Total_Medals'] 
max3=top_df.loc[top_df['Golden_Ratio'].idxmax()]
top_max_ratio=max3['Golden_Ratio']
top_country_gold=max3['Country_Name']
print(top_max_ratio)
print(top_country_gold)



# --------------
#Code starts here
data_1=data[:-1]
data_1['Total_Points']= data_1['Gold_Total']*3 + data_1['Silver_Total']*2 + data_1['Bronze_Total']*1  
most_points=max(data_1['Total_Points'])
best_country=data_1.loc[data_1['Total_Points'].idxmax(),'Country_Name']
print(most_points)
print(best_country)




# --------------
#Code starts here
best=data_1[data_1['Country_Name']=='United States']
best=best[['Gold_Total','Silver_Total','Bronze_Total']]
best.plot.bar()
plt.xlabel("United States")
plt.ylabel('Medals Tally')
plt.xticks(rotation=45)
plt.show()


