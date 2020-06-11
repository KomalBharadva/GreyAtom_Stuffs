# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data=np.genfromtxt(path, delimiter=",", skip_header=1)
print("\nData: \n\n", data)
#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
new=np.array(new_record)
census=np.concatenate([data,new])
#Code starts here



# --------------
#Code starts here
age=np.array([census[:,0]],dtype=np.intp)
print(age)
max_age=np.amax(age)
min_age=np.amin(age)
age_mean=np.mean(age)
age_std=np.std(age)
print(max_age)
print(min_age)
print(age_mean)
print(age_std)




# --------------
race_0=census[census[:,2]==0]
race_1=census[census[:,2]==1]
race_2=census[census[:,2]==2]
race_3=census[census[:,2]==3]
race_4=census[census[:,2]==4]
race_00=np.where(census[:,2]==0)
race_01=np.where(census[:,2]==1)
race_02=np.where(census[:,2]==2)
race_03=np.where(census[:,2]==3)
race_04=np.where(census[:,2]==4)
race_000=np.asarray(race_00)
race_001=np.asarray(race_01)
race_002=np.asarray(race_02)
race_003=np.asarray(race_03)
race_004=np.asarray(race_04)
len_0=race_000.size
len_1=race_001.size
len_2=race_002.size
len_3=race_003.size
len_4=race_004.size
print(len_0)
#len_1=race_01.size
#len_2=race_02.size
#len_3=race_03.size
#len_4=race_04.size
minimum=min(len_0,len_1,len_2,len_3,len_4)
print(minimum)
minority_race=3
print(minority_race)



# --------------
#Code starts here
senior_citizens1=census[census[:,0]>60]
senior_citizens=senior_citizens1.astype('int32')
working_hours=senior_citizens1[:,6]
print(working_hours)
working_hours_sum=sum(working_hours)
print(working_hours_sum)
senior_citizens_len=len(senior_citizens)
print(senior_citizens_len)
avg_working_hours=working_hours_sum/senior_citizens_len
print(avg_working_hours)



# --------------
#Code starts here
high=census[census[:,1]>10]
low=census[census[:,1]<=10]
print(high)
print(low)
avg_pay_high=np.mean(high[:,7])
avg_pay_low=np.mean(low[:,7])
print(avg_pay_high)
print(avg_pay_low)






