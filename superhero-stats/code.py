# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data= pd.read_csv(path)
data.Gender.replace('-','Agender',inplace=True)
gender_count=data.Gender.value_counts()
print(gender_count)

gender_count.plot(kind='bar')

#Code starts here 





# --------------
#Code starts here
alignment = data.Alignment.value_counts()
labels=['Character Alignment']
plt.pie(alignment)


# --------------
#Code starts here
sc_df= pd.DataFrame(data,columns=['Strength','Combat'])
ic_df= pd.DataFrame(data,columns=['Intelligence','Combat'])

sc_covariance = sc_df.cov().iloc[0,1]
ic_covariance = ic_df.cov().iloc[0,1]
sc_strength= sc_df.Strength.std()
ic_intelligence = ic_df.Intelligence.std()

sc_combat= sc_df.Combat.std()
ic_combat = ic_df.Combat.std()
sc_pearson= sc_covariance/(sc_strength*sc_combat)
ic_pearson = ic_covariance/(ic_intelligence*ic_combat)
print(float(str(sc_pearson)[:4]))
print(float(str(ic_pearson)[:4]))




# --------------
#Code starts here

total_high= data.Total.quantile(0.99)
print(total_high)

super_best= data[data['Total'] > total_high]

super_best_names= list(super_best.Name)
print(super_best_names)


# --------------
#Code starts here

fig,(ax_1,ax_2,ax_3) = plt.subplots(1,3,figsize=[20,20])
ax_1.set_title('Intelligence')
ax_1.boxplot(data.Intelligence)

ax_2.set_title('Speed')
ax_2.boxplot(data.Speed)

ax_3.set_title('Power')
ax_3.boxplot(data.Power)







