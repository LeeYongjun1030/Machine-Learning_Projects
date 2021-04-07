import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import csv

df = pd.read_csv('drug.csv', encoding='utf-8')

##### BP ############################################################
sort_by_bp=pd.DataFrame(df.groupby(['Drug','BP']).count())
sort_by_bp =sort_by_bp.reset_index()
bp_pivot = sort_by_bp.pivot(index='Drug',columns='BP',values ='Age')
bp_pivot.plot(kind='bar')
##### Sex ############################################################
sort_by_sex=pd.DataFrame(df.groupby(['Drug','Sex']).count())
sort_by_sex =sort_by_sex.reset_index()
sex_pivot = sort_by_sex.pivot(index='Drug',columns='Sex',values ='Age')
sex_pivot.plot(kind='bar')
##### Cholesterol ############################################################
sort_by_cho=pd.DataFrame(df.groupby(['Drug','Cholesterol']).count())
sort_by_cho =sort_by_cho.reset_index()
cho_pivot = sort_by_cho.pivot(index='Drug',columns='Cholesterol',values ='Age')
cho_pivot.plot(kind='bar')
##### Age ############################################################
bins = list(range(10,101,20))  ## age 데이터를 10~29, 30~49, 50~69, 70~89, 90~100로 분류
df['Age'] = pd.cut(df['Age'],bins,right=False)
sort_by_age=pd.DataFrame(df.groupby(['Drug','Age']).count())
sort_by_age =sort_by_age.reset_index()
age_pivot = sort_by_age.pivot(index='Drug',columns='Age',values ='Na_to_K')
age_pivot.plot(kind='bar')
##### Na to K ############################################################
bins = list(range(0,40,10)) ## Na_to_데이터를 0~9, 10~19, 20~29, 30~39으로 분류
df['Na_to_K'] = pd.cut(df['Na_to_K'],bins,right=False)
sort_by_na=pd.DataFrame(df.groupby(['Drug','Na_to_K']).count())
sort_by_na =sort_by_na.reset_index()
na_pivot = sort_by_na.pivot(index='Drug',columns='Na_to_K',values ='Age')
na_pivot.plot(kind='bar')

plt.show()

