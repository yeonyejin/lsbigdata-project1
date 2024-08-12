#pip install pyreadstat

import pandas as pd
import numpy as np
import seaborn as sns

raw_welfare=pd.read_spss("./data/koweps/Koweps_hpwc14_2019_beta2.sav")
raw_welfare

welfare=raw_welfare.copy()
welfare.shape
welfare.describe()

welfare=welfare.rename(
    columns = {
        "h14_g3": "sex",
        "h14_g4": "birth",
        "h14_g10": "marriage_type",
        "h14_g11": "religion",
        "p1402_8aq1": "income",
        "h14_eco9": "code_job",
        "h14_reg7": "code_region"
    }
)

welfare=welfare[["sex", "birth", "marriage_type",
                "religion", "income", "code_job", "code_region"]]
welfare.shape

welfare["sex"].dtypes
welfare["sex"].value_counts()
# welfare["sex"].isna().sum()

welfare["sex"] = np.where(welfare["sex"] == 1,'male', 'female')
welfare["sex"].value_counts()


welfare["income"].describe()
welfare["income"].isna().sum()

sex_income=welfare.dropna(subset="income") \
                  .groupby("sex", as_index=False) \
                  .agg(mean_income = ("income", "mean"))

sex_income

import seaborn as sns

sns.barplot(data=sex_income, x="sex", y="mean_income",
            hue="sex")
plt.show()
plt.clf()



welfare["birth"].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
plt.clf()


welfare["income"].isna().sum()

welfare=welfare.assign(age=2019 - welfare["birth"]+1)
welfare["age"]
sns.histplot(data=welfare, x="age")
plt.show()
plt.clf()


age_income = welfare.dropna(subset = 'income')\
                    .groupby('age', as_index=False)\
                    .agg(mean_income = ('income', 'mean'))
age_income.head()

sns.lineplot(data=age_income, x='age', y='mean_income')
plt.show()
plt.clf()

sum(welfare["income"]== 0)


# 나이별 income 칼럼 na 갯수 세기

my_df=welfare.assign(income_na=welfare["income"].isna())\
                    .groupby("age", as_index=False)\
                    .agg(n =("income_na", "sum"))

sns.barplot(data=my_df, x="age", y="n")
plt.show()
plt.clf()
### 무응답자 수 그래프임


welfare['age'].head()
welfare = welfare.assign(ageg = np.where(welfare['age']<  30, 'young',
                                np.where(welfare['age']<= 59, 'middle',
                                                              'old')))
welfare['ageg'].value_counts()

sns.countplot(data=welfare, x = 'ageg')
plt.show()
plt.clf()


ageg_income = welfare.dropna(subset = 'income')\
                     .groupby('ageg', as_index = False) \
                     .agg(mean_income = ('income', 'mean'))

sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income')
plt.show()
plt.clf()


sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income',
            order = ['young', 'middle', 'old'])
plt.show()
plt.clf()


# 나이대별 수입 분석
#cut

###str 연산 더하기 연산은 글씨를 붙일 수 있다.
###이 밑의 수식이 되게 유용할 것이라고 하심 snippet에 저장할 정도로..
bin_cut=np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare=welfare.assign(age_group = pd.cut(welfare["age"], 
                bins=bin_cut,
                labels=(np.arange(12) * 10).astype(str) + "대"))
welfare["age_group"]


vec_x=np.array([1, 7, 5, 4, 6, 3])
pd.cut(vec_x, 3)
np.arange(12)*10-1
pd.cut(vec_x, bins=bin_cut)
vec_x = np.random.randint(0, 100, 50)


#np.version.version
# (np.arange(12)*10).astype(str) + "대"

age_income = welfare.dropna(subset = 'income')\
                     .groupby('age_group', as_index = False) \
                     .agg(mean_income = ('income', 'mean'))

age_income
sns.barplot(data=age_income, x = 'age_group', y = 'mean_income')
plt.show()
plt.clf()

#판다스 데이터 프레임을 다룰 때, 변수의 타입이 
#카테고리로 설성되어 있는 경우, groupby+ agg 콤보 안먹힘
#그래서 object 타입으로 바꿔 준 후 수행

welfare["age_group"]=welfare["age_group"].astype("object")

sex_age_income = \
    welfare.dropna(subset="income")\
           .groupby(["age_group", "sex"], as_index=False)\
           .agg(mean_income=("income", "mean"))
           
sex_age_income

sns.barplot(data=sex_age_income,
            x="age_group", y="mean_income",
            hue="sex")

plt.show()
plt.clf()

===============================================
# 연령대별 상위 4% 수입 찾아보세요!

#그룹화 및 사용자 정의 함수 적용
sex_age_income = welfare.dropna(subset=["age_group", "sex"])\
           .groupby(["age_group", "sex"], as_index=False)\
           .agg(mean_income=("income", "mean"))


welfare["age_group"]=welfare["age_group"].astype("object")
sex_age_income = \
    welfare.dropna(subset="income")\
           .groupby(["age_group", "sex"], as_index=False)\
           .agg(mean_income=("income", 
                             lambda x: np.quantile(x, q=0.96)))
           
sex_age_income


===============================================

welfare["age_group"]=welfare["age_group"].astype("object")
#그룹화 및 사용자 정의 함수 적용
def my_f(vec):
    return vec.sum()

sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: my_f(x)))
    
sex_age_income


sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: np.quantile(x, q=0.96))
    
    


sex_age_income

sns.barplot(data=sex_age_income,
            x="age_group", y="mean_income", 
            hue="sex")
plt.show()
plt.clf()


#참고
welfare.dropna(subset = 'income')\
       .groupby('sex', as_index = False)[['income']]\
       .agg(['mean', 'std'])

welfare.dropna(subset = 'income')\
       .groupby('sex')[['income']]\
       .mean()

my_f(welfare.dropna(subset = 'income')\
       .groupby('sex')[['income']])
       
       
# 9-6장

welfare["code_job"]
welfare["code_job"].value_counts()

#직종데이터 불러오기
list_job=pd.read_excel("data/koweps/Koweps_Codebook_2019.xlsx",
                        sheet_name="직종코드")
list_job.head()                        
                        
welfare=welfare.merge(list_job, 
                      how="left", on= "code_job")     

welfare.dropna(subset=["job", "income"])[["income", "job"]]                        


job_income = welfare.dropna(subset = ["job", "income"])\
                    .query("sex=='female'")\
                    .groupby("job", as_index = False)\
                    .agg(mean_income = ('income', 'mean'))\
                    .sort_values("mean_income", ascending =False)\
                    .head(10)

job_income 

import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Malgun Gothic'})
sns.barplot(data = top10, y = 'job', x = 'mean_income', hue="job")                        
plt.show()
plt.clf()


bottom10 = job_income.sort_values('mean_income').head(10)
bottom10


sns.barplot(data = bottom10, y = 'job', x = 'mean_income')\
   .set(xlim=[0, 800])

##9-8

welfare.info()
welfare["marriage_type"]
df = welfare.query("marriage_type != 5")\
            .groupby("religion", as_index=False)\
            ["marriage_type"]\
            .value_counts(normalize=True) #핵심!
    
df

df.query("marriage_type == 1 ")\
  .assign(proportion=df["proportion"]*100)\
  .round(1)





