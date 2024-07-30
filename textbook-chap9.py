# 막대그래프
#p mpg["drv"].unique()
df_mpg = mpg.groupby("drv", as_index=False)\
         .agg(mean_hwy=())
         
         
# 교재 8장, p32
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

economics=pd.read_csv("data/economics.csv")
economics.head()


economics.info()
sns.lineplot(data=economics, x="date", y="unemploy")
plt.show()
plt.clf()

economics["date2"]= pd.to_datetime(economics['date'])
economics
economics.info()

economics[["date", "date2"]]
economics["date2"].dt.year
economics["date2"].dt.month
economics["date2"].dt.day
economics["date2"].dt.quarter

economics["quarter"]= economics["date2"].dt.quarter
economics[["date2", "quarter"]]

#각 날짜는 무슨요일인가?
economics["date2"].dt.day_name()

# 원래 날짜의 3일후
economics["date2"] + pd.DateOffset(days=3)
economics["date2"] + pd.DateOffset(days=30)
economics["date2"] + pd.DateOffset(months=1)
economics["date2"].dt.is_leap_year #윤년 체크
economics['year'] = economics['date2'].dt.year
economics.head()
sns.lineplot(data=economics, 
             x = 'year', y='unemploy',
             errorbar = None)
plt.show()
plt.clf()


#1월부터 12월까지 실직자 수를 가져와서
#표본평균, 표본 편차를 내서 
# 점에 해당하는 left_ci, right_ci를 알고 있다.


my_df=economics.groupby("year", as_index=False)\
         .agg( 
             mon_mean=("unemploy", "mean"),
             mon_std=("unemploy", "std"),
             mon_n=("unemploy", "count")
         )
my_df

mean + 1.96+std/sqrt(12)

my_df["left_ci"] = my_df["mon_mean"] - 1.96 * my_df["mon_std"]/np.sqrt(my_df["mon_n"])
my_df["right_ci"] = my_df["mon_mean"] + 1.96 * my_df["mon_std"]/np.sqrt(my_df["mon_n"])
my_df.head()

import matplotlib.pyplot as plt
x = my_df["year"]
y = my_df["mon_mean"]
#plt.scatter(x, y, s=3)
plt.plot(x,y, color="black")
plt.scatter(x, my_df["left_ci"], color="blue", s=1)
plt.scatter(x, my_df["right_ci"], color="red", s=1)
plt.show()
plt.clf()







