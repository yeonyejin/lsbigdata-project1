import pandas as pd
import numpy as np



house_train= pd.read_csv('data/houseprice/train.csv')
hosut_train=house_train[["Id", "YearBuilt", "SalePrice"]]
house_train.info()

#연도별 평균
house_mean=house_train.groupby("YearBuilt", as_index=False)\
           .agg(mean_year=("SalePrice", "mean"))
house_mean

house_test = pd.read_csv('data/houseprice/test.csv')

hosue_test = house_test[["Id", "YearBuilt"]]

house_test

house_test=pd.merge(house_test, house_mean, how = "left", on = "YearBuilt")
hosue_test

house_test=house_test.rename(columns={"mean_year": "SalePrice"})
house_test


house_test["SalePrice"].isna().sum()

#비어있는 테스트 세트 집들 확인인
hosue_test.loc[house_test["SalePrice"].isna()]

#집값채우기기
house_mean=house_train["SalePrice"].mean()
house_test["SalePrice"]=house_test["SalePrice"].fillna(house_mean)
#house_test.fillna()

#sun 데이터 불러오기
sub_df = pd.read_csv('data/houseprice/sample_submission.csv')
sub_df

#SalePrice 바꿔치기기
sub_df["SalePrice"]=house_test["SalePrice"]
sub_df

sub_df.to_csv("data/houseprice/sample_submission2.csv", index=False)

===========================================

hp= pd.read_csv('data/houseprice/train.csv')
hp
hp.shape
hp.head()
hp.info()

hp['SalePrice']
price_mean = hp['SalePrice'].mean()
price_mean


sub = pd.read_csv('data/houseprice/sample_submission.csv')
sub

sub["SalePrice"] = price_mean
sub


sub.to_csv("data/houseprice/sample_submission.csv", index=False)
#인덱스 없애고자 할 때, index=False로 해줌


#같은 해에 지어진 그룹을 한 그룹으로 보고 ->평균을 냄
#test.set에 있는 집값을 예측해보자.

hp["YearBuilt"]

mon_mean=hp.groupby("YearBuilt", as_index=False)\
         .agg( 
             mon_mean=("SalePrice", "mean")
         )
test = pd.read_csv('data/houseprice/test.csv')
mon_mean["mon_mean"].mean()

test_new = pd.merge(test, hp_mean, how = "left", on = "YearBuilt")
test_new
#test["SalePrice"] = hp_mean["mon_mean"]
test_new.to_csv("data/houseprice/test_new.csv", index=False)
sub.loc[[100, 144, 376, 759, 968, 993, 1009, 1010, 1121], ["SalePrice"]] = 168213

sub["SalePrice"] = test_new["mon_mean"]
sub.to_csv("data/houseprice/sample_submission_.csv", index=False)





