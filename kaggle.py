import pandas as pd
import numpy as np



house_train= pd.read_csv('data/houseprice/train.csv')
hosue_train=house_train[["Id", "YearBuilt", "GrLivArea"]]
house_train.info()

#연도별 평균
place_mean=house_train.groupby("YearBuilt", as_index=False)\
           .agg(mean_place=("GrLivArea", "mean"))
place_mean

house_test = pd.read_csv('data/houseprice/test.csv')

house_test = house_test[["Id", "YearBuilt"]]

house_test

house_test = pd.merge(house_test, place_mean, how = "left", on = "YearBuilt")
house_test

house_test=house_test.rename(columns={"mean_place": "GrLivArea"})
house_test


house_test["GrLivArea"].isna().sum()

#비어있는 테스트 세트 집들 확인인
hosue_test.loc[house_test["GrLivArea"].isna()]

#집값채우기기
place_mean=house_train["GrLivArea"].mean()
house_test["GrLivArea"]=house_test["GrLivArea"].fillna(place_mean)
#house_test.fillna()

#sun 데이터 불러오기
sub_df = pd.read_csv('data/houseprice/sample_submission.csv')
sub_df

#SalePrice 바꿔치기기
sub_df["GrLivArea"]=house_test["GrLivArea"]
sub_df

sub_df.to_csv("data/houseprice/sample_submission.csv", index=False)




=================================
house_mean2=house.groupby(["YearBuilt","GarageCars","KitchenQual"], as_index=False)\
         .agg( 
             house_mean=("SalePrice", "mean")
         )
house_mean2

#test 파일에서 id랑년도만 빼줌
house_test2 = pd.read_csv('data/houseprice/test.csv')
house_test2 = house_test2[["Id","YearBuilt","GarageCars","KitchenQual"]]
house_test2

# test 파일에 집값 평균 구한거 연도에 맞게 추가
house_test2 = pd.merge(house_test2, house_mean2, how = "left", on = ["YearBuilt","GarageCars","KitchenQual"])
house_test2

# 열이름 바꿈
house_test = house_test.rename(
    columns = {"house_mean" : "SalePrice"}
)
house_test

# 결측치 처리
house_test["SalePrice"].isna().sum()

price_mean = house["SalePrice"].mean()
price_mean

house_test = house_test.fillna(price_mean)

# submission 파일 만듬
submission = house_test[["Id","SalePrice"]]
submission

submission.to_csv("data/houseprice/sample_submission2.csv", index=False)
