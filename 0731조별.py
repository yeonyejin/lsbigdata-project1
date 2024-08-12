import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df=pd.read_csv("./data/houseprice/train.csv")
df

house=df.copy()


df = house.dropna(subset=["MoSold","SalePrice"])\
                    .groupby("MoSold", as_index = False)\
                    .agg(count = ("SalePrice","count"))\
                    .sort_values("MoSold", ascending = True)
                    
sns.barplot(data=df, x="MoSold", y="count", hue="MoSold")
plt.rcParams.update({'font.family' : 'Malgun Gothic'})
plt.xlabel("월(month)")
plt.ylabel("이사횟수(count)")
plt.show()
plt.clf()  


df2 = house[["YearBuilt", "OverallCond"]]

house_cond = df2.groupby("OverallCond", as_index = False)\
                    .agg(count = ("YearBuilt", "count"))\
                    .sort_values("count", ascending = False)
sns.barplot(data = house_cond, x = "OverallCond", y = "count", hue = "OverallCond")
plt.show()
plt.clf()


df3 = house[["BldgType", "OverallCond"]]

house_bed = df3.groupby(["OverallCond", "BldgType"], as_index = False)\
                    .agg(count = ("BldgType", "count"))\
                    .sort_values("count", ascending = False)
sns.barplot(data = house_bed, x = "OverallCond", y = "count", hue = "BldgType")
plt.show()
plt.clf()


df4 = house[["SalePrice", "Neighborhood"]]

house_bed = df4.groupby(["Neighborhood", "SalePrice"], as_index = False)\
                    .agg(region_mean = ("SalePrice", "mean"))\
                    .sort_values("region_mean", ascending = False)
sns.barplot(data = house_bed, x = "Neighborhood", y = "region_mean", hue = "Neighborhood")

plt.show()
plt.clf()       



df5 = df4["Neighborhood"].unique()
            
sns.barplot(data = house_bed, x = "Neighborhood", y = "region_mean", hue = "Neighborhood")
plt.show()
plt.clf()


house["Neighborhood"].head(10)
house["SalePrice"]

