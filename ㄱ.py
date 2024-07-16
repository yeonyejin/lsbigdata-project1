import pandas as pd
import numpy as np

df = pd.read_csv('stat.csv')
df
df = df.drop([0])      
df

df.info()
df = df.astype({'2020':'float'})
df = df.astype({'2020.1':'float'})
df = df.astype({'2020.2':'float'})
df = df.astype({'2020.3':'float'})
df = df.astype({'2021':'float'})
df = df.astype({'2021.1':'float'})
df = df.astype({'2021.2':'float'})
df = df.astype({'2021.3':'float'})
df = df.astype({'2022':'float'})
df = df.astype({'2022.1':'float'})
df = df.astype({'2022.2':'float'})
df = df.astype({'2022.3':'float'})  # 연산이 가능하게 데이터 타입을 float으로 변경경

df.info()



df = df.drop(columns = ['2020', '2021', '2022']) # 총 출산율은 필요없어서 열 삭제

df

df = df.rename(columns = {'2020.1' : '2020_2024'})
df = df.rename(columns = {'2020.2' : '2020_2529'})
df = df.rename(columns = {'2020.3' : '2020_3034'})
df = df.rename(columns = {'2021.1' : '2021_2024'})
df = df.rename(columns = {'2021.2' : '2021_2529'})
df = df.rename(columns = {'2021.3' : '2021_3034'})
df = df.rename(columns = {'2022.1' : '2022_2024'})
df = df.rename(columns = {'2022.2' : '2022_2529'})
df = df.rename(columns = {'2022.3' : '2022_3034'})

df                                                # 변수의 이름이 한 눈에 알아보기 힘들어 변경경

df['2020mean'] = (df['2020_2024'] + df['2020_2529'] + df['2020_3034']) / 3
df['2021mean'] = (df['2021_2024'] + df['2021_2529'] + df['2021_3034']) / 3
df['2022mean'] = (df['2022_2024'] + df['2022_2529'] + df['2022_3034']) / 3
df                                            
