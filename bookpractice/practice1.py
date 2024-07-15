import pandas as pd
df = pd.DataFrame({'name'    : ['김지훈', '이유진', '박동현', '김민지'],
                   'english' : [90, 80, 60, 70],
                   'math'    : [50, 60, 100, 20]})
df
df['english']
sum(df['english'])
sum(df['math'])
sum(df['english']) / 4
sum(df['math']) / 4

df = pd.DataFrame({'prodct'   : ['사과', '딸기', '수박'],
                   'price'   : [1800, 1500, 3000],
                   'volume' : [24, 38, 13]})
df
df['price']
sum(df['price']) / 3
sum(df['volume']) 

df_exam = pd.read_excel('excel_exam.xlsx')

