
import pandas as pd

df_exam = pd.read_excel('excel_exam.xlsx')
df_exam


sum(df_exam['english'])  / 20
sum(df_exam['science']) / 20

x = [1, 2, 3, 4, 5]
x

len(x)

 df = pd.DataFrame({'a' : [1, 2, 3], 
                    'b' : [4, 5, 6]})
df

len(df)

len(df_exam)                  
sum(df_exam['english']) / len(df_exam)
sum(df_exam['science']) / len(df_exam)


df_exam_novar = pd.read_excel('excel_exam_novar.xlsx')


df_csv_exam = pd.read_csv('exam.csv')
df_csv_exam




df_midterm = pd.DataFrame({'english' : [90, 80, 60, 70],
                           'math'    : [50, 60, 100, 20],
                           'nclass'  : [1, 1, 2, 2]})
df_midterm


import pandas as pd
exam = pd.read_csv('exam.csv')
exam.head(10)
exam.tail(10)
exam.shape
exam.info()
exam.describe()


mpg = pd.read_csv('mpg.csv')
mpg.head()
mpg.tail()
mpg.shape
mpg.info()
mpg.describe(include = 'all')


sum(var)
max(var)

import pandas as pd
pd.read_csv('exam.csv')
pd.DataFrame({'x' : [1, 2, 3]})

df.head()
df.info()

var = [1, 2, 3]
var.head()

type(df)
type(var)

df.head()
df.shape
 
 
 
df_raw = pd.DataFrame({'var1' : [1, 2, 1], 
                       'var2' : [2, 3, 2]})

df_raw


df_new = df_raw.copy()
df_new

df_new = df_new.rename(columns = {'var2' :  'v2'})
df_new


df_raw

df_new
 

mpg = pd.read_csv('mpg.csv')
mpg_new = mpg.copy()
mpg_new = mpg_new.rename( columns = {'cty' : 'city'})
mpg_new = mpg_new.rename( columns = {'hwy' : 'highway'})
mpg_new

df = pd.DataFrame({'var1' : [4, 3, 8],
                   'var2' : [2, 6, 1]})
df

df['var_sum'] = df['var1'] + df['var2']
df

df['var_mean']  = (df['var1'] + df['var2'])  / 2
df


mpg.head()

sum(mpg['total']) / len(mpg)
mpg[ 'total' ].mean()



mpg['total'].describe()

mpg['total'].plot.hist()


import numpy as np

