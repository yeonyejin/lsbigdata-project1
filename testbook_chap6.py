import pandas as pd
import numpy as np

# 데이터 전처리 함수
# query()
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()


exam = pd.read_csv("data/exam.csv")

#조건에 맞는 행을 걸러내는 .query()
#exam[exam["nclass"] == 1]
exam.query("nclass == 1")
exam.query("nclass != 1")
exam.query('math>50')


exam.query('nclass == 1 & math >= 50')
exam.query('english<90 | science < 50')
exam.query("nclass in [1, 2]")
exam.query("nclass == 1 | nclass == 2")
exam.query("nclass == 1 or nclass == 2")
exam.query("nclass not in [1, 2]")


exam[["nclass"]]
exam["nclass"]
exam[["id", "nclass"]]
exam.drop(columns = ["math", "english"])
exam


exam.query("nclass == 1")[["math", "english"]]
exam.query("nclass == 1")\
    [["math", "english"]]\
    .head()


#정렬하기
exam.sort_values("math", ascending = False)
exam.sort_values(["nclass", "english"], ascending = [True, False])

#변수추가
exam  = exam.assign(
    total = exam["math"] + exam["english"] + exam["science"],
    mean = (exam["math"] + exam["english"] + exam["science"])/3)\
    .sort_values("total", ascending = False)
exam.head()

# lambda 함수 사용하기        
exam2 = pd.read_csv("data/exam.csv")
exam2  = exam2.assign(
    total = lambda x: x["math"] + x["english"] + x["science"],
    mean = lambda x: (x["math"] + x["english"] + x["science"])/3)\
    .sort_values("total", ascending = False)
exam2.head()


exam2  = exam2.assign(
    total = lambda x: x["math"] + x["english"] + x["science"],
    mean = lambda x: (x["total"])/3)\
    .sort_values("total", ascending = False)
exam2.head()


#그룹을 나눠 요약을 하는 .groupby() + .agg() 콤보
exam2.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass") \ 
     .agg(mean_math = ("math", "mean"))

 exam2.agg(mean_english = ("english", "mean"))
exam2.groupby("nclass") \ 
     .agg(mean_english = ("english", "mean"))

 exam2.agg(mean_science = ("science", "mean"))
exam2.groupby("nclass") \ 
     .agg(mean_science = ("science", "mean"))
     
     
     

import pydataset
mpg = pd.read_csv("data/mpg.csv")
mpg.query('category == "suv"') \
    .assign(total=(mpg['hwy'] + mpg['cty']) / 2) \
    .groupby('manufacturer') \
    .agg(mean_tot=('total', 'mean')) \
    .sort_values('mean_tot', ascending=False) \
    .head()


import pandas as pd

test1 = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                      'midterm': [60, 80, 70, 90, 85]})

test2 = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                      'final': [70, 83, 65, 95, 80]})

test1
test2

#Left Join
total = pd.merge(test1, test2, how = 'left', on = 'id')
total

#Right Join
total = pd.merge(test1, test2, how = 'right', on = 'id')
total

#Inner Join
total = pd.merge(test1, test2, how = 'inner', on = 'id')
total

#Outer Join
total = pd.merge(test1, test2, how = 'outer', on = 'id')
total


#exam = pd.read_csv("data/exam.csv")
name = pd.DataFrame({'nclass' : [1, 2, 3, 4, 5],
                     'teacher': ['kim', 'lee', 'park', 'choi', 'jung']})
name

exam_new = pd.merge(exam, name, how = 'left', on = 'nclass')
exam_new

#데이터를 세로로 쌓는 방법
score1 = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                      'score': [60, 80, 70, 90, 85]})

score2 = pd.DataFrame({'id': [6, 7, 8, 9, 10],
                      'score': [70, 83, 65, 95, 80]})

score1 
score2

score_all = pd.concat([score1, score2])
score_all

test1
test2

pd.concat([test1, test2], axis=1)





