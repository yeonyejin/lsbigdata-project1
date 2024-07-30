import pandas as pd
import numpy as np

# 데이터 탐색 함수수
# head()
# tail()
# shape
# info()
# describe()
exam = pd.read_csv("data/exam.csv")
exam.head(10)
exam.tail(10)
exam.shape


# 메서드(함수) vs.속성 (어트리뷰트) - 깔려있는 변수
exam.shape
exam.info()
exam.describe()

type(exam)
var = [1, 2, 3]
type(var)
exam.head()
#var.head()


exam2 = exam.copy()
exam2.rename(columns={"nclass" : "class"})

exam2["total"] = exam2["math"] + exam["english"] + exam["science"]
exam2.head()


# 200 이상: pass
# 200 미만: fail

exam2["test"] = np.where(exam2["total"]>=200, 'pass', 'fail' )
exam2.head()

# 200이상: A
# 100이상: B
# 100미만: C

exam2["test2"] = np.where(exam2["total"] >= 200, "A",
                 np.where(exam2["total"] >= 100, "B", "C"))

exam2.head()

exam2["test2"].isin(["A", "C"])





import matplotlib.pyplot as plt
count_test = exam2["test"].value_counts()
count_test.plot.bar(rot = 0)
plt.show()
plt.clf()




