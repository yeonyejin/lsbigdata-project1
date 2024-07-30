
import pandas as pd

exam=pd.read_csv("data/exam.csv")
exam.loc[exam["math"] <= 50, "math"] = "-"
exam

math_mean = exam.loc[(exam["math"] != "-"), "math"].mean()
exam.loc[exam['math']=='-','math'] = math_mean
