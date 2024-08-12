# 데이터 패키지 설치
!pip install palmerpenguins
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()
penguins["species"].unique()
penguins.columns

# x: bill_length_mm
# y: bill_depth_mm
fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "species",
    size_max = 15,
    trendline="ols"
)
# 레이아웃 업데이트
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", 
    font=dict(color="white", size =20)),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="white"),
    title=dict(
            text="펭귄 종",  # 범례 제목 설정
            font=dict(color="white", size=14)  # 범례 제목 글꼴 색상과 크기 설정
            ),
        ),
)
# 점 크기 업데이트 (여기서는 size_max 사용)
fig.update_traces(
    marker=dict(
        size=10,
        opacity=0.6  # 점의 투명도 조정 (0.0 ~ 1.0)
        ))  # 점의 크기 조정

fig.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins=penguins.dropna()
x = penguins[["bill_length_mm"]]
y = penguins["bill_depth_mm"]

model.fit(x, y)
linear_fit=model.predict(x)
slope = model.coef_[0]        # 기울기
intercept = model.intercept_  # intercept


fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀직선",
        line=dict(dash="dot", color = "white")
    )
)


fig.show()


# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
penguins_dummies = pd.get_dummies(
    penguins,
    columns=['species'],
    drop_first=False)
    
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]] # 문자형 변수를 숫자형으로 변경..?
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_
model.intercept_

regline_y= model.predict(x)


import seaborn as sns
import matplotlib.pyplot as plt


sns.scatterplot(x["bill_length_mm"], y, color = "black", s=1,
)
plt.scatter(x["bill_length_mm"], regline_y )

plt.show()
plt.clf()




#model.coef_
#array([ 0.20044313, -1.93307791, -5.10331533])
#>>> model.intercept_
#np.float64(10.565261622823762)
#회귀직선 방정식


 # y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
# penguins
# species    island  bill_length_mm  ...  body_mass_g     sex  year
# Adelie     Torgersen            39.5  ...       3800.0  female  2007
# Chinstrap  Torgersen            40.5  ...       3800.0  female  2007
# Gentoo     Torgersen            40.5  ...       3800.0  female  2007
# x1, x2, x3
# 39.5, 0, 0
# 40.5, 1, 0
# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
0.2 * 40.5 -1.93 * True -5.1* False + 10.56
 
 
 
