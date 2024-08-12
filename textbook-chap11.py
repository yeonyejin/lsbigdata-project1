import json
geo = json.load(open('data/SIG.geojson', encoding = 'UTF-8'))
geo['features'][0]['properties']
geo['features'][0]['geometry']

import pandas as pd
df_pop = pd.read_csv('data/Population_SIG.csv')
df_pop.head()
df_pop.info()

df_pop['code'] = df_pop['code'].astype(str)

#!pip install folium

===========================================
import numpy as np
import matplotlib.pyplot as plt
import json

geo_seoul = json.load(open("data/SIG_Seoul.geojson", encoding = 'UTF-8'))

type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"])
len(geo_seoul["features"][0])
geo_seoul["features"][0].keys()

# 숫자가 바뀌면"구"가 바뀐다.
geo_seoul["features"][2]["properties"]
geo_seoul["features"][0]["geometry"]

# 리스트로 정보 빼오기 
coordinate_list = geo_seoul["features"][2]["geometry"]["coordinates"]
len(coordinate_list[0][0])
coordinate_list[0][0]


import numpy as np
coordinate_array=np.array(coordinate_list[0][0])
x = coordinate_array[:, 0]
y = coordinate_array[:, 1]

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
plt.clf()

#함수로 만들기
def draw_seoul(num):
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x = coordinate_array[:, 0]
    y = coordinate_array[:, 1]
    
    plt.rcParams.update({"font.family": "Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
    return None

draw_seoul(3)

geo_count=len(geo_seoul["features"])


geo_seoul
==========================================
# 구이름 만들기
gu_name = geo_seoul["features"][0]["properties"]["SIG_KOR_NM"]
gu_name

# 방법 1
gu_name = list()
for i in range(25):
    #gu_name= gu_name + [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"]]
    gu_name.append([geo_seoul["features"][i]["properties"]["SIG_KOR_NM"]])
   
gu_name

# 방법 2
gu_name= [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"] for i in range(25)]
gu_name

# x, y 판다스 데이터 프레임
import pandas as pd

def make_seouldf(num):
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x = coordinate_array[:, 0]
    y = coordinate_array[:, 1]
    
    return pd.DataFrame({"gu_name":gu_name, "x": x, "y": y})

make_seouldf(12)

result = pd.DataFrame({})

for i in range(25):
    result = pd.concat([result, make_seouldf(i)], ignore_index = True)
    
result

# 서울 그래프 그리기
import seaborn as sns
sns.scatterplot(data=result, 
    x = 'x', y = 'y', hue = 'gu_name', legend=False,
    palette = "viridis", s=2)
plt.show()
plt.clf()




# 강남 or 강남 외
import seaborn as sns
gangnam_df = result.assign(is_gangnam=np.where(result["gu_name"]=="강남구", "강남", "강남외"))
sns.scatterplot(
    data=gangnam_df, 
    x = 'x', y = 'y',legend=False, 
    palette={'강남외': "grey", '강남': "red"},
    hue = 'is_gangnam', s=2)
plt.show()
plt.clf()



gangnam_df["is_gangnam"].unique()


# 데이터 프레임 
result=pd.DataFrame({
    'gu_name': [],
    'x':[],
    'y':[]
})

=========================================
# 서울시  전체지도 그리기 (GPT)
def draw_seoul_map():
    plt.figure(figsize=(12, 12))  # 그림 크기 설정
    
    for feature in geo_seoul['features']:
        gu_name = feature['properties']["SIG_KOR_NM"]
        coordinate_list = feature["geometry"]["coordinates"]

        # 다각형의 좌표가 여러 개일 수 있으므로 반복
        for coords in coordinate_list:
            coordinate_array = np.array(coords[0])  # 첫 번째 다각형의 좌표
            x = coordinate_array[:, 0]
            y = coordinate_array[:, 1]

            plt.plot(x, y)  # 각 구를 플롯에 추가
            
    plt.rcParams.update({"font.family": "Malgun Gothic"})
    plt.title("서울시 전체 지도")
    plt.xlabel("경도")
    plt.ylabel("위도")
    plt.grid()
    plt.show()
    
# 함수 호출
draw_seoul_map()

===============================================

import numpy as np
import matplotlib.pyplot as plt
import json

geo_seoul = json.load(open("data/SIG_Seoul.geojson", encoding = "UTF-8"))
geo_seoul["features"][0]["properties"]

df_pop = pd.read_csv("data/Population_SIG.csv")
df_pop.head()
df_seoulpop=df_pop.iloc[1:26]
df_seoulpop["code"]=df_seoulpop["code"].astype(str)
df_seoulpop.info()

# 패키지 설치하기
#!pip install folium
import folium

center_x=result["x"].mean()
center_y=result["y"].mean()
# p.304
# 흰 도화지 맵 가져오기
map_sig=folium.Map(location = [37.551, 126.973],
                  zoom_start=12,
                  tiles = "cartodbpositron")
map_sig.save("map_seoul.html")

# Choropleth 코로플릿 - 구 경계선 그리기
geo_seoul
geo_seoul["features"][0]["properties"]["SIG_CD"]
folium.Choropleth(
    geo_data=geo_seoul,
    data = df_seoulpop,
    columns=("code","pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
     
map_sig.save("map_seoul.html")

# Choropleth 코로플릿 with bins
#

bins = list(df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
folium.Choropleth(
    geo_data=geo_seoul,
    data = df_seoulpop,
    columns=("code","pop"),
    fill_color = "viridis",
    bins=bins,
    key_on = "feature.properties.SIG_CD").add_to(map_sig)
    
map_sig.save("map_seoul.html")

# 점 찍는 방법
#make_seouldf(0).iloc[:,1:3].mean() # x, y, 평균값을 찾음음
folium.Marker([37.583744, 126.983800], popup="종로구").add_to(map_sig)
map_sig.save("map_seoul.html")


