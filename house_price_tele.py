import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


house_price = pd.read_csv("data/houseprice/houseprice-with-lonlat.csv")

import folium

center_x=house_price["Latitude"].mean() # 위도
center_y=house_price["Longitude"].mean() # 경도

Latitude=house_price["Latitude"]
Longitude=house_price["Longitude"]
Price = house_price["Sale_Price"]
# p.304
# 흰 도화지 맵 가져오기
map_sig=folium.Map(location = [42.034, -93.642],
                  zoom_start=12,
                  tiles = "cartodbpositron")

# 점 찍는 방법
for i in range(len(Latitude)) : 
    folium.CircleMarker([Latitude[i], Longitude[i]],  
                        popup=f"Price: ${Price[i]}",
                        radius=3, # 집의 면적으로 표현해보기
                        color='skyblue', 
                        fill_color='skyblue',
                        fill=True, 
                        fill_opacity=0.6 ).add_to(map_sig)

    
map_sig.save("house_price_map.html")


