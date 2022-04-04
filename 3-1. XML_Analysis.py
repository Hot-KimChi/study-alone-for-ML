from bs4 import BeautifulSoup 
import urllib.request as req
import os.path

url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108"
savename = "forecast.xml"
if not os.path.exists(savename):                        ## 프로그램을 실행할 때마다 XML 파일을 내려받으면 서버에 부하
    req.urlretrieve(url, savename)                      ## 처음 실행: urlretrieve를 사용해 처음 실행할 때 로컬 파일로 데이터 저장.
                                                        ## 두번째 이후 실행: 저장한 데이터를 읽어 사용.


# BeautifulSoup로 분석하기 --- (※2)
xml = open(savename, "r", encoding="utf-8").read()      ## 파일에서 XML을 읽어 들이고 BeautifulSoup로 XML로 분석.
soup = BeautifulSoup(xml, 'html.parser')


# 각 지역 확인하기 --- (※3)
info = {}
for location in soup.find_all("location"):
    name = location.find('city').string
    weather = location.find('wf').string
    if not (weather in info):
        info[weather] = []
    info[weather].append(name)

# 각 지역의 날씨를 구분해서 출력하기
for weather in info.keys():
    print("+", weather)
    for name in info[weather]:
        print("| - ", name)