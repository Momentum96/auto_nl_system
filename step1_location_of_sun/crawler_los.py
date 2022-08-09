from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import os
def make_URL(year, month, day):

    base_url = 'https://astro.kasi.re.kr/life/pageView/10'
    lat = 36.850490744236744
    lon = 127.15250390636234
    elevation = -106.09210616048128
    address = '충청남도+천안시+서북구+천안대로+1223-24'

    return base_url + '?useElevation=1' \
           + '&lat=%s&lng=%s' % (lat, lon) \
           + '&elevation=%s' % elevation \
           + '&output_range=1' \
           + '&date=%s-%s-%s' % (year, month, day)\
           + '&hour=&minute=&second=' \
           + '&address=%s' % address



def cralwer_los(url,year, month, day):
    chrome_driver_path = os.getcwd() + '/../driver/chromedriver'
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('disable-gpu')
    driver = webdriver.Chrome(chrome_driver_path, options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    tr_list = soup.select('#sun-height-table > table > tbody')[0].find_all('tr')
    table_list = []
    for html_row in tr_list:
        row = []
        count = 0
        for html_element in html_row.find_all('td'):
            # print(html_element.getText())
            if len(html_element.getText()) > 2:
                if(count<2):
                    element = transform_angle(html_element.getText())
                    row.append(element)
                    row.append(html_element.getText())
                count = count+1
            else:
                element = year+'-'+month+'-'+day + ' ' + html_element.getText()+':00:00'
                count =0
                row.append(element)


        table_list.append(row)
    return table_list

def transform_angle(str_angle: str):
    return int(str_angle.split()[0]) + int(str_angle.split()[1]) / 60 \
           + float(str_angle.split()[2]) / 3600

def Los(year, month, day):
    print("[태양의 시간별 각도 수집]")
    table = cralwer_los(make_URL(year, month, day),year, month, day)
    df = pd.DataFrame(table,columns =['Date', 'kasi_azimuth', 'kasi_azimuth_text', 'kasi_altitude', 'kasi_altitude_text'])
    print(df)
    path = os.getcwd() + '/../los_df.csv'
    df.to_csv( path, sep=',', na_rep='NaN', float_format='%.2f', \
              columns =['Date', 'kasi_azimuth', 'kasi_azimuth_text', 'kasi_altitude', 'kasi_altitude_text'], index=False)
    # to_db(df, username='root', passwd='defacto8*jj', host='210.102.142.14', port=3306,
    #       db_name='naturallight', table_name='row_altitude')

    print("[1단계 완료]")

# def to_db(pdf: pd.DataFrame, db_type='mysql', **kwargs):
#     if db_type == 'mysql':
#         from sqlalchemy import create_engine
#         args = (kwargs['username'], kwargs['passwd'], kwargs['host'], kwargs['port'], kwargs['db_name'])
#         engine = create_engine('mysql+pymysql://%s:%s@%s:%d/%s' % args, encoding='utf-8')
#         conn = engine.connect()
#
#         # db insert
#         pdf.to_sql(name=kwargs['table_name'], con=engine, if_exists='append', index=False)
#         conn.close()

if __name__ == '__main__':
    year = input("year(yyyy) :")
    month = input("month(mm) :")
    day = input("day(dd) :")
    Los(year, month, day)



