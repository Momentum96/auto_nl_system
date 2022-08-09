from distutils.log import Log

from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import os
from sqlalchemy import create_engine
import pymysql


def make_URL(year, month, day):
    base_url = 'https://astro.kasi.re.kr/life/pageView/9'
    lat = 36.850490744236744
    lon = 127.15250390636234
    address = '충청남도+천안시+서북구+천안대로+1223-24'

    return base_url + '?useElevation=1' \
           + '&lat=%s&lng=%s' % (lat, lon) \
           + '&output_range=1' \
           + '&date=%s-%s-%s' % (year, month, day)\
           + '&address=%s' % address



def cralwer_tosRS(url,year, month, day):
    chrome_driver_path = os.getcwd() + '/../driver/chromedriver'
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('disable-gpu')
    driver = webdriver.Chrome(chrome_driver_path, options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    sunrise = soup.find_all('span', {'class': 'sunrise'})[0].string
    sunset = soup.find_all('span', {'class': 'sunset'})[0].string
    sr = sunrise[0:2] + ':' + sunrise[4:-1]
    ss = sunset[0:2] + ':' + sunset[4:-1]

    table_list = []
    row = []
    row.append(year+"-"+month+"-"+day)
    row.append(sr)
    row.append(ss)
    table_list.append(row)
    return table_list


def to_db(pdf: pd.DataFrame, db_type='mysql', **kwargs):
    if db_type == 'mysql':
        from sqlalchemy import create_engine
        args = (kwargs['username'], kwargs['passwd'], kwargs['host'], kwargs['port'], kwargs['db_name'])
        engine = create_engine('mysql+pymysql://%s:%s@%s:%d/%s' % args, encoding='utf-8')
        conn = engine.connect()

        # db insert
        pdf.to_sql(name=kwargs['table_name'], con=engine, if_exists='append', index=False)

        conn.close()


def tosRS(year, month, day):
    print("[태양의 일출일몰 시간 수집]")
    table = cralwer_tosRS(make_URL(year, month, day), year, month, day)
    df = pd.DataFrame(table,columns =['Date', 'Sunup_Time', 'Sundown_Time'])

    print(df)
    path = os.getcwd() + '/../tosRS.csv'
    df.to_csv(path, sep=',', na_rep='NaN', float_format='%.2f', \
             columns =['Date', 'Sunup_Time', 'Sundown_Time'], index=False)

    to_db(df, username='root', passwd='defacto8*jj', host='210.102.142.14', port=3306,
              db_name='naturallight', table_name='sun_info')
    print("[2단계 완료]")



if __name__ == '__main__':
    year = input("year(yyyy) :")
    month = input("month(mm) :")
    day = input("day(dd) :")
    tosRS(year, month, day)



