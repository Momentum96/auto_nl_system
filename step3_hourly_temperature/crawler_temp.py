from distutils.log import Log

from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import os
from sqlalchemy import create_engine
import pymysql


def make_URL(year, month, day):
    base_url = 'https://www.weather.go.kr/plus/land/current/city.jsp'
    return base_url \
           + '?tm=%s.%s.%s.23:00'% (year, month, day) \
           + '&type=t99'\
           + '&mode=0'\
           + '&reg=100'\
           + '&auto_man=m'\
           + '&stn=232'



def cralwer_temp(url,year, month, day):
    chrome_driver_path = os.getcwd() + '/../driver/chromedriver'
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('disable-gpu')
    driver = webdriver.Chrome(chrome_driver_path, options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    tr_list = soup.select('#container > div.inner-container > div > table > tbody')[0].find_all('tr')
    table_list = []
    total_count = 0
    for html_row in tr_list:
        row = []
        count = 0
        for html_element in html_row.find_all('td'):
            if total_count < 48:
                if 'H' in html_element.getText() :
                    element = year + '-' + month + '-' + day + ' ' + html_element.getText()[-3:-1] + ':00:00'
                    count = 0
                    row.append(element)
                    total_count = total_count + 1
                    # print('total_count : %d'%total_count)

                else:
                    count = count + 1
                    if (count == 5):
                        element = html_element.getText()
                        row.append(element)
                        total_count = total_count + 1

        table_list.append(row)
    return table_list

def join(df):

    path = os.getcwd()
    astro = pd.read_csv(path+'/../los_df.csv')
    # print(df)
    # print(astro)

    all = astro.merge(df, how="inner", on=None, sort=False)
    return all


def to_db(pdf: pd.DataFrame, db_type='mysql', **kwargs):
    if db_type == 'mysql':
        from sqlalchemy import create_engine
        args = (kwargs['username'], kwargs['passwd'], kwargs['host'], kwargs['port'], kwargs['db_name'])
        engine = create_engine('mysql+pymysql://%s:%s@%s:%d/%s' % args, encoding='utf-8')
        conn = engine.connect()

        # db insert
        pdf.to_sql(name=kwargs['table_name'], con=engine, if_exists='append', index=False)

        conn.close()


def weather(year, month, day):
    print("[일일 천안 기온 수집]")
    table = cralwer_temp(make_URL(year, month, day), year, month, day)
    df = pd.DataFrame(table,columns =['Date', 'kma_temp'])
    # print(df)
    los_temp = join(df)
    print(los_temp)
    path = os.getcwd() + '/../total.csv'
    los_temp.to_csv(path, sep=',', na_rep='NaN', float_format='%.2f', \
              columns=['Date', 'kasi_azimuth', 'kasi_azimuth_text', 'kasi_altitude', 'kasi_altitude_text','kma_temp'], index=False)

    to_db(los_temp, username='root', passwd='defacto8*jj', host='210.102.142.14', port=3306,
              db_name='naturallight', table_name='row_altitude')
    print("[3단계 완료]")



if __name__ == '__main__':
    year = input("year(yyyy) :")
    month = input("month(mm) :")
    day = input("day(dd) :")
    weather(year, month, day)



