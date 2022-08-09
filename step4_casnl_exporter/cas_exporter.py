import os
from datetime import datetime, tzinfo, timezone, timedelta
import pytz
from pymongo import MongoClient
import pandas as pd
from pprint import pprint as pp
from step4_casnl_exporter import CasEntry


def road_mongodb(times, collection):
    client = MongoClient('mongodb://210.102.142.14:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false')
    filter = {
        'datetime': {
            '$gte': datetime(int(times[0][0:4]), int(times[0][5:7]), int(times[0][8:10]), int(times[0][11:13]), int(times[0][14:16]), int(times[0][17:19]), tzinfo= timezone.utc),
            '$lte': datetime(int(times[1][0:4]), int(times[1][5:7]), int(times[1][8:10]), int(times[1][11:13]), int(times[1][14:16]), int(times[1][17:19]), tzinfo= timezone.utc)
        }
    }
    sort = list({
                    'datetime': -1
                }.items())

    result = client['nl_witlab'][collection].find(
        filter=filter,
        sort=sort
    )
    dic_list = []
    dic = dict()
    for i in result:
        dic = i
        dic_list.append(dic)
    #
    # print(dic_list[0]['datetime'])
    # pp(dic_list[0])
    return dic_list


def make_ktc_to_utc(year, month, day):
    ori_date = year+"/"+month+"/"+day
    start_time = ori_date+" 00:00:00"
    end_time = ori_date+" 23:59:59"

    start_time_obj = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
    end_time_obj = datetime.strptime(end_time, '%Y/%m/%d %H:%M:%S')
    start_time_obj = start_time_obj - timedelta(hours=9)
    end_time_obj = end_time_obj - timedelta(hours=9)

    utc_start = start_time_obj.strftime('%Y/%m/%d %H:%M:%S')
    utc_stop = end_time_obj.strftime('%Y/%m/%d %H:%M:%S')
    # print(utc_start)
    # print(utc_stop)
    return [utc_start,utc_stop]


def make_utc_to_ktc(datetime_uct):

    time_obj = datetime.strptime(datetime_uct, '%Y/%m/%d %H:%M:%S')
    korea_time_obj = time_obj + timedelta(hours=9)

    ktc_datetime = korea_time_obj.strftime('%Y/%m/%d %H:%M:%S')
    return ktc_datetime


def make_row_key_val(dic):
    # 지하 4층 높이의 dic이 있다고 가정하면 지하 3층에서 지하 4층의 dic[4층 keys]가 str인지 판별하고 맞으면 key와 val을 저장, 아니면 재귀하도록 짜야할듯.

    keyrow = []
    valrow = []
    # 1차 시도시 자가 검진
    if isinstance(dic, str) or isinstance(dic, float) or isinstance(dic, int):
        keyrow.append("1차 키 미확인")
        valrow.append(dic)
        # print([keyrow, valrow])
        return[keyrow, valrow]

    elif isinstance(dic, dict):
        key_list = list(dic.keys())
        key_list = sorted(key_list, key=str.lower)

        for key in key_list:
            if isinstance(dic[key], str) or isinstance(dic[key], int):
                keyrow.append(key)
                valrow.append(dic[key])
            elif isinstance(dic[key], float):
                keyrow.append(key)
                valrow.append(format(dic[key],"40.20f"))

            else:
                key_val = make_row_key_val(dic[key])
                if isinstance(key_val,list):
                    for temp_key in key_val[0]:
                        keyrow.append(temp_key)
                    for temp_val in key_val[1]:
                        valrow.append(temp_val)

        return [keyrow, valrow]


def mongodb_to_df(dic_list, table):
    if table == 'mongo_cas':
        main_key = 'data'
    elif table == 'mongo_cas_ird':
        main_key = 'sp_ird'

    path = os.getcwd() + '/../' + table + '.csv'
    val_table = []
    df = pd.DataFrame()
    count = 0
    for dic in dic_list:
        val_table = []
        count = count + 1

        data_dic = dic[main_key]
        key_val = make_row_key_val(data_dic)
        val_table.append(key_val[1])
        keys = key_val[0]

        val_table[0].append(dic['datetime'])
        keys.append('datetime')

        if count == 1:
            df = pd.DataFrame(val_table, columns=keys)
        else:
            temp_df = pd.DataFrame(val_table, columns=keys)
            df = df.append(temp_df)
    print(table+"_df load success!")
    return df


def to_db(pdf: pd.DataFrame, db_type='mysql', **kwargs):
    if db_type == 'mysql':
        from sqlalchemy import create_engine
        args = (kwargs['username'], kwargs['passwd'], kwargs['host'], kwargs['port'], kwargs['db_name'])
        engine = create_engine('mysql+pymysql://%s:%s@%s:%d/%s' % args, encoding='utf-8')
        conn = engine.connect()

        # db insert
        pdf.to_sql(name=kwargs['table_name'], con=engine, if_exists='append', index=False)
        conn.close()


def convert_time(temp):
    col = ""
    if temp[-2:] == "AM":
        if temp[0:2] == "12":
            col = "00" + temp[2:-3]
        elif temp[1] == ":":
            col = "0" + temp[0:-3]
        else:
            col = temp[0:-3]

    elif temp[-2:] == "PM":
        if temp[0:2] == "12":
            col = temp[0:-3]
        elif temp[1] == ":":
            col = str(int(temp[0:1]) + 12) + temp[1:-3]
        else:
            col = str(int(temp[0:2]) + 12) + temp[2:-3]

    print(col)
    return col



def exporter_process(year, month, day):
    times = make_ktc_to_utc(year, month, day)
    # 처음 시작때
    cas_dic_list = road_mongodb(times, "cas")
    cas_ird_dic_list = road_mongodb(times, "cas_ird")
    mongo_cas = mongodb_to_df(cas_dic_list,"mongo_cas")
    mongo_cas_ird = mongodb_to_df(cas_ird_dic_list, "mongo_cas_ird")
    mongo_all = mongo_cas.merge(mongo_cas_ird, how="inner", on=None, sort=True)
    path = os.getcwd() + '/../mongo_all.csv'
    mongo_all.to_csv(path, sep=',', na_rep='NaN')

    # 중간시작때 쓰는거
    # path = os.getcwd() + '/../mongo_all.csv'

    mongo_all = pd.read_csv(path)
    mongo_all.rename(columns={'datetime': 'datetime_utc'}, inplace = True)
    mongo_all = CasEntry.convert_time(mongo_all,year, month, day)


    # 컬럼명 통일, mysql 기준으로.
    mongo_all = CasEntry.rename_col(mongo_all)
    mongo_all = CasEntry.change_ratio(mongo_all)
    mongo_all = CasEntry.make_copy_col(mongo_all, month)
    mongo_all = CasEntry.remove_trash(mongo_all)
    natural_tracker = CasEntry.make_natural_tracker(mongo_all)
    cas_uv_ratio = CasEntry.make_cas_uv_ratio(mongo_all)
    cas_wave_ratio = CasEntry.make_cas_wave_ratio(mongo_all)

    path1 = os.getcwd() + '/../cas_wave_ratio.csv'
    cas_wave_ratio.to_csv(path1, sep=',', na_rep='NaN')

    # mysql 에 넣을때
    to_db(natural_tracker, username='root', passwd='defacto8*jj', host='210.102.142.14', port=3306,
          db_name='cas_db', table_name='natural_tracker')
    to_db(cas_uv_ratio, username='root', passwd='defacto8*jj', host='210.102.142.14', port=3306,
          db_name='cas_db', table_name='cas_uv_ratio ')
    to_db(cas_wave_ratio, username='root', passwd='defacto8*jj', host='210.102.142.14', port=3306,
          db_name='cas_db', table_name='cas_wave_ratio')

if __name__ == '__main__':
    # road_casdata()
    year = input("year(yyyy) :")
    month = input("month(mm) :")
    day = input("day(dd) :")
    exporter_process(year, month, day)

