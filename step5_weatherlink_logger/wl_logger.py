import pandas as pd
import pymysql
import threading

def to_db(pdf: pd.DataFrame, db_type='mysql', **kwargs):
    if db_type == 'mysql':
        from sqlalchemy import create_engine
        args = (kwargs['username'], kwargs['passwd'], kwargs['host'], kwargs['port'], kwargs['db_name'])
        engine = create_engine('mysql+pymysql://%s:%s@%s:%d/%s' % args, encoding='utf-8')
        conn = engine.connect()

        # db insert
        pdf.to_sql(name=kwargs['table_name'], con=engine, if_exists='append', index=False)

        conn.close()

def from_db(db_type='mysql', **kwargs):
    if db_type == 'mysql':
        conn = pymysql.connect(host=kwargs['host'], user=kwargs['username'], password=kwargs['passwd'],
                               db=kwargs['db_name'], charset='utf8')

        curs = conn.cursor()
        sql = "select DATE_FORMAT(Date, \"%Y-%m-%d %k:%i:%S\") from row_weather order by Date desc Limit 1"
        curs.execute(sql)
        rows = curs.fetchall()
        conn.close()
        return rows[0][0]

def make_dataframe():
    f = open("C:\\Users\\GAKA\\Desktop\\download.txt", 'r')
    lines = f.readlines()
    count = 0;
    table_list = []
    row = []
    for line in lines:
        count = count + 1
        line = " ".join(line.split())
        if count>3:
            temp_row = line.split(" ")
            temp_count =0
            date =""
            row = []
            for temp in temp_row:
                temp_count = temp_count+1
                if temp_count ==1:
                    date = temp
                    continue
                elif temp_count ==2:
                    if temp[-1] == "a":
                        if temp[0:2] == "12":
                            col = "20" + date + " 00" + temp[2:-1] + ":00"
                        elif temp[1] == ":":
                            col = "20" + date + " 0" + temp[0:-1] + ":00"
                        else:
                            col = "20" + date + " " + temp[0:-1] + ":00"


                    elif temp[-1] == "p":
                        if temp[0:2] == "12":
                            col = "20" + date + " " + temp[0:-1] + ":00"
                        elif temp[1] == ":":
                            col = "20" + date + " " + str(int(temp[0:1])+12) + temp[1:-1] + ":00"
                        else:
                            col = "20" + date + " " + str(int(temp[0:2])+12) + temp[2:-1] + ":00"
                    row.append(col)
                else:
                    row.append(temp)
        # print(count)
        # print(row)
        if row:
            table_list.append(row)
    f.close()
    return table_list



def wl_log():
    table = make_dataframe()
    # print(table)
    df = pd.DataFrame(table, columns=['Date', 'TempOut', 'TempOut_High', 'TempOut_Low', 'HumOut',
                                      'DewPt', 'WindSpeed', 'WindDir', 'WindRun', 'WindSpeed_High', 'WindDir_High',
                                      'WindChill', 'HeatIndex', 'THWIndex', 'THSWIndex', 'Bar', 'Rain',
                                      'RainRate', 'SolarRad', 'SolarEnergy', 'SolarRad_High', 'UVIndex', 'UVDose',
                                      'UVDose_High', 'HeatDD', 'CoolDD', 'TempIn', 'HumIn', 'DewIn',
                                      'HeatIn', 'EMCIn', 'AirInDensity', 'WindET', 'WindSamp', 'WindTx',
                                      'ISSRecept', 'ArcInt'])
    # print(df)
    start_time = from_db(username='witlab', passwd='defacto8*', host='210.102.142.14', port=3306,
              db_name='naturallight', table_name='row_weather')
    print(start_time)
    filted_df=df[(df['Date'] > start_time)]
    print(df)
    print(filted_df)
    to_db(filted_df, username='witlab', passwd='defacto8*', host='210.102.142.14', port=3306,
              db_name='naturallight', table_name='row_weather')
    threading.Timer(3600, wl_log).start()

if __name__ == '__main__':
    wl_log()

