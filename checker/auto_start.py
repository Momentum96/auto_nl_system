from step1_location_of_sun import crawler_los as s1
from step2_time_of_sunRS import crawler_tosRS as s2
from step3_hourly_temperature import crawler_temp as s3
from step4_casnl_exporter import cas_exporter as s4


if __name__ == '__main__':
    print("start_date")
    year = input("year(yyyy) :")
    month = input("month(mm) :")
    day = input("day(dd) :")
    day2 = input("day2(dd) :")

    for i in range(int(day), int(day2)):
        if i < 10:
            # s1.Los(year, month, '0'+str(i))
            # s2.tosRS(year, month, '0'+str(i))
            # s3.weather(year, month, '0'+str(i))
            s4.exporter_process(year, month, '0'+str(i))
        else:
            # s1.Los(year, month, str(i))
            # s2.tosRS(year, month, str(i))
            # s3.weather(year, month, str(i))
            s4.exporter_process(year, month, str(i))
    # 5단계는 기상관측기 pc에서 돌아가는방식으로 따로 사용하지 않음.