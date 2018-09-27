#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 下载指数相关数据
# @Filename: IndexDataHandler
# @Date:   : 2018-09-25 17:45
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from configparser import ConfigParser
import src.settings as SETTINGS
from src.util.utils import Utils
from src.util.dataapi.CDataHandler import CDataHandler
import os
import datetime
import pandas as pd


# def load_index_cons(end_date=datetime.date.today()):
#     """下载指数成分股数据"""
#     cfg = ConfigParser()
#     cfg.read('config.ini')
#     index_cons_path = os.path.join(SETTINGS.FACTOR_DB_PATH, cfg.get('index_cons', 'data_path'))
#     index_codes = cfg.get('index_cons', 'index_codes').split(',')
#     start_date = cfg.get('index_cons', 'start_date')
#     end_date = Utils.datetimelike_to_str(end_date, dash=False)
#     for index_code in index_codes:
#         df_index_cons = CDataHandler.DataApi.get_index_cons(index_code, start_date, end_date)
#         df_index_cons['con_code'] = df_index_cons['con_code'].apply(lambda x: Utils.code_to_symbol(x))
#         df_index_cons['out_date'] = df_index_cons['out_date'].apply(lambda x: '99999999' if len(x) == 0 else x)
#         df_index_cons['index_code'] = Utils.code_to_symbol(df_index_cons.iloc[0]['index_code'][:6], True)
#
#         data_path = os.path.join(index_cons_path, '%s.csv' % Utils.code_to_symbol(index_code, True))
#         df_index_cons.to_csv(data_path, index=False)

def load_index_cons():
    """导入指数成分股数据"""
    cfg = ConfigParser()
    cfg.read('config.ini')
    raw_data_path = cfg.get('index_cons', 'raw_data_path')
    index_codes = cfg.get('index_cons', 'index_codes').split(',')
    cons_data_path = cfg.get('index_cons', 'data_path')
    for index_code in index_codes:
        df_raw_data = pd.read_csv(os.path.join(raw_data_path, '%s.csv' % index_code), header=0, names=['date', 'code', 'name', 'status'])
        df_raw_data.sort_values(by='date', inplace=True)

        df_index_cons = pd.DataFrame(columns=['code', 'in_date', 'out_date'])
        for _, index_con in df_raw_data.iterrows():
            index_con['code'] = Utils.code_to_symbol(index_con['code'])
            if index_con['status'] == '纳入':
                if index_con['code'] in df_index_cons['code'].values:
                    if df_index_cons[df_index_cons['code']==index_con['code']].iloc[-1]['out_date'] == '9999-12-31':
                        raise ValueError("指数%s成份股变动有误:%s,%s,%s,%s" % (index_code, index_con['date'], index_con['code'], index_con['name'], index_con['status']))
                df_index_cons = df_index_cons.append({'code': index_con['code'], 'in_date': index_con['date'], 'out_date': '9999-12-31'}, ignore_index=True)
            elif index_con['status'] == '剔除':
                if index_con['code'] not in df_index_cons['code'].values:
                    raise ValueError("指数%s成份股变动有误:%s,%s,%s,%s" % (index_code, index_con['date'], index_con['code'], index_con['name'], index_con['status']))
                else:
                    if df_index_cons[df_index_cons['code']==index_con['code']].iloc[-1]['out_date'] != '9999-12-31':
                        raise ValueError("指数%s成份股变动有误:%s,%s,%s,%s" % (index_code, index_con['date'], index_con['code'], index_con['name'], index_con['status']))
                idx = df_index_cons[df_index_cons['code']==index_con['code']].index[-1]
                df_index_cons.loc[idx, 'out_date'] = index_con['date']

        df_index_cons.to_csv(os.path.join(SETTINGS.FACTOR_DB_PATH, cons_data_path, '%s.csv' % Utils.code_to_symbol(index_code,True)), index=False)


if __name__ == '__main__':
    pass
    load_index_cons()
