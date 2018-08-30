#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 导入股票行业分类数据
# @Filename: IndustryClassifyHandler
# @Date:   : 2018-01-09 03:32
# @Author  : YuJun
# @Email   : yujun_mail@163.com

from configparser import ConfigParser
import os
import csv
import pandas as pd
from pandas import DataFrame
import requests
import re
import json
import datetime
from src.util.utils import Utils


def download_sw_fyjr_classify():
    """通过tushare下载申万非银金融下二级行业的分类数据"""
    df_sw_fyjr = DataFrame()
    sw_fyjr = [['sw2_490100', '证券'], ['sw2_490200', '保险'], ['sw2_490300', '多元金融']]
    for industry_info in sw_fyjr:
        url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=1000&sort=symbol&asc=1&node=%s&symbol=&_s_r_a=page' % industry_info[0]
        text = requests.get(url).text
        reg = re.compile(r'\,(.*?)\:')
        text = reg.sub(r',"\1":', text)
        text = text.replace('"{symbol', '{"symbol')
        text = text.replace('{symbol', '{"symbol"')
        jstr = json.dumps(text)
        js = json.loads(jstr)
        df = pd.DataFrame(pd.read_json(js, dtype={'code': object}), columns=['code','symbol','name','changepercent','trade','open','high','low','settlement','volume','turnoverratio'])
        df['c_name'] = industry_info[1]
        df_sw_fyjr = df_sw_fyjr.append(df[['code', 'name', 'c_name']], ignore_index=True)

    cfg = ConfigParser()
    cfg.read('config.ini')
    file_path = os.path.join(cfg.get('industry_classify', 'raw_data_path'), 'sw_fyjr_classify.csv')
    df_sw_fyjr.to_csv(file_path, index=False)


def load_industry_classify(standard='sw', date=datetime.date.today()):
    """导入个股行业分类数据"""
    cfg = ConfigParser()
    cfg.read('config.ini')
    # 读取申万一级行业信息
    sw_classify_info_path = os.path.join(cfg.get('factor_db', 'db_path'), cfg.get('industry_classify', 'classify_data_path'), 'classify_standard_sw.csv')
    df_sw_classify = pd.read_csv(sw_classify_info_path, names=['ind_code', 'ind_name'], header=0)
    # 读取申万非银金融下二级行业信息
    sw_fyjr_classify_path = os.path.join(cfg.get('industry_classify', 'raw_data_path'), 'sw_fyjr_classify.csv')
    df_sw_fyjr_classify = pd.read_csv(sw_fyjr_classify_path, dtype={'code': object}, header=0)
    # 读取股票最新行业分类原始数据，导入本系统的股票申万一级行业分类数据文件
    # 同时把非银金融一级行业替换成二级行业
    raw_data_path = os.path.join(cfg.get('industry_classify', 'raw_data_path'), 'industry_classify_sw.csv')
    classify_data = [['证券代码', '申万行业代码', '申万行业名称']]
    with open(raw_data_path, 'r', newline='') as f:
        f.readline()
        csv_reader = csv.reader(f, delimiter='\t')
        for row in csv_reader:
            code = 'SH' + row[1] if row[1][0] == '6' else 'SZ' + row[1]
            ind_name = row[0]
            if row[1] in df_sw_fyjr_classify['code'].values:
                ind_name = df_sw_fyjr_classify[df_sw_fyjr_classify.code == row[1]].iloc[0].c_name
            ind_code = df_sw_classify[df_sw_classify.ind_name == ind_name].iloc[0].ind_code
            classify_data.append([code, ind_code, ind_name])
    # 添加退市或暂停交易个股的行业分类数据
    delisted_data_path = os.path.join(cfg.get('factor_db', 'db_path'), cfg.get('industry_classify', 'classify_data_path'), 'delisted_classify_sw.csv')
    with open(delisted_data_path, 'r', newline='') as f:
        f.readline()
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            code = row[0]
            listed_date = row[2]
            delisted_date = row[3]
            ind_code = row[7]
            ind_name = row[8]
            if delisted_date > Utils.datetimelike_to_str(date, dash=False) and listed_date <= Utils.datetimelike_to_str(date, dash=False):
                classify_data.append([code, ind_code, ind_name])
    # 保存股票行业分类文件
    ind_files = ['industry_classify_sw.csv', 'industry_classify_sw_{}.csv'.format(Utils.datetimelike_to_str(date, dash=False))]
    for file_name in ind_files:
        classify_data_path = os.path.join(cfg.get('factor_db', 'db_path'), cfg.get('industry_classify', 'classify_data_path'), file_name)
        with open(classify_data_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(classify_data)
    # 检查退市股票行业分类数据中是否已包含所有的已退市股票
    _check_dlisted_indclassify()


def _check_dlisted_indclassify():
    """检查退市股票行业代码分类"""
    # 读取退市股票行业分类数据
    cfg = ConfigParser()
    cfg.read('config.ini')
    delisted_data_path = os.path.join(cfg.get('factor_db', 'db_path'), cfg.get('industry_classify', 'classify_data_path'), 'delisted_classify_sw.csv')
    df_delisted_indclassify = pd.read_csv(delisted_data_path, header=0)
    # 读取已退市个股基本信息数据
    df_stock_basics = Utils.get_stock_basics(all=True)
    df_delisted_basics = df_stock_basics[df_stock_basics['status'] == 3]
    # 检查退市股票行业分类数据中是否已包含所有的已退市股票
    df_delisted_basics = df_delisted_basics[~df_delisted_basics['symbol'].isin(df_delisted_indclassify['id'].tolist())]
    if ~df_delisted_basics.empty:
        print('\033[1;31;40m个股{}已退市, 需加入退市股票行业分类数据中.\033[0m'.format(str(df_delisted_basics['symbol'].tolist())))


if __name__ == '__main__':
    # _check_dlisted_indclassify()
    # download_sw_fyjr_classify()
    # trading_days_series = Utils.get_trading_days(start='2018-01-01', end='2018-01-09')
    # # for date in trading_days_series:
    # #     print('loading industry classify of {}.'.format(Utils.datetimelike_to_str(date, dash=True)))
    # #     load_industry_classify(date=date)
    #
    # ind_classify_path = '/Volumes/DB/FactorDB/ElementaryFactor/industry_classify/industry_classify_sw.csv'
    # df_ind_classify = pd.read_csv(ind_classify_path, names=['id', 'ind_code', 'ind_name'], header=0)
    # delisted_classify_path = '/Volumes/DB/FactorDB/ElementaryFactor/industry_classify/delisted_classify_sw.csv'
    # df_delisted_classify = pd.read_csv(delisted_classify_path, header=0)
    # for date in trading_days_series:
    #     print('loading industry classify of {}.'.format(Utils.datetimelike_to_str(date, dash=True)))
    #     df_add = df_delisted_classify[(df_delisted_classify['delist_date'] > int(date.strftime('%Y%m%d'))) & (df_delisted_classify['list_date'] <= int(date.strftime('%Y%m%d')))]
    #     if not df_add.empty:
    #         df_dst_classify = df_ind_classify.append(df_add[['id', 'ind_code', 'ind_name']], ignore_index=True)
    #     else:
    #         df_dst_classify = df_ind_classify
    #     dst_path = '/Volumes/DB/FactorDB/ElementaryFactor/industry_classify/industry_classify_sw_{}.csv'.format(Utils.datetimelike_to_str(date, dash=False))
    #     df_dst_classify.rename(columns={'id': '证券名称', 'ind_code': '申万行业代码', 'ind_name': '申万行业名称'}, inplace=True)
    #     df_dst_classify.to_csv(dst_path, index=False)

    # import requests
    # url = 'http://www.swsindex.com/downloadfiles.aspx?swindexcode=SwClass&type=530&columnid=8892'
    # resp = requests.get(url)
    # print(resp.content)

    load_industry_classify('sw', '2018-08-24')
