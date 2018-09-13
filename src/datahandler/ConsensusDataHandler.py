#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 从一致预期原始数据文件中导入一致预期数据至一致数据库
# @Filename: ConsensusDataHandler
# @Date:   : 2018-05-15 17:55
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from configparser import ConfigParser
import os
import pandas as pd
from src.util.utils import Utils
import src.settings as SETTINGS

def load_predictedearning_data(date=None):
    """
    导入股票盈利预期(未来12个月)原始数据至因子数据库
    Parameters:
    --------
    :param date: int, str, 默认为None
        导入盈利预期数据对应的交易日期, 格式: YYYY-MM-DD
        若为None则导入总体数据(从PredictedEarnings.csv文件导入);
        若指定交易日期，则导入对应日期的盈利预期数据(从PredictedEarnings_YYYYMMDD.csv文件导入)
    :return:
    """
    cfg = ConfigParser()
    cfg.read('config.ini')
    factor_db_path = cfg.get('factor_db', 'db_path')
    predictedearnings_data_path = cfg.get('consensus_data', 'predictedearnings_path')
    # if date is None:
    #     # 如果日期为None, 从PredictedEarnings.csv文件导入盈利预期数据
    #     raw_data_path = os.path.join(cfg.get('consensus_data', 'consensus_raw_path'), 'PredictedEarnings.csv')
    #     df_predictedearnings = pd.read_csv(raw_data_path, header=0, index_col=0)
    #     for strdate in df_predictedearnings.columns.values.tolist():
    #         print('Loading Predicted Earnings data of {}.'.format(strdate))
    #         predictedearnings_data = df_predictedearnings[strdate]
    #         predictedearnings_data = predictedearnings_data[predictedearnings_data.abs() > 0]
    #         predictedearnings_data.rename(index='predicted_earnings', inplace=True)
    #         dst_data_path = os.path.join(factor_db_path, predictedearnings_data_path, 'predictedearnings_{}.csv'.format(strdate))
    #         predictedearnings_data.to_csv(dst_data_path, index=True, header=True)
    # else:
    #     # 如果指定交易日期, 从PredictedEarnings_YYYYMMDD.csv文件导入盈利预期数据
    #     str_date = date.replace('-', '')
    #     print('Loaidng Predicted Earning data of {}.'.format(str_date))
    #     raw_data_path = os.path.join(cfg.get('consensus_data', 'consensus_raw_path'), 'PredictedEarnings_{}.csv'.format(str_date))
    #     df_predictedearnings = pd.read_csv(raw_data_path, header=0, index_col=0)
    #     df_predictedearnings = df_predictedearnings[df_predictedearnings.abs() > 0]
    #     df_predictedearnings.rename(index='predicted_earnings', inplace=True)
    #     dst_data_path = os.path.join(factor_db_path, predictedearnings_data_path, 'predictedearnings_{}.csv'.format(str_date))
    #     df_predictedearnings.to_csv(dst_data_path, index=True, header=True)

    if date is None:
        raw_data_path = os.path.join(cfg.get('consensus_data', 'consensus_raw_path'), 'predicted_earnings.csv')
    else:
        str_date = str(date).replace('-', '')
        raw_data_path = os.path.join(cfg.get('consensus_data', 'consensus_raw_path'), 'predicted_earnings_{}.csv'.format(str_date))
    df_predictedearnings = pd.read_csv(raw_data_path, header=0, index_col=0, encoding=SETTINGS.DATA_ENCODING_TYPE)
    for strdate in df_predictedearnings.columns.values.tolist():
        print('Loading Predicted Earnings data of {}.'.format(strdate))
        predictedearnings_data = df_predictedearnings[strdate]
        predictedearnings_data = predictedearnings_data[predictedearnings_data.abs() > 0]
        predictedearnings_data.rename(index='predicted_earnings', inplace=True)
        dst_data_path = os.path.join(factor_db_path, predictedearnings_data_path, 'predictedearnings_{}.csv'.format(strdate))
        predictedearnings_data.to_csv(dst_data_path, index=True, header=True, encoding=SETTINGS.DATA_ENCODING_TYPE)

def load_predictedgrowth_data(date=None):
    """
    导入股票预期增长率原始数据至因子数据库
    Parameters:
    --------
    :param date: int, str, 默认为None
        导入预期增长数据对应的日期（可以为日、月、年）, 格式: YYYY-MM-DD/YYYYMMDD, YYYY-MM/YYYYMM, YYYY
    :return:
    """
    cfg = ConfigParser()
    cfg.read('config.ini')
    factor_db_path = cfg.get('factor_db', 'db_path')
    predictedgrowth_data_path = cfg.get('consensus_data', 'predictedgrowth_path')
    if date is None:
        raw_yoydata_path = os.path.join(cfg.get('consensus_data', 'consensus_raw_path'), 'consensus_growth_yoy.csv')
        raw_cagrdata_path = os.path.join(cfg.get('consensus_data', 'consensus_raw_path'), 'consensus_growth_cagr.csv')
    else:
        str_date = str(date).replace('-', '')
        raw_yoydata_path = os.path.join(cfg.get('consensus_data', 'consensus_raw_path'), 'consensus_growth_yoy_{}.csv'.format(str_date))
        raw_cagrdata_path = os.path.join(cfg.get('consensus_data', 'consensus_raw_path'), 'consensus_growth_cagr_{}.csv'.format(str_date))
    df_growth_yoy = pd.read_csv(raw_yoydata_path, header=0, index_col=0, encoding=SETTINGS.DATA_ENCODING_TYPE)
    df_growth_cagr = pd.read_csv(raw_cagrdata_path, header=0, index_col=0, encoding=SETTINGS.DATA_ENCODING_TYPE)
    if df_growth_cagr.columns.values.tolist() != df_growth_yoy.columns.values.tolist():
        print('columns of yoy data is not same to cagr data!')
        return
    for strdate in df_growth_yoy.columns.values.tolist():
        print('Loading consensus growth data of {}.'.format(strdate))
        yoy_data = df_growth_yoy[strdate]
        yoy_data = yoy_data[yoy_data.abs() > 0]
        yoy_data.rename('growth_1y', inplace=True)
        cagr_data = df_growth_cagr[strdate]
        cagr_data = cagr_data[cagr_data.abs() > 0]
        cagr_data.rename('growth_2y', inplace=True)
        df_growth_data = pd.concat([yoy_data, cagr_data], axis=1)
        df_growth_data.index.name = 'code'
        dst_data_path = os.path.join(factor_db_path, predictedgrowth_data_path, 'consensus_growth_data_{}.csv'.format(strdate))
        df_growth_data.to_csv(dst_data_path, na_rep='NaN', encoding=SETTINGS.DATA_ENCODING_TYPE)


if __name__ == '__main__':
    pass
    # load_predictedearning_data(201808)
    # load_predictedgrowth_data(201808)

    # ----------填充predicted earnings数据-----------------
    # start_date = '2018-06-29'
    # end_date = '2018-08-31'
    # trading_days_series = Utils.get_trading_days(start_date, end_date)
    # predictedearning_data_path = '/Volumes/DB/FactorDB/ElementaryFactor/consensus_data/predicted_earnings'
    # df_predictedearning = pd.DataFrame()
    # for trading_day in trading_days_series:
    #     file_path = os.path.join(predictedearning_data_path, 'predictedearnings_{}.csv'.format(Utils.datetimelike_to_str(trading_day, dash=False)))
    #     if os.path.isfile(file_path):
    #         df_predictedearning = pd.read_csv(file_path, header=0)
    #     else:
    #         if not df_predictedearning.empty:
    #             print('save predicted earning data of {}.'.format(Utils.datetimelike_to_str(trading_day, dash=True)))
    #             df_predictedearning.to_csv(file_path, index=False)
    # ---------------------------------------------------

    # ----------填充predicted growth数据-----------------
    # start_date = '2018-06-29'
    # end_date = '2018-08-31'
    # trading_days_series = Utils.get_trading_days(start_date, end_date)
    # predicted_growth_path = '/Volumes/DB/FactorDB/ElementaryFactor/consensus_data/growth_data'
    # df_predictedgrowth = pd.DataFrame()
    # for trading_day in trading_days_series:
    #     file_path = os.path.join(predicted_growth_path, 'consensus_growth_data_{}.csv'.format(Utils.datetimelike_to_str(trading_day, dash=False)))
    #     if os.path.isfile(file_path):
    #         df_predictedgrowth = pd.read_csv(file_path, header=0)
    #     else:
    #         if not df_predictedgrowth.empty:
    #             print('save predicted growth data of {}.'.format(Utils.datetimelike_to_str(trading_day, dash=True)))
    #             df_predictedgrowth.to_csv(file_path, index=False)
    # ---------------------------------------------------
