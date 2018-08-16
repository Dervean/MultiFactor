#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: alpha模型文件
# @Filename: AlphaModel
# @Date:   : 2018-08-09 01:57
# @Author  : YuJun
# @Email   : yujun_mail@163.com

import os
import pandas as pd
import numpy as np
import src.settings as SETTINGS
import src.alphamodel.alphafactors.cons as alphafactor_ct
from src.util.utils import Utils
from src.riskmodel.RiskModel import BarraModel


def _calc_Orthogonalized_factorloading(factor_name, start_date, end_date=None, month_end=True, save=False):
    """
    计算alpha因子经正交化后的因子载荷
    Parameters:
    --------
    :param factor_name: str
        alpha因子名称, e.g: SmartMoney
    :param start_date: datetime-like, str
        开始日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param end_date: datetime-like, str, 默认None
        结束日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param month_end: bool, 默认True
        是否只计算月末日期的因子载荷
    :param save: bool, 默认False
        是否保存计算结果
    :return: pd.DataFrame
    --------
        因子经正交化后的因子载荷
        0. 日期, 为计算日期的下一个交易日
        1. id, 证券代码
        2. factorvalue, 因子载荷
        如果end_date=None，返回start_date对应的因子载荷数据
        如果end_date!=None，返回最后一天的对应的因子载荷数据
        如果没有计算数据，返回None
    """
    start_date = Utils.to_date(start_date)
    if end_date is not None:
        end_date = Utils.to_date(end_date)
        trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
    else:
        trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)

    riskmodel = BarraModel()
    for calc_date in trading_days_series:
        if month_end and (not Utils.is_month_end(calc_date)):
            continue

        # 读取目标因子原始载荷经标准化后的载荷值
        target_factor_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+factor_name.upper()+'_CT')['db_file'], 'standardized', factor_name)
        df_targetfactor_loading = Utils.read_factor_loading(target_factor_path, Utils.datetimelike_to_str(calc_date, dash=False), drop_na=True)
        df_targetfactor_loading.drop(columns='date', inplace=True)
        df_targetfactor_loading.rename(columns={'factorvalue': factor_name}, inplace=True)

        # 读取风险模型中的风格因子载荷矩阵
        df_stylefactor_loading = riskmodel.get_StyleFactorloading_matrix(calc_date)
        df_stylefactor_loading.renmae(columns={'code': 'id'}, inplace=True)

        # 读取alpha因子载荷矩阵数据(经正交化后的载荷值)
        df_alphafactor_loading = pd.DataFrame()
        for alphafactor_name in alphafactor_ct.ALPHA_FACTORS:
            if alphafactor_name == factor_name:
                break
            factorloading_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+alphafactor_name.upper()+'_CT')['db_file'], 'orthogonalized', alphafactor_name)
            factor_loading = Utils.read_factor_loading(factorloading_path, Utils.datetimelike_to_str(calc_date, dash=False), drop_na=True)
            factor_loading.drop(columns='date', inplace=True)
            factor_loading.rename(columns={'factorvalue': alphafactor_name}, inplace=True)

            if df_alphafactor_loading.empty:
                df_alphafactor_loading = factor_loading
            else:
                df_alphafactor_loading = pd.merge(left=df_alphafactor_loading, right=factor_loading, how='inner', on='id')

        # 合并目标因子载荷、风格因子载荷与alpha因子载荷
        df_factorloading = pd.merge(left=df_targetfactor_loading, right=df_stylefactor_loading, how='inner', on='id')
        if not df_alphafactor_loading.empty:
            df_factorloading = pd.merge(left=df_stylefactor_loading, right=df_alphafactor_loading, how='inner', on='id')

        # 构建目标因子载荷向量、风格与alpha因子载荷矩阵
        df_factorloading.set_index('id', inplace=True)
        arr_targetfactor_loading = np.array(df_factorloading[factor_name])
        stylealphafactor_names = df_factorloading.columns.tolist()
        stylealphafactor_names.remove(factor_name)
        arr_stylealphafactor_loading = np.array(df_factorloading[stylealphafactor_names])

        # 将arr_targetfactor_loading对arr_stylealphafactor_loading进行截面回归, 得到的残差即为正交化后的因子载荷


if __name__ == '__main__':
    pass
