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
import statsmodels.api as sm
import cvxpy as cvx
import datetime
import src.settings as SETTINGS
import src.alphamodel.alphafactors.cons as alphafactor_ct
import src.riskmodel.riskfactors.cons as riskfactor_ct
import src.alphamodel.cons as alphamodel_ct
from src.util.utils import Utils
from src.riskmodel.RiskModel import Barra
from src.portfolio.portfolio import CWeightHolding, CPortfolio
from src.alphamodel.alphafactors import *


def test_alpha_factor(factor_name, start_date, end_date):
    """
    alpha因子检验
    Parameters:
    --------
    :param factor_name: str
        因子名称, e.g: SmartMoney
    :param start_date: datetime-like, str
        开始日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param end_date: datetime-like, str
        结束日期, e.g: YYYY-MM-DD, YYYYMMDD
    :return:
    --------
    """
    # 计算因子载荷值
    _calc_alphafactor_loading(start_date=start_date, end_date=end_date, factor_name=factor_name, multi_proc=SETTINGS.CONCURRENCY_ON, test=True)

    # 计算因子正交化后的因子载荷
    _calc_Orthogonalized_factorloading(factor_name=factor_name, start_date=start_date, end_date=end_date, month_end=True, save=True)

    # 计算最小波动纯因子组合
    _calc_MVPFP(factor_name=factor_name, start_date=start_date, end_date=end_date, month_end=True, save=True)


def _calc_alphafactor_loading(start_date, end_date=None, factor_name=None, multi_proc=False, test=False):
    """
    计算alpha因子因子载荷值(原始载荷值及去极值标准化后载荷值)
    Parameters:
    --------
    :param start_date: datetime-like, str
        开始日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param end_date: datetime-like, str, 默认为None
        结束日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param factor_name: str, 默认为None
        alpha因子名称, e.g: SmartMoney
        factor_namea为None时, 计算所有alpha因子载荷值; 不为None时, 计算指定alpha因子的载荷值
    :param multi_proc: bool, 默认为None
        是否进行并行计算
    :param test: bool, 默认为False
        是否是进行因子检验
    :return: 保存因子载荷值(原始载荷值及去极值标准化后的载荷值)
    """
    # param_cons = eval('alphafactor_ct.'+factor_name.upper() + '_CT')
    start_date = Utils.to_date(start_date)
    if end_date is None:
        trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
    else:
        end_date = Utils.to_date(end_date)
        trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)

    for calc_date in trading_days_series:
        if factor_name is None:
            for alphafactor_name in alphafactor_ct.ALPHA_FACTORS:
                CAlphaFactor = eval(alphafactor_name+'()')
                CAlphaFactor.calc_factor_loading(calc_date, month_end=True, save=True, multi_proc=multi_proc)
        else:
            if (not test) and (factor_name not in alphafactor_ct.ALPHA_FACTORS):
                raise ValueError("alpha因子类: %s, 不存在." % factor_name)
            CAlphaFactor = eval(factor_name + '()')
            CAlphaFactor.calc_factor_loading(calc_date, month_end=True, save=True, multi_proc=multi_proc)


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
    :return: dict
    --------
        因子经正交化后的因子载荷
        0. date, 为计算日期的下一个交易日
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

    CRiskModel = Barra()
    orthog_factorloading = {}
    for calc_date in trading_days_series:
        if month_end and (not Utils.is_month_end(calc_date)):
            continue

        # 读取目标因子原始载荷经标准化后的载荷值
        target_factor_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+factor_name.upper()+'_CT')['db_file'], 'standardized', factor_name)
        df_targetfactor_loading = Utils.read_factor_loading(target_factor_path, Utils.datetimelike_to_str(calc_date, dash=False), drop_na=True)
        df_targetfactor_loading.drop(columns='date', inplace=True)
        df_targetfactor_loading.rename(columns={'factorvalue': factor_name}, inplace=True)

        # 读取风险模型中的风格因子载荷矩阵
        df_stylefactor_loading = CRiskModel.get_StyleFactorloading_matrix(calc_date)
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
        Y = arr_targetfactor_loading
        X = sm.add_constant(arr_stylealphafactor_loading)
        model = sm.OLS(Y, X)
        results = model.fit()

        datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
        orthog_factorloading = {'date': [datelabel]*len(results.resid), 'id': df_factorloading.index.tolist(), 'factorvalue': results.resid}

        # 保存正交化后的因子载荷
        if save:
            orthog_factorloading_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+factor_name.upper()+'_CT')['db_file'], 'orthogonalized', factor_name)
            Utils.factor_loading_persistent(orthog_factorloading_path, Utils.datetimelike_to_str(calc_date, dash=False), orthog_factorloading, ['date', 'id', 'factorvalue'])

    return orthog_factorloading


def _get_factorloading(factor_name, date, factor_type):
    """
    读取个股因子载荷数据
    Parameters:
    --------
    :param factor_name: str
        alpha因子名称, e.g: SmartMoney
    :param date: datetime-like, str
        日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param factor_type: str
        因子类型:
        'raw': 原始因子, 'standardized': 去极值标准化后的因子, 'orthogonalized': 正交化后的因子
    :return: pd.DataFrame
    --------
        因子载荷向量数据
        0. date: 日期
        1. id: 证券代码
        2. factorvalue: 因子值
    """
    date = Utils.datetimelike_to_str(date, dash=False)
    factorloading_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+factor_name.upper()+'.CT')['db_file'], factor_type, factor_name)
    df_factorloading = Utils.read_factor_loading(factorloading_path, date, drop_na=True)
    return df_factorloading


def _calc_MVPFP(factor_name, start_date, end_date=None, month_end=True, save=False):
    """
    构建目标因子的最小波动纯因子组合(Minimum Volatility Pure Factor Portfolio, MVPFP)
    Parameters:
    --------
    :param factor_name: str
        alpha因子名称, e.g: SmartMoney
    :param start_date: datetime-like, str
        开始日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param end_date: datetime-like, str, 默认为None
        结束日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param month_end: bool, 默认为True
        是否只计算月末日期的因子载荷
    :param save: bool, 默认为False
        是否保存计算结果
    :return: CWeightHolding类
        最小波动纯因子组合权重数据
    --------
    具体优化算法:暴露1单位目标因子敞口, 同时保持其余所有风险因子的敞口为0, 并具有最小预期波动率的组合
    Min: W'VW
    s.t. W'X_beta = 0
         W'x_target = 1
    其中: W: 最小波动纯因子组合对应的权重
         V: 个股协方差矩阵
         X_beta: 个股风格因子载荷矩阵
         x_target: 个股目标因子载荷向量
    """
    start_date = Utils.to_date(start_date)
    if end_date is None:
        trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
    else:
        end_date = Utils.to_date(end_date)
        trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)

    CRiskModel = Barra()
    mvpfp_holding = CWeightHolding()
    for calc_date in trading_days_series:
        if month_end and (not Utils.is_month_end(calc_date)):
            continue
        # 取得/计算calc_date的个股协方差矩阵数据
        stock_codes, arr_stocks_covmat = CRiskModel.calc_stocks_covmat(calc_date)
        # 取得个股风格因子载荷矩阵数据
        df_stylefactor_loading = CRiskModel.get_StyleFactorloading_matrix(calc_date)
        # df_stylefactor_loading.set_index('code', inplace=True)
        # df_stylefactor_loading = df_stylefactor_loading.loc[stock_codes]    # 按个股顺序重新排列
        # arr_stylefactor_loading = np.array(df_stylefactor_loading)
        # 取得个股目标因子载荷向量数据(正交化后的因子载荷)
        df_targetfactor_loading = _get_factorloading(factor_name, calc_date, alphafactor_ct.FACTORLOADING_TYPE['ORTHOGONALIZED'])
        df_targetfactor_loading.drop(columns='date', inplace=True)
        df_targetfactor_loading.rename(columns={'id': 'code', 'factorvalue': factor_name}, inplace=True)

        df_factorloading = pd.merge(left=df_stylefactor_loading, right=df_targetfactor_loading, how='inner', on='code')
        df_factorloading.set_index('code', inplace=True)

        df_stylefactor_loading = df_factorloading.loc[stock_codes, riskfactor_ct.STYLE_RISK_FACTORS]
        arr_stylefactor_laoding = np.array(df_stylefactor_loading)

        df_targetfactor_loading = df_factorloading.loc[stock_codes, factor_name]
        arr_targetfactor_loading = np.array(df_targetfactor_loading)

        # 优化计算最小波动纯因子组合权重
        V = arr_stocks_covmat
        X_beta = arr_stylefactor_laoding
        x_target = arr_targetfactor_loading
        N = len(stock_codes)
        w = cvx.Variable((N, 1))
        risk = cvx.quad_form(w, V)
        constraints = [cvx.matmul(w.T * X_beta) == 0,
                       cvx.matmul(w.T * x_target) == 1]
        prob = cvx.Problem(cvx.Minimize(risk), constraints)
        prob.solve()
        if prob.status == cvx.OPTIMAL:
            datelabel = Utils.datetimelike_to_str(calc_date, dash=False)
            df_holding = pd.DataFrame({'date': [datelabel]*len(stock_codes), 'code': stock_codes, 'weight': w.value})
            mvpfp_holding.from_dataframe(df_holding)
            if save:
                holding_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+factor_name.upper()+'.CT')['db_file'], 'mvpfp', '{}_{}.csv'.format(factor_name, datelabel))
                mvpfp_holding.save_data(holding_path)
        else:
            raise cvx.SolverError("%s优化计算%s最小纯因子组合失败。" % (Utils.datetimelike_to_str(calc_date), factor_name))

    return mvpfp_holding


# TODO _calc_alphafactor_performance()
def _calc_mvpfp_performance(factor_name, start_date, end_date):
    """
    计算最小波动纯因子组合的绩效
    Parameters:
    --------
    :param factor_name: str
        因子名称, e.g: SmartMoney
    :param start_date: datetime-like, str
        开始日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param end_date: datetime-like, str
        结束日期, e.g: YYYY-MM-DD, YYYYMMDD
    :return:
    """
    start_date = Utils.to_date(start_date)
    end_date = Utils.to_date(end_date)
    # 读取mvpfp组合持仓数据, 构建Portfolio
    mvpfp_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+factor_name.upper()+'.CT')['db_file'], 'mvpfp')
    if not os.path.isdir(mvpfp_path):
        raise NotADirectoryError("%s因子的mvpfp组合文件夹不存在.")
    mvpfp_port = CPortfolio('weight_holding')
    for mvpfp_filename in os.listdir(mvpfp_path):
        if os.path.splitext(mvpfp_filename)[1] != '.csv':
            continue
        mvpfp_date = Utils.to_date(mvpfp_filename.split('.')[0])
        if mvpfp_date < start_date or mvpfp_date > end_date:
            continue
        mvpfp_filepath = os.path.join(mvpfp_path, mvpfp_filename)
        mvpfp_port.load_holdings_fromfile(mvpfp_filepath)
    # 遍历持仓数据, 计算组合绩效
    df_daily_performance = pd.DataFrame(columns=alphamodel_ct.FACTOR_PERFORMANCE_HEADER['daily_performance'])
    df_daily_performance.loc[0, 'port_daily_ret'] = 0.0
    df_daily_performance.loc[0, 'bnk_daily_ret'] = 0.0
    df_daily_performance.loc[0, 'hedge_daily_ret'] = 0.0
    df_daily_performance.loc[0, 'port_nav'] = 1.0
    df_daily_performance.loc[0, 'bnk_nav'] = 1.0
    df_daily_performance.loc[0, 'hedge_nav'] = 1.0
    df_daily_performance.loc[0, 'port_accu_ret'] = 0.0
    df_daily_performance.loc[0, 'bnk_accu_ret'] = 0.0
    df_daily_performance.loc[0, 'hedge_accu_ret'] = 0.0
    mvpfp_holdings = mvpfp_port.holdings
    prev_holdingdate = curr_holding_date = None
    holding_dates = list(mvpfp_holdings.keys())
    df_daily_performance.loc[0, 'date'] = holding_dates[0]
    if end_date > holding_dates[-1]:
        holding_dates += [end_date]
    mvpfp_daily_performance = pd.Series(index=alphamodel_ct.FACTOR_PERFORMANCE_HEADER['daily_performance'])
    for holding_date in holding_dates:
        prev_holdingdate = curr_holding_date
        curr_holding_date = holding_date
        if prev_holdingdate is None:
            continue
        holding_data = mvpfp_holdings[prev_holdingdate]
        trading_days_series = Utils.get_trading_days(start=prev_holdingdate+datetime.timedelta(days=1), end=curr_holding_date)
        for calc_date in trading_days_series:
            mvpfp_daily_performance['date'] = calc_date
            daily_ret = 0
            for _, holding in holding_data.holding.iterrows():
                ret = Utils.calc_interval_ret(holding['code'], start=trading_days_series[0], end=calc_date)
                if ret is not None:
                    daily_ret += ret * holding['weight']
            mvpfp_daily_performance['port_daily_ret'] = daily_ret
            mvpfp_daily_performance['bnk_daily_ret'] = Utils.calc_interval_ret(alphamodel_ct.BENCHMARK, start=trading_days_series[0], end=calc_date)
            mvpfp_daily_performance['hedge_daily_ret'] = mvpfp_daily_performance['port_daily_ret'] - mvpfp_daily_performance['bnk_daily_ret']

            df_daily_performance = df_daily_performance.append(mvpfp_daily_performance, ignore_index=True)

    for k in range(1, len(df_daily_performance)):
        df_daily_performance.loc[k, 'port_nav'] = df_daily_performance.loc[k-1, 'port_nav'] * (1 + df_daily_performance.loc[k, 'port_daily_ret'])
        df_daily_performance.loc[k, 'bkn_nav'] = df_daily_performance.loc[k-1, 'bnk_nav'] * (1 + df_daily_performance.loc[k, 'bnk_daily_ret'])
        df_daily_performance.loc[k, 'hedge_nav'] = df_daily_performance.loc[k-1, 'hedge_nav'] * (1 + df_daily_performance.loc[k, 'hedge_daily_ret'])
        df_daily_performance.loc[k, 'port_accu_ret'] = df_daily_performance.loc[k, 'port_nav'] - 1
        df_daily_performance.loc[k, 'bnk_accu_ret'] = df_daily_performance.loc[k, 'bnk_nav'] - 1
        df_daily_performance.loc[k, 'hedge_accu_ret'] = df_daily_performance.loc[k, 'hedge_nav'] - 1


# TODO _save_mvpfp_performance()
def _save_mvpfp_performance(performance_data, performance_filepath, performance_type, save_type):
    """
    保存最小波动纯因子组合的绩效数据
    Parameters:
    --------
    :param performance_data: pd.DataFrame
        绩效数据
    :param performance_filepath:
    :param performance_type:
    :param save_type:
    :return:
    """


if __name__ == '__main__':
    pass

