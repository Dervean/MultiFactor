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
import logging
import math
import src.settings as SETTINGS
import src.alphamodel.alphafactors.cons as alphafactor_ct
import src.riskmodel.riskfactors.cons as riskfactor_ct
import src.alphamodel.cons as alphamodel_ct
from src.util.utils import Utils
from src.riskmodel.RiskModel import Barra
from src.portfolio.portfolio import CWeightHolding, CPortfolio
from src.alphamodel.alphafactors.SmartMoney import SmartMoney
from src.alphamodel.alphafactors.APM import APM
from src.alphamodel.alphafactors.IntradayMomentum import IntradayMomentum


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


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

    # 计算最小波动纯因子组合的绩效
    _calc_mvpfp_performance(factor_name=factor_name, start_date=start_date, end_date=end_date)

    # 计算最小波动纯因子组合的汇总绩效
    _calc_mvpfp_summary(factor_name=factor_name, calc_date=end_date)


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
        df_stylefactor_loading.rename(columns={'code': 'id'}, inplace=True)

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
            df_factorloading = pd.merge(left=df_factorloading, right=df_alphafactor_loading, how='inner', on='id')

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
    factorloading_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+factor_name.upper()+'_CT')['db_file'], factor_type, factor_name)
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

### ------------------------------------------------------------------------------ ###
        """
        # 取得/计算calc_date的个股协方差矩阵数据
        df_stocks_covmat = CRiskModel.calc_stocks_covmat(calc_date)
        df_stocks_covmat.reset_index(inplace=True)
        # 取得个股风险因子(含行业因子和风格因子)载荷矩阵数据
        df_riskfactor_loading = CRiskModel.get_factorloading_matrix(calc_date, riskfactor_ct.RISK_FACTORS_NOMARKET)
        # 取得个股目标因子载荷向量数据(正交化后的因子载荷)
        df_targetfactor_loading = _get_factorloading(factor_name, calc_date, alphafactor_ct.FACTORLOADING_TYPE['ORTHOGONALIZED'])
        df_targetfactor_loading.drop(columns='date', inplace=True)
        df_targetfactor_loading.rename(columns={'id': 'code', 'factorvalue': factor_name}, inplace=True)

        # 合并数据, 提取共有个股的数据
        df_factorloading = pd.merge(left=df_riskfactor_loading, right=df_targetfactor_loading, how='inner', on='code')
        df_factorloading = pd.merge(left=df_factorloading, right=df_stocks_covmat, how='inner', on='code')
        df_factorloading.set_index('code', inplace=True)

        df_riskfactor_loading = df_factorloading.loc[:, riskfactor_ct.RISK_FACTORS_NOMARKET]
        arr_riskfactor_laoding = np.array(df_riskfactor_loading)

        df_targetfactor_loading = df_factorloading.loc[:, factor_name]
        arr_targetfactor_loading = np.array(df_targetfactor_loading)

        df_stocks_covmat = df_factorloading.loc[:, df_factorloading.index]
        arr_stocks_covmat = np.array(df_stocks_covmat)

        # 优化计算最小波动纯因子组合权重
        N = len(arr_stocks_covmat)
        V = arr_stocks_covmat
        X_beta = arr_riskfactor_laoding
        x_target = arr_targetfactor_loading.reshape((N, 1))
        w = cvx.Variable(N)
        risk = cvx.quad_form(w, V)
        constraints = [cvx.matmul(w.T, X_beta) == 0,
                       cvx.matmul(w.T, x_target) == 1]
        prob = cvx.Problem(cvx.Minimize(risk), constraints)
        prob.solve()
        if prob.status == cvx.OPTIMAL:
            datelabel = Utils.datetimelike_to_str(calc_date, dash=False)
            df_holding = pd.DataFrame({'date': [datelabel]*N, 'code': df_factorloading.index.tolist(), 'weight': w.value})
            mvpfp_holding.from_dataframe(df_holding)
            if save:
                holding_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+factor_name.upper()+'._CT')['db_file'], 'mvpfp', '{}_{}.csv'.format(factor_name, datelabel))
                mvpfp_holding.save_data(holding_path)
        else:
            raise cvx.SolverError("%s优化计算%s最小纯因子组合失败。" % (Utils.datetimelike_to_str(calc_date), factor_name))
        """
### ------------------------------------------------------------------------------ ###

        # 读取风险模型相关数据: 风险因子暴露矩阵, 风险因子协方差矩阵, 特质波动率方差
        df_riskfactor_loading, arr_covmat, spec_var = CRiskModel.get_riskmodel_data(calc_date, factors=riskfactor_ct.RISK_FACTORS, cov_type='cov', var_type='var')

        # 读取目标因子载荷数据(正交化后的因子载荷)
        df_targetfactor_loading = _get_factorloading(factor_name, calc_date, alphafactor_ct.FACTORLOADING_TYPE['ORTHOGONALIZED'])
        df_targetfactor_loading.drop(columns='date', inplace=True)
        df_targetfactor_loading.rename(columns={'id': 'code', 'factorvalue': factor_name}, inplace=True)

        df_riskfactor_loading.reset_index(inplace=True)
        df_factorloading = pd.merge(left=df_riskfactor_loading, right=df_targetfactor_loading, how='inner', on='code')
        df_factorloading.set_index('code', inplace=True)

        df_riskfactor_loading = df_factorloading[riskfactor_ct.RISK_FACTORS]
        targetfactor_loading = df_factorloading[factor_name]
        spec_var = spec_var[df_factorloading.index]

        n = len(df_riskfactor_loading)          # 个股数量
        F = np.array(df_riskfactor_loading)     # 个股风险因子载荷矩阵
        sigma = arr_covmat                      # 风险因子协方差矩阵
        D = np.diag(spec_var)                   # 个股特质波动率方差矩阵(对角矩阵)
        w = cvx.Variable(n)                     # 个股权重向量
        f = F.T * w
        risk = cvx.quad_form(f, sigma) + cvx.quad_form(w, D)

        x_alpha = np.array(targetfactor_loading)

        constraints = [w*F == 0,
                       w*x_alpha == 1]
        prob = cvx.Problem(cvx.Minimize(risk), constraints)

        # prob = cvx.Problem(cvx.Minimize(cvx.sum(cvx.abs(w*F))), [w*x_alpha == 1])

        prob.solve(verbose=True)
        if prob.status == cvx.OPTIMAL:
            datelabel = Utils.datetimelike_to_str(calc_date, dash=False)
            df_holding = pd.DataFrame({'date': [datelabel]*n, 'code': targetfactor_loading.index.tolist(), 'weight': w.value})
            df_holding.sort_values(by='weight', ascending=False, inplace=True)
            mvpfp_holding.from_dataframe(df_holding)
            if save:
                holding_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('alphafactor_ct.'+factor_name.upper()+'_CT')['db_file'], 'mvpfp', '{}_{}.csv'.format(factor_name, datelabel))
                mvpfp_holding.save_data(holding_path)
        else:
            raise cvx.SolverError("%s优化计算%s最小纯因子组合失败。" % (Utils.datetimelike_to_str(calc_date), factor_name))

    return mvpfp_holding


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
    df_daily_performance = pd.DataFrame(columns=alphamodel_ct.FACTOR_PERFORMANCE_HEADER['daily_performance'])       # 日度绩效
    df_monthly_performance = pd.DataFrame(columns=alphamodel_ct.FACTOR_PERFORMANCE_HEADER['monthly_performance'])   # 月度绩效

    df_daily_performance.loc[0, 'daily_ret'] = 0.0
    df_daily_performance.loc[0, 'nav'] = 1.0
    df_daily_performance.loc[0, 'accu_ret'] = 0.0

    mvpfp_holdings = mvpfp_port.holdings
    prev_holdingdate = curr_holding_date = None
    prevmonth_idx = 0
    holding_dates = list(mvpfp_holdings.keys())
    df_daily_performance.loc[0, 'date'] = holding_dates[0]
    if end_date > holding_dates[-1]:
        holding_dates += [end_date]
    mvpfp_daily_performance = pd.Series(index=alphamodel_ct.FACTOR_PERFORMANCE_HEADER['daily_performance'])
    mvpfp_monthly_performance = pd.Series(index=alphamodel_ct.FACTOR_PERFORMANCE_HEADER['monthly_performance'])
    for holding_date in holding_dates:
        prev_holdingdate = curr_holding_date
        curr_holding_date = holding_date
        if prev_holdingdate is None:
            continue
        prevmonth_idx = df_daily_performance.index[-1]
        holding_data = mvpfp_holdings[prev_holdingdate]
        trading_days_series = Utils.get_trading_days(start=prev_holdingdate+datetime.timedelta(days=1), end=curr_holding_date)
        for calc_date in trading_days_series:
            mvpfp_daily_performance['date'] = calc_date
            daily_ret = 0
            # TODO 增加并行计算个股绩效的功能
            for _, holding in holding_data.holding.iterrows():
                ret = Utils.calc_interval_ret(holding['code'], start=trading_days_series[0], end=calc_date)
                if ret is not None:
                    daily_ret += ret * holding['weight']
            mvpfp_daily_performance['daily_ret'] = daily_ret
            mvpfp_daily_performance['nav'] = df_daily_performance.iloc[-1]['nav'] * (1 + daily_ret)
            mvpfp_daily_performance['accu_ret'] = mvpfp_daily_performance['nav'] - 1

            df_daily_performance = df_daily_performance.append(mvpfp_daily_performance, ignore_index=True)

        mvpfp_monthly_performance['date'] = curr_holding_date
        mvpfp_monthly_performance['monthly_ret'] = df_daily_performance.iloc[-1]['nav'] / df_daily_performance.loc[prevmonth_idx, 'nav'] - 1.0
        df_monthly_performance = df_monthly_performance.append(mvpfp_monthly_performance, ignore_index=True)

    # for k in range(1, len(df_daily_performance)):
    #     df_daily_performance.loc[k, 'nav'] = df_daily_performance.loc[k-1, 'nav'] * (1 + df_daily_performance.loc[k, 'daily_ret'])
    #     df_daily_performance.loc[k, 'accu_ret'] = df_daily_performance.loc[k, 'nav'] - 1

    # 保存数据
    _save_mvpfp_performance(df_daily_performance, factor_name, 'daily', 'a')
    _save_mvpfp_performance(df_monthly_performance, factor_name, 'monthly', 'a')


def _calc_mvpfp_summary(factor_name, calc_date):
    """
    计算最小波动纯因子组合的汇总绩效数据
    Parameters:
    --------
    :param factor_name: str
        alpha因子名称, e.g: SmartMoney
    :param calc_date: datetime-like, str
        计算日期, e.g: YYYY-MM-DD, YYYYMMDD
    :return:
        计算汇总绩效数据, 并保存
    """
    calc_date = Utils.to_date(calc_date)
    dailyperformance_filepath = os.path.join(SETTINGS.FACTOR_DB_PATH,
                                             eval('alphafactor_ct.'+factor_name.uppper()+'.CT')['db_file'],
                                             'performance/performance_daily.csv')
    df_daily_performance = pd.read_csv(dailyperformance_filepath, parse_dates=[0], header=0)
    df_daily_performance = df_daily_performance[df_daily_performance['date'] <= calc_date]

    monthlyperformance_filepath = os.path.join(SETTINGS.FACTOR_DB_PATH,
                                               eval('alphafactor_ct.'+factor_name.upper()+'.CT')['db_file'],
                                               'performance/performance_monthly.csv')
    df_monthly_performance = pd.read_csv(monthlyperformance_filepath, parse_dates=[0], header=0)
    df_monthly_performance = df_monthly_performance[df_monthly_performance['date'] <= calc_date]
    if len(df_monthly_performance) < 12:
        logging.info("alpha因子'%s'的历史月度绩效数据长度小于12个月, 不计算汇总绩效数据." % factor_name)
        return
    summary_performance = pd.Series(index=alphamodel_ct.FACTOR_PERFORMANCE_HEADER['summary_performance'])
    for k in alphamodel_ct.SUMMARY_PERFORMANCE_MONTH_LENGTH:
        if k == 'total':
            daily_performance = df_daily_performance
            monthly_performance = df_monthly_performance
            summary_performance['type'] = k
        else:
            if not isinstance(k, int):
                raise TypeError("计算因子汇总绩效的时间区间类型除了'total'外, 应该为整型.")
            if len(df_monthly_performance) >= k:
                monthly_performance = df_monthly_performance.iloc[-k:]
            else:
                logging.info("alpha因子'%s'的历史月度绩效数据长度小于%d个月, 不予计算该历史时间长度的汇总绩效." % (factor_name, k))
                continue
            daily_performance = df_daily_performance[df_daily_performance['date'] >= monthly_performance.iloc[0]['date']]
            summary_performance['type'] = str(k) + 'm'

        summary_performance['date'] = daily_performance.iloc[-1]['date']
        summary_performance['total_ret'] = daily_performance.iloc[-1]['nav'] / daily_performance.iloc[0]['nav'] - 1.0
        summary_performance['annual_ret'] = math.pow(summary_performance['total_ret']+1, 12/k)
        summary_performance['volatility'] = np.std(daily_performance['daily_ret']) * math.sqrt(250)
        summary_performance['monthly_winrate'] = len(monthly_performance[monthly_performance['monthly_ret'] > 0]) / k
        summary_performance['IR'] = summary_performance['annual_ret'] / summary_performance['volatility']

        fmax_drawdown = 0.0
        for m in range(1, len(daily_performance)):
            fdrawdown = daily_performance.iloc[m]['nav'] / max(daily_performance.iloc[:m]['nav']) - 1.0
            if fdrawdown < fmax_drawdown:
                fmax_drawdown = fdrawdown
        summary_performance['max_drawdown'] = fmax_drawdown

        _save_mvpfp_performance(summary_performance, factor_name, 'summary', 'a')


def _save_mvpfp_performance(performance_data, factor_name, performance_type, save_type):
    """
    保存最小波动纯因子组合的绩效数据
    Parameters:
    --------
    :param performance_data: pd.DataFrame
        绩效数据(包含日度时间序列数据, 月度时间序列数据, summary data)
    :param factor_name: str
        alpha因子名称, e.g: SmartMoney
    :param performance_type: str
        绩效数据类型, 'daily'=日度时间序列数据, 'monthly'=月度时间序列数据, 'summary'=汇总数据
    :param save_type: str
        保存类型, 'a'=新增, 'w'=覆盖
    :return:
    """
    if not isinstance(performance_data, pd.DataFrame):
        raise TypeError("绩效数据必须为pd.DataFrame类型.")
    if performance_data.empty:
        logging.info('绩效数据为空, 未保存.')
        return

    if performance_type == 'daily':
        performance_filepath = os.path.join(SETTINGS.FACTOR_DB_PATH,
                                            eval('alphafactor_ct.'+factor_name.upper()+'_CT')['db_file'],
                                            'performance/performance_daily.csv')
    elif performance_type == 'monthly':
        performance_filepath = os.path.join(SETTINGS.FACTOR_DB_PATH,
                                            eval('alphafactor_ct.' + factor_name.upper() + '_CT')['db_file'],
                                            'performance/performance_monthly.csv')
    elif performance_type == 'summary':
        if not isinstance(performance_data, pd.Series):
            raise TypeError("‘summary’类型的绩效数据类型应该为pd.Series.")
        performance_filepath = os.path.join(SETTINGS.FACTOR_DB_PATH,
                                            eval('alphafactor_ct.' + factor_name.upper() + '_CT')['db_file'],
                                            'performance/performance_{}.csv'.format(performance_data['type']))
    else:
        raise ValueError("绩效数据类型有误, 应为'daily'=日度绩效时间序列数据, 'monthly'=月度绩效时间序列数据, 'summary'=绩效汇总数据.")

    if save_type == 'a':
        if performance_type == 'daily':
            if os.path.isfile(performance_filepath):
                df_performance_data = pd.read_csv(performance_filepath, parse_dates=[0], header=0)
                df_performance_data = df_performance_data[df_performance_data['date'] <= performance_data.loc[0, 'date']]
                if not df_performance_data.empty:
                    performance_data['nav'] *= df_performance_data.loc[0, 'nav']
                    performance_data['accu_ret'] = performance_data['nav'] - 1
                Utils.save_timeseries_data(performance_data, performance_filepath, 'a')
        elif performance_type in ['monthly', 'summary']:
            Utils.save_timeseries_data(performance_data, performance_filepath, 'a')
        else:
            raise ValueError("绩效数据类型有误, 应为'daily'=日度绩效时间序列数据, 'monthly'=月度绩效时间序列数据, 'summary'=绩效汇总数据.")
    elif save_type == 'w':
        Utils.save_timeseries_data(performance_data, performance_filepath, 'w')
    else:
        raise ValueError("保存类型有误, 应为'a'=新增, 'w'=覆盖.")


if __name__ == '__main__':
    # pass
    # _calc_alphafactor_loading(start_date='2018-08-31', end_date='2018-08-31', factor_name='IntradayMomentum', multi_proc=False, test=True)
    # _calc_Orthogonalized_factorloading(factor_name='SmartMoney', start_date='2018-06-29', end_date='2018-06-29', month_end=True, save=True)
    _calc_MVPFP(factor_name='SmartMoney', start_date='2018-06-29', end_date='2018-06-29', month_end=True, save=True)
