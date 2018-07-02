#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 风险模型的Beta类风险因子
# @Filename: Beta
# @Date:   : 2018-03-21 18:10
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.factors.factor import Factor
import src.riskmodel.riskfactors.cons as risk_ct
import src.factors.cons as factor_ct
from src.util.utils import Utils
# import src.util.cons as util_ct
from src.util.dataapi.CDataHandler import CDataHandler
import pandas as pd
import numpy as np
import logging
import os
import statsmodels.api as sm
import datetime
from multiprocessing import Pool, Manager
import time


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class DBETA(Factor):
    """贝塔因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.DBETA_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股BETA因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如600000或SH600000
        :param calc_date: datetime-like, str
            计算日期, 格式YYYY-MM-DD
        :return: pd.Series
        --------
            个股的BETA因子和HSIGMA因子载荷
            0. code: 个股代码
            1. beta: BETA因子载荷
            2. hsigma: HSIGMA因子载荷
            若计算失败, 返回None
        """
        # 取得个股复权行情数据
        df_secu_quote = Utils.get_secu_daily_mkt(code, end=calc_date, ndays=risk_ct.DBETA_CT.trailing+1, fq=True)
        if df_secu_quote is None:
            return None
        df_secu_quote.reset_index(drop=True, inplace=True)
        # 取得基准复权行情数据
        benchmark_code = risk_ct.DBETA_CT.benchmark
        df_benchmark_quote = Utils.get_secu_daily_mkt(benchmark_code, end=calc_date, fq=True)
        if df_benchmark_quote is None:
            return None
        df_benchmark_quote = df_benchmark_quote[df_benchmark_quote['date'].isin(list(df_secu_quote['date']))]
        df_benchmark_quote.reset_index(drop=True, inplace=True)
        # 计算个股和基准的日收益率序列
        arr_secu_close = np.array(df_secu_quote.iloc[1:]['close'])
        arr_secu_preclose = np.array(df_secu_quote.shift(1).iloc[1:]['close'])
        arr_secu_daily_ret = arr_secu_close / arr_secu_preclose - 1.
        arr_benchmark_close = np.array(df_benchmark_quote.iloc[1:]['close'])
        arr_benchmark_preclose = np.array(df_benchmark_quote.shift(1).iloc[1:]['close'])
        arr_benchmark_daily_ret = arr_benchmark_close / arr_benchmark_preclose - 1.
        # 计算权重(指数移动加权平均)
        T = len(arr_benchmark_daily_ret)
        time_spans = sorted(range(T), reverse=True)
        alpha = 1 - np.exp(np.log(0.5)/risk_ct.DBETA_CT.half_life)
        x = [1-alpha] * T
        y = [alpha] * (T-1)
        y.insert(0, 1)
        weights = np.float_power(x, time_spans) * y
        # 采用加权最小二乘法计算Beta因子载荷及hsigma
        arr_benchmark_daily_ret = sm.add_constant(arr_benchmark_daily_ret)
        cap_model = sm.WLS(arr_secu_daily_ret, arr_benchmark_daily_ret, weights=weights)
        result = cap_model.fit()
        beta = result.params[1]
        hsigma = np.sqrt(result.mse_resid)
        return pd.Series([Utils.code_to_symbol(code), beta, hsigma], index=['code', 'beta', 'hsigma'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如600000, SH600000
        :param calc_date: datetime-like or str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列中
        """
        logging.info('[%s] Calc BETA factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        beta_data = None
        try:
            beta_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if beta_data is None:
            beta_data = pd.Series([Utils.code_to_symbol(code), np.nan, np.nan], index=['code', 'beta', 'hsigma'])
        q.put(beta_data)

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的因子载荷, 并保存至因子数据库
        Parameters:
        --------
        :param start_date: datetime-like, str
            开始日期, 格式: YYYY-MM-DD or YYYYMMDD
        :param end_date: datetime-like, str
            结束日期, 如果为None, 则只计算start_date日期的因子载荷, 格式: YYYY-MM-DD or YYYYMMDD
        :param month_end: bool, 默认为True
            如果为True, 则只计算月末时点的因子载荷
        :param save: bool, 默认True
            是否保存至因子数据库
        :param kwargs:
        :return: dict
            因子载荷
        """
        # 取得交易日序列及股票基本信息表
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列, 计算筹码分布因子载荷
        dict_beta = {}
        dict_hsigma = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc BETA factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股BETA因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.DBETA_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []        # 个股代码list
            betas = []      # BETA因子值
            hsigmas = []    # HSIGMA因子值

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算BETA因子和HSIGMA因子值,
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's BETA and HSIGMA factor data." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    beta_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if beta_data is None:
                        ids.append(Utils.code_to_symbol(stock_info.symbol))
                        betas.append(np.nan)
                        hsigmas.append(np.nan)
                    else:
                        ids.append(beta_data['code'])
                        betas.append(beta_data['beta'])
                        hsigmas.append(beta_data['hsigma'])
            else:
                # 采用多进程并行计算BETA因子和HSIGMA因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    beta_data = q.get(True)
                    ids.append(beta_data['code'])
                    betas.append(beta_data['beta'])
                    hsigmas.append(beta_data['hsigma'])

            date_label = Utils.get_trading_days(calc_date, ndays=2)[1]
            dict_beta = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': betas}
            dict_hsigma = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': hsigmas}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_beta, ['date', 'id', 'factorvalue'])
                hsigma_path = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.HSIGMA_CT.db_file)
                Utils.factor_loading_persistent(hsigma_path, Utils.datetimelike_to_str(calc_date, dash=False), dict_hsigma, ['date', 'id', 'factorvalue'])
            # 休息180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_beta


class Beta(Factor):
    """风险因子的Beta因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.BETA_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        pass

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        pass

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        com_factors = []
        for com_factor in risk_ct.BETA_CT.component:
            com_factors.append(eval(com_factor + '()'))
        cls._calc_synthetic_factor_loading(start_date=start_date, end_date=end_date, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'], com_factors=com_factors)

    @classmethod
    def calc_factor_loading_(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的因子载荷, 并保存至因子数据库
        Parameters:
        --------
        :param start_date: datetime-like, str
            开始日期, 格式: YYYY-MM-DD or YYYYMMDD
        :param end_date: datetime-like, str
            结束日期, 如果为None, 则只计算start_date日期的因子载荷, 格式: YYYY-MM-DD or YYYYMMDD
        :param month_end: bool, 默认为True
            如果为True, 则只计算月末时点的因子载荷
        :param save: bool, 默认为True
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程, False=采用单进程, 默认为False
        :return: dict
            因子载荷数据
        """
        # 取得交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算Beta因子下各个成分因子的因子载荷
        if 'multi_proc' not in kwargs:
            kwargs['multi_proc'] = False
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            # 计算各成分因子的因子载荷
            for com_factor in risk_ct.BETA_CT.component:
                factor = eval(com_factor + '()')
                factor.calc_factor_loading(start_date=calc_date, end_date=None, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'])
            # 合成Beta因子载荷
            Beta_factor = pd.DataFrame()
            for com_factor in risk_ct.BETA_CT.component:
                factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, eval('risk_ct.' + com_factor + '_CT')['db_file'])
                factor_loading = Utils.read_factor_loading(factor_path, Utils.datetimelike_to_str(calc_date, dash=False))
                factor_loading.drop(columns='date', inplace=True)
                factor_loading[com_factor] = Utils.normalize_data(Utils.clean_extreme_value(np.array(factor_loading['factorvalue']).reshape((len(factor_loading), 1))))
                factor_loading.drop(columns='factorvalue', inplace=True)
                if Beta_factor.empty:
                    Beta_factor = factor_loading
                else:
                    Beta_factor = pd.merge(left=Beta_factor, right=factor_loading, how='inner', on='id')
            Beta_factor.set_index('id', inplace=True)
            weight = pd.Series(risk_ct.BETA_CT.weight)
            Beta_factor = (Beta_factor * weight).sum(axis=1)
            Beta_factor.name = 'factorvalue'
            Beta_factor.index.name = 'id'
            Beta_factor = pd.DataFrame(Beta_factor)
            Beta_factor.reset_index(inplace=True)
            Beta_factor['date'] = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            # 保存Beta因子载荷
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), Beta_factor.to_dict('list'), ['date', 'id', 'factorvalue'])


if __name__ == '__main__':
    # pass
    # BETA.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=False)
    Beta.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=False)
