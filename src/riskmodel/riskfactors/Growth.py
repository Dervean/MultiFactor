#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 风险因子中的成长因子
# @Filename: Growth
# @Date:   : 2018-05-28 16:08
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.factors.factor import Factor
import src.riskmodel.riskfactors.cons as risk_ct
import src.factors.cons as factor_ct
from src.util.utils import Utils, ConsensusType
import src.util.cons as utils_con
from src.util.dataapi.CDataHandler import  CDataHandler
import pandas as pd
import numpy as np
import logging
import os
import datetime
from multiprocessing import Pool, Manager
import time
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class EGRLF(Factor):
    """分析师预期盈利长期增长率因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.EGRLF_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股EGRLF因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的EGRLF因子载荷
            0. code
            1. egrlf
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        calc_date = Utils.to_date(calc_date)
        # 读取个股的预期盈利增长率数据
        earningsgrowth_data = Utils.get_consensus_data(calc_date, code, ConsensusType.PredictedEarningsGrowth)
        if earningsgrowth_data is None:
            # 如果个股的预期盈利增长率数据不存在, 那么用过去3年净利润增长率代替
            hist_growth_data = Utils.get_hist_growth_data(code, calc_date, 3)
            if hist_growth_data is None:
                return None
            if np.isnan(hist_growth_data['netprofit']):
                return None
            egrlf = hist_growth_data['netprofit']
        else:
            egrlf = earningsgrowth_data['growth_2y']

        return pd.Series([code, egrlf], index=['code', 'egrlf'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列
        """
        logging.info('[{}] Calc EGRLF factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        egrlf_data = None
        try:
            egrlf_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if egrlf_data is not None:
            q.put(egrlf_data)

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
        :param save: bool, 默认为True
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程, False=采用单进程, 默认为False
        :return: dict
            因子载荷数据
        """
        # 取得交易日序列及股票基本信息表
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列, 计算egrlf因子载荷
        dict_egrlf = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc EGRLF factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的EGRLF因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.EGRLF_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []        # 个股代码list
            egrlfs = []     # EGRLF因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算EGRLF因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's EGRLF factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    egrlf_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if egrlf_data is not None:
                        ids.append(egrlf_data['code'])
                        egrlfs.append(egrlf_data['egrlf'])
            else:
                # 采用多进程并行计算EGRLF因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    egrlf_data = q.get(True)
                    ids.append(egrlf_data['code'])
                    egrlfs.append(egrlf_data['egrlf'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_egrlf = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': egrlfs}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_egrlf, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_egrlf


class EGRSF(Factor):
    """分析师预期盈利短期增长率因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.EGRSF_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股EGRSF因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的EGRSF因子载荷
            0. code
            1. egrsf
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        calc_date = Utils.to_date(calc_date)
        # 读取个股的预期盈利增长率数据
        earningsgrowth_data = Utils.get_consensus_data(calc_date, code, ConsensusType.PredictedEarningsGrowth)
        if earningsgrowth_data is None:
            # 如果个股的预期盈利增长率数据不存在, 那么用过去1年净利润增长率代替
            hist_growth_data = Utils.get_hist_growth_data(code, calc_date, 1)
            if hist_growth_data is None:
                return None
            if np.isnan(hist_growth_data['netprofit']):
                return None
            egrsf = hist_growth_data['netprofit']
        else:
            egrsf = earningsgrowth_data['growth_1y']

        return pd.Series([code, egrsf], index=['code', 'egrsf'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码,   如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列
        """
        logging.info('[{}] Calc EGRSF factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        egrsf_data = None
        try:
            egrsf_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if egrsf_data is not None:
            q.put(egrsf_data)

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
        :param save: bool, 默认为True
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程, False=采用单进程, 默认为False
        :return: dict
            因子载荷数据
        """
        # 取得交易日序列及股票基本信息表
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列, 计算egrsf因子载荷
        dict_egrsf = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc EGRSF factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的EGRSF因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.EGRSF_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []        # 个股代码list
            egrsfs = []     # EGRSF因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算EGRSF因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's EGRSF factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    egrsf_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if egrsf_data is not None:
                        ids.append(egrsf_data['code'])
                        egrsfs.append(egrsf_data['egrsf'])
            else:
                # 采用多进程并行计算EGRSF因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    egrsf_data = q.get(True)
                    ids.append(egrsf_data['code'])
                    egrsfs.append(egrsf_data['egrsf'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_egrsf = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': egrsfs}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_egrsf, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_egrsf


def _get_prevN_years_finbasicdata(date, code, years):
    """
    读取过去n年的主要财务指标数据, 其中每股数据会经过复权因子调整
    :param date: datetime-like
        日期
    :param code: str
        个股代码, 格式: SH600000
    :param years: int
        返回的报告期年数
    :return: list of pd.Series
    """
    year = date.year
    month = date.month
    if month in (1, 2, 3, 4):
        # report_dates = [datetime.datetime(year-5, 12, 31),
        #                 datetime.datetime(year-4, 12, 31),
        #                 datetime.datetime(year-3, 12, 31),
        #                 datetime.datetime(year-2, 12, 31)]
        report_dates = [datetime.datetime(year-n, 12, 31) for n in range(years, 1, -1)]
        is_ttm = True
    elif month in (5, 6, 7, 8):
        # report_dates = [datetime.datetime(year-5, 12, 31),
        #                 datetime.datetime(year-4, 12, 31),
        #                 datetime.datetime(year-3, 12, 31),
        #                 datetime.datetime(year-2, 12, 31),
        #                 datetime.datetime(year-1, 12, 31)]
        report_dates = [datetime.datetime(year-n, 12, 31) for n in range(years, 0, -1)]
        is_ttm = False
    else:
        # report_dates = [datetime.datetime(year-4, 12, 31),
        #                 datetime.datetime(year-3, 12, 31),
        #                 datetime.datetime(year-2, 12, 31),
        #                 datetime.datetime(year-1, 12, 31)]
        report_dates = [datetime.datetime(year-n, 12, 31) for n in range(years-1, 0, -1)]
        is_ttm = True

    df_mkt_data = Utils.get_secu_daily_mkt(code, end=date, fq=True)    # 个股复权行情, 用于调整每股数据

    prevN_years_finbasicdata = []
    for report_date in report_dates:
        fin_basic_data = Utils.get_fin_basic_data(code, report_date, date_type='report_date')
        if fin_basic_data is None:
            return None
        fin_basic_data = fin_basic_data.to_dict()
        df_extract_mkt = df_mkt_data[df_mkt_data.date <= report_date.strftime('%Y-%m-%d')]
        if not df_extract_mkt.empty:
            fq_factor = df_extract_mkt.iloc[-1]['factor']
            # 调整每股数据
            fin_basic_data['BasicEPS'] *= fq_factor
            fin_basic_data['UnitNetAsset'] *= fq_factor
            fin_basic_data['UnitNetOperateCashFlow'] *= fq_factor
            # 计算调整后的主营业务收入
            fin_basic_data['MainOperateRevenue_adj'] = fin_basic_data['MainOperateRevenue'] / fq_factor
        else:
            fin_basic_data['MainOperateRevenue_adj'] = fin_basic_data['MainOperateRevenue']
        prevN_years_finbasicdata.append(fin_basic_data)
    if is_ttm:
        ttm_fin_basic_data = Utils.get_ttm_fin_basic_data(code, date)
        if ttm_fin_basic_data is None:
            return None
        ttm_fin_basic_data = ttm_fin_basic_data.to_dict()
        df_extract_mkt = df_mkt_data[df_mkt_data.date <= ttm_fin_basic_data['ReportDate'].strftime('%Y-%m-%d')]
        if not df_extract_mkt.empty:
            fq_factor = df_extract_mkt.iloc[-1]['factor']
            # 调整每股数据
            ttm_fin_basic_data['BasicEPS'] *= fq_factor
            # 计算调整后的主营业务收入
            ttm_fin_basic_data['MainOperateRevenue_adj'] = ttm_fin_basic_data['MainOperateRevenue'] / fq_factor
        else:
            ttm_fin_basic_data['MainOperateRevenue_adj'] = ttm_fin_basic_data['MainOperateRevenue']
        prevN_years_finbasicdata.append(ttm_fin_basic_data)
    return prevN_years_finbasicdata


class EGRO(Factor):
    """盈利增长率因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.EGRO_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股EGRO因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的EGRO因子载荷
            0. code
            1. egro
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        calc_date = Utils.to_date(calc_date)
        # 读取过去5年的主要财务指标数据
        years = 5
        prevN_years_finbasicdata = _get_prevN_years_finbasicdata(calc_date, code, years)
        if prevN_years_finbasicdata is None:
            return None
        # 复权因子调整后的EPS对年度t进行线性回归(OLS), 计算斜率beta
        arr_eps = np.asarray([fin_basicdata['BasicEPS'] for fin_basicdata in prevN_years_finbasicdata])
        if any(np.isnan(arr_eps)):
            return None
        arr_t = np.arange(1, years+1)
        arr_t = sm.add_constant(arr_t)
        model = sm.OLS(arr_eps, arr_t)
        results = model.fit()
        beta = results.params[1]
        # 计算平均EPS
        avg_eps = np.mean(arr_eps)
        if abs(avg_eps) < utils_con.TINY_ABS_VALUE:
            return None
        # egro = beta / avg_eps
        egro = beta / avg_eps

        return pd.Series([code, egro], index=['code', 'egro'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列
        """
        logging.info('[{}] Calc EGRO factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        egro_data = None
        try:
            egro_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if egro_data is not None:
            q.put(egro_data)

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
        :param save: bool, 默认为True
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程, False=采用单进程, 默认为False
        :return: dict
            因子载荷数据
        """
        # 取得交易日序列及股票基本信息表
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列, 计算egro因子载荷
        dict_egro = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc EGRO factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股egro因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.EGRO_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []        # 个股代码list
            egros = []      # EGRO因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算EGRO因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's EGRO factor laoding." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    egro_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if egro_data is not None:
                        ids.append(egro_data['code'])
                        egros.append(egro_data['egro'])
            else:
                # 采用多进程并行计算EGRO因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    egro_data = q.get(True)
                    ids.append(egro_data['code'])
                    egros.append(egro_data['egro'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_egro = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': egros}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_egro, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_egro


class SGRO(Factor):
    """营业收入增长率因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.SGRO_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股SGRO因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的SGRO因子载荷
            0. code
            1. sgro
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        calc_date = Utils.to_date(calc_date)
        # 读取过去5年的主要财务指标数据
        years = 5
        prevN_years_finbasicdata = _get_prevN_years_finbasicdata(calc_date, code, years)
        if prevN_years_finbasicdata is None:
            return None
        # 复权因子调整后的主营业务收入对年度t进行线性回归(OLS), 计算斜率beta
        arr_revenue = np.asarray([fin_basicdata['MainOperateRevenue_adj'] for fin_basicdata in prevN_years_finbasicdata])
        if any(np.isnan(arr_revenue)):
            return None
        arr_t = np.arange(1, years+1)
        arr_t = sm.add_constant(arr_t)
        model = sm.OLS(arr_revenue, arr_t)
        results = model.fit()
        beta = results.params[1]
        # 计算平均revenue
        avg_revenue = np.mean(arr_revenue)
        if abs(avg_revenue) < utils_con.TINY_ABS_VALUE:
            return None
        # sgro = beta / avg_revenue
        sgro = beta / avg_revenue

        return pd.Series([code, sgro], index=['code', 'sgro'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters
        -------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列
        """
        logging.info('[{}] Calc SGRO factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        sgro_data = None
        try:
            sgro_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if sgro_data is not None:
            q.put(sgro_data)

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
        :param save: bool, 默认为True
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程, False=采用单进程, 默认为False
        :return: dict
            因子载荷数据
        """
        # 取得交易日序列及股票基本信息表
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列, 计算sgro因子载荷
        dict_sgro = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc SGRO factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股sgro因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.SGRO_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []        # 个股代码list
            sgros = []      # SGRO因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算SGRO因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's SGRO factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    sgro_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if sgro_data is not None:
                        ids.append(sgro_data['code'])
                        sgros.append(sgro_data['sgro'])
            else:
                # 采用多进程并行计算SGRO因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    sgro_data = q.get(True)
                    ids.append(sgro_data['code'])
                    sgros.append(sgro_data['sgro'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_sgro = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': sgros}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_sgro, ['date', 'id',  'factorvalue'])
            # 暂停280秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_sgro


class Growth(Factor):
    """风险因子中的成长因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.GROWTH_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        pass

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        pass

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
        # 遍历交易日序列, 计算growth因子下各个成分因子的因子载荷
        if 'multi_proc' not in kwargs:
            kwargs['multi_proc'] = False
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            # 计算各成分因子的因子载荷
            for com_factor in risk_ct.GROWTH_CT.component:
                factor = eval(com_factor + '()')
                factor.calc_factor_loading(start_date=calc_date, end_date=None, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'])
            # 合成Growth因子载荷
            growth_factor = pd.DataFrame()
            for com_factor in risk_ct.GROWTH_CT.component:
                factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, eval('risk_ct.' + com_factor + '_CT')['db_file'])
                factor_loading = Utils.read_factor_loading(factor_path, Utils.datetimelike_to_str(calc_date, dash=False))
                factor_loading.drop(columns='date', inplace=True)
                factor_loading[com_factor] = Utils.normalize_data(Utils.clean_extreme_value(np.array(factor_loading['factorvalue']).reshape((len(factor_loading), 1))))
                factor_loading.drop(columns='factorvalue', inplace=True)
                if growth_factor.empty:
                    growth_factor = factor_loading
                else:
                    growth_factor = pd.merge(left=growth_factor, right=factor_loading, how='inner', on='id')
            growth_factor.set_index('id', inplace=True)
            weight = pd.Series(risk_ct.GROWTH_CT.weight)
            growth_factor = (growth_factor * weight).sum(axis=1)
            growth_factor.name = 'factorvalue'
            growth_factor.index.name = 'id'
            growth_factor = pd.DataFrame(growth_factor)
            growth_factor.reset_index(inplace=True)
            growth_factor['date'] = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            # 保存growth因子载荷
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), growth_factor.to_dict('list'), ['date', 'id', 'factorvalue'])


if __name__ == '__main__':
    pass
    # EGRLF.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    # EGRLF.calc_secu_factor_loading('000046', '2017-12-29')
    # EGRSF.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    # EGRO.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    # EGRO.calc_secu_factor_loading('SZ300591', '2017-12-29')
    # SGRO.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=False)
    # SGRO.calc_secu_factor_loading('300607', '2017-12-29')
    Growth.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
