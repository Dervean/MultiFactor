#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 风险模型中的流动性因子
# @Filename: Liquidity
# @Date:   : 2018-05-10 01:14
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.factors.factor import Factor
import src.riskmodel.riskfactors.cons as risk_ct
import src.factors.cons as factor_ct
from src.util.utils import Utils
from src.util.dataapi.CDataHandler import CDataHandler
import pandas as pd
import numpy as np
import math
import logging
import os
import datetime
from multiprocessing import Pool, Manager
import time
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class STOM(Factor):
    """流动性风险因子中的月度换手率因子"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.STOM_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股的STOM因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like or str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的STOM因子载荷
            0. code
            1. stom
            如果计算失败, 返回None
        """
        # 读取个股过去252个交易日的日行情数据（非复权）
        df_mkt_data = Utils.get_secu_daily_mkt(code, end=calc_date, ndays=252, fq=False)
        if df_mkt_data is None or df_mkt_data.empty:
            return None
        # stom
        days = risk_ct.STOM_CT.month_days * risk_ct.STOM_CT.months
        if len(df_mkt_data) >= days:
            stom = math.log(df_mkt_data.iloc[-days:]['turnover1'].sum()/risk_ct.STOM_CT.months)
        else:
            stom = math.log(df_mkt_data['turnover1'].sum()/risk_ct.STOM_CT.months)
        
        return pd.Series([Utils.code_to_symbol(code), stom], index=['code', 'stom'])
    
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
        logging.info('[{}] Calc STOM factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        stom_data = None
        try:
            stom_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if stom_data is None:
            stom_data = pd.Series([Utils.code_to_symbol(code), np.nan], index=['code', 'stom'])
        q.put(stom_data)

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
        # all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列, 计算STOM因子载荷
        dict_stom = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc STOM factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的STOM因子值
            # s = (calc_date - datetime.timedelta(days=risk_ct.STOM_CT.listed_days)).strftime('%Y%m%d')
            # stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            s = calc_date - datetime.timedelta(days=risk_ct.STOM_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s, False)
            ids = []        # 个股代码list
            stoms = []      # STOM因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算STOM因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's STOM factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    stom_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if stom_data is None:
                        ids.append(Utils.code_to_symbol(stock_info.symbol))
                        stoms.append(np.nan)
                    else:
                        ids.append(stom_data['code'])
                        stoms.append(stom_data['stom'])
            else:
                # 采用多进程并行计算STOM因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    stom_data = q.get(True)
                    ids.append(stom_data['code'])
                    stoms.append(stom_data['stom'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_stom = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': stoms}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_stom, ['date', 'id', 'factorvalue'])


class STOQ(Factor):
    """流动性因子中的季度平均换手率因子"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.STOQ_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股的STOQ因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like or str
            计算日期, 格式：YYYY-MM-DD
        :return: pd.Series
        --------
            个股的STOQ因子载荷
            0. code
            1. stoq
            如果计算失败, 返回None
        """
        # 读取个股过去252个交易日的日行情数据（非复权）
        df_mkt_data = Utils.get_secu_daily_mkt(code, end=calc_date, ndays=252, fq=False)
        if df_mkt_data is None or df_mkt_data.empty:
            return None
        # stoq
        days = risk_ct.STOQ_CT.month_days * risk_ct.STOQ_CT.months
        if len(df_mkt_data) >= days:
            stoq = math.log(df_mkt_data.iloc[-days:]['turnover1'].sum()/risk_ct.STOQ_CT.months)
        else:
            stoq = math.log(df_mkt_data['turnover1'].sum()/risk_ct.STOQ_CT.months)

        return pd.Series([Utils.code_to_symbol(code), stoq], index=['code', 'stoq'])

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
        logging.info('[{}] Calc STOQ factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        stoq_data = None
        try:
            stoq_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if stoq_data is None:
            stoq_data = pd.Series([Utils.code_to_symbol(code), np.nan], index=['code', 'stoq'])
        q.put(stoq_data)

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
        # all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列, 计算STOQ因子载荷
        dict_stoq = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc STOQ factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的STOQ因子值
            # s = (calc_date - datetime.timedelta(days=risk_ct.STOQ_CT.listed_days)).strftime('%Y%m%d')
            # stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            s = calc_date - datetime.timedelta(days=risk_ct.STOQ_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s, False)
            ids = []    # 个股代码list
            stoqs = []  # STOQ因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算STOQ因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's STOQ factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    stoq_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if stoq_data is None:
                        ids.append(Utils.code_to_symbol(stock_info.symbol))
                        stoqs.append(np.nan)
                    else:
                        ids.append(stoq_data['code'])
                        stoqs.append(stoq_data['stoq'])
            else:
                # 采用多进程并行计算STOQ因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子值
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc(), args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    stoq_data = q.get(True)
                    ids.append(stoq_data['code'])
                    stoqs.append(stoq_data['stoq'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_stoq = {'date': [date_label]*len(ids), 'id':ids, 'factorvalue': stoqs}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_stoq, ['date', 'id', 'factorvalue'])


class STOA(Factor):
    """流动性因子中的年度平均换手率因子"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.STOA_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股的STOA因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like or str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的STOA因子载荷
            0. code
            1. stoa
            如果计算失败, 返回None
        """
        # 读取个股过去252个交易日的日行情数据（非复权）
        df_mkt_data = Utils.get_secu_daily_mkt(code, end=calc_date, ndays=252, fq=False)
        if df_mkt_data is None or df_mkt_data.empty:
            return None
        # stoa
        days = risk_ct.STOA_CT.month_days * risk_ct.STOA_CT.months
        if len(df_mkt_data) >= days:
            stoa = math.log(df_mkt_data.iloc[-days:]['turnover1'].sum()/risk_ct.STOA_CT.months)
        else:
            stoa = math.log(df_mkt_data['turnover1'].sum()/risk_ct.STOA_CT.months)

        return pd.Series([Utils.code_to_symbol(code), stoa], index=['code', 'stoa'])

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
        logging.info('[{}] Calc STOA factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        stoa_data = None
        try:
            stoa_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if stoa_data is None:
            stoa_data = pd.Series([Utils.code_to_symbol(code), np.nan], index=['code', 'stoa'])
        q.put(stoa_data)

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
        # 取得交易序列及股票基本信息表
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列, 计算STOA因子载荷
        dict_stoa = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc STOA factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的STOA因子值
            # s = (calc_date - datetime.timedelta(days=risk_ct.STOA_CT.listed_days)).strftime('%Y%m%d')
            # stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            s = calc_date - datetime.timedelta(days=risk_ct.STOA_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s, False)
            ids = []    # 个股代码list
            stoas = []  # STOA因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算STOA因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's STOA factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    stoa_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if stoa_data is None:
                        ids.append(Utils.code_to_symbol(stock_info.symbol))
                        stoas.append(np.nan)
                    else:
                        ids.append(stoa_data['code'])
                        stoas.append(stoa_data['stoa'])
            else:
                # 采用多进程并行计算STOA因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子值
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    stoa_data = q.get(True)
                    ids.append(stoa_data['code'])
                    stoas.append(stoa_data['stoa'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_stoa = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': stoas}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_stoa, ['date', 'id', 'factorvalue'])


class Liquidity(Factor):
    """风险因子中的流动性因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.LIQUIDITY_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        pass

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        pass

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        com_factors = []
        for com_factor in risk_ct.LIQUIDITY_CT.component:
            com_factors.append(eval(com_factor + '()'))
        cls._calc_synthetic_factor_loading(start_date=start_date, end_date=end_date, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'], com_factors=com_factors)


class Liquidity_(Factor):
    """流动性因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.LIQUIDITY_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股的LIQUIDITY因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的LIQUIDILITY因子载荷
            0. code
            1. stom 月度换手率
            2. stoq 季度换手率
            3. stoa 年度换手率
            4. liquidity
            如果计算失败, 返回None
        """
        # 读取个股过去252个交易日的日行情数据（非复权）
        stom_days = risk_ct.LIQUIDITY_CT.stom_days
        stoq_months = risk_ct.LIQUIDITY_CT.stoq_months
        stoa_months = risk_ct.LIQUIDITY_CT.stoa_months
        df_mkt_data = Utils.get_secu_daily_mkt(code, end=calc_date, ndays=stoa_months*stom_days, fq=False)
        if df_mkt_data is None or df_mkt_data.empty:
            return None
        # stom
        if len(df_mkt_data) >= stom_days:
            stom = math.log(df_mkt_data.iloc[-stom_days:]['turnover1'].sum())
        else:
            stom = math.log(df_mkt_data['turnover1'].sum())
        # stoq
        stoq_days = stom_days * stoq_months
        if len(df_mkt_data) >= stoq_days:
            stoq = math.log(df_mkt_data.iloc[-stoq_days:]['turnover1'].sum()/stoq_months)
        else:
            stoq = math.log(df_mkt_data['turnover1'].sum()/stoq_months)
        # stoa
        stoa = math.log(df_mkt_data['turnover1'].sum()/stoa_months)
        # liquidity = 0.35*stom + 0.35*stoq + 0.3*stoa
        stom_weight = risk_ct.LIQUIDITY_CT.stom_weight
        stoq_weight = risk_ct.LIQUIDITY_CT.stoq_weight
        stoa_weight = risk_ct.LIQUIDITY_CT.stoa_weight
        liquidity = stom_weight * stom + stoq_weight * stoq + stoa_weight * stoa

        return pd.Series([Utils.code_to_symbol(code), stom, stoq, stoa, liquidity], index=['code', 'stom', 'stoq', 'stoa', 'liquidity'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式:YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列
        """
        logging.info('[{}] Calc Liquidity factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        liquidity_data = None
        try:
            liquidity_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if liquidity_data is not None:
            q.put(liquidity_data)

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
        # 遍历交易日序列, 计算LIQUIDITY因子载荷
        dict_raw_liquidity = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            dict_stom = None
            dict_stoq = None
            dict_stoa = None
            dict_raw_liquidity = None
            logging.info('[%s] Calc Liquidity factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股，计算个股LIQUIDITY因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.LIQUIDITY_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []
            stoms = []
            stoqs = []
            stoas = []
            raw_liquidities = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算LIQUIDITY因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's LIQUIDITY factor loading." % (Utils.datetimelike_to_str(calc_date, dash=True), stock_info.symbol))
                    liquidity_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if liquidity_data is not None:
                        ids.append(liquidity_data['code'])
                        stoms.append(liquidity_data['stom'])
                        stoqs.append(liquidity_data['stoq'])
                        stoas.append(liquidity_data['stoa'])
                        raw_liquidities.append(liquidity_data['liquidity'])
            else:
                # 采用多进程计算LIQUIDITY因子值
                q = Manager().Queue()
                p = Pool(4)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    liquidity_data = q.get(True)
                    ids.append(liquidity_data['code'])
                    stoms.append(liquidity_data['stom'])
                    stoqs.append(liquidity_data['stoq'])
                    stoas.append(liquidity_data['stoa'])
                    raw_liquidities.append(liquidity_data['liquidity'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_stom = dict({'date': [date_label]*len(ids), 'id': ids, 'factorvalue': stoms})
            dict_stoq = dict({'date': [date_label]*len(ids), 'id': ids, 'factorvalue': stoqs})
            dict_stoa = dict({'date': [date_label]*len(ids), 'id': ids, 'factorvalue': stoas})
            dict_raw_liquidity = dict({'date': [date_label]*len(ids), 'id': ids, 'factorvalue': raw_liquidities})
            # 读取Size因子值, 将流动性因子与Size因子正交化
            size_factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.SIZE_CT.db_file)
            df_size = Utils.read_factor_loading(size_factor_path, Utils.datetimelike_to_str(calc_date, dash=False))
            df_size.drop(columns='date', inplace=True)
            df_size.rename(columns={'factorvalue': 'size'}, inplace=True)
            df_liquidity = pd.DataFrame(dict({'id': ids, 'liquidity': raw_liquidities}))
            df_liquidity = pd.merge(left=df_liquidity, right=df_size, how='inner', on='id')
            arr_liquidity = Utils.normalize_data(Utils.clean_extreme_value(np.array(df_liquidity['liquidity']).reshape((len(df_liquidity), 1))))
            arr_size = Utils.normalize_data(Utils.clean_extreme_value(np.array(df_liquidity['size']).reshape((len(df_liquidity), 1))))
            model = sm.OLS(arr_liquidity, arr_size)
            results = model.fit()
            df_liquidity['liquidity'] = results.resid
            df_liquidity.drop(columns='size', inplace=True)
            df_liquidity.rename(columns={'liquidity': 'factorvalue'}, inplace=True)
            df_liquidity['date'] = date_label
            # 保存因子载荷
            if save:
                str_date = Utils.datetimelike_to_str(calc_date, dash=False)
                factor_header = ['date', 'id', 'factorvalue']
                Utils.factor_loading_persistent(cls._db_file, 'stom_{}'.format(str_date), dict_stom, factor_header)
                Utils.factor_loading_persistent(cls._db_file, 'stoq_{}'.format(str_date), dict_stoq, factor_header)
                Utils.factor_loading_persistent(cls._db_file, 'stoa_{}'.format(str_date), dict_stoa, factor_header)
                Utils.factor_loading_persistent(cls._db_file, 'rawliquidity_{}'.format(str_date), dict_raw_liquidity, factor_header)
                Utils.factor_loading_persistent(cls._db_file, str_date, df_liquidity.to_dict('list'), factor_header)

            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_raw_liquidity


if __name__ == '__main__':
    # pass
    Liquidity.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=False)
