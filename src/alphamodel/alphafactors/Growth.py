#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 成长类因子
# @Filename: Growth
# @Date:   : 2018-10-17 16:37
# @Author  : YuJun
# @Email   : yujun_mail@163.com

import src.settings as SETTINGS
from src.factors.factor import Factor
import src.alphamodel.alphafactors.cons as alphafactor_ct
from src.util.utils import Utils
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool, Manager
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class OperateRevenueYoY(Factor):
    """营业收入同比增长率"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.OPERATEREVENUEYOY_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股营业收入同比增长因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, e.g: 600000, SH600000
        :param calc_date: datetime-like, str
            因子载荷计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
        --------
            index为:
            0.code
            1.optrevn_yoy
            如果计算失败, 返回None
        """
        # 读取同比增长数据
        yoy_growth_data = Utils.get_fin_yoygrowth_data(code, calc_date)
        if yoy_growth_data is None:
            return None
        if np.isnan(yoy_growth_data['MainOperateRevenue']):
            return None

        return pd.Series([Utils.code_to_symbol(code), yoy_growth_data['MainOperateRevenue']], index=['code', 'optrevn_yoy'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如600000 or SH600000
        :param calc_date: datetime-like or str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列中
        """
        logging.debug('[%s] Calc OperateRevenue YoY factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        optreven_yoy_data = None
        try:
            optreven_yoy_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if optreven_yoy_data is not None:
            q.put(optreven_yoy_data)

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的因子载荷, 并保存至因子数据库
        Parameters:
        --------
        :param start_date: datetime-like or str
            开始日期, 格式: YYYY-MM-DD or YYYYMMDD
        :param end_date: datetime-like, str
            结束日期, 如果为None, 则只计算start_date日期的因子载荷, 格式:YYYY-MM-DD or YYYYMMDD
        :param month_end: bool, 默认True
            如果为True, 则只计算月末时点的因子载荷
        :param save: bool, 默认True
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程并行计算, False=采用单进程计算, 默认为False
        :return: dict
            因子载荷
        --------
        """
        # 取得交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算营业收入同比增长因子载荷
        dict_optrevnyoy = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc OperateRevenue YoY factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股营业收入同比增长率因子载荷
            s = calc_date - datetime.timedelta(days=alphafactor_ct.OPERATEREVENUEYOY_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            optrevnyoys = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's OperateRevenue YoY factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    optrevn_yoy_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if optrevn_yoy_data is not None:
                        ids.append(optrevn_yoy_data['code'])
                        optrevnyoys.append(optrevn_yoy_data['optrevn_yoy'])
            else:
                # 采用多进程计算
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    optrevn_yoy_data = q.get(True)
                    ids.append(optrevn_yoy_data['code'])
                    optrevnyoys.append(optrevn_yoy_data['optrevn_yoy'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_optrevnyoy = {'date': [datelabel]*len(ids), 'id': ids, 'factorvalue': optrevnyoys}
            # 计算去极值标准化后的因子载荷
            df_std_optrevnyoy = Utils.normalize_data(pd.DataFrame(dict_optrevnyoy), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_optrevnyoy['factorvalue'] = round(df_std_optrevnyoy['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_optrevnyoy, 'OperateRevenueYoY', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_optrevnyoy, 'OperateRevenueYoY', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_optrevnyoy


class OperateProfitYoY(Factor):
    """营业利润同比增长率"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.OPERATEPROFITYOY_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股营业利润同比增长率因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, e.g: 600000, SH600000
        :param calc_date: datetime-like, str
            因子载荷计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
            个股的营业利润同比增长率因子数据
        --------
            pd.Series的index为:
            0.code
            1.optprofit_yoy
            如果计算失败, 返回None
        """
        yoy_growth_data = Utils.get_fin_yoygrowth_data(code, calc_date)
        if yoy_growth_data is None:
            return None
        if np.isnan(yoy_growth_data['OperateProfit']):
            return None

        return pd.Series([Utils.code_to_symbol(code), yoy_growth_data['OperateProfit']], index=['code', 'optprofit_yoy'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如600000 or SH600000
        :param calc_date: datetime-like or str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列中
        """
        logging.debug('[%s] Calc OperateProfit YoY factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        optprofit_yoy_data = None
        try:
            optprofit_yoy_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if optprofit_yoy_data is not None:
            q.put(optprofit_yoy_data)

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的因子载荷, 并保存至因子数据库
        Parameters:
        --------
        :param start_date: datetime-like or str
            开始日期, 格式: YYYY-MM-DD or YYYYMMDD
        :param end_date: datetime-like, str
            结束日期, 如果为None, 则只计算start_date日期的因子载荷, 格式:YYYY-MM-DD or YYYYMMDD
        :param month_end: bool, 默认True
            如果为True, 则只计算月末时点的因子载荷
        :param save: bool, 默认True
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程并行计算, False=采用单进程计算, 默认为False
        :return: dict
            因子载荷
        """
        # 取得交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算营业利润同比增长率因子载荷
        dict_optprofityoy = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc OperateProfit YoY factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股营业利润同比增长率因子载荷
            s = calc_date - datetime.timedelta(days=alphafactor_ct.OPERATEPROFITYOY_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            optprofityoys = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's OperateProfit YoY factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    optprofit_yoy_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if optprofit_yoy_data is not None:
                        ids.append(optprofit_yoy_data['code'])
                        optprofityoys.append(optprofit_yoy_data['optprofit_yoy'])
            else:
                # 采用多进程计算
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    optprofit_yoy_data = q.get(True)
                    ids.append(optprofit_yoy_data['code'])
                    optprofityoys.append(optprofit_yoy_data['optprofit_yoy'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_optprofityoy = {'date': [datelabel]*len(ids), 'id': ids, 'factorvalue': optprofityoys}
            # 计算去极值标准化后的因子载荷
            df_std_optprofityoy = Utils.normalize_data(pd.DataFrame(dict_optprofityoy), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_optprofityoy['factorvalue'] = round(df_std_optprofityoy['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_optprofityoy, 'OperateProfitYoY', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_optprofityoy, 'OperateProfitYoY', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_optprofityoy


class NetProfitYoY(Factor):
    """净利润同比增长率"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.NETPROFITYOY_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股净利润同比增长率因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, e.g: 600000, SH600000
        :param calc_date: datetime-like, str
            因子载荷计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
            个股的净利润同比增长率因子数据
        --------
            pd.Series的index为:
            0.code
            1.netprofit_yoy
            如果计算失败, 返回None
        """
        yoy_growth_data = Utils.get_fin_yoygrowth_data(code, calc_date)
        if yoy_growth_data is None:
            return None
        if np.isnan(yoy_growth_data['NetProfit']):
            return None

        return pd.Series([Utils.code_to_symbol(code), yoy_growth_data['NetProfit']], index=['code', 'netprofit_yoy'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如600000 or SH600000
        :param calc_date: datetime-like or str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列中
        """
        logging.debug('[%s] Calc NetProfit YoY factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        netprofit_yoy_data = None
        try:
            netprofit_yoy_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if netprofit_yoy_data is not None:
            q.put(netprofit_yoy_data)

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的因子载荷, 并保存至因子数据库
        Parameters:
        --------
        :param start_date: datetime-like or str
            开始日期, 格式: YYYY-MM-DD or YYYYMMDD
        :param end_date: datetime-like, str
            结束日期, 如果为None, 则只计算start_date日期的因子载荷, 格式:YYYY-MM-DD or YYYYMMDD
        :param month_end: bool, 默认True
            如果为True, 则只计算月末时点的因子载荷
        :param save: bool, 默认True
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程并行计算, False=采用单进程计算, 默认为False
        :return: dict
            因子载荷
        """
        # 取得交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算净利润同比增长率因子载荷
        dict_netprofityoy = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc NetProfit YoY factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股净利润同比增长率因子载荷
            s = calc_date - datetime.timedelta(days=alphafactor_ct.NETPROFITYOY_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            netprofityoys = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's NetProfit YoY factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    netprofit_yoy_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if netprofit_yoy_data is not None:
                        ids.append(netprofit_yoy_data['code'])
                        netprofityoys.append(netprofit_yoy_data['netprofit_yoy'])
            else:
                # 采用多进程计算
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    netprofit_yoy_data = q.get(True)
                    ids.append(netprofit_yoy_data['code'])
                    netprofityoys.append(netprofit_yoy_data['netprofit_yoy'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_netprofityoy = {'date': [datelabel] * len(ids), 'id': ids, 'factorvalue': netprofityoys}
            # 计算去极值标准化后的因子载荷
            df_std_netprofityoy = Utils.normalize_data(pd.DataFrame(dict_netprofityoy), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_netprofityoy['factorvalue'] = round(df_std_netprofityoy['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_netprofityoy, 'NetProfitYoY', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_netprofityoy, 'NetProfitYoY', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_netprofityoy


class OperateCashFlowYoY(Factor):
    """经营现金流同比增长率"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.OPERATECASHFLOWYOY_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股经营现金流同比增长率因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, e.g: 600000, SH600000
        :param calc_date: datetime-like, str
            因子载荷计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
            个股的经营现金流同比增长率因子数据
        --------
            pd.Series的index为:
            0.code
            1.optcashflow_yoy
            如果计算失败, 返回None
        """
        yoy_growth_data = Utils.get_fin_yoygrowth_data(code, calc_date)
        if yoy_growth_data is None:
            return None
        if np.isnan(yoy_growth_data['NetOperateCashFlow']):
            return None

        return pd.Series([Utils.code_to_symbol(code), yoy_growth_data['NetOperateCashFlow']], index=['code', 'optcashflow_yoy'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如600000 or SH600000
        :param calc_date: datetime-like or str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列中
        """
        logging.debug('[%s] Calc OperateCashFlow YoY factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        optcashflow_yoy_data = None
        try:
            optcashflow_yoy_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if optcashflow_yoy_data is not None:
            q.put(optcashflow_yoy_data)

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的因子载荷, 并保存至因子数据库
        Parameters:
        --------
        :param start_date: datetime-like or str
            开始日期, 格式: YYYY-MM-DD or YYYYMMDD
        :param end_date: datetime-like, str
            结束日期, 如果为None, 则只计算start_date日期的因子载荷, 格式:YYYY-MM-DD or YYYYMMDD
        :param month_end: bool, 默认True
            如果为True, 则只计算月末时点的因子载荷
        :param save: bool, 默认True
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程并行计算, False=采用单进程计算, 默认为False
        :return: dict
            因子载荷
        """
        # 取得交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算经营现金流同比增长率因子载荷
        dict_optcashflowyoy = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc OpterateCashFlow YoY factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股净利润同比增长率因子载荷
            s = calc_date - datetime.timedelta(days=alphafactor_ct.OPERATECASHFLOWYOY_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            optcashflowyoys = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's OperateCashFlow YoY factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    optcashflow_yoy_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if optcashflow_yoy_data is not None:
                        ids.append(optcashflow_yoy_data['code'])
                        optcashflowyoys.append(optcashflow_yoy_data['optcashflow_yoy'])
            else:
                # 采用多进程计算
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    optcashflow_yoy_data = q.get(True)
                    ids.append(optcashflow_yoy_data['code'])
                    optcashflowyoys.append(optcashflow_yoy_data['optcashflow_yoy'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_optcashflowyoy = {'date': [datelabel] * len(ids), 'id': ids, 'factorvalue': optcashflowyoys}
            # 计算去极值标准化后的因子载荷
            df_std_optcashflowyoy = Utils.normalize_data(pd.DataFrame(dict_optcashflowyoy), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_optcashflowyoy['factorvalue'] = round(df_std_optcashflowyoy['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_optcashflowyoy, 'OperateCashFlowYoY', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_optcashflowyoy, 'OperateCashFlowYoY', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_optcashflowyoy


if __name__ == '__main__':
    pass
    # OperateRevenueYoY.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=False)
    # OperateProfitYoY.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=True)
    # NetProfitYoY.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=False)
    OperateCashFlowYoY.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=True)
