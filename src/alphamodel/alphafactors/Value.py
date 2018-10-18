#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 估值类因子
# @Filename: Value
# @Date:   : 2018-10-16 11:40
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


class EPTTM(Factor):
    """市盈率倒数TTM"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.EPTTM_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股EPTTM因子的载荷
        EPTTM = 净利润TTN值/总市值
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
            1.epttm
            如果计算失败, 返回None
        """
        # 读取TTM财务数据
        ttm_fin_data = Utils.get_ttm_fin_basic_data(code, calc_date)
        if ttm_fin_data is None:
            return None
        if np.isnan(ttm_fin_data['NetProfit']):
            return None
        # 读取个股总市值数据
        cap_data = Utils.get_secu_cap_data(code, calc_date)
        if cap_data is None:
            return None
        if np.isnan(cap_data['total_cap']):
            return None
        # EPTTM = 净利润TTM/总市值
        epttm = ttm_fin_data['NetProfit'] * 10000 / cap_data['total_cap']

        return pd.Series([Utils.code_to_symbol(code), epttm], index=['code', 'epttm'])

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
        logging.debug('[%s] Calc EPTTM factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        epttm_data = None
        try:
            epttm_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if epttm_data is not None:
            q.put(epttm_data)

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
        # 取得交易日序列及股票基本信息表
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算EPTTM因子载荷
        dict_epttm = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc EPTTM factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股EPTTM因子值
            s = calc_date - datetime.timedelta(days=alphafactor_ct.EPTTM_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            epttms = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算EPTTM因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's EPTTM factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    epttm_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if epttm_data is not None:
                        ids.append(epttm_data['code'])
                        epttms.append(epttm_data['epttm'])
            else:
                # 采用多进程进行并行计算EPTTM因子值
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    epttm_data = q.get(True)
                    ids.append(epttm_data['code'])
                    epttms.append(epttm_data['epttm'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_epttm = {'date': [datelabel]*len(ids), 'id': ids, 'factorvalue': epttms}
            # 计算去极值标准化后的因子载荷
            df_std_epttm = Utils.normalize_data(pd.DataFrame(dict_epttm), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_epttm['factorvalue'] = round(df_std_epttm['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_epttm, 'EPTTM', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_epttm, 'EPTTM', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_epttm


class SPTTM(Factor):
    """市销率倒数TTM"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.SPTTM_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股SPTTM因子的载荷
        SPTTM = 营业收入TTM / 总市值
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
            1.spttm
            如果计算失败, 返回None
        """
        # 读取TTM财务数据
        ttm_fin_data = Utils.get_ttm_fin_basic_data(code, calc_date)
        if ttm_fin_data is None:
            return None
        if np.isnan(ttm_fin_data['MainOperateRevenue']):
            return None
        # 读取个股总市值数据
        cap_data = Utils.get_secu_cap_data(code, calc_date)
        if cap_data is None:
            return None
        if np.isnan(cap_data['total_cap']):
            return None
        # SPTTM = 营业收入 / 总市值
        spttm = ttm_fin_data['MainOperateRevenue'] * 10000 / cap_data['total_cap']

        return pd.Series([Utils.code_to_symbol(code), spttm], index=['code', 'spttm'])

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
        logging.debug('[%s] Calc SPTTM factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        spttm_data = None
        try:
            spttm_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if spttm_data is not None:
            q.put(spttm_data)

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
        # 遍历交易日序列, 计算SPTTM因子载荷
        dict_spttm = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc SPTTM factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股SPTTM因子值
            s = calc_date - datetime.timedelta(days=alphafactor_ct.SPTTM_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            spttms = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算SPTTM因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's SPTTM factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    spttm_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if spttm_data is not None:
                        ids.append(spttm_data['code'])
                        spttms.append(spttm_data['spttm'])
            else:
                # 采用多进程并行计算SPTTM因子值
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    spttm_data = q.get(True)
                    ids.append(spttm_data['code'])
                    spttms.append(spttm_data['spttm'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_spttm = {'date': [datelabel]*len(ids), 'id': ids, 'factorvalue': spttms}
            # 计算去极值标准化后的因子载荷
            df_std_spttm = Utils.normalize_data(pd.DataFrame(dict_spttm), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_spttm['factorvalue'] = round(df_std_spttm['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_spttm, 'SPTTM', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_spttm, 'SPTTM', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_spttm

if __name__ == '__main__':
    pass
    # EPTTM.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=False)
    SPTTM.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=False)
