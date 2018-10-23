#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 财务质量因子
# @Filename: Quality
# @Date:   : 2018-10-22 12:11
# @Author  : YuJun
# @Email   : yujun_mail@163.com


import src.settings as SETTINGS
from src.factors.factor import Factor
import src.alphamodel.alphafactors.cons as alphafactor_ct
from src.util.utils import Utils
import src.util.cons as util_ct
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool, Manager
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class ROE(Factor):
    """净资产收益率"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.ROE_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股净资产收益率因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, e.g: SH600000, 600000
        :param calc_date: datetime-like, str
            因子载荷计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
        --------
            Series的index为:
            0.code
            1.roe
            如果计算失败, 返回None
        """
        # 读取个股财务摘要数据
        fin_basic_data = Utils.get_fin_basic_data(code, calc_date, 'trading_date')
        if fin_basic_data is None:
            return None
        if np.isnan(fin_basic_data['NetProfit']):
            return None
        # 取得报告期年初的个股财务摘要数据
        fin_report_date = Utils.get_fin_report_date(calc_date)
        fin_report_date = datetime.datetime(fin_report_date.year-1, 12, 31)
        beg_fin_basic_data = Utils.get_fin_basic_data(code, fin_report_date, 'report_date')

        # 计算平均净资产 = (期初净资产 + 期末净资产) / 2
        beg_net_asset = end_net_asset = 0.0
        if beg_fin_basic_data is None:
            beg_net_asset = 0.0
        else:
            if np.isnan(beg_fin_basic_data['ShareHolderEquity']):
                beg_net_asset = 0.0
            else:
                beg_net_asset = beg_fin_basic_data['ShareHolderEquity']
        if np.isnan(fin_basic_data['ShareHolderEquity']):
            end_net_asset = 0.0
        else:
            end_net_asset= fin_basic_data['ShareHolderEquity']

        if abs(beg_net_asset) < util_ct.TINY_ABS_VALUE and abs(end_net_asset) < util_ct.TINY_ABS_VALUE:
            return None
        elif abs(beg_net_asset) < util_ct.TINY_ABS_VALUE:
            avg_net_asset = end_net_asset
        elif abs(end_net_asset) < util_ct.TINY_ABS_VALUE:
            avg_net_asset = beg_net_asset
        else:
            avg_net_asset = (beg_net_asset + end_net_asset) / 2.0

        # if abs(beg_fin_basic_data['ShareHolderEquity']) < util_ct.TINY_ABS_VALUE:
        #     return None

        froe = fin_basic_data['NetProfit'] / avg_net_asset
        return pd.Series([Utils.code_to_symbol(code), froe], index=['code', 'roe'])

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
        logging.debug('[%s] Calc ROE factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        roe_data = None
        try:
            roe_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if roe_data is not None:
            q.put(roe_data)

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
        # 遍历交易日序列, 计算ROE因子载荷
        dict_roe = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc ROE factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股ROE因子载荷
            s = calc_date - datetime.timedelta(days=alphafactor_ct.ROE_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            roes = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's ROE factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    roe_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if roe_data is not None:
                        ids.append(roe_data['code'])
                        roes.append(roe_data['roe'])
            else:
                # 采用多进程计算
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    roe_data = q.get(True)
                    ids.append(roe_data['code'])
                    roes.append(roe_data['roe'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_roe = {'date': [datelabel]*len(ids), 'id': ids, 'factorvalue': roes}
            # 计算去极值标准化后的因子载荷
            df_std_roe = Utils.normalize_data(pd.DataFrame(dict_roe), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_roe['factorvalue'] = round(df_std_roe['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_roe, 'ROE', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_roe, 'ROE', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_roe


class ROEQ(Factor):
    """单季度净资产收益率因子"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.ROEQ_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股单季度净资产收益率因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, e.g: SH600000, 600000
        :param calc_date: datetime-like, str
            因子载荷计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
        --------
            Series的index为:
            0.code
            1.roeq
        """
        # 读取个股单季度财报数据
        fin_quarter_basicdata = Utils.get_fin_singlequarter_basicdata(code, calc_date, 'trading_date')
        if fin_quarter_basicdata is None:
            return None
        if np.isnan(fin_quarter_basicdata['NetProfit']):
            return None

        # 平均净资产 = (季度初净资产 + 季度末净资产) / 2
        if np.isnan(fin_quarter_basicdata['BegShareHolderEquity']):
            beg_net_asset = 0.0
        else:
            beg_net_asset = fin_quarter_basicdata['BegShareHolderEquity']
        if np.isnan(fin_quarter_basicdata['EndShareHolderEquity']):
            end_net_asset = 0.0
        else:
            end_net_asset = fin_quarter_basicdata['EndShareHolderEquity']

        if abs(beg_net_asset) < util_ct.TINY_ABS_VALUE and abs(end_net_asset) < util_ct.TINY_ABS_VALUE:
            return None
        elif abs(beg_net_asset) < util_ct.TINY_ABS_VALUE:
            avg_net_asset = end_net_asset
        elif abs(end_net_asset) < util_ct.TINY_ABS_VALUE:
            avg_net_asset = beg_net_asset
        else:
            avg_net_asset = (beg_net_asset + end_net_asset) / 2.0

        froeq = fin_quarter_basicdata['NetProfit'] / avg_net_asset
        return pd.Series([Utils.code_to_symbol(code), froeq], index=['code', 'roeq'])

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
        logging.debug('[%s] Calc ROEQ factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        roeq_data = None
        try:
            roeq_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if roeq_data is not None:
            q.put(roeq_data)

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
        # 遍历交易日序列, 计算ROEQ因子载荷
        dict_roeq = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc ROEQ factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股ROEQ因子载荷
            s = calc_date - datetime.timedelta(days=alphafactor_ct.ROEQ_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            roeqs = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's ROEQ factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    roeq_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if roeq_data is not None:
                        ids.append(roeq_data['code'])
                        roeqs.append(roeq_data['roeq'])
            else:
                # 采用多进程计算
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    roeq_data = q.get(True)
                    ids.append(roeq_data['code'])
                    roeqs.append(roeq_data['roeq'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_roeq = {'date': [datelabel]*len(ids), 'id': ids, 'factorvalue': roeqs}
            # 计算去极值标准化后的因子载荷
            df_std_roeq = Utils.normalize_data(pd.DataFrame(dict_roeq), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_roeq['factorvalue'] = round(df_std_roeq['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_roeq, 'ROEQ', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_roeq, 'ROEQ', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_roeq


class ROA(Factor):
    """总资产收益率因子"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.ROA_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股总资产收益率因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, e.g: SH600000, 600000
        :param calc_date: datetime-like, str
            因子载荷计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
        --------
            Series的index为:
            0.code
            1.roa
            如果计算失败, 返回None
        """
        # 读取个股财务摘要数据
        fin_basic_data = Utils.get_fin_basic_data(code, calc_date, 'trading_date')
        if fin_basic_data is None:
            return None
        if np.isnan(fin_basic_data['NetProfit']):
            return None
        # 读取报告期年初的个股财务摘要数据
        fin_report_date = Utils.get_fin_report_date(calc_date)
        fin_report_date = datetime.datetime(fin_report_date.year-1, 12, 31)
        beg_fin_basic_data = Utils.get_fin_basic_data(code, fin_report_date, 'report_date')

        # 计算平均总资产 = (期初总资产 + 期末总资产) / 2
        beg_total_asset = end_total_asset = 0.0
        if beg_fin_basic_data is None:
            beg_total_asset = 0.0
        else:
            if np.isnan(beg_fin_basic_data['TotalAsset']):
                beg_total_asset = 0.0
            else:
                beg_total_asset = beg_fin_basic_data['TotalAsset']
        if np.isnan(fin_basic_data['TotalAsset']):
            end_total_asset = 0.0
        else:
            end_total_asset = fin_basic_data['TotalAsset']

        if abs(beg_total_asset) < util_ct.TINY_ABS_VALUE and abs(end_total_asset) < util_ct.TINY_ABS_VALUE:
            return None
        elif abs(beg_total_asset) < util_ct.TINY_ABS_VALUE:
            avg_total_asset = end_total_asset
        elif abs(end_total_asset) < util_ct.TINY_ABS_VALUE:
            avg_total_asset = beg_total_asset
        else:
            avg_total_asset = (beg_total_asset + end_total_asset) / 2.0

        froa = fin_basic_data['NetProfit'] / avg_total_asset
        return pd.Series([Utils.code_to_symbol(code), froa], index=['code', 'roa'])

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
        logging.debug('[%s] Calc ROA factor os %s.' % (Utils.datetimelike_to_str(calc_date), code))
        roa_data = None
        try:
            roa_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if roa_data is not None:
            q.put(roa_data)

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
        # 遍历交易日序列, 计算ROA因子载荷
        dict_roa = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc ROA factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股ROA因子载荷
            s = calc_date - datetime.timedelta(days=alphafactor_ct.ROA_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            roas = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's ROA factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    roa_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if roa_data is not None:
                        ids.append(roa_data['code'])
                        roas.append(roa_data['roa'])
            else:
                # 采用多进程计算
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    roa_data = q.get(True)
                    ids.append(roa_data['code'])
                    roas.append(roa_data['roa'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_roa = {'date': [datelabel] * len(ids), 'id': ids, 'factorvalue': roas}
            # 计算去极值标准化后的因子载荷
            df_std_roa = Utils.normalize_data(pd.DataFrame(dict_roa), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_roa['factorvalue'] = round(df_std_roa['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_roa, 'ROA', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_roa, 'ROA', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_roa


class ROAQ(Factor):
    """单季度总资产收益率因子"""

    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.ROAQ_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股单季度总资产收益率因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, e.g: SH600000, 600000
        :param calc_date: datetime-like, str
            因子载荷计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
        --------
            Series的index为:
            0.code
            1.roaq
        """
        # 读取个股单季度财报数据
        fin_quarter_basicdata = Utils.get_fin_singlequarter_basicdata(code, calc_date, 'trading_date')
        if fin_quarter_basicdata is None:
            return None
        if np.isnan(fin_quarter_basicdata['NetProfit']):
            return None

        # 平均总资产 = (季度初总资产 + 季度末总资产) / 2
        if np.isnan(fin_quarter_basicdata['BegTotalAsset']):
            beg_total_asset = 0.0
        else:
            beg_total_asset = fin_quarter_basicdata['BegTotalAsset']
        if np.isnan(fin_quarter_basicdata['EndTotalAsset']):
            end_total_asset = 0.0
        else:
            end_total_asset = fin_quarter_basicdata['EndTotalAsset']

        if abs(beg_total_asset) < util_ct.TINY_ABS_VALUE and abs(end_total_asset) < util_ct.TINY_ABS_VALUE:
            return None
        elif abs(beg_total_asset) < util_ct.TINY_ABS_VALUE:
            avg_total_asset = end_total_asset
        elif abs(end_total_asset) < util_ct.TINY_ABS_VALUE:
            avg_total_asset = beg_total_asset
        else:
            avg_total_asset = (beg_total_asset + end_total_asset) / 2.0

        froaq = fin_quarter_basicdata['NetProfit'] / avg_total_asset
        return pd.Series([Utils.code_to_symbol(code), froaq], index=['code', 'roaq'])

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
        logging.debug('[%s] Calc ROAQ factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        roaq_data = None
        try:
            roaq_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if roaq_data is not None:
            q.put(roaq_data)

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
        if end_date is not None:
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算ROAQ因子载荷
        dict_roaq = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc ROAQ factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股ROAQ因子载荷
            s = calc_date - datetime.timedelta(days=alphafactor_ct.ROAQ_CT.listed_days)
            stock_basics = Utils.get_stock_basics(s)
            ids = []
            roaqs = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's ROAQ factor." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    roaq_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if roaq_data is not None:
                        ids.append(roaq_data['code'])
                        roaqs.append(roaq_data['roaq'])
            else:
                # 采用多进程计算
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    roaq_data = q.get(True)
                    ids.append(roaq_data['code'])
                    roaqs.append(roaq_data['roaq'])

            datelabel = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_roaq = {'date': [datelabel]*len(ids), 'id': ids, 'factorvalue': roaqs}
            # 计算去极值标准化后的因子载荷
            df_std_roaq = Utils.normalize_data(pd.DataFrame(dict_roaq), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_roaq['factorvalue'] = round(df_std_roaq['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_roaq, 'ROAQ', 'raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_roaq, 'ROAQ', 'standardized', columns=['date', 'id', 'factorvalue'])

        return dict_roaq


if __name__ == '__main__':
    pass
    # ROE.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=True)
    # ROA.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=True)
    # ROEQ.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=True)
    ROAQ.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=True)
