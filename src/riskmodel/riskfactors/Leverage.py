#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 风险因子中的杠杆因子
# @Filename: Leverage
# @Date:   : 2018-05-23 14:28
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.factors.factor import Factor
import src.riskmodel.riskfactors.cons as risk_ct
import src.factors.cons as factor_ct
import src.util.cons as utils_con
from src.util.utils import Utils
from src.util.dataapi.CDataHandler import  CDataHandler
import pandas as pd
import numpy as np
import logging
import os
import datetime
from multiprocessing import Pool, Manager
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class MLEV(Factor):
    """市场杠杆因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.MLEV_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股MLEV因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如Sh600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的MLEV因子载荷
            0. code
            1. mlev
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        report_date = Utils.get_fin_report_date(calc_date)
        # 读取个股最新财务报表摘要数据
        fin_summary_data = Utils.get_fin_summary_data(code, report_date)
        # ld为个股长期负债的账面价值, 如果缺失长期负债数据, 则用负债总计代替
        if fin_summary_data is None:
            return None
        ld = fin_summary_data['TotalNonCurrentLiabilities']
        if np.isnan(ld):
            ld = fin_summary_data['TotalLiabilities']
        if np.isnan(ld):
            return None
        ld *= 10000.0
        # pe为优先股账面价值, 对于A股pe设置为0
        pe = 0.0
        # 读取个股市值数据
        lncap_path = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.LNCAP_CT.db_file)
        lncap_factor_loading = Utils.read_factor_loading(lncap_path, Utils.datetimelike_to_str(calc_date, dash=False), code)
        if lncap_factor_loading.empty:
            return None
        me = np.exp(lncap_factor_loading['factorvalue'])
        # mlev = (me + pe + ld)/me
        mlev = (me + pe + ld) / me

        return pd.Series([code, mlev], index=['code', 'mlev'])

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
        logging.info('[{}] Calc MLEV factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        mlev_data = None
        try:
            mlev_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if mlev_data is not None:
            q.put(mlev_data)

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
        # 遍历交易日序列, 计算MLEV因子载荷
        dict_mlev = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc MLEV factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的MLEV因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.MLEV_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []    # 个股代码list
            mlevs = []  # MLEV因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算MLEV因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's MLEV factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    mlev_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if mlev_data is not None:
                        ids.append(mlev_data['code'])
                        mlevs.append(mlev_data['mlev'])
            else:
                # 采用多进程并行计算MLEV因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    mlev_data = q.get(True)
                    ids.append(mlev_data['code'])
                    mlevs.append(mlev_data['mlev'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_mlev = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': mlevs}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_mlev, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_mlev


class DTOA(Factor):
    """资产负债比因子"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.DTOA_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股DTOA因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的DTOA因子载荷
            0. code
            1. dtoa
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        report_date = Utils.get_fin_report_date(calc_date)
        # 读取最新主要财务指标数据
        fin_basic_data = Utils.get_fin_basic_data(code, report_date)
        if fin_basic_data is None:
            return None
        # td为负债总额, ta为总资产
        td = fin_basic_data['TotalLiability']
        if np.isnan(td):
            return None
        ta = fin_basic_data['TotalAsset']
        if np.isnan(ta):
            return None
        if abs(ta) < utils_con.TINY_ABS_VALUE:
            return None
        # dtoa = td / ta
        dtoa = td / ta

        return pd.Series([code, dtoa], index=['code', 'dtoa'])

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
        logging.info('[{}] Calc DTOA factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        dtoa_data = None
        try:
            dtoa_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if dtoa_data is not None:
            q.put(dtoa_data)

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
        # 遍历交易日序列, 计算DTOA因子载荷
        dict_dtoa = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc DTOA factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的DTOA因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.DTOA_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []    # 个股代码list
            dtoas = []  # DTOA因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算DTOA因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Cacl %s's DTOA factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    dtoa_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if dtoa_data is not None:
                        ids.append(dtoa_data['code'])
                        dtoas.append(dtoa_data['dtoa'])
            else:
                # 采用多进程并行计算DTOA因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    dtoa_data = q.get(True)
                    ids.append(dtoa_data['code'])
                    dtoas.append(dtoa_data['dtoa'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_dtoa = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': dtoas}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_dtoa, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_dtoa


class BLEV(Factor):
    """账面杠杆因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.BLEV_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股BLEV因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的BLEV因子载荷
            0. code
            1. blev
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        report_date = Utils.get_fin_report_date(calc_date)
        # 读取个股最新财务报表摘要数据
        fin_summary_data = Utils.get_fin_summary_data(code, report_date)
        if fin_summary_data is None:
            return None
        be = fin_summary_data['TotalShareholderEquity']
        if np.isnan(be):
            return None
        if abs(be) < utils_con.TINY_ABS_VALUE:
            return None
        ld = fin_summary_data['TotalNonCurrentLiabilities']
        if np.isnan(ld):
            ld = fin_summary_data['TotalLiabilities']
            if np.isnan(ld):
                return None
        pe = 0
        # blev = (be + pe + ld) / be
        blev = (be + pe + ld) / be

        return pd.Series([code, blev], index=['code', 'blev'])

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
        logging.info('[{}] Calc BLEV factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        blev_data = None
        try:
            blev_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if blev_data is not None:
            q.put(blev_data)

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
        # 遍历交易日序列, 计算BLEV因子载荷
        dict_blev = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc BLEV factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的BLEV因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.BLEV_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []    # 个股代码list
            blevs = []  # BLEV因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算BLEV因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's BLEV factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    blev_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if blev_data is not None:
                        ids.append(blev_data['code'])
                        blevs.append(blev_data['blev'])
            else:
                # 采用多进程并行计算BLEV因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    blev_data = q.get(True)
                    ids.append(blev_data['code'])
                    blevs.append(blev_data['blev'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_blev = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': blevs}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_blev, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_blev


class Leverage(Factor):
    """风险因子中的杠杆因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.LEVERAGE_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        pass

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        pass

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        com_factors = []
        for com_factor in risk_ct.LEVERAGE_CT.component:
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
        # 遍历交易日序列, 计算Leverage因子下各个成分因子的因子载荷
        if 'multi_proc' not in kwargs:
            kwargs['multi_proc'] = False
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            # 计算各成分因子的因子载荷
            for com_factor in risk_ct.LEVERAGE_CT.component:
                factor = eval(com_factor + '()')
                factor.calc_factor_loading(start_date=calc_date, end_date=None, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'])
            # 计算Leverage因子载荷
            leverage_factor = pd.DataFrame()
            for com_factor in risk_ct.LEVERAGE_CT.component:
                factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, eval('risk_ct.' + com_factor + '_CT')['db_file'])
                factor_loading = Utils.read_factor_loading(factor_path, Utils.datetimelike_to_str(calc_date, dash=False))
                factor_loading.drop(columns='date', inplace=True)
                factor_loading.rename(columns={'factorvalue': com_factor}, inplace=True)
                factor_loading[com_factor] = Utils.normalize_data(Utils.clean_extreme_value(np.array(factor_loading[com_factor]).reshape((len(factor_loading), 1))))
                if leverage_factor.empty:
                    leverage_factor = factor_loading
                else:
                    leverage_factor = pd.merge(left=leverage_factor, right=factor_loading, how='inner', on='id')
            leverage_factor.set_index('id', inplace=True)
            weight = pd.Series(risk_ct.LEVERAGE_CT.weight)
            leverage_factor = (leverage_factor * weight).sum(axis=1)
            leverage_factor.name = 'factorvalue'
            leverage_factor.index.name = 'id'
            leverage_factor = pd.DataFrame(leverage_factor)
            leverage_factor.reset_index(inplace=True)
            leverage_factor['date'] = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            # 保存Leverage因子载荷
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), leverage_factor.to_dict('list'), ['date', 'id', 'factorvalue'])


if __name__ == '__main__':
    pass
    # MLEV.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    # DTOA.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    # BLEV.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    Leverage.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=False)
