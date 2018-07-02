#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 风险因子中的盈利预期因子
# @Filename: EarningsYield
# @Date:   : 2018-05-16 16:07
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.factors.factor import Factor
import src.riskmodel.riskfactors.cons as risk_ct
import src.factors.cons as factor_ct
from src.util.utils import Utils, ConsensusType
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

class EPFWD(Factor):
    """预期盈利市值比因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.EPFWD_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股EPFWD因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的EPFWD因子载荷
            0. code
            1. epfwd
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        # 读取个股的预期盈利数据
        predictedearnings_data =  Utils.get_consensus_data(calc_date, code, ConsensusType.PredictedEarings)
        if predictedearnings_data is None:
            # 如果个股的预期盈利数据不存在, 那么代替ttm净利润
            ttm_fin_data = Utils.get_ttm_fin_basic_data(code, calc_date)
            if ttm_fin_data is None:
                return None
            predictedearnings_data = pd.Series([code, ttm_fin_data['NetProfit']], index=['code', 'predicted_earnings'])
        fpredictedearnings = predictedearnings_data['predicted_earnings']
        if np.isnan(fpredictedearnings):
            return None
        # 读取个股市值
        size_path = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.LNCAP_CT.db_file)
        size_factor_loading = Utils.read_factor_loading(size_path, Utils.datetimelike_to_str(calc_date, dash=False), code)
        if size_factor_loading.empty:
            return None
        # epfwd = 盈利预期/市值
        epfwd = fpredictedearnings * 10000.0 / np.exp(size_factor_loading['factorvalue'])

        return pd.Series([code, epfwd], index=['code', 'epfwd'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        ---------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :param q: 队列, 用于进程间通信
        :return: 添加因子载荷至队列
        """
        logging.info('[{}] Calc EPFWD factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        epfwd_data = None
        try:
            epfwd_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if epfwd_data is None:
            epfwd_data = pd.Series([Utils.code_to_symbol(code), np.nan], index=['code', 'epfwd'])
        q.put(epfwd_data)

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
        # 遍历交易日序列, 计算EPFWD因子载荷
        dict_epfwd = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc EPFWD factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的EPFWD因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.EPFWD_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []        # 个股代码list
            epfwds = []     # EPFWD因子值list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算EPFWD因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's EPFWD factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    epfwd_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if epfwd_data is None:
                        ids.append(Utils.code_to_symbol(stock_info.symbol))
                        epfwds.append(np.nan)
                    else:
                        ids.append(epfwd_data['code'])
                        epfwds.append(epfwd_data['epfwd'])
            else:
                # 采用多进程并行计算EPFWD因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    epfwd_data = q.get(True)
                    ids.append(epfwd_data['code'])
                    epfwds.append(epfwd_data['epfwd'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_epfwd = {'date': [date_label]*len(ids), 'id':ids, 'factorvalue': epfwds}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_epfwd, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_epfwd


class CETOP(Factor):
    """现金流量市值比因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.CETOP_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股CETOP因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的CETOP因子载荷
            0. code
            1. cetop
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        # 读取个股的主要财务指标数据ttm值
        ttm_fin_data = Utils.get_ttm_fin_basic_data(code, calc_date)
        if ttm_fin_data is None:
            return None
        ttm_cash = ttm_fin_data['NetOperateCashFlow']
        if np.isnan(ttm_cash):
            return None
        # 读取个股市值
        lncap_path = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.LNCAP_CT.db_file)
        lncap_data = Utils.read_factor_loading(lncap_path, Utils.datetimelike_to_str(calc_date, dash=False), code)
        if lncap_data.empty:
            return None
        secu_cap = np.exp(lncap_data['factorvalue'])
        # cetop = 经营活动现金流ttm值/市值
        cetop = ttm_cash * 10000 / secu_cap

        return pd.Series([code, cetop], index=['code', 'cetop'])

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
        logging.info('[{}] Calc CETOP factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        cetop_data = None
        try:
            cetop_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if cetop_data is None:
            cetop_data = pd.Series([Utils.code_to_symbol(code), np.nan], index=['code', 'cetop'])
        q.put(cetop_data)

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
        # 遍历交易日序列, 计算CETOP因子载荷
        dict_cetop = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc CETOP factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的CETOP因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.CETOP_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []
            cetops = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算CETOP因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's CETOP factor loading." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    cetop_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if cetop_data is None:
                        ids.append(Utils.code_to_symbol(stock_info.symbol))
                        cetops.append(np.nan)
                    else:
                        ids.append(cetop_data['code'])
                        cetops.append(cetop_data['cetop'])
            else:
                # 采用多进程并行计算CETOP因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    cetop_data = q.get(True)
                    ids.append(cetop_data['code'])
                    cetops.append(cetop_data['cetop'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_cetop = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': cetops}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_cetop, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_cetop


class ETOP(Factor):
    """盈利市值比因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.ETOP_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股ETOP因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的ETOP因子载荷
            0. code
            1. etop
            如果计算失败, 返回None
        """
        code = Utils.code_to_symbol(code)
        # 读取个股的ttm净利润
        ttm_fin_data = Utils.get_ttm_fin_basic_data(code, calc_date)
        if ttm_fin_data is None:
            return None
        ttm_netprofit = ttm_fin_data['NetProfit']
        if np.isnan(ttm_netprofit):
            return None
        # 读取个股市值
        lncap_path = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.LNCAP_CT.db_file)
        lncap_data = Utils.read_factor_loading(lncap_path, Utils.datetimelike_to_str(calc_date, dash=False), code)
        if lncap_data.empty:
            return None
        secu_cap = np.exp(lncap_data['factorvalue'])
        # etop = ttm净利润/市值
        etop = ttm_netprofit * 10000 / secu_cap

        return pd.Series([code, etop], index=['code', 'etop'])

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
        logging.info('[{}] Calc ETOP factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        etop_data = None
        try:
            etop_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if etop_data is None:
            etop_data = pd.Series([Utils.code_to_symbol(code), np.nan], index=['code', 'etop'])
        q.put(etop_data)

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
        # 遍历交易日序列, 计算ETOP因子载荷
        dict_etop = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc ETOP factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的ETOP因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.ETOP_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []
            etops = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算ETOP因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's ETOP factor loading." % (Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                    etop_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if etop_data is None:
                        ids.append(Utils.code_to_symbol(stock_info.symbol))
                        etops.append(np.nan)
                    else:
                        ids.append(etop_data['code'])
                        etops.append(etop_data['etop'])
            else:
                # 采用多进程并行计算ETOP因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    etop_data = q.get(True)
                    ids.append(etop_data['code'])
                    etops.append(etop_data['etop'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_etop = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': etops}
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_etop, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_etop


class EarningsYield(Factor):
    """风险因子中的盈利预期因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.EARNINGSYIELD_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        pass

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        pass

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        com_factors = []
        for com_factor in risk_ct.EARNINGSYIELD_CT.component:
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
        # 遍历交易日序列, 计算EarningsYield因子下各个成分因子的因子载荷
        if 'multi_proc' not in kwargs:
            kwargs['multi_proc'] = False
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            for com_factor in risk_ct.EARNINGSYIELD_CT.component:
                factor = eval(com_factor + '()')
                factor.calc_factor_loading(start_date=calc_date, end_date=None, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'])
            # 计算EarningsYield因子载荷
            earningsyield_factor = pd.DataFrame()
            for com_factor in risk_ct.EARNINGSYIELD_CT.component:
                factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, eval('risk_ct.' + com_factor + '_CT')['db_file'])
                factor_loading = Utils.read_factor_loading(factor_path, Utils.datetimelike_to_str(calc_date, dash=False))
                factor_loading.drop(columns='date', inplace=True)
                factor_loading.rename(columns={'factorvalue': com_factor}, inplace=True)
                factor_loading[com_factor] = Utils.normalize_data(Utils.clean_extreme_value(np.array(factor_loading[com_factor]).reshape((len(factor_loading), 1))))
                if earningsyield_factor.empty:
                    earningsyield_factor = factor_loading
                else:
                    earningsyield_factor = pd.merge(left=earningsyield_factor, right=factor_loading, how='inner', on='id')
            earningsyield_factor.set_index('id', inplace=True)
            weight = pd.Series(risk_ct.EARNINGSYIELD_CT.weight)
            earningsyield_factor = (earningsyield_factor * weight).sum(axis=1)
            earningsyield_factor.name = 'factorvalue'
            earningsyield_factor.index.name = 'id'
            earningsyield_factor = pd.DataFrame(earningsyield_factor)
            earningsyield_factor.reset_index(inplace=True)
            earningsyield_factor['date'] = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            # 保存EarningsYield因子载荷
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), earningsyield_factor.to_dict('list'), ['date', 'id', 'factorvalue'])


if __name__ == '__main__':
    pass
    # EPFWD.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    # CETOP.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    # ETOP.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    EarningsYield.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=False)
