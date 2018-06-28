#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 风险模型中的市值因子
# @Filename: Size
# @Date:   : 2018-04-12 22:29
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.factors.factor import Factor
import src.riskmodel.riskfactors.cons as risk_ct
import src.factors.cons as factor_ct
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


class LNCAP(Factor):
    """市值因子中的总市值自然对数类(Natural log of market cap)"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.LNCAP_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股LNCAP因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param calc_date: datetime-like, str
            计算日期, 格式: YYYY-MM-DD
        :return: pd.Series
        --------
            个股的LNCAP因子载荷
            0. code
            1. lncap
            如果计算失败, 返回None
        """
        # 取得个股的非复权收盘价
        df_secu_quote = Utils.get_secu_daily_mkt(code, start=calc_date, fq=False, range_lookup=True)
        if df_secu_quote is None:
            return None
        secu_close = df_secu_quote['close']
        # 取得个股最新的A股总股本数据
        cap_struct = Utils.get_cap_struct(code, calc_date)
        if cap_struct is None:
            return None
        total_share = cap_struct.total - cap_struct.liquid_b - cap_struct.liquid_h
        # 计算总市值的自然对数值
        lncap = np.log(secu_close * total_share)
        liquid_cap = secu_close * cap_struct.liquid_a
        return pd.Series([Utils.code_to_symbol(code), lncap, liquid_cap], index=['code', 'lncap', 'liquid_cap'])

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
        logging.info('[{}] Calc LNCAP factor of {}.'.format(Utils.datetimelike_to_str(calc_date), code))
        lncap_data = None
        try:
            lncap_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if lncap_data is not None:
            q.put(lncap_data)

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
        # 遍历交易日序列, 计算LNCAP因子载荷
        dict_lncap = None
        dict_liquidcap = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc LNCAP factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股的LNCAP因子值
            s = (calc_date - datetime.timedelta(days=risk_ct.LNCAP_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = all_stock_basics[all_stock_basics.list_date < s]
            ids = []    # 个股代码list
            lncaps = [] # LNCAP因子值list
            liquid_caps = []    # 流通市值因子list

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算LNCAP因子值
                for _, stock_info in stock_basics.iterrows():
                    logging.info("[%s] Calc %s's LNCAP factor loading." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    lncap_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if lncap_data is not None:
                        ids.append(lncap_data['code'])
                        lncaps.append(lncap_data['lncap'])
                        liquid_caps.append(lncap_data['liquid_cap'])
            else:
                # 采用多进程并行计算LNCAP因子值
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(4)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    lncap_data = q.get(True)
                    ids.append(lncap_data['code'])
                    lncaps.append(lncap_data['lncap'])
                    liquid_caps.append(lncap_data['liquid_cap'])

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_lncap = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': lncaps}
            dict_liquidcap = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': liquid_caps}
            liquidcap_path = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.LNCAP_CT.liquidcap_dbfile)
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_lncap, ['date', 'id', 'factorvalue'])
                Utils.factor_loading_persistent(liquidcap_path, Utils.datetimelike_to_str(calc_date, dash=False), dict_liquidcap, ['date', 'id', 'factorvalue'])
            # 暂停180秒
            logging.info('Suspending for 180s.')
            # time.sleep(180)
        return dict_lncap


class Size(Factor):
    """风险因子中的市值因子类"""
    _db_file = os.path.join(factor_ct.FACTOR_DB.db_path, risk_ct.SIZE_CT.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        pass

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        pass

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        cls._calc_synthetic_factor_loading(start_date=start_date, end_date=end_date, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'])

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
        # 遍历交易日序列, 计算Size因子下各个成分因子的因子载荷
        if 'multi_proc' not in kwargs:
            kwargs['multi_proc'] = False
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            for com_factor in risk_ct.SIZE_CT.component:
                factor = eval(com_factor + '()')
                factor.calc_factor_loading(start_date=calc_date, end_date=None, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'])
            # 计算Size因子载荷
            size_factor = pd.DataFrame()
            for com_factor in risk_ct.SIZE_CT.component:
                factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, eval('risk_ct.' + com_factor + '_CT')['db_file'])
                factor_loading = Utils.read_factor_loading(factor_path, Utils.datetimelike_to_str(calc_date, dash=False))
                factor_loading.drop(columns='date', inplace=True)
                factor_loading.rename(columns={'factorvalue': com_factor}, inplace=True)
                # factor_loading[com_factor] = Utils.normalize_data(Utils.clean_extreme_value(np.array(factor_loading[com_factor]).reshape((len(factor_loading),1))))
                if size_factor.empty:
                    size_factor = factor_loading
                else:
                    size_factor = pd.merge(left=size_factor, right=factor_loading, how='inner', on='id')
            # 读取个股行业分类数据, 添加至size_factor中
            df_industry_classify = Utils.get_industry_classify()
            size_factor = pd.merge(left=size_factor, right=df_industry_classify[['id', 'ind_code']])
            # 取得含缺失值的因子载荷数据
            missingdata_factor = size_factor.loc[[ind for ind, data in size_factor.iterrows() if data.hasnans]]
            # 剔除size_factor中的缺失值
            size_factor.dropna(axis='index', how='any', inplace=True)
            # 对size_factor去极值、标准化
            size_factor = Utils.normalize_data(size_factor, id='id', columns=risk_ct.SIZE_CT.component, treat_outlier=True, weight='cap', calc_date=calc_date)
            # 把missingdata_factor中的缺失值替换为行业均值
            ind_codes = set(missingdata_factor['ind_code'])
            ind_mean_factor = {}
            for ind_code in ind_codes:
                ind_mean_factor[ind_code] = size_factor[size_factor['ind_code'] == ind_code].mean()
            missingdata_label = {ind: missingdata_factor.columns[missingdata.isna()].tolist() for ind, missingdata in missingdata_factor.iterrows()}
            for ind, cols in missingdata_label.items():
                missingdata_factor.loc[ind, cols] = ind_mean_factor[missingdata_factor.loc[ind, 'ind_code']][cols]
            # 把missingdata_factor和size_factor合并
            size_factor = pd.concat([size_factor, missingdata_factor])
            # 删除ind_code列
            size_factor.drop(columns='ind_code', inplace=True)
            # 合成Size因子
            size_factor.set_index('id', inplace=True)
            weight = pd.Series(risk_ct.SIZE_CT.weight)
            size_factor = (size_factor * weight).sum(axis=1)
            size_factor.name = 'factorvalue'
            size_factor.index.name = 'id'
            size_factor = pd.DataFrame(size_factor)
            size_factor.reset_index(inplace=True)
            size_factor['date'] = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            # 保存Size因子载荷
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), size_factor.to_dict('list'), ['date','id','factorvalue'])



if __name__ == '__main__':
    # pass
    # LNCAP.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
    Size.calc_factor_loading(start_date='2017-12-29', end_date=None, month_end=False, save=True, multi_proc=True)
