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
        if np.isnan(fin_basic_data['ShareHolderEquity']):
            return None
        if abs(fin_basic_data['ShareHolderEquity']) < util_ct.TINY_ABS_VALUE:
            return None

        froe = fin_basic_data['NetProfit'] / fin_basic_data['ShareHolderEquity']
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


if __name__ == '__main__':
    pass
    ROE.calc_factor_loading(start_date='2018-09-28', end_date='2018-09-28', month_end=True, save=True, multi_proc=True)
