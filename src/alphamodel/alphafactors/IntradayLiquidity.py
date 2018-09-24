#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 日内流动性因子
# @Filename: IntradayLiquidity
# @Date:   : 2018-09-23 17:02
# @Author  : YuJun
# @Email   : yujun_mail@163.com


import src.settings as SETTINGS
from src.factors.factor import Factor
import src.alphamodel.alphafactors.cons as alphafactor_ct
from src.util.utils import Utils
import pandas as pd
import numpy as np
import os
import logging
from multiprocessing import Pool, Manager
import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class IntradayLiquidity(Factor):
    """日内流动性因子类"""
    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.INTRADAYLIQUIDITY.db_file)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股日内各时点的流动性因子载荷
        Parameters
        --------
        :param code: str
            个股代码, e.g: 600000, SH600000
        :param calc_date: datetime-like, str
            因子载荷计算日期: e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
        --------
            日内各时段的流动性载荷(换手率), 各个index对应的含义如下:
            0. code: 个股代码
            1. liq1: 第一个小时换手率
            2. liq2: 第二个小时换手率
            3. liq3: 第三个小时换手率
            4. liq4: 第四个小时换手率
            若计算失败, 返回None
        """
        calc_date = Utils.to_date(calc_date)
        # 取得过去90天的交易日序列, 按日期降序排列
        trading_days = Utils.get_trading_days(end=calc_date, ndays=90, ascending=False)
        # 读取个股在过去90个交易日中的最近days_num个交易日的分钟行情数据
        be_enough, mkt_data = Utils.get_min_mkts_fq(code, trading_days, alphafactor_ct.INTRADAYLIQUIDITY.days_num)
        if not be_enough:
            return None
        # 读取个股最新的股本数据
        capstruct_data = Utils.get_cap_struct(code, calc_date)
        if capstruct_data is None:
            return None
        # 计算个股各个时段的换手率值
        mkt_data['time'] = mkt_data['datetime'].apply(lambda x: x[11:])
        liq1 = mkt_data[(mkt_data['time'] >='09:30:00') & (mkt_data['time'] <= '10:30:00')]['vol'].sum()*100 / capstruct_data['liquid_a']
        liq2 = mkt_data[(mkt_data['time'] >='10:31:00') & (mkt_data['time'] <= '11:30:00')]['vol'].sum()*100 / capstruct_data['liquid_a']
        liq3 = mkt_data[(mkt_data['time'] >='13:00:00') & (mkt_data['time'] <= '14:00:00')]['vol'].sum()*100 / capstruct_data['liquid_a']
        liq4 = mkt_data[(mkt_data['time'] >='14:01:00') & (mkt_data['time'] <= '15:00:00')]['vol'].sum()*100 / capstruct_data['liquid_a']

        return pd.Series([liq1, liq2, liq3, liq4], index=['liq1', 'liq2', 'liq3', 'liq4'])

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters:
        --------
        :param code: str
            个股代码, e.g: 600000, SH600000
        :param calc_date: datetime-like, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param q: 队列, 用于进程间通信
        :return: 计算并添加因子载荷至队列中
        """
        logging.debug('[%s] Calc Intraday Liquidity of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        liquidity_data = None
        try:
            liquidity_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if liquidity_data is not None:
            liquidity_data['id'] = Utils.code_to_symbol(code)
            q.put(liquidity_data)

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的因子载荷, 并保存至因子数据库
        Parameters:
        --------
        :param start_date: datetime-like, str
            开始计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param end_date: datetime-like, str
            结束计算日期, e.g: YYYY-MM-DD, YYYYMMDD。默认为None
            如果为None, 则只计算start_date日期的因子载荷
        :param month_end: bool, 默认为True
            如果为True, 则只计算月末时点的因子载荷
        :param save: bool, 默认为False
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程并行计算, False=采用单进程计算, 默认为False
        :return: 因子载荷数据, pd.DataFrame
        --------
            因子载荷数据, pd.DataFrame
            0. date: 日期
            1. id: 证券代码
            2. liq1: 第一个小时换手率
            3. liq2: 第二个小时换手率
            4. liq3: 第三个小时换手率
            5. liq4: 第四个小时换手率
        """
        # 取得交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算日内流动性因子值
        dict_intraday_liquidity = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue

            # 计算日内各时段流动性因子
            dict_intraday_liquidity = {'date': [], 'id': [], 'liq1': [], 'liq2': [], 'liq3': [], 'liq4': []}
            # 遍历个股, 计算个股日内流动性因子值
            s = calc_date - datetime.timedelta(days=alphafactor_ct.INTRADAYLIQUIDITY.listed_days)
            stock_basics = Utils.get_stock_basics(s).iloc[:100]

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程进行计算
                for _, stock_info in stock_basics.iterrows():
                    liquidity_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if liquidity_data is not None:
                        logging.debug("[%s] %s's intraday liquidity = (%0.4f,%0.4f,%0.4f,%0.4f)" % (Utils.datetimelike_to_str(calc_date),stock_info.symbol, liquidity_data.liq1, liquidity_data.liq2, liquidity_data.liq3, liquidity_data.liq4))
                        dict_intraday_liquidity['id'].append(Utils.code_to_symbol(stock_info.symbol))
                        dict_intraday_liquidity['liq1'].append(round(liquidity_data['liq1'], 6))
                        dict_intraday_liquidity['liq2'].append(round(liquidity_data['liq2'], 6))
                        dict_intraday_liquidity['liq3'].append(round(liquidity_data['liq3'], 6))
                        dict_intraday_liquidity['liq4'].append(round(liquidity_data['liq4'], 6))
            else:
                # 采用多进程并行计算日内流动性因子值
                q = Manager().Queue()
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    liquidity_data = q.get(True)
                    dict_intraday_liquidity['id'].append(liquidity_data['id'])
                    dict_intraday_liquidity['liq1'].append(round(liquidity_data['liq1'], 6))
                    dict_intraday_liquidity['liq2'].append(round(liquidity_data['liq2'], 6))
                    dict_intraday_liquidity['liq3'].append(round(liquidity_data['liq3'], 6))
                    dict_intraday_liquidity['liq4'].append(round(liquidity_data['liq4'], 6))

            date_label = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            dict_intraday_liquidity['date'] = [date_label] * len(dict_intraday_liquidity['id'])

            dict_liq1 = {'date': dict_intraday_liquidity['date'], 'id': dict_intraday_liquidity['id'],
                         'factorvalue': dict_intraday_liquidity['liq1']}
            df_std_liq1 = Utils.normalize_data(pd.DataFrame(dict_liq1), columns='factorvalue', treat_outlier=True,
                                               weight='eq')
            dict_liq2 = {'date': dict_intraday_liquidity['date'], 'id': dict_intraday_liquidity['id'],
                         'factorvalue': dict_intraday_liquidity['liq2']}
            df_std_liq2 = Utils.normalize_data(pd.DataFrame(dict_liq2), columns='factorvalue', treat_outlier=True,
                                               weight='eq')
            dict_liq3 = {'date': dict_intraday_liquidity['date'], 'id': dict_intraday_liquidity['id'],
                         'factorvalue': dict_intraday_liquidity['liq3']}
            df_std_liq3 = Utils.normalize_data(pd.DataFrame(dict_liq3), columns='factorvalue', treat_outlier=True,
                                               weight='eq')
            dict_liq4 = {'date': dict_intraday_liquidity['date'], 'id': dict_intraday_liquidity['id'],
                         'factorvalue': dict_intraday_liquidity['liq4']}
            df_std_liq4 = Utils.normalize_data(pd.DataFrame(dict_liq4), columns='factorvalue', treat_outlier=True,
                                               weight='eq')

            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(os.path.join(cls._db_file, 'liq1'), Utils.datetimelike_to_str(calc_date, dash=False), dict_liq1, 'liq1', factor_type='raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(os.path.join(cls._db_file, 'liq1'), Utils.datetimelike_to_str(calc_date, dash=False), df_std_liq1, 'liq1', factor_type='standardized', columns=['date', 'id', 'factorvalue'])

                cls._save_factor_loading(os.path.join(cls._db_file, 'liq2'), Utils.datetimelike_to_str(calc_date, dash=False), dict_liq2, 'liq2', factor_type='raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(os.path.join(cls._db_file, 'liq2'), Utils.datetimelike_to_str(calc_date, dash=False), df_std_liq2, 'liq2', factor_type='standardized', columns=['date', 'id', 'factorvalue'])

                cls._save_factor_loading(os.path.join(cls._db_file, 'liq3'), Utils.datetimelike_to_str(calc_date, dash=False), dict_liq3, 'liq3', factor_type='raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(os.path.join(cls._db_file, 'liq3'), Utils.datetimelike_to_str(calc_date, dash=False), df_std_liq3, 'liq3', factor_type='standardized', columns=['date', 'id', 'factorvalue'])

                cls._save_factor_loading(os.path.join(cls._db_file, 'liq4'), Utils.datetimelike_to_str(calc_date, dash=False), dict_liq4, 'liq4', factor_type='raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(os.path.join(cls._db_file, 'liq4'), Utils.datetimelike_to_str(calc_date, dash=False), df_std_liq4, 'liq4', factor_type='standardized', columns=['date', 'id', 'factorvalue'])

        return dict_intraday_liquidity


if __name__ == '__main__':
    pass
