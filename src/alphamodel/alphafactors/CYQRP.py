#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 筹码分布因子
# @Filename: CYQ
# @Date:   : 2018-02-23 18:12
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.factors.factor import Factor
import src.settings as SETTINGS
import src.alphamodel.alphafactors.cons as alphafactor_ct
from src.util.utils import Utils
import pandas as pd
from pandas import Series
import numpy as np
import os
from pathlib import Path
import logging
import datetime
from multiprocessing import Pool, Manager
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class CYQRP(Factor):
    """筹码分布因子类"""
    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.CYQRP_CT.db_file)      # 因子对应的数据库文件名

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股筹码分布数据
        Parameters:
        --------
        :param code: str
            个股代码, 如600000或SH600000
        :param calc_date: datetime-like, str
            计算日期, 格式YYYY-MM-DD
        :return: tuple(code, close, cyq_data)
        --------
            1. code
            2. close: float
            个股在calc_date的收盘价
            3. cyq_data: pd.Series
            个股从IPO开始、至calc_date为止的筹码分布数据
            Series的index为筹码价格, values为对应每个筹码价格的持仓比例
            若计算失败, 返回None
        """
        # 读取个股IPO数据
        ipo_data = Utils.get_ipo_info(code)
        if ipo_data is None:
            return None
        if ipo_data['发行价格'][:-1] == '--':
            return None
        ipo_price = float(ipo_data['发行价格'][:-1])
        # 读取个股上市以来的日复权行情数据
        mkt_data = Utils.get_secu_daily_mkt(code, end=calc_date, fq=True)
        secu_close = mkt_data.iloc[-1]['close']
        # 计算每天的均价
        mkt_data['vwap'] = np.around(mkt_data['amount'] / mkt_data['vol'] * mkt_data['factor'], 2)
        mkt_data.dropna(axis=0, how='any', inplace=True)
        # 行情数据按日期降序排列
        mkt_data.sort_values(by='date', ascending=False, inplace=True)
        mkt_data.reset_index(drop=True, inplace=True)
        # 计算筹码分布
        cyq_data = mkt_data[['vwap', 'turnover1']]
        cyq_data = cyq_data.append(Series([ipo_price, 0], index=['vwap', 'turnover1']), ignore_index=True)
        cyq_data['minusTR'] = 1 - cyq_data['turnover1']
        cyq_data['cumprod_TR'] = cyq_data['minusTR'].cumprod().shift(1)
        cyq_data.loc[0, 'cumprod_TR'] = 1.
        cyq_data['cyq'] = cyq_data['turnover1'] * cyq_data['cumprod_TR']
        secu_cyq = cyq_data['cyq'].groupby(cyq_data['vwap']).sum()
        # 如果筹码价格数量小于30个, 返回None
        if len(secu_cyq) < 30:
            return None
        secu_cyq = secu_cyq[secu_cyq.values > 0.00001]
        return (Utils.code_to_symbol(code), secu_close, secu_cyq)

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
        logging.debug('[%s] Calc CYQ factor of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        cyq_data = None
        try:
            cyq_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if cyq_data is not None:
            q.put(cyq_data)

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
        # 遍历交易日序列, 计算筹码分布因子载荷
        dict_cyq = {}
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[%s] Calc CYQ factor loading.' % Utils.datetimelike_to_str(calc_date))
            # 遍历个股, 计算个股筹码分布因子值
            s= (calc_date - datetime.timedelta(days=alphafactor_ct.CYQRP_CT.listed_days)).strftime('%Y%m%d')
            stock_basics = Utils.get_stock_basics(s)

            secu_cyq_path = Path(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.CYQRP_CT.db_file,
                                 'secu_cyq/%s' % calc_date.strftime('%Y-%m-%d'))
            if not secu_cyq_path.exists():
                secu_cyq_path.mkdir()
            ids = []
            rps = []

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程计算筹码分布数据, 及当前价格的相对位置(=当前价格-平均成本)/平均成本
                for _, stock_info in stock_basics.iterrows():
                    logging.debug("[%s] Calc %s's cyq data." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol))
                    secu_cyq = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if secu_cyq is not None:
                        secu_code, secu_close, cyq_data = secu_cyq
                        # 保存个股的筹码分布数据
                        cyq_data.to_csv(Path(secu_cyq_path, '%s.csv' % secu_code), header=True)
                        # 计算当前价格的相对位置
                        avg_cyq = np.sum(np.array(cyq_data.index) * np.array(cyq_data.values))
                        relative_position = round((secu_close - avg_cyq) / avg_cyq, 6)
                        ids.append(secu_code)
                        rps.append(relative_position)
            else:
                # 采用多进程进行并行计算筹码分布数据, 及当前价格的相对位置(=当前价格-平均成本)/平均成本
                q = Manager().Queue()   # 队列, 用于进程间通信, 存储每个进程计算的因子载荷
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)             # 进程池, 最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q, ))
                p.close()
                p.join()
                while not q.empty():
                    secu_cyq = q.get(True)
                    secu_code, secu_close, cyq_data = secu_cyq
                    # 保存个股的筹码分布数据
                    cyq_data.to_csv(Path(secu_cyq_path, '%s.csv' % secu_code), header=True)
                    # 计算当前价格的相对位置
                    avg_cyq = np.sum(np.array(cyq_data.index) * np.array(cyq_data.values))
                    relative_position = round((secu_close - avg_cyq) / avg_cyq, 6)
                    ids.append(secu_code)
                    rps.append(relative_position)

            date_label = Utils.get_trading_days(calc_date, ndays=2)[1]
            dict_cyq = {'date': [date_label]*len(ids), 'id': ids, 'factorvalue': rps}
            # 计算去极值标准化后的因子载荷
            df_std_cyq = Utils.normalize_data(pd.DataFrame(dict_cyq), columns='factorvalue', treat_outlier=True, weight='eq')
            df_std_cyq['factorvalue'] = round(df_std_cyq['factorvalue'], 6)
            # 保存因子载荷至因子数据库
            if save:
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_cyq, 'CYQRP', factor_type='raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_cyq, 'CYQRP', factor_type='standardized', columns=['date', 'id', 'factorvalue'])
            # 休息90秒
            # logging.info('Suspending for 100s.')
            # time.sleep(100)
        return dict_cyq


if __name__ == '__main__':
    # pass
    CYQRP.calc_factor_loading(start_date='2018-8-31', end_date='2018-08-31', month_end=True, save=True, multi_proc=False)
    # CYQ._calc_factor_loading('000722', '2006-12-29')
