#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Filename: utils
# @Date:   : 2017-11-27 17:27
# @Author  : YuJun
# @Email   : yujun_mail@163.com

import os
import pandas as pd
import numpy as np
from pandas import Series
from pandas import DataFrame
import csv
import datetime
import calendar
import shelve
from enum import Enum, auto
from src.util import cons as ct
import src.riskmodel.riskfactors.cons as risk_ct
from src.util.Cache import Cache
import tushare as ts
from collections import Iterable


class SecuTradingStatus(Enum):
    """
    个股的交易状态：正常、停牌、涨停、跌停
    """
    Normal = auto()         # 正常交易
    Suspend = auto()        # 停牌
    LimitUp = auto()        # 涨停
    LimitDown = auto()      # 跌停

class ConsensusType(object):
    """
    一致预期数据类型
    """
    PredictedEarings = auto()           # 预期盈利
    PredictedEarningsGrowth = auto()    # 预期盈利增长率


class Utils(object):

    _DataCache = Cache(500)    # 数据缓存

    @classmethod
    def get_stock_basics(cls, date=None, remove_st=False, remove_suspension=False, all=False):
        """
        读取上市交易个股列表信息
        Parameters:
        --------
        :param date: datetime-like or str
            交易日期, 格式: YYYY-MM-DD or YYYYMMDD
            默认为None, 读取最新的上市交易个股信息列表
        :param remove_st: bool
            是否剔除st个股, 默认为False, 即不剔除
        :param remove_suspension: bool
            是否剔除停牌个股, 默认为False, 即不剔除
        :param all: bool
            是否返回全部个股代码, 包括已退市个股, 默认为False
            如果为True, 那么其他参数失效
        :return: pd.DataFrame
        --------
            0. symbol: 个股代码, e.g: SH600000
            1. name: 个股简称
            2. list_date: 上市日期
            3. delist_date: 退市日期, 未退市的该值等于99999999
            4. status: 状态, 1=正常交易, 3=已退市
            5. market: 交易市场, SH=上交所, SZ=深交所
            6. currency: 报价货币
        """
        if date is None:
            date = datetime.date.today()
        date = cls.datetimelike_to_str(date, dash=False)

        df_stock_basics = pd.read_csv(os.path.join(ct.DB_PATH, ct.BASIC_INFO_PATH, 'stock_basics.csv'), dtype={'list_date': str, 'delist_date': str})
        if all:
            df_stock_basics.reset_index(drop=True, inplace=True)
            return df_stock_basics
        df_stock_basics = df_stock_basics[(df_stock_basics['list_date'] <= date) & (df_stock_basics['delist_date'] > date)]
        if remove_st:
            st_stocks = cls.get_st_stocks(date)
            df_stock_basics = df_stock_basics[~df_stock_basics['symbol'].isin(st_stocks)]
        if remove_suspension:
            df_suspension_info = cls.get_suspension_info(date)
            df_stock_basics = df_stock_basics[~df_stock_basics['symbol'].isin(df_suspension_info['symbol'])]
        df_stock_basics.reset_index(drop=True, inplace=True)
        return df_stock_basics

    @classmethod
    def get_st_stocks(cls, date=None):
        """
        读取st个股元组
        Parameters:
        --------
        :param date: datetime-like or str
            交易日期, e.g. YYYY-MM-DD or YYYYMMDD
            默认为None, 读取最新的st个股列表
        :return: set
        --------
            st个股代码set
        """
        if date is None:
            date = datetime.date.today()
        date = int(cls.datetimelike_to_str(date, dash=False))
        df_st_info = pd.read_csv(os.path.join(ct.DB_PATH, ct.BASIC_INFO_PATH, 'st_info.csv'))
        df_st_info = df_st_info[(df_st_info['st_start'] <= date) & (df_st_info['st_end'] > date)]
        if df_st_info.empty:
            return set()
        else:
            return set(df_st_info['code'])

    @classmethod
    def get_suspension_info(cls, date):
        """
        读取指定日期个股停牌信息数据
        Parameters:
        --------
        :param date: datetime-like or str
            交易日期, e.g: YYYY-MM-DD or YYYYMMDD
        :return: pd.DataFrame
        --------
            0. symbol:      个股代码
            1. name:        个股简称
            2. list_date:   上市日期
            3. delist_date: 退市日期
            4. status:      状态, 1=上市, 3=退市
            5. market:      交易市场, SH=上交所, SZ=深交所
            6. currency:    交易货币
            读取失败, 返回None
        """
        suspension_info_path = os.path.join(ct.DB_PATH, ct.SUSPENSION_INOF_PATH, '{}.csv'.format(cls.datetimelike_to_str(date, dash=False)))
        if os.path.exists(suspension_info_path):
            df_suspension_info = pd.read_csv(suspension_info_path, header=0)
            return df_suspension_info
        else:
            return None

    @classmethod
    def calc_interval_ret(cls, secu_code, start=None, end=None, ndays=None):
        """
        计算证券区间收益率
        计算区间分三种方式指定：
        （1）指定开始、结束日期，即start和end不为空，此时忽略参数ndays
        （2）指定开始日期、天数，即start和ndays不为空，而end为空
        （3）指定结束日期、天数，即end和ndays不为空，而start为空
        --------
        :param secu_code: string
            证券代码，e.g. 600000
        :param start: string or datetime-like
            开始日期，格式：YYYY-MM-DD
        :param end: string or datetime-like
            结束日期，格式：YYYY-MM-DD
        :param ndays: int
            交易日天数
        :return:
        --------
            float，证券的区间收益率
            计算失败，返回None
        """
        symbol = _code_to_symbol(secu_code)
        file_path = '%s.csv' % (os.path.join(ct.DB_PATH, ct.MKT_DAILY_FQ, symbol))
        headers = ['code', 'date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'turnover1', 'turnover2', 'factor']
        df_mkt = pd.read_csv(file_path, names=headers, header=0)
        # 如果start或end是datetime.datetime/datetime.date类型，将其转化为字符串
        if isinstance(start, datetime.datetime) or isinstance(start, datetime.date):
            start = start.strftime('%Y-%m-%d')
        if isinstance(end, datetime.datetime) or isinstance(end, datetime.date):
            end = end.strftime('%Y-%m-%d')
        if start is not None and end is not None:
            # 取得开始日期前一交易日和结束日期之间的索引值列表，根据这个索引值列表取得复权行情切片，计算区间收益率
            inds = list(df_mkt[(df_mkt.date >= start) & (df_mkt.date <= end)].index.values)
            if len(inds) > 0:
                if inds[0] > 0:
                    inds.insert(0, inds[0]-1)
                    df_mkt = df_mkt.ix[inds]
                    interval_ret = df_mkt.iloc[-1, 5] / df_mkt.iloc[0, 5] - 1.0
                else:
                    # 如果开始日期小于等于该证券上市日期，那么起始价格取发行价(若有)或上市日期开盘价(若没有发行价)
                    df_mkt = df_mkt.loc[inds]
                    ipo_info = cls.get_ipo_info(symbol)
                    if ipo_info is None:
                        interval_ret = df_mkt.iloc[-1, 5] / df_mkt.iloc[0, 2] -1.0
                    else:
                        if ipo_info['发行价格'][:-1] != '--':
                            ipo_price = float(ipo_info['发行价格'][:-1])
                            interval_ret = df_mkt.iloc[-1, 5] / ipo_price - 1.0
                        else:
                            interval_ret = df_mkt.iloc[-1, 5] / df_mkt.iloc[0, 2] - 1.0
                # interval_ret = df_mkt.iloc[-1, 5]/df_mkt.iloc[0, 5] - 1.0
            else:
                # 如果在指定的开始、结束日期间该证券没有行情数据，返回None
                interval_ret = None
        elif start is not None and ndays is not None:
            try:
                start_ind = df_mkt[df_mkt.date >= start].iloc[0].name
            except IndexError:
                start_ind = -1
            if start_ind < 0:
                interval_ret = None
            else:
                if start_ind <= df_mkt.iloc[0].name:
                    start_close = df_mkt.iloc[0].open
                else:
                    start_close = df_mkt.loc[start_ind-1, 'close']
                end_ind = start_ind + ndays - 1
                if end_ind > df_mkt.iloc[-1].name:
                    end_ind = df_mkt.iloc[-1].name
                end_close = df_mkt.loc[end_ind, 'close']
                interval_ret = end_close / start_close - 1.0
        elif end is not None and ndays is not None:
            try:
                end_ind = df_mkt[df_mkt.date <= end].iloc[-1].name
            except IndexError:
                end_ind = -1
            if end_ind < 0:
                interval_ret = None
            else:
                end_close = df_mkt.loc[end_ind, 'close']
                start_ind = end_ind - ndays
                if start_ind < df_mkt.iloc[0].name:
                    start_close = df_mkt.iloc[0].open
                else:
                    start_close = df_mkt.loc[start_ind, 'close']
                interval_ret = end_close / start_close - 1.0
        else:
            interval_ret = None
        return interval_ret

    # 交易日序列静态变量，Series
    utils_trading_days = Series()

    @classmethod
    def get_trading_days(cls, start=None, end=None, ndays=None, ascending=True):
        """
        取得交易日列表，分三种方式取得
        （1）指定开始、结束日期，即start和end不为None，此时忽略参数ndays
        （2）指定开始日期和天数，即start和ndays不为None，而end为None
        （3）指定结束日期和天数，即end和ndays为None，而start为None
        --------
        :param start: datetime-like or str
            开始日期，格式：YYYY-MM-DD
        :param end: datetime-like or str
            结束日期，格式：YYYY-MM-DD
        :param ndays: int
            交易日天数
        :param ascending: bool，默认True
            是否升序排列
        :return:
        --------
            Series of pandas.Timestamp，交易日列表，默认按交易日升序排列
        """
        if len(Utils.utils_trading_days) == 0:
            # ts_conn = ts.get_apis()
            # df_SZZS = ts.bar(code='000001', conn=ts_conn, asset='INDEX')
            # ts.close_apis(ts_conn)
            # Utils.utils_trading_days = Series(df_SZZS.index).sort_values()

            calendar_path = os.path.join(ct.DB_PATH, ct.BASIC_INFO_PATH, 'trading_days.csv')
            df_trading_days = pd.read_csv(calendar_path, parse_dates=[0])
            Utils.utils_trading_days = df_trading_days['trading_day']
        if start is not None:
            start = cls.to_date(start)
        if end is not None:
            end = cls.to_date(end)
        if start is not None and end is not None:
            trading_days = Utils.utils_trading_days[(Utils.utils_trading_days >= start) & (Utils.utils_trading_days <= end)]
        elif start is not None and ndays is not None:
            trading_days = Utils.utils_trading_days[Utils.utils_trading_days >= start].iloc[:ndays]
        elif end is not None and ndays is not None:
            trading_days = Utils.utils_trading_days[Utils.utils_trading_days <= end].iloc[-ndays:]
        elif start is not None:
            trading_days = Utils.utils_trading_days[Utils.utils_trading_days >= start]
        elif end is not None:
            trading_days = Utils.utils_trading_days[Utils.utils_trading_days <= end]
        elif ndays is not None:
            trading_days = Utils.utils_trading_days.iloc[-ndays:]
        else:
            trading_days = Utils.utils_trading_days
        trading_days = trading_days.reset_index(drop=True)
        if not ascending:
            trading_days = trading_days.sort_values(ascending=False)
        return trading_days

    @classmethod
    def get_prev_n_day(cls, end, ndays=1):
        """
        取得截止日期前的第几个交易日
        :param end: datetime-like, str
            截止日期
        :param ndays: int
            天数
        :return: pandas.Timestamp
        """
        end = cls.to_date(end)
        trading_days = cls.get_trading_days(end=end)
        if trading_days.iloc[-1] == end:
            return trading_days.iloc[-(ndays+1)]
        else:
            return trading_days.iloc[-ndays]

    @classmethod
    def get_next_n_day(cls, start, ndays=1):
        """
        取得起始日期后的第几个交易日
        Parameters:
        --------
        :param start: datetime-like, str
            起始日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param ndays: int
            第几天
        :return: pandas.Timestamp
        """
        start = cls.to_date(start)
        trading_days = cls.get_trading_days(start=start)
        if trading_days.iloc[0] == start:
            return trading_days.iloc[ndays]
        else:
            return trading_days.iloc[(ndays-1)]

    @classmethod
    def is_month_end(cls, trading_day):
        """
        是否是月末的交易日
        :param trading_day: datetime-like, str
        :return: bool
        """
        trading_day = Utils.to_date(trading_day)
        trading_days = Utils.get_trading_days(start=trading_day, ndays=2)
        if trading_day != trading_days[0]:
            return False
        else:
            if trading_day.month == trading_days[1].month:
                return False
            else:
                return True

    @classmethod
    def is_month_start(cls, trading_day):
        """
        是否是月初的交易日
        :param trading_day: datetime-like, str
        :return:
        """
        trading_day = cls.to_date(trading_day)
        trading_days = cls.get_trading_days(end=trading_day, ndays=2)
        if trading_day != trading_days[1]:
            return False
        else:
            if trading_day.month == trading_days[0].month:
                return False
            else:
                return True

    @classmethod
    def get_secu_daily_mkt(cls, secu_code, start=None, end=None, ndays=None,fq=False, range_lookup=False):
        """
        读取证券的日行情数据
        Parameters
        --------
        :param secu_code: str
            证券代码，e.g. 600000 or SH600000
        :param start: datetime-like, str
            开始日期，格式：YYYY-MM-DD
        :param end: datetime-like, str，默认None
            结束日期，格式：YYYY-MM-DD；当end=None & ndays=None时，只取start一天的行情数据
        :param ndays: int, 默认None
            行情天数
        :param fq: bool，默认False
            是否读取复权行情数据
        :param range_lookup: bool，默认False
            是否模糊匹配，False=精确匹配，True=模糊匹配，该参数仅对end=None时适用，
            当range_lookup=False，返回trading_day=start的行情数据，没有行情数据返回空数据。
            当range_lookup=True，如果start没有行情数据时，返回start之前最近一个交易日行情数据.
        :return:
        --------
            1. 指定开始、结束日期, 即start和end不为None, 此时忽略ndays, 返回pd.DataFrame
            2. 指定开始日期和天数, 即start和ndays不为None、而end为None, 返回pd.DataFrame
            3. 指定结束日期和天数, 即end和ndays不为None、而start为None, 返回pd.DataFrame
            4. 仅指定开始日期, 即start不为None、而end和ndays为None, 返回pd.Series, 返回start当天或之前最新的行情(根据range_lookup参数确定)
            5. 仅指定结束日期, 即end不为None、而start和ndays为None, 返回pd.DataFrame, 从上市日到end的日行情数据
            code    证券代码
            date    日期
            open    开盘价
            high    最高价
            low     最低价
            close   收盘价
            vol     成交量
            amount  成交金额
            turnover1   流通盘换手率
            turnover2   全流通换手率
            factor  复权系数
        """
        symbol = _code_to_symbol(secu_code)
        if fq:
            file_path = '%s.csv' % os.path.join(ct.DB_PATH, ct.MKT_DAILY_FQ, symbol)
            if not os.path.exists(file_path):
                return None
            df_mkt = pd.read_csv(file_path, names=ct.MKT_DAILY_FQ_HEADER, header=0)
        else:
            file_path = '%s.csv' % os.path.join(ct.DB_PATH, ct.MKT_DAILY_NOFQ, symbol)
            if not os.path.exists(file_path):
                return None
            df_mkt = pd.read_csv(file_path, names=ct.MKT_DAILY_NOFQ_HEADER, header=0)
        if len(df_mkt) == 0:
            return None
        if start is not None:
            start = cls.datetimelike_to_str(start, dash=True)
        if end is not None:
            end = cls.datetimelike_to_str(end, dash=True)
        if start is not None and end is not None:
            mkt_data = df_mkt[(df_mkt.date >= start) & (df_mkt.date <= end)]
        elif start is not None and ndays is not None:
            mkt_data = df_mkt[df_mkt.date >= start].head(ndays)
        elif end is not None and ndays is not None:
            mkt_data = df_mkt[df_mkt.date <= end].tail(ndays)
        elif start is not None:
            if range_lookup:
                # mkt_data = df_mkt[df_mkt.date <= start].iloc[-1]
                mkt_data = df_mkt[df_mkt.date <= start]
                if mkt_data.empty:
                    mkt_data = pd.Series()
                else:
                    mkt_data = mkt_data.iloc[-1]
            else:
                mkt_data = df_mkt[df_mkt.date == start]
                if mkt_data.shape[0] == 0:
                    mkt_data = Series()
                else:
                    mkt_data = mkt_data.iloc[0]
        elif end is not None:
            # if range_lookup:
            #     mkt_data = df_mkt[df_mkt.date <= end].iloc[-1]
            # else:
            #     mkt_data = df_mkt[df_mkt.date == end]
            #     if mkt_data.shape[0] == 0:
            #         mkt_data = Series()
            #     else:
            #         mkt_data = mkt_data.iloc[0]
            mkt_data = df_mkt[df_mkt.date <= end]
        else:
            mkt_data = None
        if (isinstance(mkt_data, pd.DataFrame) or isinstance(mkt_data, pd.Series)) and mkt_data.empty:
            mkt_data = None
        return mkt_data

    @classmethod
    def get_min_mkt(cls, code, trade_date, index=False, fq=False):
        """
        获取证券（个股或指数）指定日期的分钟行情数据
        Parameters:
        --------
        :param code: string
            证券代码，如600000,SH600000,SZ000002
        :param trade_date: datetime-like, str
            交易日，当类型为str时格式为YYYY-MM-DD
        :param index: bool,默认False
        :param fq: bool,默认False
            是否复权
        :return:
            证券分钟行情数据，DataFrame
        --------
        DataFrame
            0: code，个股代码，如SH600000
            1: time，时间，格式YYYY-MM-DD hh:mm:ss
            2: open，开盘价
            3: high，最高价
            4: low，最低价
            5: close，收盘价
            6: volume，成交量(手)
            7: amount，成交金额(元)
            8: factor，复权系数
            如果没有行情数据返回None
        """
        symbol = cls.code_to_symbol(code, index)
        str_date = cls.datetimelike_to_str(trade_date)
        if fq:
            mkt_file_path = os.path.join(ct.DB_PATH, ct.MKT_MIN_FQ, str_date, '%s.csv' % symbol)
        else:
            mkt_file_path = os.path.join(ct.DB_PATH, ct.MKT_MIN_NOFQ, str_date, '%s.csv' % symbol)
        key = '%s_1min_mkt_%s' % (symbol, cls.to_date(trade_date).strftime('%Y%m%d'))
        df_mkt_min = cls._DataCache.get(key)
        if df_mkt_min is None and os.path.isfile(mkt_file_path):
            df_mkt_min = pd.read_csv(mkt_file_path, names=ct.MKT_MIN_FQ_HEADER, skiprows=[0])
            cls._DataCache.set(key, df_mkt_min)
        # else:
        #     df_mkt_min = None
        return df_mkt_min

    @classmethod
    def get_min_mkts_fq(cls, code, days, ret_num):
        """
        获取个股指定日期的复权分钟行情数据，日期范围由days列表指定，返回ret_num天的数据
        Parameters:
        ------
        :param code:string
            个股代码，如SH600000或600000
        :param days:list-like of string/datetime like, YYYY-MM-DD
            日期列表
        :param ret_num:int
            返回的交易日数量
        :return:
            例如：如果要取得浦发银行过去30个交易日中的10个交易日复权分钟行情数据，那么参数设置为：
                 code=SH600000, days为过去30个交易日列表, ret_num=10
        ------
          DataFrame
            0: code，个股代码，如SH600000
            1: time，时间，格式YYYY-MM-DD hh:mm:ss
            2: open，开盘价
            3: high，最高价
            4: low，最低价
            5: close，收盘价
            6: volume，成交量(手)
            7: amount，成交金额(元)
            8: factor，复权系数
            如果给定的日期范围内读取分钟数据天数小于ret_num天，那么be_enough=False，否则be_enough=True
        """
        # cfg = ConfigParser()
        # cfg.read('config.ini')
        # db_path = cfg.get('factor_db', 'db_path')   # 读取因子数据库路径
        # db_path = factor_ct.FACTOR_DB.db_path
        df_min_mkt = DataFrame()
        k = 0
        for trading_date in days:
            mkt_file_path = os.path.join(ct.DB_PATH, ct.MKT_MIN_FQ, Utils.datetimelike_to_str(trading_date), '%s.csv' %
                                         Utils.code_to_symbol(code))
            if os.path.isfile(mkt_file_path):
                # 读取个股每天的分钟行情数据
                df = pd.read_csv(mkt_file_path,
                                 names=['code', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'factor'],
                                 skiprows=[0])
                # 计算每分钟的涨跌幅，每天第一分钟的涨跌幅=close/open-1
                df['ret'] = df['close'] / df['close'].shift(1) - 1.0
                df.ix[0, 'ret'] = df.ix[0, 'close'] / df.ix[0, 'open'] - 1.0
                # 拼接数据
                df_min_mkt = df_min_mkt.append(df, ignore_index=True)
                k += 1
                if k >= ret_num:
                    break
        be_enough = True
        if k < ret_num:
            be_enough = False
        return be_enough, df_min_mkt

    # 个股股本结构数据静态变量
    utils_cap_struct = DataFrame()

    @classmethod
    def get_cap_struct(cls, code, date):
        """
        读取个股指定日期最新的股本结构数据
        Parameters:
        --------
        :param code: str
            股票代码，如600000或SH600000
        :param date: datetime-like or str
            日期
        :return: pd.Series
            个股截止指定日期的最新股本结构数据
        --------
            0: code, 代码, str
            1: date, 变更日期, str
            2: reason, 变更原因, str
            3: total, 总股本, float
            4: liquid_a, 流通A股, float
            5: liquid_b, 流通B股, float
            6: liquid_h, 流通H股, float
            如果截止指定日期最新股本结构数据不存在，返回None
        """
        code = cls.code_to_symbol(code)
        str_date = cls.datetimelike_to_str(cls.to_date(date))
        # 如果utils_cap_struct变量为空，那么先导入
        if cls.utils_cap_struct.shape[0] == 0:
            cap_struct_path = os.path.join(ct.DB_PATH, ct.CAP_STRUCT, 'cap_struct.csv')
            cls.utils_cap_struct = pd.read_csv(cap_struct_path, names=ct.CAP_STRUCT_HEADER, header=0)
        cap_struct = cls.utils_cap_struct[(cls.utils_cap_struct.code == code) & (cls.utils_cap_struct.date <= str_date)]
        if cap_struct.shape[0] > 0:
            return cap_struct.iloc[-1]
        else:
            return None

    @classmethod
    def get_fin_basic_data(cls, code, date, date_type='report_date'):
        """
        读取个股最新的主要财务指标数据
        Parameter:
        --------
        :param code: str
            证券代码，如600000或SH600000
        :param date: datetime-like or str
            日期，格式：YYYY-MM-DD or YYYYMMDD
        :param date_type: str, 默认为'report_date'
            日期类型, 'report_date': 报告期; 'trading_date': 交易日
        :return: pd.Series
        --------
            0. ReportDate
            1. BasicEPS:基本每股收益（元）
            2. UnitNetAsset:每股净资产（元）
            3. UnitNetOperateCashFlow:每股经营活动净现金流（元）
            4. MainOperateRevenue:主营业务收入（万元）
            5. MainOperateProfit:主营业务利润（万元）
            6. OperateProfit:营业利润（万元）
            7. InvestIncome:投资收益（万元）
            8. NonOperateNetIncome:营业外收支净额（万元）
            9. TotalProfit:利润总额（万元）
            10. NetProfit:净利润（万元）
            11. DeductedNetPorfit:扣除非经常性损益后净利润（万元）
            12. NetOperateCashFlow:经营活动现金流净额（万元）
            13. CashEquivalentsChg:现金及现金等价物增加额（万元）
            14. TotalAsset:总资产（万元）
            15. CurrentAsset:流动资产（万元）
            16. TotalLiability:总负债（万元）
            17. CurrentLiability:流动负债（万元）
            18. ShareHolderEquity:归属母公司股东权益（万元）
            19. ROE:净资产收益率加权（%）
            读取失败，返回None
        """
        code = cls.code_to_symbol(code)
        date = cls.to_date(date)
        if date_type == 'report_date':
            pass
        elif date_type == 'trading_date':
            date = cls.get_fin_report_date(date)
        else:
            return None
        if not cls.is_fin_report_date(date):
            return None
        fin_basic_data_path = os.path.join(ct.DB_PATH, ct.FIN_BASIC_DATA_PATH, '%s.csv' % code)
        df_fin_basic_data = pd.read_csv(fin_basic_data_path, na_values='--', parse_dates=[0],
                                        names=ct.FIN_BASIC_DATA_HEADER, header=0)
        fin_basic_data = df_fin_basic_data[df_fin_basic_data.ReportDate == date]
        if fin_basic_data.shape[0] == 0:
            return None
        else:
            return fin_basic_data.iloc[0]

    @classmethod
    def get_ttm_fin_basic_data(cls, code, date):
        """
        读取个股最新ttm主要财务指标数据
        Parameters:
        --------
        :param code: str
            个股代码，如SH600000或600000
        :param date: datetime-like or str
            日期，格式YYYY-MM-DD or YYYYMMDD
        :return: pd.Series
        --------
        0. ReportDate: 报告期
        1. BasicEPS: 基本每股收益(元)
        2. MainOperateRevenue: 主营业务收入（万元）
        3. MainOperateProfit: 主营业务利润（万元）
        4. OperateProfit: 营业利润（万元）
        5. InvestIncome: 投资收益（万元）
        6. NonOperateNetIncome: 营业外收益净额（万元）
        7. TotalProfit: 利润总额（万元）
        8. NetProfit: 净利润（万元）
        9. DeductedNetProfit: 扣除非经常性损益后净利润（万元）
        10. NetOperateCashFlow: 经营活动现金流净额（万元）
        读取失败，返回None
        """
        code = cls.code_to_symbol(code)
        date = cls.to_date(date)
        if date.month in (5, 6, 7, 8):
            date1 = datetime.datetime(date.year, 3, 31)
            date2 = datetime.datetime(date.year-1, 12, 31)
            date3 = datetime.datetime(date.year-1, 3, 31)
        elif date.month in (9, 10):
            date1 = datetime.datetime(date.year, 6, 30)
            date2 = datetime.datetime(date.year-1, 12, 31)
            date3 = datetime.datetime(date.year-1, 6, 30)
        elif date.month in (11, 12):
            date1 = datetime.datetime(date.year, 9, 30)
            date2 = datetime.datetime(date.year-1, 12, 31)
            date3 = datetime.datetime(date.year-1, 9, 30)
        else:
            date1 = datetime.datetime(date.year-1, 9, 30)
            date2 = datetime.datetime(date.year-2, 12, 31)
            date3 = datetime.datetime(date.year-2, 9, 30)
        fin_basic_data1 = cls.get_fin_basic_data(code, date1)
        if fin_basic_data1 is None:
            return None
        fin_basic_data2 = cls.get_fin_basic_data(code, date2)
        if fin_basic_data2 is None:
            return None
        fin_basic_data3 = cls.get_fin_basic_data(code, date3)
        if fin_basic_data3 is None:
            return None
        ttm_fin_basic_data = {}
        ttm_fin_basic_data['ReportDate'] = date1
        ttm_fin_basic_data['BasicEPS'] = fin_basic_data1['BasicEPS'] + fin_basic_data2['BasicEPS'] - fin_basic_data3['BasicEPS']
        ttm_fin_basic_data['MainOperateRevenue'] = fin_basic_data1['MainOperateRevenue'] + fin_basic_data2['MainOperateRevenue'] - fin_basic_data3['MainOperateRevenue']
        ttm_fin_basic_data['MainOperateProfit'] = fin_basic_data1['MainOperateProfit'] + fin_basic_data2['MainOperateProfit'] - fin_basic_data3['MainOperateProfit']
        ttm_fin_basic_data['OperateProfit'] = fin_basic_data1['OperateProfit'] + fin_basic_data2['OperateProfit'] - fin_basic_data3['OperateProfit']
        ttm_fin_basic_data['InvestIncome'] = fin_basic_data1['InvestIncome'] + fin_basic_data2['InvestIncome'] - fin_basic_data3['InvestIncome']
        ttm_fin_basic_data['NonOperateNetIncome'] = fin_basic_data1['NonOperateNetIncome'] + fin_basic_data2['NonOperateNetIncome'] - fin_basic_data3['NonOperateNetIncome']
        ttm_fin_basic_data['TotalProfit'] = fin_basic_data1['TotalProfit'] + fin_basic_data2['TotalProfit'] - fin_basic_data3['TotalProfit']
        ttm_fin_basic_data['NetProfit'] = fin_basic_data1['NetProfit'] + fin_basic_data2['NetProfit'] - fin_basic_data3['NetProfit']
        ttm_fin_basic_data['DeductedNetProfit'] = fin_basic_data1['DeductedNetProfit'] + fin_basic_data2['DeductedNetProfit'] - fin_basic_data3['DeductedNetProfit']
        ttm_fin_basic_data['NetOperateCashFlow'] = fin_basic_data1['NetOperateCashFlow'] + fin_basic_data2['NetOperateCashFlow'] - fin_basic_data3['NetOperateCashFlow']
        return Series(ttm_fin_basic_data)

    @classmethod
    def get_fin_summary_data(cls, code, report_date):
        """
        读取个股最新的财务报表摘要数据
        Parameters:
        --------
        :param code: str
            证券代码, 如SH600000, 600000
        :param report_date: datetime-like, str
            报告期, 格式: YYYY-MM-DD or YYYYMMDD
        :return: pd.Series
        --------
            0. ReportDate
            1. OperatingIncome:营业收入(万元)
            2. OperatingCost:营业成本(万元)
            3. OperatingProfit:营业利润(万元)
            4. TotalProfit:利润总额(万元)
            5. IncomeTax:所得税费用(万元)
            6. NetProfit:净利润(万元)
            7. EarningsPerShare:基本每股收益
            8. Cash:货币资金(万元)
            9. AccountsReceivable:应收账款(万元)
            10. Inventories:存货(万元)
            11. TotalCurrentAssets:流动资产合计(万元)
            12. NetFixedAssets:固定资产净额(万元)
            13. TotalAssets:资产总计(万元)
            14. TotalCurrentLiabilities:流动负债合计(万元)
            15. TotalNonCurrentLiabilities:非流动负债合计(万元)
            16. TotalLiabilities:负债合计(万元)
            17. TotalShareholderEquity:所有者权益(或股东权益)合计(万元)
            18. InitialCashAndCashEquivalentsBalance:期初现金及现金等价物余额(万元)
            19. NetCashFlowsFromOperatingActivities:经营活动产生的现金流量净额(万元)
            20. NetCashFlowsFromInvestingActivities:投资活动产生的现金流量净额(万元)
            21. NetCashFlowsFromFinancingActivities:筹资活动产生的现金流量净额(万元)
            22. NetIncreaseInCashAndCashEquivalents:现金及现金等价物增加额(万元)
            23. FinalCashAndCashEquivalentsBalance:期末现金及现金等价物余额(万元)
            读取失败, 返回None
        """
        code = cls.code_to_symbol(code)
        date = cls.to_date(report_date)
        if not cls.is_fin_report_date(date):
            return None
        fin_summary_data_path = os.path.join(ct.DB_PATH, ct.FIN_SUMMARY_DATA_PATH, '%s.csv' % code)
        df_fin_summary_data = pd.read_csv(fin_summary_data_path, na_values='--', parse_dates=[0],
                                          names=ct.FIN_SUMMARY_DATA_HEADER, header=0)
        fin_summary_data = df_fin_summary_data[df_fin_summary_data.ReportDate == date]
        if fin_summary_data.shape[0] == 0:
            return None
        else:
            return fin_summary_data.iloc[0]

    @classmethod
    def get_ttm_fin_summary_data(cls, code, date):
        """
        读取个股最新ttm的财务报表摘要数据
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param date: datetime-like or str
            日期, 格式:YYYY-MM-DD or YYYYMMDD
        :return: pd.Series
        --------
        0. ReportDate:报告期
        1. OperatingIncome:营业收入(万元)
        2. OperatingCost:营业成本(万元)
        3. OperatingProfit:营业利润(万元)
        4. TotalProfit:利润总额(万元)
        5. NetProfit:净利润(万元)
        6. EarningsPerShare:基本每股收益
        7.
        """

    @classmethod
    def get_hist_growth_data(cls, code, date, years):
        """
        计算个股过去N年复合增长率数据
        Parameters:
        --------
        :param code: str
            个股代码, 如SH600000, 600000
        :param date: datetime-like or str
            日期, 格式: YYYY-MM-DD or YYYYMMDD
        :param years: int
            过去N年
        :return: pd.Series
        --------
            0. date: 日期
            1. code: 代码
            2. revenue: 主营业务收入复合增长率
            2. netprofit: 净利润复合增长率
            读取失败, 返回None
        """
        code = cls.code_to_symbol(code)
        date = cls.to_date(date)
        # 读取最新的ttm主要财务数据
        latest_ttm_data = cls.get_ttm_fin_basic_data(code, date)
        if latest_ttm_data is None:
            return None
        # 读取N年前的ttm主要财务数据
        firstDayWeekDay, monthRange = calendar.monthrange(date.year-years, date.month)
        prevN_date = datetime.datetime(date.year-years, date.month, monthRange)
        prevN_ttm_data = cls.get_ttm_fin_basic_data(code, prevN_date)
        if prevN_ttm_data is None:
            return None
        # 计算复合增长率数据
        growth_data = pd.Series([date, code, 0.0, 0.0], index=['date', 'code', 'revenue', 'netprofit'])
        if np.isnan(prevN_ttm_data['MainOperateRevenue']):
            growth_data['revenue'] = np.nan
        elif abs(prevN_ttm_data['MainOperateRevenue']) < ct.TINY_ABS_VALUE:
            growth_data['revenue'] = np.nan
        elif np.isnan(latest_ttm_data['MainOperateRevenue']):
            growth_data['revenue'] = np.nan
        elif prevN_ttm_data['MainOperateRevenue'] > ct.TINY_ABS_VALUE and latest_ttm_data['MainOperateRevenue'] > ct.TINY_ABS_VALUE:
            growth_data['revenue'] = pow(latest_ttm_data['MainOperateRevenue']/prevN_ttm_data['MainOperateRevenue'], 1/years) - 1
        else:
            growth_data['revenue'] = (latest_ttm_data['MainOperateRevenue'] - prevN_ttm_data['MainOperateRevenue']) / abs(prevN_ttm_data['MainOperateRevenue']) / years

        if np.isnan(prevN_ttm_data['NetProfit']):
            growth_data['netprofit'] = np.nan
        elif abs(prevN_ttm_data['NetProfit']) < ct.TINY_ABS_VALUE:
            growth_data['netprofit'] = np.nan
        elif np.isnan(latest_ttm_data['NetProfit']):
            growth_data['netprofit'] = np.nan
        elif prevN_ttm_data['NetProfit'] > ct.TINY_ABS_VALUE and latest_ttm_data['NetProfit'] > ct.TINY_ABS_VALUE:
            growth_data['netprofit'] = pow(latest_ttm_data['NetProfit']/prevN_ttm_data['NetProfit'], 1/years) - 1
        else:
            growth_data['netprofit'] = (latest_ttm_data['NetProfit'] - prevN_ttm_data['NetProfit']) / abs(prevN_ttm_data['NetProfit']) / years

        return growth_data

    @classmethod
    def get_consensus_data(cls, date, code=None, consensus_type=ConsensusType.PredictedEarings):
        """
        读取个股指定交易日期的一致预期数据
        :param date: datetime-like, str
            交易日期, 格式: YYYY-MM-DD
        :param code: str
            个股代码, 如SH600000, 600000
        :param consensus_type: ConsensusType
            一致预期数据类型
        :return:
            如果code=None, 返回pd.DataFrame, 所有个股的指定一致预期数据
            如果code<>None, 返回pd.Series, 该指定个股的指定一致预期数据
        """
        str_date = Utils.datetimelike_to_str(date, dash=False)
        if consensus_type == ConsensusType.PredictedEarings:
            consensus_path = os.path.join(ct.DB_PATH ,ct.CONSENSUS_PATH, '{}/{}_{}.csv'.format('predicted_earnings' ,'predictedearnings', str_date))
        elif consensus_type == ConsensusType.PredictedEarningsGrowth:
            consensus_path = os.path.join(ct.DB_PATH, ct.CONSENSUS_PATH, '{}/{}_{}.csv'.format('growth_data', 'consensus_growth_data', str_date))
        else:
            return None
        df_consensus = pd.read_csv(consensus_path, header=0)
        if code is None:
            return df_consensus
        else:
            code = cls.code_to_symbol(code)
            secu_consensus = df_consensus[df_consensus['code'] == code]
            if secu_consensus.empty:
                return None
            else:
                return secu_consensus.iloc[0]

    @classmethod
    def is_fin_report_date(cls, date):
        """
        给定的日期是否为财务报告日期
        Parameters:
        --------
        :param date: datetime-like or str
            日期
        :return: bool
        """
        date = cls.to_date(date)
        year = date.year
        report_dates = [datetime.datetime(year, 3, 31), datetime.datetime(year, 6, 30),
                        datetime.datetime(year, 9, 30), datetime.datetime(year, 12, 31)]
        if date in report_dates:
            return True
        else:
            return False

    @classmethod
    def get_fin_report_date(cls, trading_day):
        """
        根据交易日日期返回最新财报的报告期日期
        Parameters:
        --------
        :param trading_day: datetime-like or str
            交易日期
        :return: datetime.datetime
            最新财报的报告期日期
        --------
            规则：5、6、7、8月采用年报或一季报数据（年报、一季报4月底全部公告完毕）
                 9、10月采用中报数据（中报8月底全部公告完毕）
                 11、12月及下一年1、2、3、4月采用三季报数据（三季报10月底全部公告完毕）
        """
        trading_day = cls.to_date(trading_day)
        year = trading_day.year
        month = trading_day.month
        day = trading_day.day
        if month in (5,6,7,8):
            month = 3
            day = 31
        elif month in (9, 10):
            month = 6
            day = 30
        elif month in (11, 12):
            month = 9
            day = 30
        elif month in (1, 2, 3, 4):
            year -= 1
            month = 9
            day = 30
        return datetime.datetime(year, month, day)

    @classmethod
    def get_ind_dist(cls, code):
        """
        读取个股的行业分布
        Parameters:
        --------
        :param code: str
            个股代码，如600000或SH600000
        :return: pd.Series
        --------
            个股的申万一级行业分布
            Series的index为行业代码，值为0(个股不属于该行业)或1(个股属于该行业)
        """
        code = cls.code_to_symbol(code)
        # 读取申万一级行业信息表
        sw_classify_info_path = os.path.join(ct.DB_PATH, ct.INDUSTRY_CLASSIFY_DATA_PATH, 'classify_standard_sw.csv')
        df_sw_classify = pd.read_csv(sw_classify_info_path, names=ct.SW_INDUSTRY_CLASSIFY_HEADER, header=0)
        df_sw_classify = df_sw_classify.set_index('ind_code', drop=False)
        # 读取个股的行业分类信息
        secu_ind_classify_path = os.path.join(ct.DB_PATH, ct.INDUSTRY_CLASSIFY_DATA_PATH, 'industry_classify_sw.csv')
        df_secu_ind_classify = pd.read_csv(secu_ind_classify_path,names=['id', 'ind_code', 'ind_name'], header=0)
        secu_ind = df_secu_ind_classify[df_secu_ind_classify.id == code].iloc[0].ind_code
        # 构建个股的申万一级行业分布Series
        secu_ind_dist = df_sw_classify['ind_code'] == secu_ind
        secu_ind_dist = secu_ind_dist.astype(int)
        return secu_ind_dist

    @classmethod
    def get_industry_classify(cls, date=None):
        """
        读取行业分类数据（申万一级行业）
        Parameters:
        --------
        :param date: datetime-like, str
            日期
        :return: pd.DataFrame
        --------
            行业分类数据
            0. id: 个股代码
            1. ind_code: 行业代码
            2. ind_name: 行业名称
            读取失败, 返回None
        """
        if date is None:
            sw_classify_data_path = os.path.join(ct.DB_PATH, ct.INDUSTRY_CLASSIFY_DATA_PATH, 'industry_classify_sw.csv')
        else:
            sw_classify_data_path = os.path.join(ct.DB_PATH, ct.INDUSTRY_CLASSIFY_DATA_PATH, 'industry_classify_sw_{}.csv'.format(cls.datetimelike_to_str(date, dash=False)))
            if not os.path.exists(sw_classify_data_path):
                sw_classify_data_path = os.path.join(ct.DB_PATH, ct.INDUSTRY_CLASSIFY_DATA_PATH, 'industry_classify_sw.csv')
        if not os.path.exists(sw_classify_data_path):
            return None
        df_ind_classify = pd.read_csv(sw_classify_data_path, names=['id', 'ind_code', 'ind_name'], header=0)
        return df_ind_classify

    @classmethod
    def get_ipo_info(cls, code=None):
        """
        读取个股IPO信息数据
        Parameters:
        --------
        :param code: str
            个股代码, 如600000 or SH600000
        :return: pd.DataFrame or pd.Series
        --------
        1. 成立日期
        2. 上市日期
        3. 发行方式
        4. 面值
        5. 发行数量
        6. 发行价格
        7. 募资资金总额
        8. 发行费用
        9. 发行中签率
        10. 发行市盈率
        11. 发行后每股收益
        12. 发行后每股净资产
        13. 上市首日开盘价
        14. 上市首日收盘价
        15. 上市首日换手率
        16. 主承销商
        17. 上市保荐人
        18. 会计师事务所
        19. 代码
        """
        ipo_info_path = os.path.join(ct.DB_PATH, ct.IPO_INFO_PATH, 'ipo_info.csv')
        df_ipo_info = pd.read_csv(ipo_info_path, header=0)
        if code is not None:
            code = cls.code_to_symbol(code)
            if code not in df_ipo_info[df_ipo_info['代码'] == code].values:
                return None
            else:
                return df_ipo_info[df_ipo_info['代码'] == code].iloc[0]
        else:
            return df_ipo_info

    @classmethod
    def trading_status(cls, code, trading_day):
        """
        返回个股在指定交易日的交易状态：正常、停牌、涨停、跌停
        Parameters
        --------
        :param code: str
            个股代码，如SH600000
        :param trading_day: datetime-like, str
            交易日
        :return:
            正常交易: SecuTradingStatus.Normal
            停牌：SecuTradingStatus.Suspend
            涨停：SecuTradingStatus.LimitUp
            跌停：SecuTradingStatus.LimitDown
        """
        symbol = cls.code_to_symbol(code)
        str_date = cls.datetimelike_to_str(cls.to_date(trading_day), dash=True)
        file_path = '%s.csv' % os.path.join(ct.DB_PATH, ct.MKT_DAILY_NOFQ, symbol)
        df_daily_mkt = pd.read_csv(file_path, names=ct.MKT_DAILY_NOFQ_HEADER, header=0)
        df_daily_mkt = df_daily_mkt[df_daily_mkt.date <= str_date].iloc[-2:]
        if len(df_daily_mkt) > 0:
            if df_daily_mkt.iloc[-1].date != str_date:
                return SecuTradingStatus.Suspend
            else:
                if abs(df_daily_mkt.iloc[-1].high - df_daily_mkt.iloc[-1].low) > 0.01:
                    return SecuTradingStatus.Normal
                else:
                    if len(df_daily_mkt) == 2:
                        if df_daily_mkt.iloc[-1].low > df_daily_mkt.iloc[-2].close * 1.099:
                            return SecuTradingStatus.LimitUp
                        elif df_daily_mkt.iloc[-1].high < df_daily_mkt.iloc[-2].close * 0.901:
                            return SecuTradingStatus.LimitDown
                        else:
                            return SecuTradingStatus.Normal
                    else:
                        if df_daily_mkt.iloc[-1].low > df_daily_mkt.iloc[-1].open * 1.099:
                            return SecuTradingStatus.LimitUp
                        elif df_daily_mkt.iloc[-1].high < df_daily_mkt.iloc[-1].open * 0.901:
                            return SecuTradingStatus.LimitDown
                        else:
                            return SecuTradingStatus.Normal
        else:
            return SecuTradingStatus.Suspend

    @classmethod
    def factor_loading_persistent(cls, db_file, str_key, dict_factor_loading, columns=None):
        """
        持久化因子载荷
        Parameters
        --------
        :param db_file: str
            因子载荷数据文件，绝对路径
        :param str_key: str
            持久化因子载荷时用到的key，一般为日期，格式YYYYMMDD
        :param dict_factor_loading: dict, pd.DataFrame
            因子载荷值
        :param columns: sequence, 默认=None
            输出的列，并按指定顺序输出
        :return:
        """
        persistence_type = ct.FACTOR_LOADING_PERSISTENCE_TYPE.split(',')
        for perst_type in persistence_type:
            if perst_type == 'shelve':
                db = shelve.open(db_file, flag='c', protocol=None, writeback=False)
                try:
                    db[str_key] = dict_factor_loading
                    db.sync()
                finally:
                    db.close()
            elif perst_type == 'csv':
                db_file += '_%s.csv' % str_key
                if isinstance(dict_factor_loading, dict):
                    DataFrame(dict_factor_loading).to_csv(db_file, index=False, columns=columns, na_rep='NaN')
                elif isinstance(dict_factor_loading, DataFrame):
                    dict_factor_loading.to_csv(db_file, index=False, columns=columns, na_rep='NaN')
                else:
                    raise TypeError("保存的因子载荷数据类型应该为dict或pd.DataFrame")

    @classmethod
    def read_factor_loading(cls, db_file, str_key, code=None, nan_value=None, drop_na=False):
        """
        从因子载荷持久化文件中读取指定str_key的因子载荷值
        Parameters
        --------
        :param db_file: str
            因子载荷数据文件，绝对路径
        :param str_key: str
            键值，一般为日期，格式为YYYYMMDD
        :param code: str, 默认为None
            个股代码, 如SH600000, 600000
        :param nan_value: object, 默认为None
            如果不为None，那么缺失值用nan_value替换
        :param drop_na: bool, 默认False
            是否删除含有NaN值的行
        :return: DataFrame or Series，因子载荷
        --------
            DataFrame(code==None) or Series(code<>None):
            0: date, 日期，str，格式YYYY-MM-DD
            1: factorvalue,因子载荷
            2: id, 证券代码，如SH600000
        """
        using_type = ct.USING_PERSISTENCE_TYPE
        if using_type == 'shelve':
            db = shelve.open(db_file, flag='c', protocol=None, writeback=False)
            try:
                dict_factor_loading = db[str_key]
                df_factor_loading = DataFrame(dict_factor_loading)
            except KeyError:
                df_factor_loading = DataFrame()
            finally:
                db.close()
        elif using_type == 'csv':
            db_file += '_%s.csv' % str_key
            if not os.path.isfile(db_file):
                raise FileExistsError("因子文件%s不存在." % db_file)
            df_factor_loading = pd.read_csv(db_file, header=0)
        else:
            df_factor_loading = DataFrame()
        if (code is not None) and (not df_factor_loading.empty):
            code = cls.code_to_symbol(code)
            df_factor_loading = df_factor_loading[df_factor_loading['id'] == code]
            if not df_factor_loading.empty:
                df_factor_loading = df_factor_loading.iloc[0]
        if nan_value is not None:
            df_factor_loading = df_factor_loading.fillna(nan_value)
        if drop_na:
            df_factor_loading.dropna(axis=0, how='any', inplace=True)
        return df_factor_loading

    @classmethod
    def clean_extreme_value(cls, arr_data, method='MAD'):
        """
        对数据进行去极值处理
        :param arr_data: np.array
            需要进行去极值的原始数据，数组的每一列各自进行去极值操作
        :param method: 去极值算法
        :return: np.array
            去极值处理后的数据
        """
        raw_data = arr_data.copy()
        m = np.median(raw_data, axis=0)     # 原始数据的中位数
        mad = np.median(np.fabs(raw_data - m), axis=0)
        fupper = m + mad * ct.CLEAN_EXTREME_VALUE_MULTI_CONST
        flower = m - mad * ct.CLEAN_EXTREME_VALUE_MULTI_CONST
        for k in range(raw_data.shape[1]):
            if method == 'MAD':
                raw_data[:, k][raw_data[:, k] > fupper[k]] = fupper[k]
                raw_data[:, k][raw_data[:, k] < flower[k]] = flower[k]
        return raw_data

    @classmethod
    def normalize_data(cls, raw_data, id=None, columns=None, treat_outlier=False, weight='eq', calc_date=None):
        """
        对数据进行标准化
        :param raw_data: pd.DataFrame or np.ndarray
            需要进行标准化处理的原始数据
        :param id: str, 默认None
            个股代码列名
            当raw_data为pd.DataFrame, weight='cap'时, 需指定个股代码列名, 用于计算权重
        :param columns: str, list, 默认为None
            当raw_data为pd.DataFrame时, 指定需要标准化的列; 如果为None, 计算除参数id指定的列外其他所有列的标准化
            当raw_data为np.ndarray时, 计算所有列的标准化
        :param treat_outlier: bool, 默认False
            是否处理极值
        :param weight: str
            计算标准化时均值的加权方式
            'eq': 采用等权重计算均值; 'cap': 采用流通市值加权计算均值, 此时需要用参数id指定的个股证券列计算流通市值
        :param calc_date: datetime-like or str, 默认None
            标准化计算日期, 当weight='cap'时，用于读取个股市值数据
        :return: pd.DataFrame, np.ndarray, 与raw_data类型一致
            标准化处理后的数据
        """
        # raw_data = arr_data.copy()
        # u = np.mean(raw_data, axis=0)
        # s = np.std(raw_data, axis=0)
        # return (raw_data - u)/s

        if isinstance(raw_data, np.ndarray):
            arr_data = raw_data.copy()
            if treat_outlier:
                arr_data = cls.clean_extreme_value(arr_data)
            u = np.mean(arr_data, axis=0)
            s = np.std(arr_data, axis=0)
            return (arr_data - u)/s
        elif isinstance(raw_data, pd.DataFrame):
            df_data = raw_data.copy()
            if weight == 'eq':
                if columns is None:
                    cols = [col for col in df_data.columns.tolist() if col != id]
                elif isinstance(columns, Iterable):
                    cols = [col for col in df_data.columns.tolist() if (col in columns) & (col != id)]
                else:
                    cols = [col for col in df_data.columns.tolist() if (col == columns) & (col != id)]
                arr_data = np.array(df_data[cols]).reshape((len(df_data), len(cols)))
                if treat_outlier:
                    arr_data = cls.clean_extreme_value(arr_data)
                u = np.mean(arr_data, axis=0)
                s = np.std(arr_data, axis=0)
                df_data[cols] = (arr_data - u)/s
                return df_data
            elif weight == 'cap':
                if id is None:
                    return raw_data
                else:
                    cap_factor_path = os.path.join(ct.DB_PATH, risk_ct.LNCAP_CT.liquidcap_dbfile)
                    df_cap_factor = cls.read_factor_loading(cap_factor_path, cls.datetimelike_to_str(calc_date, dash=False), drop_na=True)
                    df_cap_factor.drop(columns='date', inplace=True)
                    df_cap_factor.rename(columns={'id': id, 'factorvalue': 'cap'}, inplace=True)
                    df_data = pd.merge(left=df_data, right=df_cap_factor, how='inner', on=id)
                    df_data['cap'] = df_data['cap'] / df_data['cap'].sum()
                    if columns is None:
                        cols = [col for col in df_data.columns.tolist() if col not in [id, 'cap']]
                    elif isinstance(columns, Iterable) and not isinstance(columns, str):
                        cols = [col for col in df_data.columns.tolist() if (col in columns) & (col not in [id, 'cap'])]
                    else:
                        cols = [col for col in df_data.columns.tolist() if (col == columns) & (col not in [id, 'cap'])]
                    arr_data = np.array(df_data[cols]).reshape((len(df_data), len(cols)))
                    if treat_outlier:
                        arr_data = cls.clean_extreme_value(arr_data)
                    weight = np.array(df_data['cap']).reshape((1, len(df_data)))
                    u = np.dot(weight, arr_data)
                    s = np.std(arr_data, axis=0)
                    df_data[cols] = (arr_data - u)/s
                    df_data.drop(columns='cap', inplace=True)
                    return df_data

    @classmethod
    def get_backtest_data(cls, backtest_path, start_date):
        """
        读取截止start_date前的回测数据，包括port_data和port_nav数据
        Parameters
        --------
        :param backtest_path: str
            回测数据文件夹路径
        :param start_date: datetime-like or str
            回测开始日期，格式：YYYY-MM-DD or YYYYMMDD
        :return: pandas.DataFrame
            0: port_data, 截止start_date日期前组合最新的持仓数据
            1: port_nav, 截止start_date日期前组合最新的净值数据
        """
        start_date = cls.to_date(start_date)
        port_data = None
        port_nav = None
        # 遍历backtest_path文件夹下的文件，读取截止start_date日期前组合最新的持仓数据及净值数据
        port_data_dates = []
        for backtest_file in os.listdir(backtest_path):
            if os.path.splitext(backtest_file)[0][:9] == 'port_data':
                port_data_dates.append(os.path.splitext(backtest_file)[0][-8:])
            if os.path.splitext(backtest_file)[0] == 'port_nav':
                port_nav = pd.read_csv(os.path.join(backtest_path, backtest_file))
                port_nav = port_nav[port_nav.date < start_date.strftime('%Y-%m-%d')]
        port_data_dates = [x for x in port_data_dates if x < start_date.strftime('%Y%m%d')]
        if len(port_data_dates) > 0:
            port_data_date = max(port_data_dates)
            port_data_path = os.path.join(backtest_path, 'port_data_%s.csv' % port_data_date)
            port_data = pd.read_csv(port_data_path)
        return port_data, port_nav

    @classmethod
    def port_data_to_wind(cls, port_data_path, start=None, end=None):
        """
        把回测的port_data数据转换为wind的模拟组合持仓数据（权重形式）
        Parameters:
        --------
        :param port_data_path: str
            port_data数据文件夹路径
        :param start: datetime-like or str
            开始日期，默认None
        :param end: datetime-like or str
            结束日期，默认None
            当start和end都为None时，转换文件夹下所有port_data数据
            当start和end有一个为None时，转换start或end指定日期的port_data数据
            当start和end都不为None时，转换开始、结束日期(含)之间的port_data数据
        :return:
        """
        if not os.path.exists(port_data_path):
            return
        if not os.path.exists(os.path.join(port_data_path, 'port_nav.csv')):
            return
        else:
            df_port_nav = pd.read_csv(os.path.join(port_data_path, 'port_nav.csv'), header=0)
        if not os.path.exists(os.path.join(port_data_path, 'wind')):
            os.mkdir(os.path.join(port_data_path, 'wind'))
        wind_port_datas = [['证券代码', '持仓权重', '成本价格', '调整日期', '证券类型']]
        if start is None and end is None:
            for port_data_file in os.listdir(port_data_path):
                if port_data_file[:9] == 'port_data':
                    wind_data_file = os.path.join(port_data_path, 'wind', port_data_file)
                    print('processing file %s.' % port_data_file)
                    wind_port_data = _port_data_to_wind(os.path.join(port_data_path, port_data_file), wind_data_file, df_port_nav)
                    if len(wind_port_data) > 0:
                        for row in wind_port_data[1:]:
                            wind_port_datas.append(row)
        # 保存
        with open(os.path.join(port_data_path, 'wind', 'agg_wind_port_data.csv'), 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(wind_port_datas)

    @classmethod
    def datetimelike_to_str(cls, datetime_like, dash=True):
        if isinstance(datetime_like, datetime.datetime) or isinstance(datetime_like, datetime.date):
            if dash:
                return datetime_like.strftime('%Y-%m-%d')
            else:
                return datetime_like.strftime('%Y%m%d')
        else:
            if dash:
                return datetime_like
            else:
                return datetime_like.replace('-', '')

    @classmethod
    def to_date(cls, date_like):
        if isinstance(date_like, datetime.datetime) or isinstance(date_like, datetime.date):
            return date_like
        else:
            return datetime.datetime.strptime(date_like.replace('-', ''), '%Y%m%d')

    @classmethod
    def code_to_symbol(cls, code, index=False):
        if not index:
            return _code_to_symbol(code)
        else:
            return _code_to_index_symbol(code)

    @classmethod
    # 转换为适用于tushare格式的代码
    def code_to_tssymbol(cls, code, index=False):
        if len(code) != 6:
            return code
        else:
            if not index:
                return '%s.SH' % code if code[:1] in ['5', '6', '9'] else '%s.SZ' % code
            else:
                return '%s.SZ' % code if code[:3] == '399' else '%s.SH' % code


def _code_to_symbol(code):
    """
    生成本系统的证券代码symbol
    :param code:原始代码，如600000, 600000,SH, 600000SH
    :return: 本系统证券代码, 如SH600000
    """
    if '.' in code:
        codes = code.split('.')
        if codes[0].upper() in ['SH', 'SZ']:
            code = codes[1]
            mkt_code = codes[0].upper()
        elif codes[1].upper() in ['SH', 'SZ']:
            code = codes[0]
            mkt_code = codes[1].upper()
        else:
            mkt_code = ''
        code = mkt_code + code
    else:
        if code[:2].upper() in ['SH', 'SZ']:
            mkt_code = code[:2].upper()
            code = code[2:]
        elif code[-2:].upper() in ['SH', 'SZ']:
            mkt_code = code[-2:].upper()
            code = code[:-2]
        else:
            mkt_code = ''
        code = mkt_code + code

    if len(code) != 6:
        return code
    else:
        return 'SH%s' % code if code[:1] in ['5', '6', '9'] else 'SZ%s' % code


def _code_to_index_symbol(code):
    """
    生成本系统的指数代码symbol
    :param code: 原始代码，如000001（上证综指）
    :return:
    """
    if len(code) != 6:
        return code
    else:
        return 'SZ%s' % code if code[:3] == '399' else 'SH%s' % code


def _symbol_to_windcode(symbol):
    """
    将本系统的证券代码symbol转换为wind代码
    :param symbol: str
        本系统证券代码，如SH600000
    :return: str
        wind证券代码，如600000.SH
    """
    return '%s.%s' % (symbol[2:], symbol[:2])


def _port_data_to_wind(port_data_file, wind_data_file, df_port_nav):
    """
    把回测的port_data数据转换为wind的模拟组合持仓数据（权重形式）
    Parameters:
    --------
    :param port_data_file: str
        port_data数据文件路径
    :param wind_data_file: wind模拟组合持仓数据文件路径
    :param df_port_nav: pandas.DataFrame
        组合净值数据columns=['date', 'nav']
    :return:
    """
    df_port_data = pd.read_csv(port_data_file, header=0)
    if df_port_data.shape[0] > 0:
        fweight = 1.0 / df_port_data.shape[0]
    else:
        fweight = 0.0
    str_weight = '%.4f%%' % round(fweight * 100, 4)
    dst_rows = [['证券代码', '持仓权重', '成本价格', '调整日期', '证券类型']]
    str_date = df_port_data.iloc[0].date
    # fnav = df_port_nav[df_port_nav.date < str_date].iloc[-1].nav * 100000000.0
    # dst_rows.append(['Asset', str(fnav), '1', str_date])
    for _, port_data in df_port_data.iterrows():
        str_date = port_data.date
        wind_code = _symbol_to_windcode(port_data.id)
        df_mkt_data = Utils.get_secu_daily_mkt(port_data.id, str_date)
        str_buy_price = str(round(df_mkt_data.amount/df_mkt_data.vol, 2))
        dst_rows.append([wind_code, str_weight, str_buy_price, str_date, '股票'])
    with open(wind_data_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(dst_rows)
    return dst_rows


if __name__ == '__main__':
    pass
    # test calc_interval_ret
    # ret = Utils.calc_interval_ret('603329', start='2017-12-29', end='2017-12-29')
    # print('ret = %0.4f' % ret)
    # test get_trading_days
    # trading_days = Utils.get_trading_days(start=datetime.datetime.strptime('2017-01-01', '%Y-%m-%d'), end='20171031', ndays=10)
    # print(len(trading_days))
    # trading_days = Utils.get_trading_days(start=pd.Timestamp('2017-01-01'), end='2017-10-31', ndays=10)
    # print(len(trading_days))
    # trading_days = Utils.get_trading_days(start='2016-10-01', end='2017-10-31', ndays=10)
    # print(len(trading_days))
    # trading_days = Utils.get_trading_days()
    # print(len(trading_days))
    # test datetimelike_to_str
    # print(Utils.datetimelike_to_str('2017-12-07', dash=False))
    # print(Utils.datetimelike_to_str('20171207'))
    # print(Utils.datetimelike_to_str(datetime.date(2017, 12, 7), dash=False))
    # test get_secu_daily_mkt
    # mkt = Utils.get_secu_daily_mkt('600827', '2015-03-05', range_lookup=False)
    # print(mkt)
    # print(mkt.shape)
    # Utils.port_data_to_wind('/Volumes/DB/FactorDB/FactorBackTest/IntradayMomentum')
    # df = ts.get_industry_classified('sw')
    # print(df.head())
    # secu_ind_dist = Utils.get_ind_dist('600000')
    # print(secu_ind_dist)
    # ipo_info = Utils.get_ipo_info('603329')
    # print(ipo_info)
    # st_stocks = Utils.get_st_stocks()
    # print(st_stocks)
    # stock_basics = Utils.get_stock_basics(date='2017-12-29', remove_st=True, remove_suspension=True)
    # print(stock_basics.head(100))
    # print(_code_to_symbol('000300.sH'))
    # df_ind_classify = Utils.get_industry_classify('2009-12-31')
    # print(df_ind_classify.head())


    # 检查缺失的个股行情
    # df_dlisted_stocks = Utils.get_stock_basics('2015-01-05', all=True)
    # df_dlisted_stocks = df_dlisted_stocks[df_dlisted_stocks['status'] == 3]
    # for _, stock in df_dlisted_stocks.iterrows():
    #     mkt_path = os.path.join('/Volumes/DB/Data/Stk_DAY_FQ/Stk_DAY_FQ_WithHS_20171222/{}.csv'.format(stock['symbol']))
    #     if os.path.isfile(mkt_path):
    #         print('{} mkt data exists.'.format(stock['symbol']))
    #     else:
    #         print('\033[1;31;40m{} {} mkt data does not exists.\033[0m'.format(stock['symbol'], stock['name']))

    # 个股每日收益率文件重命名
    # ret_path = '/Volumes/DB/FactorDB/riskmodel/dailyret'
    # for factorret_file_name in os.listdir(ret_path):
    #     if (factorret_file_name[:8] != 'dailyret') and (os.path.splitext(factorret_file_name)[1] == '.csv'):
    #         os.rename(os.path.join(ret_path, factorret_file_name), os.path.join(ret_path, 'dailyret_'+factorret_file_name))
