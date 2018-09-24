#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 日内动量因子
# @Filename: IntradayMomentum
# @Date:   : 2017-12-28 17:29
# @Author  : YuJun
# @Email   : yujun_mail@163.com


import src.settings as SETTINGS
from src.factors.factor import Factor
import src.alphamodel.alphafactors.cons as alphafactor_ct
from src.util.utils import Utils, SecuTradingStatus
from src.util.dataapi.CDataHandler import CDataHandler
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import os
import math
import datetime
import calendar
import logging
from multiprocessing import Pool, Manager
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class IntradayMomentum(Factor):
    """日内动量因子类"""
    __days = alphafactor_ct.INTRADAYMOMENTUM_CT.days_num     # 读取过去多少天的分钟行情数据进行因子载荷计算
    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.INTRADAYMOMENTUM_CT.db_file)     # 因子对应的数据库文件名

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股日内各时点动量值
        Parameters
        --------
        :param code: str
            个股代码，如600000或SH600000
        :param calc_date: datetime-like, str
            因子载荷计算日期，格式YYYY-MM-DD
        :return: pd.Series
        --------
            日内个时点的动量值，各个index对应的含义如下：
            0. m0: 隔夜时段动量
            1. m1: 第一个小时动量
            2. m2: 第二个小时动量
            3. m3: 第三个小时动量
            4. m4: 第四个小时动量
            5. m_normal: 传统动量
            若计算失败，返回None
        """
        # 取得过去90天的交易日序列，按日期降序排列
        trading_days = Utils.get_trading_days(end=calc_date, ndays=90, ascending=False)
        # 取得个股过去90天中的最近21天的1分钟行情数据，根据每天的分钟行情读取日内5个时点的价格，并计算日内收益值
        mkt_data = DataFrame()
        mkt_data_header = ['date', 'p0930', 'p1030', 'p1130', 'p1400', 'p1500']
        intra_day_ret = DataFrame()
        ret_header = ['date', 'r0', 'r1', 'r2', 'r3', 'r4']
        k = 0
        for trading_day in trading_days:
            df_1m_mkt = Utils.get_min_mkt(code, trading_day, fq=True)
            if df_1m_mkt is None:
                continue
            # 计算日内5个时点的价格
            time_label = '%s 09:31:00' % trading_day.strftime('%Y-%m-%d')
            p0930 = df_1m_mkt[df_1m_mkt.datetime == time_label].iloc[0].open
            time_label = '%s 10:30:00' % trading_day.strftime('%Y-%m-%d')
            p1030 = df_1m_mkt[df_1m_mkt.datetime == time_label].iloc[0].close
            time_label = '%s 11:30:00' % trading_day.strftime('%Y-%m-%d')
            p1130 = df_1m_mkt[df_1m_mkt.datetime == time_label].iloc[0].close
            time_label = '%s 14:00:00' % trading_day.strftime('%Y-%m-%d')
            p1400 = df_1m_mkt[df_1m_mkt.datetime == time_label].iloc[0].close
            time_label = '%s 15:00:00' % trading_day.strftime('%Y-%m-%d')
            p1500 = df_1m_mkt[df_1m_mkt.datetime == time_label].iloc[0].close
            s = Series([trading_day, p0930, p1030, p1130, p1400, p1500], index=mkt_data_header)
            mkt_data = mkt_data.append(s, ignore_index=True)
            # 计算日内收益
            if k > 0:
                r0 = math.log(mkt_data.iloc[k-1].p0930 / mkt_data.iloc[k].p1500)
                r1 = math.log(mkt_data.iloc[k-1].p1030 / mkt_data.iloc[k-1].p0930)
                r2 = math.log(mkt_data.iloc[k-1].p1130 / mkt_data.iloc[k-1].p1030)
                r3 = math.log(mkt_data.iloc[k-1].p1400 / mkt_data.iloc[k-1].p1130)
                r4 = math.log(mkt_data.iloc[k-1].p1500 / mkt_data.iloc[k-1].p1400)

                # r0 = mkt_data.iloc[k - 1].p0930 / mkt_data.iloc[k].p1500 -1.0
                # r1 = mkt_data.iloc[k - 1].p1030 / mkt_data.iloc[k - 1].p0930 - 1.0
                # r2 = mkt_data.iloc[k - 1].p1130 / mkt_data.iloc[k - 1].p1030 - 1.0
                # r3 = mkt_data.iloc[k - 1].p1400 / mkt_data.iloc[k - 1].p1130 - 1.0
                # r4 = mkt_data.iloc[k - 1].p1500 / mkt_data.iloc[k - 1].p1400 - 1.0

                s = Series([mkt_data.iloc[k-1].date, r0, r1, r2, r3, r4], index=ret_header)
                intra_day_ret = intra_day_ret.append(s, ignore_index=True)
            k += 1
            if k > cls.__days:
                break
        if k <= cls.__days:
            return None
        intra_day_ret = intra_day_ret.sort_values(by='date')
        # mkt_data = mkt_data.sort_values(by='date')
        # mkt_data = mkt_data.reset_index(drop=True)
        # 计算传统动量因子值，=过去20日的涨跌幅
        m_normal = math.log(mkt_data.iloc[0].p1500 / mkt_data.iloc[-1].p1500)

        # m_normal = mkt_data.iloc[0].p1500 / mkt_data.iloc[-1].p1500 - 1.0

        # 遍历上述取得的行情数据，计算每日的日内收益值
        # intra_day_ret = DataFrame()
        # ret_header = ['date', 'r0', 'r1', 'r2', 'r3', 'r4']
        # for k in range(1, len(mkt_data)):
        #     r0 = math.log(mkt_data.iloc[k].p0930 / mkt_data.iloc[k-1].p1500)
        #     r1 = math.log(mkt_data.iloc[k].p1030 / mkt_data.iloc[k].p0930)
        #     r2 = math.log(mkt_data.iloc[k].p1130 / mkt_data.iloc[k].p1030)
        #     r3 = math.log(mkt_data.iloc[k].p1400 / mkt_data.iloc[k].p1130)
        #     r4 = math.log(mkt_data.iloc[k].p1500 / mkt_data.iloc[k].p1400)
        #     s = Series([mkt_data.iloc[k].date, r0, r1, r2, r3, r4], index=ret_header)
        #     intra_day_ret = intra_day_ret.append(s, ignore_index=True)
        intra_day_ret = intra_day_ret.set_index('date')
        # 个股的日内各时点的动量因子值等于过去20个交易日各个r_i累加
        intra_day_momentum = intra_day_ret.sum()
        intra_day_momentum.index = ['m0', 'm1', 'm2', 'm3', 'm4']
        intra_day_momentum['m_normal'] = m_normal
        return intra_day_momentum

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters
        --------
        :param code: str
            个股代码，如600000或SH600000
        :param calc_date: datetime-like or str
            计算日期，格式：YYYY-MM-DD
        :param q: 队列，用于进程间通信
        :return: 添加因子载荷至队列中
        """
        logging.info('[%s] Calc Intraday Momentum of %s.' % (Utils.datetimelike_to_str(calc_date), code))
        momentum_data = None
        try:
            momentum_data = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if momentum_data is not None:
            q.put((Utils.code_to_symbol(code), momentum_data.m0, momentum_data.m1, momentum_data.m2,
                   momentum_data.m3, momentum_data.m4, momentum_data.m_normal))

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的因子载荷，并保存至因子数据库
        Parameters
        --------
        :param start_date: datetime-like, str
            开始日期，格式：YYYY-MM-DD or YYYYMMDD
        :param end_date: datetime-like, str
            结束日期，如果为None，则只计算start_date日期的因子载荷，格式：YYYY-MM-DD or YYYYMMDD
        :param month_end: bool，默认True
            如果为True，则只计算月末时点的因子载荷
        :param save: bool，默认False
            是否保存至因子数据库
        :param kwargs:
            'multi_proc': bool, True=采用多进程并行计算, False=采用单进程计算, 默认为False
        :return: 因子载荷，DataFrame
        --------
            因子载荷，DataFrame
            0. date: 日期
            1. id: 证券symbol
            2. m0: 隔夜时段动量
            3. m1: 第一个小时动量
            4. m2: 第二个小时动量
            5. m3: 第三个小时动量
            6. m4: 第四个小时动量
            7. m_normal: 传统动量
        """
        # 取得交易日序列及股票基本信息表
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列，计算日内动量因子值
        dict_intraday_momentum = None
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue

            # 计算日内各时段动量因子
            dict_intraday_momentum = {'date': [], 'id': [], 'm0': [], 'm1': [],
                                      'm2': [], 'm3': [], 'm4': [], 'm_normal': []}
            # 遍历个股，计算个股日内动量值
            s = (calc_date - datetime.timedelta(days=90)).strftime('%Y%m%d')
            stock_basics = Utils.get_stock_basics(s)

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程进行计算
                for _, stock_info in stock_basics.iterrows():
                    momentum_data = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    if momentum_data is not None:
                        logging.info("[%s] %s's intraday momentum = (%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f)" % (calc_date.strftime('%Y-%m-%d'),stock_info.symbol, momentum_data.m0, momentum_data.m1, momentum_data.m2, momentum_data.m3, momentum_data.m4, momentum_data.m_normal))
                        dict_intraday_momentum['id'].append(Utils.code_to_symbol(stock_info.symbol))
                        dict_intraday_momentum['m0'].append(round(momentum_data.m0, 6))
                        dict_intraday_momentum['m1'].append(round(momentum_data.m1, 6))
                        dict_intraday_momentum['m2'].append(round(momentum_data.m2, 6))
                        dict_intraday_momentum['m3'].append(round(momentum_data.m3, 6))
                        dict_intraday_momentum['m4'].append(round(momentum_data.m4, 6))
                        dict_intraday_momentum['m_normal'].append(round(momentum_data.m_normal, 6))
            else:
                # 采用多进程并行计算日内动量因子载荷
                q = Manager().Queue()   # 队列，用于进程间通信，存储每个进程计算的因子载荷
                p = Pool(SETTINGS.CONCURRENCY_KERNEL_NUM)             # 进程池，最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    momentum_data = q.get(True)
                    dict_intraday_momentum['id'].append(momentum_data[0])
                    dict_intraday_momentum['m0'].append(round(momentum_data[1], 6))
                    dict_intraday_momentum['m1'].append(round(momentum_data[2], 6))
                    dict_intraday_momentum['m2'].append(round(momentum_data[3], 6))
                    dict_intraday_momentum['m3'].append(round(momentum_data[4], 6))
                    dict_intraday_momentum['m4'].append(round(momentum_data[5], 6))
                    dict_intraday_momentum['m_normal'].append(round(momentum_data[6], 6))

            date_label = Utils.get_trading_days(calc_date, ndays=2)[1]
            dict_intraday_momentum['date'] = [date_label] * len(dict_intraday_momentum['id'])

            # 保存因子载荷至因子数据库
            if save:
                # Utils.factor_loading_persistent(cls._db_file, calc_date.strftime('%Y%m%d'), dict_intraday_momentum)
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_intraday_momentum, 'periodmomentum', factor_type='raw')

            # 计算日内各时段动量因子的Rank IC值向量, 并保存
            # cls._calc_periodmomentum_ic(calc_date, 'month')

            # 计算最优化权重
            if alphafactor_ct.INTRADAYMOMENTUM_CT['optimized']:
                cls._optimize_periodmomentum_weight(calc_date)

            # 计算合成日内动量因子
            if alphafactor_ct.INTRADAYMOMENTUM_CT['synthesized']:
                logging.info('[%s] calc synthetic intraday momentum factor loading.' % Utils.datetimelike_to_str(calc_date))
                dict_intraday_momentum = {'date': [], 'id': [], 'factorvalue': []}
                # 读取日内个时段动量因子值
                # period_momentum_path = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.INTRADAYMOMENTUM_CT.db_file, 'raw/periodmomentum')
                # df_factor_loading = Utils.read_factor_loading(period_momentum_path, Utils.datetimelike_to_str(calc_date, False))
                df_factor_loading = cls._get_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), factor_name='periodmomentum', factor_type='raw', drop_na=False)
                if df_factor_loading.shape[0] <= 0:
                    logging.info("[%s] It doesn't exist intraday momentum factor loading." % Utils.datetimelike_to_str(calc_date))
                    return
                df_factor_loading.fillna(0, inplace=True)
                # 读取因子最优权重
                factor_weight = cls._get_factor_weight(calc_date)
                if factor_weight is None:
                    logging.info("[%s] It doesn't exist factor weight.")
                    return
                # 计算合成动量因子, 合成之前先对日内各时段动量因子进行去极值和标准化处理
                arr_factor_loading = np.array(df_factor_loading[['m0', 'm1', 'm2', 'm3', 'm4']])
                arr_factor_loading = Utils.normalize_data(arr_factor_loading, treat_outlier=True)
                arr_factor_weight = np.array(factor_weight.drop('date')).reshape((5, 1))
                arr_synthetic_factor = np.dot(arr_factor_loading, arr_factor_weight)
                dict_intraday_momentum['date'] = list(df_factor_loading['date'])
                dict_intraday_momentum['id'] = list(df_factor_loading['id'])
                dict_intraday_momentum['factorvalue'] = list(arr_synthetic_factor.astype(float).round(6).reshape((arr_synthetic_factor.shape[0],)))
                # 标准化合成动量因子
                df_std_intradaymonmentum = Utils.normalize_data(pd.DataFrame(dict_intraday_momentum), columns='factorvalue', treat_outlier=True, weight='eq')
                # 保存合成因子
                if save:
                    # Utils.factor_loading_persistent(synthetic_db_file, Utils.datetimelike_to_str(calc_date, False), dict_intraday_momentum)
                    cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_intraday_momentum, 'IntradayMomentum', factor_type='raw', columns=['date', 'id', 'factorvalue'])
                    cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_intradaymonmentum, 'IntradayMomentum', factor_type='standardized', columns=['date', 'id', 'factorvalue'])

            # 休息360秒
            # logging.info('Suspending for 360s.')
            # time.sleep(360)
        return dict_intraday_momentum

    @classmethod
    def _get_factor_weight(cls, date=None):
        """
        取得日内各时点动量因子的权重
        --------
        :param date: datetime-like or str
            日期, 默认为None
            如果date=None, 返回全部权重数据
        :return: pd.Series, pd.DataFrame
            各时点权重信息
        --------
            0. date: 日期
            1. w0: 第一个时点动量因子的权重
            2. w1: 第二个时点动量因子的权重
            3. w2: 第三个时点动量因子的权重
            4. w3: 第四个时点动量因子的权重
            5. w4: 第五个时点动量因子的权重
            读取不到数据，返回None
        """

        weight_file_path = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.INTRADAYMOMENTUM_CT.optimal_weight_file)
        if not os.path.isfile(weight_file_path):
            return None
        df_optimal_weight = pd.read_csv(weight_file_path, parse_dates=[0], header=0)
        df_optimal_weight.sort_values(by='date', inplace=True)

        if date is None:
            if df_optimal_weight.empty:
                return None
            else:
                return df_optimal_weight
        else:
            date = Utils.to_date(date)
            df_weight = df_optimal_weight[df_optimal_weight.date <= date]
            if df_weight.shape[0] > 0:
                return df_weight.iloc[-1]
            else:
                df_weight = df_optimal_weight[df_optimal_weight.date >= date]
                if df_weight.shape[0] > 0:
                    return df_weight.iloc[0]
                else:
                    return None

    @classmethod
    def _save_factor_weight(cls, weight_data, save_type='a'):
        """
        保存日内各时点动量因子的权重数据
        Parameters:
        --------
        :param weight_data: pd.Series, pd.DataFrame
            权重数据, columns=['date', 'w0', 'w1', 'w2', 'w3', 'w4']
        :param save_type: str
            保存方式, 'a'=新增, 'w'=覆盖, 默认为'a'
        :return:
        """
        # TODO 该方法弃用, 改用utils.save_timeseries_data()接口
        factor_weight_path = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.INTRADAYMOMENTUM_CT['optimal_weight_file'])
        if save_type == 'a':
            df_optimal_weight = cls._get_factor_weight()
            if isinstance(weight_data, pd.Series):
                if df_optimal_weight is None:
                    df_optimal_weight = pd.DataFrame().append(weight_data, ignore_index=True)
                else:
                    if weight_data['date'] in df_optimal_weight['date'].values:
                        idx = df_optimal_weight[df_optimal_weight['date']==weight_data['date']].index
                        df_optimal_weight.drop(index=idx, inplace=True)
                    df_optimal_weight = df_optimal_weight.append(weight_data, ignore_index=True)
                    df_optimal_weight.sort_values(by='date', inplace=True)
            elif isinstance(weight_data, pd.DataFrame):
                if df_optimal_weight is None:
                    df_optimal_weight = weight_data
                else:
                    dates = set(df_optimal_weight['date']).intersection(set(weight_data['date']))
                    if len(dates) > 0:
                        df_optimal_weight.set_index('date', inplace=True)
                        df_optimal_weight.drop(index=dates, inplace=True)
                        df_optimal_weight.reset_index(inplace=True)
                    df_optimal_weight = df_optimal_weight.append(weight_data, ignore_index=True)
                    df_optimal_weight.sort_values(by='date', inplace=True)
            else:
                raise TypeError("参数weight_data类型必须为pd.Series或pd.DataFrame.")
            df_optimal_weight.to_csv(factor_weight_path, index=False)
        elif save_type == 'w':
            if isinstance(weight_data, pd.Series):
                pd.DataFrame().append(weight_data, ignore_index=True).to_csv(factor_weight_path, index=False)
            elif isinstance(weight_data, pd.DataFrame):
                weight_data.to_csv(factor_weight_path, index=False)
            else:
                raise TypeError("参数weight_data类型必须为pd.Series或pd.DataFrame.")
        else:
            raise ValueError("保存类型应该为'a'=新增, 'w'=覆盖.")

    @classmethod
    def _calc_periodmomentum_ic(cls, calc_date, date_interval_type='month'):
        """
        计算日内各时段动量因子的Rank IC值向量
        Parameters:
        --------
        :param calc_date: datetime-like, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param date_interval_type: str
            个股收益率计算的时间长度, 'month'=月度收益, 'day'=日收益
        :return: pd.Series
        --------
            IC值向量
            0. date, 日期
            1. IC0, 隔夜时段动量因子IC
            2. IC1, 第1小时动量因子IC
            3. IC2, 第2小时动量因子IC
            4. IC3, 第3小时动量因子IC
            5. IC4, 第4小时动量因子IC
        """
        calc_date = Utils.to_date(calc_date)
        # 读取日内各时段动量因子载荷数据
        df_period_mom = cls._get_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False),
                                                factor_name='periodmomentum', factor_type='raw', drop_na=True)
        if df_period_mom.empty:
            return None

        if date_interval_type == 'month':
            # 读取个股下个月的月度收益率数据
            ret_start, ret_end = Utils.next_month(calc_date)
        elif date_interval_type == 'day':
            ret_start = ret_end = Utils.get_trading_days(start=calc_date, ndays=2)[1]

        df_period_mom['ret'] = np.nan
        for idx, factorloading_data in df_period_mom.iterrows():
            fret = Utils.calc_interval_ret(factorloading_data['id'], start=ret_start, end=ret_end)
            if fret is not None:
                df_period_mom.loc[idx, 'ret'] = fret
        df_period_mom.dropna(inplace=True)
        # 计算Rank IC值
        df_period_mom.drop(columns=['date', 'id', 'm_normal'], inplace=True)
        df_spearman_corr = df_period_mom.corr(method='spearman')
        rank_IC = df_spearman_corr.loc['ret', ['m0', 'm1', 'm2', 'm3', 'm4']].round(4)
        rank_IC['date'] = calc_date
        # 保存Rank IC值
        ic_filepath = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.INTRADAYMOMENTUM_CT['factor_ic_file'])
        Utils.save_timeseries_data(rank_IC, ic_filepath, save_type='a', columns=['date', 'm0', 'm1', 'm2', 'm3', 'm4'])

        return rank_IC

    @classmethod
    def _optimize_periodmomentum_weight(cls, calc_date):
        """
        优化计算日内各时段动量因子载荷的权重
        Parameters:
        --------
        :param calc_date: datetime-like, str
            计算日期
        :return: pd.Series
        --------
            日内各时段动量因子载荷的优化权重向量
            0. date, 日期, datetimelike
            1. w0, 隔夜时段动量因子权重
            2. w1, 第1小时动量因子权重
            3. w2, 第2小时动量因子权重
            4. w3, 第3小时动量因子权重
            5. w4, 第4小时动量因子权重
        """
        calc_date = Utils.to_date(calc_date)
        # 读取过去60个月日内各时段动量因子IC时间序列值
        ic_filepath = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.INTRADAYMOMENTUM_CT['factor_ic_file'])
        df_ic = pd.read_csv(ic_filepath, header=0, parse_dates=[0])
        df_ic = df_ic[df_ic['date'] <= calc_date].iloc[-60:]
        # 计算IC的均值和协方差矩阵
        df_ic.drop(columns='date', inplace=True)
        ic_mean = np.mat(df_ic.mean(axis=0)).reshape((df_ic.shape[1], 1))
        ic_cov = np.mat(df_ic.cov())
        # 计算日内时段因子的最优权重
        optimal_weights = ic_cov.I * ic_mean
        optimal_weights /= optimal_weights.sum()
        optimal_weights = np.array(optimal_weights).flatten().tolist()
        optimal_weights.insert(0, calc_date)
        optimal_weights = pd.Series(optimal_weights, index=['date', 'w0', 'w1', 'w2', 'w3', 'w4'])
        # 保存最优权重
        weight_filepath = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.INTRADAYMOMENTUM_CT['optimal_weight_file'])
        Utils.save_timeseries_data(optimal_weights, weight_filepath, save_type='a', columns=['date', 'w0', 'w1', 'w2', 'w3', 'w4'])

def mom_backtest(start, end):
    """
    日内动量因子历史回测
    Parameters:
    --------
    :param start: datetime-like, str
        回测开始日期，格式YYYY-MM-DD or YYYYMMDD
    :param end: datetime-like, str
        回测结束日期，格式：YYYY-MM-DD or YYYYMMDD
    :return:
    """
    # 取得开始、结束日期之间的交易日序列
    trading_days = Utils.get_trading_days(start, end)
    # 读取截止开始日期前最新的组合回测数据
    prev_trading_day = Utils.get_prev_n_day(trading_days.iloc[0], 1)
    backtest_path = os.path.join(alphafactor_ct.FACTOR_DB.db_path, alphafactor_ct.INTRADAYMOMENTUM_CT.backtest_path)
    factor_data, port_nav = Utils.get_backtest_data(backtest_path, trading_days.iloc[0])
    if port_nav is None:
        port_nav = DataFrame({'date': [prev_trading_day.strftime('%Y-%m-%d')], 'nav': [1.0]})
    # 遍历交易日，如果是月初，则读取动量因子载荷值；如果不是月初，则进行组合估值
    for trading_day in trading_days:
        if factor_data is None:
            nav = port_nav[port_nav.date == prev_trading_day.strftime('%Y-%m-%d')].iloc[0].nav
        else:
            nav = port_nav[port_nav.date == factor_data.iloc[0].date].iloc[0].nav
        interval_ret = 0.0
        # 月初进行调仓
        if Utils.is_month_start(trading_day):
            logging.info('[%s] 月初调仓.' % Utils.datetimelike_to_str(trading_day, True))
            # 调仓前，先计算按均价卖出原先组合个股在当天的组合nav
            if factor_data is not None:
                for ind, factor_info in factor_data.iterrows():
                    daily_mkt = Utils.get_secu_daily_mkt(factor_info.id, trading_day, fq=True, range_lookup=True)
                    if daily_mkt.date == trading_day.strftime('%Y-%m-%d'):
                        vwap_price = daily_mkt.amount / daily_mkt.vol * daily_mkt.factor
                    else:
                        vwap_price = daily_mkt.close
                    interval_ret += vwap_price / factor_info.buyprice - 1.0
                interval_ret /= float(len(factor_data))
                nav *= (1.0 + interval_ret)
            # 读取factor_data
            factor_data_path = os.path.join(alphafactor_ct.FACTOR_DB.db_path, alphafactor_ct.INTRADAYMOMENTUM_CT.synthetic_db_file)
            factor_data = Utils.read_factor_loading(factor_data_path, Utils.datetimelike_to_str(prev_trading_day, False))
            # 遍历factor_data，剔除在调仓日期没有正常交易（如停牌）、及涨停的个股
            ind_to_be_deleted = []
            for ind, factor_info in factor_data.iterrows():
                trading_status = Utils.trading_status(factor_info.id, trading_day)
                if trading_status == SecuTradingStatus.Suspend or trading_status == SecuTradingStatus.LimitUp:
                    ind_to_be_deleted.append(ind)
            factor_data = factor_data.drop(ind_to_be_deleted, axis=0)
            # 对factor_data按因子值升序排列，取前10%个股
            factor_data = factor_data.sort_values(by='factorvalue', ascending=True)
            factor_data = factor_data.iloc[:int(len(factor_data)*0.1)]
            # 遍历factor_data，添加买入价格，并估值计算当天调仓后的组合收益
            factor_data['buyprice'] = 0.0
            interval_ret = 0.0
            for ind, factor_info in factor_data.iterrows():
                daily_mkt = Utils.get_secu_daily_mkt(factor_info.id, trading_day, fq=True, range_lookup=False)
                assert len(daily_mkt) > 0
                factor_data.loc[ind, 'buyprice'] = daily_mkt.amount / daily_mkt.vol * daily_mkt.factor
                interval_ret += daily_mkt.close / factor_data.loc[ind, 'buyprice'] - 1.0
            interval_ret /= float(len(factor_data))
            nav *= (1.0 + interval_ret)
            # 保存factor_data
            port_data_path = os.path.join(alphafactor_ct.FACTOR_DB.db_path, alphafactor_ct.INTRADAYMOMENTUM_CT.backtest_path,
                                          'port_data_%s.csv' % Utils.datetimelike_to_str(trading_day, False))
            factor_data.to_csv(port_data_path, index=False)
        else:
            # 非调仓日，对组合进行估值
            logging.info('[%s] 月中估值.' % Utils.datetimelike_to_str(trading_day, True))
            if factor_data is not None:
                for ind, factor_info in factor_data.iterrows():
                    daily_mkt = Utils.get_secu_daily_mkt(factor_info.id, trading_day, fq=True, range_lookup=True)
                    interval_ret += daily_mkt.close / factor_info.buyprice - 1.0
                interval_ret /= float(len(factor_data))
                nav *= (1.0 + interval_ret)
        # 添加nav
        port_nav = port_nav.append(Series({'date': trading_day.strftime('%Y-%m-%d'), 'nav': nav}), ignore_index=True)
        # 设置prev_trading_day
        prev_trading_day = trading_day
    # 保存port_nav
    port_nav_path = os.path.join(alphafactor_ct.FACTOR_DB.db_path, alphafactor_ct.INTRADAYMOMENTUM_CT.backtest_path, 'port_nav.csv')
    port_nav.to_csv(port_nav_path, index=False)


if __name__ == '__main__':
    # pass
    # IntradayMomentum.calc_factor_loading(start_date='2013-01-01', end_date='2017-12-31', month_end=True, save=True, synthetic_factor=True)
    # mom_backtest('2013-01-04', '2017-12-31')
    IntradayMomentum._calc_periodmomentum_ic('2018-07-31', 'month')
