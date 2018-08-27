#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Filename: SmartMoney
# @Date:   : 2017-10-30 17:50
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.factors.factor import Factor
# import tushare as ts
import datetime
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from pandas import Series
import math
from src.util.utils import Utils, SecuTradingStatus
import src.settings as SETTINGS
import src.alphamodel.alphafactors.cons as alphafactor_ct
from src.util.dataapi.CDataHandler import CDataHandler
from multiprocessing import Pool, Manager
import logging
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class SmartMoney(Factor):
    """聪明钱因子类"""
    __days = alphafactor_ct.SMARTMONEY_CT.days_num   # 读取过去多少天的分钟行情进行因子载荷计算
    _db_file = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.SMARTMONEY_CT.db_file)   # 因子对应数据库文件名

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股的聪明钱因子载荷
        :param code: 个股代码，如SH600000或600000
        :param calc_date: 用于读取分钟行情的交易日期列表
        :return: float
            个股的SmartQ因子载荷值，无法计算返回None
        """
        #  取得过去30天的交易日期
        trading_days = Utils.get_trading_days(end=calc_date, ndays=30, ascending=False)
        # 取得过去self.__days天交易日的分钟行情数据
        be_enough, df_min_mkt = Utils.get_min_mkts_fq(code, trading_days, cls.__days)
        # 计算SmartMoney因子载荷值
        if be_enough:
            # 1.计算指标S_t = abs(R_t)/sqrt(V_t), R_t=第t分钟涨跌幅, V_t=第t分钟成交量
            df_min_mkt['ind_s'] = df_min_mkt.apply(lambda x: abs(x.ret)*10000/math.sqrt(x.volume*100.0) if x.volume > 0 else 0, axis=1)
            # 2.降序排列指标S
            df_min_mkt = df_min_mkt.sort_values(by='ind_s', ascending=False)
            # 3.计算累积成交量、累积成交金额
            df_min_mkt['accum_volume'] = df_min_mkt['volume'].cumsum()
            df_min_mkt['accum_amount'] = df_min_mkt['amount'].cumsum()
            # 4.找到累积成交量占比前20%找到累积成交量占比前20%的交易，视为聪明钱（smart）交易, 那么聪明钱的情绪因子Q=VWAP_{smart}/VWAP_{all}
            # total_volume = df_min_mkt.iloc[df_min_mkt.shape[0]-1]['accum_volume'] * 100
            # total_amount = df_min_mkt.iloc[df_min_mkt.shape[0]-1]['accum_amount']
            # smart_volume = int(df_min_mkt.iloc[df_min_mkt.shape[0]-1]['accum_volume'] * 0.2)
            total_volume = df_min_mkt.iloc[-1].accum_volume * 100
            total_amount = df_min_mkt.iloc[-1].accum_amount
            smart_volume = int(df_min_mkt.iloc[-1].accum_volume * 0.2)
            vwap_all = total_amount / total_volume
            # vwap_smart = 0.0
            smart_mkt = df_min_mkt[df_min_mkt.accum_volume > smart_volume].iloc[0]
            vwap_smart = smart_mkt.accum_amount / (smart_mkt.accum_volume*100.0)
            # for row in df_min_mkt.itertuples():
            #     if row.accum_volume > smart_volume:
            #         vwap_smart = row.accum_amount / (row.accum_volume*100.0)
            #         break
            smart_q = round(vwap_smart / vwap_all, 6)
        else:
            smart_q = None
        # 返回个股的SmartMoney因子载荷值
        return smart_q

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters
        --------
        :param code: str
            个股代码，如600000或SH600000
        :param calc_date: datetime-like or str
            计算日期
        :param q: 队列，用于进程间通信
        :return: 添加因子载荷至队列q中
        """
        logging.info('[%s] Calc SmartQ of %s.' % (calc_date.strftime('%Y-%m-%d'), code))
        smart_q = None
        try:
            smart_q = cls._calc_factor_loading(code, calc_date)
        except Exception as e:
            print(e)
        if smart_q is not None:
            q.put((Utils.code_to_symbol(code), smart_q))

    @classmethod
    def calc_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的因子载荷，并保存至因子数据库
        Parameters
        --------
        :param start_date: datetime-like, str
            开始日期
        :param end_date: datetime-like, str，默认None
            结束日期，如果为None，则只计算start_date日期的因子载荷
        :param month_end: bool，默认True
            只计算月末时点的因子载荷
        :param save: 是否保存至因子数据库，默认为False
        :param kwargs:
            'multi_proc': bool, True=采用多进程并行计算, False=采用单进程计算, 默认为False
        :return: 因子载荷，DataFrame
        --------
            因子载荷,DataFrame
            0. date, 日期, 为计算日期的下一个交易日
            1: id, 证券代码
            2: factorvalue, 因子载荷
            如果end_date=None，返回start_date对应的因子载荷数据
            如果end_date!=None，返回最后一天的对应的因子载荷数据
            如果没有计算数据，返回None
        """
        # 0.取得交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 取得样本个股信息
        # all_stock_basics = CDataHandler.DataApi.get_secu_basics()
        # 遍历交易日序列，计算SMartQ因子载荷
        dict_factor = None
        for calc_date in trading_days_series:
            dict_factor = {'id': [], 'factorvalue': []}
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            # 1.获取用于读取分钟行情的交易日列表（过去30天的交易日列表，降序排列）
            # trading_days = _get_trading_days(calc_date, 30)
            # trading_days = Utils.get_trading_days(end=calc_date, ndays=30, ascending=False)
            # 2.取得样本个股信息
            # stock_basics = ts.get_stock_basics()
            s = (calc_date - datetime.timedelta(days=90)).strftime('%Y%m%d')
            stock_basics = Utils.get_stock_basics(s)
            # 3.遍历样本个股代码，计算Smart_Q因子载荷值
            dict_factor = {'date': None, 'id': [], 'factorvalue': []}

            if 'multi_proc' not in kwargs:
                kwargs['multi_proc'] = False
            if not kwargs['multi_proc']:
                # 采用单进程进行计算
                for _, stock_info in stock_basics.iterrows():
                    # code = '%s%s' % ('SH' if code[:2] == '60' else 'SZ', code)
                    factor_loading = cls._calc_factor_loading(stock_info.symbol, calc_date)
                    print("[%s]Calculating %s's SmartMoney factor loading = %.4f." % (calc_date.strftime('%Y-%m-%d'), stock_info.symbol, -1.0 if factor_loading is None else factor_loading))
                    if factor_loading is not None:
                        # df_factor.ix[code, 'factorvalue'] = factor_loading
                        dict_factor['id'].append(Utils.code_to_symbol(stock_info.symbol))
                        dict_factor['factorvalue'].append(factor_loading)
            else:
                # 采用多进程并行计算SmartQ因子载荷
                q = Manager().Queue()   # 队列，用于进程间通信，存储每个进程计算的因子载荷值
                p = Pool(4)             # 进程池，最多同时开启4个进程
                for _, stock_info in stock_basics.iterrows():
                    p.apply_async(cls._calc_factor_loading_proc, args=(stock_info.symbol, calc_date, q,))
                p.close()
                p.join()
                while not q.empty():
                    smart_q = q.get(True)
                    dict_factor['id'].append(smart_q[0])
                    dict_factor['factorvalue'].append(smart_q[1])

            date_label = Utils.get_trading_days(calc_date, ndays=2)[1]
            dict_factor['date'] = [date_label] * len(dict_factor['id'])
            # 4.计算去极值标准化后的因子载荷
            df_std_factor = Utils.normalize_data(pd.DataFrame(dict_factor), columns='factorvalue', treat_outlier=True, weight='eq')
            # 5.保存因子载荷至因子数据库
            if save:
                # Utils.factor_loading_persistent(cls._db_file, calc_date.strftime('%Y%m%d'), dict_factor)
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), dict_factor, 'SmartMoney', factor_type='raw', columns=['date', 'id', 'factorvalue'])
                cls._save_factor_loading(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), df_std_factor, 'SmartMoney', factor_type='standardized', columns=['date', 'id', 'factorvalue'])
            # 休息300秒
            logging.info('Suspending for 360s.')
            time.sleep(360)
        return dict_factor


def smartq_backtest(start, end):
    """
    SmartQ因子的历史回测
    Parameters:
    --------
    :param start: datetime-like, str
        回测开始日期，格式：YYYY-MM-DD，开始日期应该为月初
    :param end: datetime-like, str
        回测结束日期，格式：YYYY-MM-DD
    :return:
    """
    # 取得开始结束日期间的交易日序列
    trading_days = Utils.get_trading_days(start, end)
    # 读取截止开始日期前最新的组合回测数据
    prev_trading_day = Utils.get_prev_n_day(trading_days.iloc[0], 1)
    backtest_path = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.SMARTMONEY_CT.backtest_path)
    factor_data, port_nav = Utils.get_backtest_data(backtest_path, trading_days.iloc[0])
    # factor_data = None  # 记录每次调仓时最新入选个股的SmartQ因子信息，pd.DataFrame<date,factorvalue,id,buprice>
    if port_nav is None:
        port_nav = DataFrame({'date': [prev_trading_day.strftime('%Y-%m-%d')], 'nav': [1.0]})
    # 遍历交易日，如果是月初，则读取SmartQ因子载荷值，进行调仓；如果不是月初，则进行组合估值
    t = 0   # 记录调仓次数
    for trading_day in trading_days:
        if factor_data is None:
            nav = port_nav[port_nav.date == prev_trading_day.strftime('%Y-%m-%d')].iloc[0].nav
        else:
            nav = port_nav[port_nav.date == factor_data.iloc[0].date].iloc[0].nav
        interval_ret = 0.0
        # 月初进行调仓
        if Utils.is_month_start(trading_day):
            logging.info('[%s] 月初调仓.' % Utils.datetimelike_to_str(trading_day, True))
            # 调仓前，先计算组合按均价卖出原先组合个股在当天的估值
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
            factor_data = Utils.read_factor_loading(SmartMoney.get_db_file(), Utils.datetimelike_to_str(prev_trading_day, False))
            # 遍历factor_data, 计算每个个股过去20天的涨跌幅，并剔除在调仓日没有正常交易（如停牌）及涨停的个股
            ind_to_be_deleted = []
            factor_data['ret20'] = np.zeros(len(factor_data))
            for ind, factor_info in factor_data.iterrows():
                trading_status = Utils.trading_status(factor_info.id, trading_day)
                if trading_status == SecuTradingStatus.Suspend or trading_status == SecuTradingStatus.LimitUp:
                    ind_to_be_deleted.append(ind)
                fret20 = Utils.calc_interval_ret(factor_info.id, end=prev_trading_day, ndays=20)
                if fret20 is None:
                    if ind not in ind_to_be_deleted:
                        ind_to_be_deleted.append(ind)
                else:
                    factor_data.loc[ind, 'ret20'] = fret20
            factor_data = factor_data.drop(ind_to_be_deleted, axis=0)
            # 对factor_data过去20天涨跌幅降序排列，剔除涨幅最大的20%个股
            k = int(factor_data.shape[0]*0.2)
            factor_data = factor_data.sort_values(by='ret20', ascending=False).iloc[k:]
            del factor_data['ret20']    # 删除ret20列
            # 对factor_data按因子值升序排列，取前10%个股
            factor_data = factor_data.sort_values(by='factorvalue', ascending=True)
            k = int(factor_data.shape[0]*0.1)
            factor_data = factor_data.iloc[:k]
            # 遍历factor_data，添加买入价格，并估值计算当天调仓后的组合收益
            factor_data['buyprice'] = 0.0
            interval_ret = 0.0
            for ind, factor_info in factor_data.iterrows():
                daily_mkt = Utils.get_secu_daily_mkt(factor_info.id, trading_day, fq=True, range_lookup=False)
                assert len(daily_mkt) > 0
                factor_data.loc[ind, 'buyprice'] = daily_mkt.amount / daily_mkt.vol * daily_mkt.factor
                interval_ret += daily_mkt.close / factor_data.loc[ind, 'buyprice'] - 1.0
            interval_ret /= float(factor_data.shape[0])
            nav *= (1.0 + interval_ret)
            # 保存factor_data
            port_data_path = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.SMARTMONEY_CT.backtest_path,
                                          'port_data_%s.csv' % Utils.datetimelike_to_str(trading_day, False))
            factor_data.to_csv(port_data_path, index=False)
            t += 1
            if t % 6 == 0:
                logging.info('Suspended for 300s.')
                time.sleep(300)
        else:
            # 非调仓日，对组合进行估值
            logging.info('[%s] 月中估值.' % Utils.datetimelike_to_str(trading_day, True))
            if factor_data is not None:
                for ind, factor_info in factor_data.iterrows():
                    daily_mkt = Utils.get_secu_daily_mkt(factor_info.id, trading_day, fq=True, range_lookup=True)
                    interval_ret += daily_mkt.close / factor_info.buyprice - 1.0
                interval_ret /= float(factor_data.shape[0])
                nav *= (1.0 + interval_ret)
        # 添加nav
        port_nav = port_nav.append(Series({'date': Utils.datetimelike_to_str(trading_day, True), 'nav':nav}),
                                   ignore_index=True)
        # 设置prev_trading_day
        prev_trading_day = trading_day
    # 保存port_nav
    port_nav_path = os.path.join(SETTINGS.FACTOR_DB_PATH, alphafactor_ct.SMARTMONEY_CT.backtest_path, 'port_nav.csv')
    port_nav.to_csv(port_nav_path, index=False)


# def SmartMoney_BackTest(start, end):
#     """
#     聪明钱因子的历史回测
#     --------
#     :param start: string
#         测试开始日期，格式：YYYY-MM-DD，开始日期应该为月初的前一个交易日，即月末交易日
#     :param end: string
#         测试结束日期，格式：YYYY-MM-DD
#     :return:
#     """
#     # 定义组合持仓变量、组合净值序列变量
#     port_0_holding = DataFrame()    # 多头组合的最新持仓,columns:<date,code,hold_vol,buyprice>
#     port_1_holding = DataFrame()    # 1至5组组合的最新持仓,columns:<date,code,hold_vol,buyprice>
#     port_2_holding = DataFrame()
#     port_3_holding = DataFrame()
#     port_4_holding = DataFrame()
#     port_5_holding = DataFrame()
#     port_nav = DataFrame()          # 组合净值序列,columns:<date(index),port_0,port_1,port_2,port_3,port_4,port_5,port_ls>
#     # 组合规模设置为1亿元
#     port_init_scale = 100000000.0
#     # 取得开始、结束日期区间内的交易日列表
#     trading_days = Utils.get_trading_days(start=start, end=end)
#     # 初始化组合净值序列
#     port_nav_header = ['date', 'port_0', 'port_1', 'port_2', 'port_3', 'port_4', 'port_5', 'port_ls']
#     port_nav = port_nav.append(Series([trading_days[0], port_init_scale, port_init_scale, port_init_scale, port_init_scale, port_init_scale, port_init_scale, 0.0], index=port_nav_header), ignore_index=True)
#     port_nav.set_index('date', drop=True, inplace=True)
#     # 遍历交易日，如果是月初，调仓；否则更新组合估值
#     pre_trading_day = trading_days[0]
#     for trading_day in trading_days[1:]:
#         # 如果是第一个交易日或月初，那么进行构建组合或调仓
#         if trading_day == trading_days[1] or trading_day.month != pre_trading_day.month:
#             logging.debug('%s,月初进行调仓' % trading_day.strftime('%Y-%m-%d'))
#             # 1.取得全体A股作为样本股
#             stock_basics = ts.get_stock_basics()
#             # 2.剔除样本股中的ST股和上市未满60天的新股
#             d = int((pre_trading_day - datetime.timedelta(days=90)).strftime('%Y%m%d'))
#             stock_basics = stock_basics[(stock_basics.timeToMarket < d) & (stock_basics.timeToMarket > 0)]
#             # 3.在样本股中剔除最近20天涨幅最大的20%个股
#             stock_basics['ret20'] = np.zeros(stock_basics.shape[0])
#             for code, _ in stock_basics.iterrows():
#                 stock_basics.ix[code, 'ret20'] = Utils.calc_interval_ret(secu_code=code, end=pre_trading_day, ndays=20)
#             stock_basics = stock_basics.sort_values(by='ret20', ascending=False, inplace=False).iloc[int(stock_basics.shape[0]*0.2):]
#             # 4.在样本股中选取Q因子最小的10%，组成SMART组合。涨停、停牌不买入，跌停、停牌不卖出
#             # 4.1.从因子数据库中读取处于pre_trading_day当天的SmartMoney因子载荷，如果数据库中不存在当天的因子载荷，则计算之
#             #     取得（或计算的）因子载荷的格式为DataFrame，索引名称为'ID'，因子载荷项列名为'factorvalue'
#             # db = shelve.open(SmartMoney.get_db_file(), flag='c', protocol=None, writeback=False)
#             # if pre_trading_day.strftime('%Y%m%d') in db:
#             #     try:
#             #         df_smartmoney = db[pre_trading_day.strftime('%Y%m%d')]
#             #     except KeyError:
#             #         df_smartmoney = SmartMoney.calc_factor_loading(pre_trading_day.date())
#             #     finally:
#             #         db.close()
#             # else:
#             #     df_smartmoney = SmartMoney.calc_factor_loading(pre_trading_day.date())
#             df_smartmoney = Utils.read_factor_loading(SmartMoney.get_db_file(), pre_trading_day.strftime('%Y%m%d'))
#             if df_smartmoney.shape[0] == 0:
#                 df_smartmoney = SmartMoney.calc_factor_loading(pre_trading_day, month_end=False, save=False)
#             # 4.2.遍历stock_basics，从因子载荷DataFrame中抽取对应的值
#             stock_basics['smart_money'] = np.zeros(stock_basics.shape[0])
#             for code, _ in stock_basics.iterrows():
#                 if Utils.code_to_symbol(code) in df_smartmoney.index:
#                     stock_basics.ix[code, 'smart_money'] = df_smartmoney.ix[Utils.code_to_symbol(code), 'factorvalue']
#                 else:
#                     stock_basics.ix[code, 'smart_money'] = -1.0
#             # 4.3.剔除因子载荷小于0的数据，并按照因子载荷升序排列
#             stock_basics = stock_basics[stock_basics.smart_money > 0]
#             stock_basics.sort_values(by='smart_money', ascending=True, inplace=True)
#             # 5.构建组合：等权重，规模=1亿元，格式DataFrame<date,code,hold_vol,close_FQ>，
#             # 5.1.构建5个分组合，并计算5个分组和多空组合的净值
#             num_in_group = int(stock_basics.shape[0]/5)
#             sub_port_header = ['date', 'code', 'hold_vol', 'buyprice']
#             # 5.1.1.构建分组合1的持仓，并计算分组合1的净值
#             logging.debug('构建分组合1')
#             port_1_holding = DataFrame()
#             for code, _ in stock_basics.iloc[0:num_in_group].iterrows():    # 添加持仓（剔除停牌股）
#                 mkt_data = Utils.get_secu_daily_mkt(code, trading_day, fq=True)
#                 if len(mkt_data) > 0:
#                     port_1_holding = port_1_holding.append(
#                         Series([trading_day, code, 0, mkt_data.open], index=sub_port_header), ignore_index=True)
#             hold_num = port_1_holding.shape[0]
#             port_nav.ix[trading_day, 'port_1'] = 0.0
#             for ind, holding in port_1_holding.iterrows():   # 遍历分组合1持仓，计算每个持仓的持仓量，以及分组合1净值
#                 port_1_holding.ix[ind, 'hold_vol'] = port_nav.iloc[-2].port_1 / hold_num / holding.buyprice
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_1'] += port_1_holding.ix[ind, 'hold_vol'] * mkt_data.close
#             port_1_holding.to_csv(os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.SMARTMONEY_CT.backtest_path,
#                                                'port_1_%s.csv' % trading_day.strftime('%Y%m%d')), index=False, columns=sub_port_header)
#             # 5.1.2.构建分组合2的持仓，并计算分组合2的净值
#             logging.debug('构建分组合2')
#             port_2_holding = DataFrame()
#             for code, _ in stock_basics.iloc[num_in_group:num_in_group * 2].iterrows():     # 添加持仓（剔除停牌股）
#                 mkt_data = Utils.get_secu_daily_mkt(code, trading_day, fq=True)
#                 if len(mkt_data) > 0:
#                     port_2_holding = port_2_holding.append(
#                         Series([trading_day, code, 0, mkt_data.open], index=sub_port_header), ignore_index=True)
#             hold_num = port_2_holding.shape[0]
#             port_nav.ix[trading_day, 'port_2'] = 0.0
#             for ind, holding in port_2_holding.iterrows():  # 遍历分组合2持仓，计算每个持仓的持仓量，以及分组合2净值
#                 port_2_holding.ix[ind, 'hold_vol'] = port_nav.iloc[-2].port_2 / hold_num / holding.buyprice
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_2'] += port_2_holding.ix[ind, 'hold_vol'] * mkt_data.close
#             port_2_holding.to_csv(os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.SMARTMONEY_CT.backtest_path,
#                                                'port_2_%s.csv' % trading_day.strftime('%Y%m%d')), index=False, columns=sub_port_header)
#             # 5.1.3.构建分组合3的持仓，并计算分组合3的净值
#             logging.debug('构建分组合3')
#             port_3_holding = DataFrame()
#             for code, _ in stock_basics.iloc[num_in_group * 2:num_in_group * 3].iterrows():     # 添加持仓（剔除停牌股）
#                 mkt_data = Utils.get_secu_daily_mkt(code, trading_day, fq=True)
#                 if len(mkt_data) > 0:
#                     port_3_holding = port_3_holding.append(
#                         Series([trading_day, code, 0, mkt_data.open], index=sub_port_header), ignore_index=True)
#             hold_num = port_3_holding.shape[0]
#             port_nav.ix[trading_day, 'port_3'] = 0.0
#             for ind, holding in port_3_holding.iterrows():  # 遍历分组合3持仓，计算每个持仓的持仓量，以及分组合3净值
#                 port_3_holding.ix[ind, 'hold_vol'] = port_nav.iloc[-2].port_3 / hold_num / holding.buyprice
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_3'] += port_3_holding.ix[ind, 'hold_vol'] * mkt_data.close
#             port_3_holding.to_csv(os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.SMARTMONEY_CT.backtest_path,
#                                                'port_3_%s.csv' % trading_day.strftime('%Y%m%d')), index=False, columns=sub_port_header)
#             # 5.1.4.构建分组合4的持仓，并计算分组合4的净值
#             logging.debug('构建分组合4')
#             port_4_holding = DataFrame()
#             for code, _ in stock_basics.iloc[num_in_group * 3:num_in_group * 4].iterrows():     # 添加持仓（剔除停牌股）
#                 mkt_data = Utils.get_secu_daily_mkt(code, trading_day, fq=True)
#                 if len(mkt_data) > 0:
#                     port_4_holding = port_4_holding.append(
#                         Series([trading_day, code, 0, mkt_data.open], index=sub_port_header), ignore_index=True)
#             hold_num = port_4_holding.shape[0]
#             port_nav.ix[trading_day, 'port_4'] = 0.0
#             for ind, holding in port_4_holding.iterrows():  # 遍历分组合4持仓，计算每个持仓的持仓量，以及分组合4净值
#                 port_4_holding.ix[ind, 'hold_vol'] = port_nav.iloc[-2].port_4 / hold_num / holding.buyprice
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_4'] += port_4_holding.ix[ind, 'hold_vol'] * mkt_data.close
#             port_4_holding.to_csv(os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.SMARTMONEY_CT.backtest_path,
#                                                'port_4_%s.csv' % trading_day.strftime('%Y%m%d')), index=False, columns=sub_port_header)
#             # 5.1.5.构建分组合5的持仓，并计算分组合5的净值
#             logging.debug('构建分组合5')
#             port_5_holding = DataFrame()
#             for code, _ in stock_basics.iloc[num_in_group * 4:].iterrows():     # 添加持仓（剔除停牌股）
#                 mkt_data = Utils.get_secu_daily_mkt(code, trading_day, fq=True)
#                 if len(mkt_data) > 0:
#                     port_5_holding = port_5_holding.append(
#                         Series([trading_day, code, 0, mkt_data.open], index=sub_port_header), ignore_index=True)
#             hold_num = port_5_holding.shape[0]
#             port_nav.ix[trading_day, 'port_5'] = 0.0
#             for ind, holding in port_5_holding.iterrows():  # 遍历分组合5持仓，计算每个持仓的持仓量，以及分组合净值
#                 port_5_holding.ix[ind, 'hold_vol'] = port_nav.iloc[-2].port_5 / hold_num / holding.buyprice
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_5'] += port_5_holding.ix[ind, 'hold_vol'] * mkt_data.close
#             port_5_holding.to_csv(os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.SMARTMONEY_CT.backtest_path,
#                                                'port_5_%s.csv' % trading_day.strftime('%Y%m%d')), index=False, columns=sub_port_header)
#             # 5.1.6.计算多空组合的净值
#             port_nav.ix[trading_day, 'port_ls'] = port_nav.ix[trading_day, 'port_1'] - port_nav.ix[trading_day, 'port_5']
#             # 5.2.构建多头组合的持仓，并计算多头组合的净值
#             logging.debug('构建多头组合')
#             port_0_holding = DataFrame()
#             for code, _ in stock_basics.iloc[0:int(stock_basics.shape[0] * factor_ct.SMARTMONEY_CT.constituent_ratio)].iterrows():
#                 mkt_data = Utils.get_secu_daily_mkt(code, trading_day, fq=True)
#                 if len(mkt_data) > 0:
#                     port_0_holding = port_0_holding.append(
#                         Series([trading_day, code, 0, mkt_data.open], index=sub_port_header), ignore_index=True)
#             hold_num = port_0_holding.shape[0]
#             port_nav.ix[trading_day, 'port_0'] = 0.0
#             for ind, holding in port_0_holding.iterrows():  # 遍历多头组合持仓，计算每个持仓的持仓量，以及多头组合净值
#                 port_0_holding.ix[ind, 'hold_vol'] = port_nav.iloc[-2].port_0 / hold_num / holding.buyprice
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_0'] += port_0_holding.ix[ind, 'hold_vol'] * mkt_data.close
#             port_0_holding.to_csv(os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.SMARTMONEY_CT.backtest_path,
#                                                'port_0_%s.csv' % trading_day.strftime('%Y%m%d')), index=False, columns=sub_port_header)
#         # 如果不是第一个交易日或月初，计算组合净值
#         else:
#             logging.debug('%s,组合估值' % trading_day.strftime('%Y-%m-%d'))
#             # 遍历分组合1，计算分组合1净值
#             port_nav.ix[trading_day, 'port_1'] = 0.0
#             for _, holding in port_1_holding.iterrows():
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 # try:
#                 port_nav.ix[trading_day, 'port_1'] += holding.hold_vol * mkt_data.close
#                 # except AttributeError:
#                 #     print('Mkt data Error, code = %s, date = %s.' % (holding.code, trading_day.strftime('%Y-%m-%d')))
#             # 遍历分组合2，计算分组合2净值
#             port_nav.ix[trading_day, 'port_2'] = 0.0
#             for _, holding in port_2_holding.iterrows():
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_2'] += holding.hold_vol * mkt_data.close
#             # 遍历分组合3，计算分组合3净值
#             port_nav.ix[trading_day, 'port_3'] = 0.0
#             for _, holding in port_3_holding.iterrows():
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_3'] += holding.hold_vol * mkt_data.close
#             # 遍历分组合4，计算分组合4净值
#             port_nav.ix[trading_day, 'port_4'] = 0.0
#             for _, holding in port_4_holding.iterrows():
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_4'] += holding.hold_vol * mkt_data.close
#             # 遍历分组合5，计算分组合5净值
#             port_nav.ix[trading_day, 'port_5'] = 0.0
#             for _, holding in port_5_holding.iterrows():
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_5'] += holding.hold_vol * mkt_data.close
#             # 多空组合的净值等于分组合1的净值减去分组合5的净值
#             port_nav.ix[trading_day, 'port_ls'] = port_nav.ix[trading_day, 'port_1'] - port_nav.ix[trading_day, 'port_5']
#             # 遍历多头组合，计算多头组合的净值
#             port_nav.ix[trading_day, 'port_0'] = 0.0
#             for _, holding in port_0_holding.iterrows():
#                 mkt_data = Utils.get_secu_daily_mkt(holding.code, trading_day, fq=True, range_lookup=True)
#                 port_nav.ix[trading_day, 'port_0'] += holding.hold_vol * mkt_data.close
#         # 更改pre_trading_day
#         pre_trading_day = trading_day
#     # 保存组合净值序列数据
#     port_nav.to_csv(os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.SMARTMONEY_CT.backtest_path, 'port_nav.csv'),
#                     index=True, columns=port_nav_header)


if __name__ == '__main__':
    # 计算SmartQ因子载荷
    # SmartMoney.calc_factor_loading(start_date='2018-01-01',end_date='2018-01-05', month_end=True, save=True)
    # 模拟组合历史回测
    smartq_backtest('2016-11-19', '2018-01-03')
