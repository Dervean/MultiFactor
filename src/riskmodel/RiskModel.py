#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 风险模型类文件
# @Filename: RIskModel
# @Date:   : 2018-06-21 20:03
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.util.utils import Utils, SecuTradingStatus
from src.riskmodel.riskfactors.Size import Size
from src.riskmodel.riskfactors.Beta import Beta
from src.riskmodel.riskfactors.Momentum import Momentum
from src.riskmodel.riskfactors.ResVolatility import ResVolatility
from src.riskmodel.riskfactors.NonlinearSize import NonlinearSize
from src.riskmodel.riskfactors.Value import Value
from src.riskmodel.riskfactors.Liquidity import Liquidity
from src.riskmodel.riskfactors.EarningsYield import EarningsYield
from src.riskmodel.riskfactors.Growth import Growth
from src.riskmodel.riskfactors.Leverage import Leverage
import pandas as pd
import src.settings as SETTINGS
import src.riskmodel.cons as riskmodel_ct
import src.riskmodel.riskfactors.cons as riskfactor_ct
import os
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class Barra(object):
    """Barra风险模型基类"""

    def calc_factorloading(self, start_date, end_date=None, multi_prc=False):
        """
        计算风险因子的因子载荷
        Parameters:
        --------
        :param start_date: datetime-like, str
            计算开始日期, 格式: YYYY-MM-DD
        :param end_date: datetime-like, str
            计算结束日期, 格式: YYYY-MM-DD
        :param multi_prc: bool
            是否并行计算, 默认为False
        :return: None
        """
        # 读取交易日序列
        start_date = Utils.to_date(start_date)
        if not end_date is None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(start=start_date, ndays=1)
        # 遍历交易日序列, 计算风险因子的因子载荷
        for calc_date in trading_days_series:
            Size.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Beta.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Momentum.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            ResVolatility.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            NonlinearSize.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Value.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Liquidity.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            EarningsYield.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Growth.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Leverage.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)

            self._calc_secu_dailyret(calc_date)
            self._calc_IndFactorloading(calc_date)
            self._calc_StyleFactorloading(calc_date)

    def estimate_factor_ret(self, start_date, end_date=None):
        """
        估计风险因子的因子报酬
        Parameters:
        --------
        :param start_date:
        :param end_date:
        :return:
        """

    def _calc_secu_dailyret(self, start_date, end_date=None):
        """
        计算个股日收益率数据
        Parameters:
        --------
        :param start_date: datetime-like, str
            计算开始日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param end_date: datetime-like, str
            计算结束日期
        :return:
            计算全体个股日收益率数据, 保存至数据库
        """
        # 读取交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算每个交易日的所有上市交易个股的日收益率数据
        for calc_date in trading_days_series:
            logging.info('[{}] Calc daily return data.'.format(Utils.datetimelike_to_str(calc_date)))
            df_dailyret = pd.DataFrame()
            # 读取在calc_date上市交易的A股代码
            stock_basics = Utils.get_stock_basics(calc_date)
            # 遍历个股, 计算日收益率（剔除停牌的个股）
            for _, stock_info in stock_basics.iterrows():
                logging.debug('[{}] Calc daily ret of {}.'.format(Utils.datetimelike_to_str(calc_date), stock_info.symbol))
                # 剔除停牌个股
                trading_status = Utils.trading_status(stock_info['symbol'], calc_date)
                if trading_status == SecuTradingStatus.Suspend:
                    continue
                # 计算日收益率
                daily_ret = Utils.calc_interval_ret(stock_info['symbol'], calc_date, calc_date)
                if daily_ret is None:
                    continue
                df_dailyret = df_dailyret.append(pd.Series([stock_info.symbol, round(daily_ret, 6)], index=['code', 'ret']), ignore_index=True)
            # 保存每个交易日的收益率数据
            # dailyret_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.DAILY_RET_PATH, '{}.csv'.format(Utils.datetimelike_to_str(calc_date, dash=False)))
            # df_dailyret.to_csv(dailyret_path, index=False, encoding='utf-8')

            dailyret_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.DAILY_RET_PATH)
            Utils.factor_loading_persistent(dailyret_path, Utils.datetimelike_to_str(calc_date, dash=False), df_dailyret)

    def _calc_IndFactorloading(self, date):
        """
        计算风险模型行业因子载荷矩阵, 保存至数据库
        Parameters:
        --------
        :param date: datetime-like, str
            计算日期
        :return:
        """
        logging.info('[{}] Calc industry factor loading matrix.'.format(Utils.datetimelike_to_str(date)))
        # 读取指定日期的行业分类数据
        df_IndClassify_data = Utils.get_industry_classify(date)
        # 构造行业因子载荷矩阵, 并保存至数据库
        if not df_IndClassify_data is None:
            df_IndClassify_data = df_IndClassify_data.set_index('id')
            df_IndClassify_data.index.name = 'code'
            df_IndFactorloading = pd.get_dummies(df_IndClassify_data['ind_code'])
            df_IndFactorloading.reset_index(inplace=True)
            indfactorloading_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.INDUSTRY_FACTORLOADING_PATH)
            Utils.factor_loading_persistent(indfactorloading_path, Utils.datetimelike_to_str(date, dash=False), df_IndFactorloading)
            # df_IndFactorloading.to_csv(os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.INDUSTRY_FACTORLOADING_PATH, 'ind_factorloading_{}.csv'.format(Utils.datetimelike_to_str(date, dash=False))))

    def _calc_StyleFactorloading(self, date):
        """
        计算风险模型中风格因子载荷矩阵, 保存至数据库
        Parameters:
        --------
        :param date: datetime-like, str
            计算日期
        :return:
        """
        logging.info('[{}] Calc style factor loading matrix.'.format(Utils.datetimelike_to_str(date)))
        df_stylefactorloading_matrix = pd.DataFrame()
        for risk_factor in riskfactor_ct.RISK_FACTORS:
            factorloading_path = os.path.join(SETTINGS.FACTOR_DB_PATH, eval('riskfactor_ct.%s_CT' % risk_factor.upper())['db_file'])
            df_factor_loading = Utils.read_factor_loading(factorloading_path, Utils.datetimelike_to_str(date, dash=False))
            df_factor_loading.drop(columns='date', inplace=True)
            df_factor_loading.rename(index=str, columns={'factorvalue': risk_factor}, inplace=True)
            if df_stylefactorloading_matrix.empty:
                df_stylefactorloading_matrix = df_factor_loading
            else:
                df_stylefactorloading_matrix = pd.merge(left=df_stylefactorloading_matrix, right=df_factor_loading, how='inner', on='id')
        if not df_stylefactorloading_matrix.empty:
            df_stylefactorloading_matrix.rename(index=str, columns={'id': 'code'}, inplace=True)

        stylefactorloading_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.STYLE_FACTORLOADING_PATH)
        Utils.factor_loading_persistent(stylefactorloading_path, Utils.datetimelike_to_str(date, dash=False), df_stylefactorloading_matrix)

    def _get_cap_weight(self, date):
        """
        读取指定日期上市个股的流通市值权重, 剔除停牌个股
        Parameters:
        --------
        :param date: datetime-like, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.DataFrame, 个股流通市值权重数据
        --------
            0. code: 个股代码
            1. weight: 流通市值权重
            计算失败, 返回None
        """
        date = Utils.to_date(date)
        # 读取个股流通市值数据
        cap_data_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskfactor_ct.LNCAP_CT.liquidcap_dbfile)
        df_cap_data = Utils.read_factor_loading(cap_data_path, Utils.datetimelike_to_str(date, dash=False))
        # 计算个股流通市值权重
        sum_cap = df_cap_data['factorvalue'].sum()
        df_cap_data['weight'] = df_cap_data['factorvalue'] / sum_cap
        # 读取个股停牌信息
        df_suspension_info = Utils.get_suspension_info(date)
        if df_suspension_info is None:
            return None
        # 个股流通市值数据剔除停牌个股
        df_cap_data = df_cap_data[~df_cap_data['id'].isin(df_suspension_info['symbol'])]

        df_cap_data.drop(columns=['date', 'factorvalue'], inplace=True)
        df_cap_data.rename(columns={'id': 'code'}, inplace=True)
        df_cap_data.reset_index(drop=True, inplace=True)
        return df_cap_data

    def _get_secu_dailyret(self, date):
        """
        读取个股日收益率数据向量
        Parameters:
        --------
        :param date: datetime-like, str
            读取日期
        :return: pd.DataFrame
        --------
            0. code: 个股代码
            1. ret: 个股日收益率
            读取失败, 返回None
        """
        secu_dailyret_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.DAILY_RET_PATH)
        df_secudailyret = Utils.read_factor_loading(secu_dailyret_path, Utils.datetimelike_to_str(date, dash=False))
        if df_secudailyret.empty:
            return None
        else:
            return df_secudailyret

    def _get_IndFactorloading_matrix(self, date):
        """
        读取行业因子载荷矩阵
        Parameters:
        --------
        :param date: datetime-like, str
            读取日期
        :return: pd.DataFrame
        --------
            0. code: 个股代码
            1...30: 行业因子载荷
            读取失败, 返回None
        """
        indfactorloading_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.INDUSTRY_FACTORLOADING_PATH)
        df_IndFactorloading = Utils.read_factor_loading(indfactorloading_path, Utils.datetimelike_to_str(date, dash=False))
        if df_IndFactorloading.empty:
            return None
        else:
            return df_IndFactorloading

    def _get_StyleFactorloading_matrix(self, date):
        """
        读取风格因子载荷矩阵
        Parameters:
        --------
        :param date: datetime-like, str
            读取日期
        :return: pd.DataFrame
        --------
            0. code: 个股代码
            1...10: 风格因子载荷
            读取失败, 返回None
        """
        stylefactorloading_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.STYLE_FACTORLOADING_PATH)
        df_StyleFactorloading = Utils.read_factor_loading(stylefactorloading_path, Utils.datetimelike_to_str(date, dash=False))
        if df_StyleFactorloading.empty:
            return None
        else:
            return df_StyleFactorloading


if __name__ == '__main__':
    BarraModel = Barra()
    # BarraModel.calc_factorloading('2017-12-29')
    # BarraModel._calc_secu_dailyret('2017-12-29')
    # BarraModel._get_cap_weight('2017-12-29')
    # BarraModel._calc_IndFactorloading('2017-12-29')
    # BarraModel._calc_StyleFactorloading('2017-12-29')
    # print(BarraModel._get_IndFactorloading_matrix('2017-12-29').head())
    # print(BarraModel._get_StyleFactorloading_matrix('2017-12-29').head())
    print(BarraModel._get_secu_dailyret('2018-01-02').head())
