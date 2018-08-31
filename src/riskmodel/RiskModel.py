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
import numpy as np
import src.settings as SETTINGS
import src.riskmodel.cons as riskmodel_ct
import src.riskmodel.riskfactors.cons as riskfactor_ct
import os
import logging
import cvxpy as cvx
from src.util.algo import Algo
import datetime
import math
from src.portfolio.portfolio import CWeightHolding, CPortHolding, load_holding_data


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
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(start=start_date, ndays=1)
        # 遍历交易日序列, 计算风险因子的因子载荷
        for calc_date in trading_days_series:
            # Size.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            # Beta.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            # Momentum.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            # ResVolatility.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            # NonlinearSize.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            # Value.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            # Liquidity.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            # EarningsYield.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            # Growth.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            # Leverage.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            #
            # self._calc_secu_dailyret(calc_date)
            self._calc_IndFactorloading(calc_date)
            self._calc_StyleFactorloading(calc_date)

    def estimate_factor_ret(self, start_date, end_date=None):
        """
        估计风险因子的因子报酬
        Parameters:
        --------
        :param start_date: datetime-like, str
            开始日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param end_date: datetime-like, str
            结束日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.series
        --------
            风险因子的因子报酬
            0. date: 日期
            1. market: country factor
            2...31. industry factor
            32...41. style factor
        """
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)

        df_residual_rets = pd.DataFrame()
        for trading_day in trading_days_series:
            logging.info('Estimate risk factor return of {}.'.format(Utils.datetimelike_to_str(trading_day)))
            df_estimator_matrix = pd.DataFrame()
            # 读取个股流通市值权重
            cap_weight = self._get_cap_weight(trading_day)
            cap_weight = cap_weight.set_index('code')
            df_estimator_matrix = cap_weight
            # 读取个股下一个交易日的日收益率序列
            next_trading_day = Utils.get_next_n_day(start=trading_day, ndays=1)
            daily_ret = self._get_secu_dailyret(next_trading_day)
            daily_ret = daily_ret.set_index('code')
            df_estimator_matrix = pd.merge(left=df_estimator_matrix, right=daily_ret, how='inner', left_index=True, right_index=True)
            # 读取行业因子载荷矩阵
            ind_factorloading = self._get_IndFactorloading_matrix(trading_day)
            ind_factorloading = ind_factorloading.set_index('code')
            df_estimator_matrix = pd.merge(left=df_estimator_matrix, right=ind_factorloading, how='inner', left_index=True, right_index=True)
            # 读取风格因子载荷矩阵
            style_factorloading = self._get_StyleFactorloading_matrix(trading_day)
            style_factorloading = style_factorloading.set_index('code')
            df_estimator_matrix = pd.merge(left=df_estimator_matrix, right=style_factorloading, how='inner', left_index=True, right_index=True)

            # 优化计算因子报酬
            ind_factor_labels = ind_factorloading.columns.tolist()
            style_factor_labels = style_factorloading.columns.tolist()
            cap_weight = np.asarray(df_estimator_matrix[cap_weight.columns.tolist()])
            daily_ret = np.asarray(df_estimator_matrix[daily_ret.columns.tolist()])
            ind_factorloading = np.asarray(df_estimator_matrix[ind_factor_labels])
            style_factorloading = np.asarray(df_estimator_matrix[style_factor_labels])

            I = ind_factorloading.shape[1]      # number of industry factors
            S = style_factorloading.shape[1]    # number of style factors
            f_c = cvx.Variable()                # return of country factor
            f_i = cvx.Variable((I, 1))          # return of industry factors
            f_s = cvx.Variable((S, 1))          # return of style factors
            objective = cvx.Minimize(cap_weight.T * (daily_ret - f_c - ind_factorloading * f_i - style_factorloading * f_s) ** 2)
            constraints = [cvx.matmul(cap_weight.T, ind_factorloading) * f_i == 0]
            prob = cvx.Problem(objective, constraints)
            prob.solve()
            # print('status: ', prob.status)
            # print('optimal value: ', prob.value)
            # print('country factor ret: ', f_c.value)
            # print('industry factor ret:')
            # print(f_i.value)
            # print('stryle factor ret: ')
            # print(f_s.value)
            # 保存风险因子报酬
            if prob.status == cvx.OPTIMAL:
                factor_ret = np.concatenate((f_i.value, f_s.value), axis=0)
                factor_ret = np.insert(factor_ret, 0, f_c.value)
                # df_factor_ret = pd.DataFrame(factor_ret.reshape(1, len(factor_ret)), columns=['market'] + ind_factor_labels + style_factor_labels)
                ser_factor_ret = pd.Series(factor_ret, index=['market'] + ind_factor_labels + style_factor_labels)
                self._save_factor_ret(trading_day, ser_factor_ret, 'a')
                # print(ser_factor_ret)

                # 计算残差值
                arr_daily_ret = np.asarray(df_estimator_matrix['ret']).reshape((len(df_estimator_matrix), 1))   # 个股下一交易日收益率序列

                df_estimator_matrix.drop(columns=['weight', 'ret'], inplace=True)
                df_estimator_matrix.insert(loc=0, column='market', value=np.ones(len(df_estimator_matrix)))
                arr_factor_loading = np.asarray(df_estimator_matrix)    # 个股风险因子载荷矩阵

                arr_factor_ret = np.asarray(ser_factor_ret).reshape((len(ser_factor_ret), 1))   # 风险因子报酬向量

                arr_residual = np.around(arr_daily_ret - np.dot(arr_factor_loading, arr_factor_ret),8)

                # ser_residual = pd.Series(arr_residual.flatten(), index=df_estimator_matrix.index,name='residual')
                # ser_residual['date'] = datetime.datetime(trading_day.year, trading_day.month, trading_day.day)
                df_residual = pd.DataFrame(arr_residual.reshape((1, len(arr_residual))), index=[trading_day], columns=df_estimator_matrix.index)
                df_residual_rets = df_residual_rets.append(df_residual)
                # self._save_residual_ret(trading_day, ser_residual, 'a')

            else:
                logging.info('\033[1;31;40m{}优化计算风险因子报酬无解.\033[0m'.format(Utils.datetimelike_to_str(trading_day, dash=True)))

        # df_residual_rets.index.name = 'date'
        self._save_residual_ret(start_date, df_residual_rets, 'a')

    def calc_factor_covmat(self, start_date, end_date=None, calc_mode='cov'):
        """
        计算风险因子协方差矩阵
        Parameters:
        --------
        :param start_date: datetime-like, str
            开始日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param end_date: datetime-like, str
            结束日期, e.g: YYYY-MM-DD, YYYYMMDD, 默认为None
        :param calc_mode: str
            计算模式,
            'cov'=计算因子朴素协方差矩阵和最终协方差矩阵, 并保存
            'naive'=只计算因子朴素协方差矩阵, 并保存
        :return:
        """
        # 读取交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is None:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        else:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)

        for calc_date in trading_days_series:
            logging.info('Calc factor covmat of {}.'.format(Utils.datetimelike_to_str(calc_date, dash=True)))
            # 计算风险因子朴素协方差矩阵, 并保存
            factor_names, naive_covmat = self._naive_factor_covmat(calc_date)
            self._save_factor_covmat(calc_date, naive_covmat, factor_names, 'naive')
            # print('factor names:')
            # print(factor_names)
            # print('naive covmat :')
            # print(naive_covmat)
            if calc_mode == 'cov':
                # 对风险因子朴素协方差矩阵进行Newey_West调整
                nw_adj_covmat = self._covmat_NeweyWest_adj(naive_covmat, calc_date)
                # print('Newey_West adjusted covmat:')
                # print(nw_adj_covmat)
                # 进行波动率偏误调整
                lambda_F = self._vol_RegimeAdj_multiplier(calc_date)
                riskfactor_covmat = lambda_F ** 2 * nw_adj_covmat
                # 保存风险因子协方差矩阵
                self._save_factor_covmat(calc_date, riskfactor_covmat, factor_names, 'cov')

    def calc_spec_varmat(self, start_date, end_date=None, calc_mode='var'):
        """
        计算特质收益率方差矩阵
        Parameters:
        --------
        :param start_date: datetime-like, str
            开始日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param end_date: datetime-like, str
            结束日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param calc_mode: str
            计算模式
            'var'=计算特质收益率朴素方差及最终特质收益率方差矩阵数据, 并保存
            'naive'=只计算特质收益率朴素方差矩阵数据, 并保存
        :return:
        """
        # 读取交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is None:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        else:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)

        for calc_date in trading_days_series:
            logging.info('Calc specific risk variance matrix of {}.'.format(Utils.datetimelike_to_str(calc_date, dash=True)))
            # 计算特质收益率朴素方差数据, 并保存
            naive_specvar = self._naive_spec_var(calc_date)
            self._save_spec_var(calc_date, naive_specvar, 'naive')
            if calc_mode == 'var':
                # 对特质收益率朴素方差矩阵进行Newey_West调整
                nw_spec_var = self._specvar_NeweyWest_adj(naive_specvar, calc_date)
                # 进行波动率偏误调整

                # 保存特质波动率矩阵数据
                self._save_spec_var(calc_date, nw_spec_var, 'var')

    def calc_stocks_covmat(self, calc_date):
        """
        计算个股协方差矩阵
        Parameters:
        --------
        :param calc_date: datetime-like, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: tuple(list, np.array)
                 pd.DataFrame
        --------
            返回一个元组, 第一个元素为个股代码list, 第二个元素为个股协方差矩阵
            返回个股协方差数据, index和columns均为个股代码
        """
        calc_date = Utils.to_date(calc_date)
        # 取得个股风险因子载荷数据
        df_factorloading = self._get_factorloading_matrix(calc_date)
        # 取得个股特质波动率方差数据
        df_specvar = self._get_spec_var('var', calc_date)[Utils.datetimelike_to_str(calc_date, dash=False)]

        tmp = pd.merge(left=df_factorloading, right=df_specvar, how='inner', on='code')
        arr_factorloading = np.array(tmp.loc[:, riskfactor_ct.RISK_FACTORS_NOMARKET])    # 个股因子载荷矩阵, 记为X
        arr_specvar = np.diagflat(tmp['spec_var'].tolist())                     # 个股特质波动率方差矩阵, 记为sigma

        # 取得风险因子协方差矩阵数据, 记为F
        arr_factor_covmat = self._get_factor_covmat('cov', calc_date, factors=riskfactor_ct.RISK_FACTORS_NOMARKET)[Utils.datetimelike_to_str(calc_date, dash=False)]

        # 个股协方差矩阵V = XFX'+sigma
        V = np.linalg.multi_dot([arr_factorloading, arr_factor_covmat, arr_factorloading.T]) + arr_specvar

        # return tmp['code'].tolist(), V
        secu_codes = tmp['code'].tolist()
        df_covmat = pd.DataFrame(V, index=secu_codes, columns=secu_codes)
        df_covmat.index.name = 'code'
        df_covmat.columns.name = 'code'
        return df_covmat

    def get_riskmodel_data(self, date, factors=None, cov_type='cov', var_type='var'):
        """
        读取风险模型相关的数据, 包含: 因子暴露矩阵, 因子协方差矩阵, 特质波动率方差矩阵
        Parameters:
        --------
        :param date: datetime-like, str
            日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param factors: str, list of str
            涉及的因子或因子列表, 默认为None, 即所有风险因子
        :param cov_type: str
            因子协方差矩阵类型, 'naive'=朴素协方差因子矩阵, 'cov'=最终协方差矩阵, 默认为'cov'
        :param var_type: str
            特质波动率方差矩阵类型, 'naive'=朴素特质波动率方差矩阵, 'var'=最终特质波动率方差矩阵, 默认为'var'
        :return: tuple(pd.DataFrame, np.array, pd.Series)
        --------
            风险模型相关数据:
            [1]. 第一个元素为个股风险因子暴露矩阵(pd.DataFrame):
                 0. code: 个股代码(index)
                 1...n: factor_name: 风险因子载荷
            [2]. 第二个元素为风险因子协方差矩阵(np.array)
            [3]. 第三个元素为个股特质波动率方差矩阵(pd.Series)
                 index为个股代码, Series数值为个股对应的特质方差
        """
        # 取得因子暴露矩阵
        df_factorloading = self.get_factorloading_matrix(date, factors=factors)

        # 取得特质波动率方差矩阵
        df_specvar = self.get_spec_var(date, var_type=var_type)

        # 取得因子协方差矩阵
        arr_covmat = self.get_factor_covmat(date, cov_type=cov_type, factors=factors)

        # factors = df_factorloading.columns.tolist()
        tmp = pd.merge(left=df_factorloading, right=df_specvar, how='inner', on='code')
        tmp.set_index('code', inplace=True)
        df_factorloading = tmp[factors]
        spec_var = tmp['spec_var']

        return df_factorloading, arr_covmat, spec_var

    def _save_factor_ret(self, date, data, save_type='a'):
        """
        保存风险因子报酬数据
        Parameters:
        --------
        :param date: datetime-like, str
            日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param data: pd.Series
            因子报酬数据
        :param save_type: str
            保存方式, 'a'=新增, 'w'=覆盖, 默认为'a'
        :return:
        """
        date = Utils.to_date(date)
        factor_ret_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.FACTOR_RETURN_PATH)

        data.name = date
        if save_type == 'a':
            df_factor_ret = self._get_factor_ret()
            if df_factor_ret is None:
                df_factor_ret = pd.DataFrame().append(data, ignore_index=True)
                df_factor_ret.index = [date]
                df_factor_ret.index.name = 'date'
                df_factor_ret.to_csv(factor_ret_path)
            else:
                df_factor_ret.loc[date] = data
                df_factor_ret.sort_index(inplace=True)
                df_factor_ret.to_csv(factor_ret_path)
        elif save_type == 'w':
            data.to_csv(factor_ret_path, index=True)

    def _save_residual_ret(self, date, data, save_type='a'):
        """
        保存个股风险模型中的残差收益率(特质收益率)数据
        Parameters:
        --------
        :param date: datetime-like, str
            日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param data: pd.Series, pd.DataFrame
            个股残差收益率数据
        :param save_type: str
            保存方式, 'a'=新增, 'w'=覆盖, 默认为'a'
        :return:
        """
        date = Utils.to_date(date)
        residual_ret_path =os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.RISKMODEL_RESIDUAL_PATH)

        data.name = date
        if save_type == 'a':
            df_residual_ret = self._get_residual_ret()
            if isinstance(data, pd.Series):
                if df_residual_ret is None:
                    df_residual_ret = pd.DataFrame().append(data, ignore_index=True)
                    df_residual_ret.index = [date]
                    df_residual_ret.index.name = 'date'
                    df_residual_ret.to_csv(residual_ret_path)
                else:
                    data.name = date
                    if date in df_residual_ret.index:
                        df_residual_ret.drop(index=date, inplace=True)
                    df_residual_ret = df_residual_ret.append(data, ignore_index=False)
                    df_residual_ret.sort_index(inplace=True)
                    df_residual_ret.to_csv(residual_ret_path)
            elif isinstance(data, pd.DataFrame):
                if df_residual_ret is None:
                    df_residual_ret = data
                    df_residual_ret.to_csv(residual_ret_path)
                else:
                    dates = [d for d in data.index if d in df_residual_ret.index]
                    if len(dates) > 0:
                        df_residual_ret.drop(index=dates, inplace=True)
                    df_residual_ret = df_residual_ret.append(data)
                    df_residual_ret.sort_index(inplace=True)
                    if df_residual_ret.index.name != 'date':
                        df_residual_ret.index.name = 'date'
                    df_residual_ret.to_csv(residual_ret_path)
            else:
                raise TypeError("参数data类型必须为pd.Series或pd.DataFrame.")
        elif save_type == 'w':
            data.to_csv(residual_ret_path, index=True)

    def _get_factor_ret(self, start=None, end=None, ndays=None, factors=None):
        """
        读取风险因子报酬数据
        Parameters:
        --------
        :param start: datetime-like, str
            开始日期, e.g: YYYY-MM-DD or YYYYMMDD
        :param end: datetime-like, str
            结束日期, e.g: YYYY-MM-DD or YYYYMMDD
        :param ndays: int
            天数
        :param factors: str, list
            需要返回的风险因子, 默认为None, 即返回所有风险因子
        :return: pd.DataFrame
        --------
            返回风险因子报酬数据
            0. date: 日期(index)
            1. market: country factor
            2...31. industry factor
            32...41. style factor

            根据提供的参数不同, 返回不同数据:
            1. start和end都不为None: 返回开始、结束日期区间内数据
            2. start为None, end和ndays不为None: 返回截止结束日期(含)的前ndays天数据
            3. end为None, start和ndays不为None: 返回自从开始日期(含)的后ndays天数据
            4. start和ndays为None, end不为None: 返回截止结束日期(含)的所有数据
            5. end和ndays为None, start不为None: 返回自从开始日期(含)开始的所有数据
            6. start和end都为None, 返回所有数据
        """
        factor_ret_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.FACTOR_RETURN_PATH)
        if not os.path.isfile(factor_ret_path):
            return None
        df_factorret = pd.read_csv(factor_ret_path, header=0, index_col=0, parse_dates=[0])
        if factors is not None:
            df_factorret = df_factorret[factors]
        if df_factorret.empty:
            return None
        if (start is not None) and (end is not None):
            start = Utils.to_date(start)
            end = Utils.to_date(end)
            return df_factorret.loc[start: end]
        elif end is not None:
            end = Utils.to_date(end)
            if ndays is not None:
                return df_factorret.loc[: end].tail(ndays)
            else:
                return df_factorret.loc[: end]
        elif start is not None:
            start = Utils.to_date(start)
            if ndays is not None:
                return df_factorret.loc[start:].head(ndays)
            else:
                return df_factorret.loc[start:]
        else:
            return df_factorret

    def _get_residual_ret(self, start=None, end=None, ndays=None):
        """
        读取风险模型的残差收益率数据
        Parameters:
        --------
        :param start: datetime-like, str
            开始日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param end: datetime-like, str
            结束日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param ndays: int
            天数
        :return: pd.DataFrame
        --------
            返回风险模型的残差收益率数据

            根据提供的参数不同, 返回不同数据:
            1. start和end都不为None: 返回开始、结束日期区间内数据
            2. start为None, end和ndays不为None: 返回截止结束日期(含)的前ndays天数据
            3. end为None, start和ndays不为None: 返回自从开始日期(含)的后ndays天数据
            4. start和ndays为None, end不为None: 返回截止结束日期(含)的所有数据
            5. end和ndays为None, start不为None: 返回自从开始日期(含)开始的所有数据
            6. start和end都为None, 返回所有数据
        """
        residual_ret_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.RISKMODEL_RESIDUAL_PATH)
        if not os.path.isfile(residual_ret_path):
            return None
        df_residual_ret = pd.read_csv(residual_ret_path, header=0, index_col=0, parse_dates=[0])
        if df_residual_ret.empty:
            return None
        if (start is not None) and (end is not None):
            start = Utils.to_date(start)
            end = Utils.to_date(end)
            df_residual_ret = df_residual_ret[start: end]
        elif end is not None:
            end = Utils.to_date(end)
            if ndays is not None:
                df_residual_ret = df_residual_ret.loc[: end].tail(ndays)
            else:
                df_residual_ret = df_residual_ret.loc[: end]
        elif start is not None:
            start = Utils.to_date(start)
            if ndays is not None:
                df_residual_ret = df_residual_ret.loc[start:].head(ndays)
            else:
                df_residual_ret = df_residual_ret.loc[start:]
        else:
            return df_residual_ret

        if df_residual_ret.empty:
            return None
        else:
            return df_residual_ret

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
                df_dailyret = df_dailyret.append(pd.Series([stock_info.symbol, calc_date, round(daily_ret, 6)], index=['code', 'date', 'ret']), ignore_index=True)
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
        if df_IndClassify_data is not None:
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
        for risk_factor in riskfactor_ct.STYLE_RISK_FACTORS:
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
            df_secudailyret = df_secudailyret.drop(columns='date')
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

    def get_StyleFactorloading_matrix(self, date):
        """读取风格因子载荷矩阵"""
        return self._get_StyleFactorloading_matrix(date)

    def _get_factorloading_matrix(self, date, factors=None):
        """
        读取风险因子载荷矩阵数据
        Parameters:
        --------
        :param date: datetime-like, str
            读取日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param factors: str, list
            需要返回的因子
        :return: pd.DataFrame
        """
        date = Utils.to_date(date)
        ind_factorloading = self._get_IndFactorloading_matrix(date)
        style_factorloading = self._get_StyleFactorloading_matrix(date)
        factorloading_matrix = pd.merge(left=ind_factorloading, right=style_factorloading, how='inner', on='code')
        for mf in riskfactor_ct.MARKET_FACTORS:
            factorloading_matrix[mf] = 1.0
        if factors is not None:
            if isinstance(factors, str):
                if factors not in riskfactor_ct.RISK_FACTORS:
                    raise ValueError("给定的因子名称不是风险因子, 无法返回风险因子载荷值.")
                factors = ['code', factors]
            elif isinstance(factors, list):
                if not all([factor in riskfactor_ct.RISK_FACTORS for factor in factors]):
                    raise ValueError("给定的因子中含非风险因子, 无法提供风险因子载荷值.")
                factors = ['code'] + factors
            else:
                raise TypeError("factors参数类型只能为str或list")
            factorloading_matrix = factorloading_matrix[factors]
        return factorloading_matrix

    def get_factorloading_matrix(self, date, factors=None):
        """读取风险因子载荷矩阵数据"""
        return self._get_factorloading_matrix(date, factors)

    def _naive_factor_covmat(self, date):
        """
        计算风险模型因子朴素协方差矩阵, 采用EWMA算法计算
        Parameters:
        --------
        :param date: datetime-lie, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: tuple(str, np.array)
        --------
            返回一个元组, 第一个元素是因子名称list, 第二个元素是因子朴素协方差矩阵
        """
        # 读取风险因子报酬序列数据
        ewma_param = riskmodel_ct.FACTOR_COVMAT_PARAMS['EWMA']
        date = Utils.to_date(date)
        df_factor_ret = self._get_factor_ret(end=date, ndays=ewma_param['trailing'], factors=riskfactor_ct.RISK_FACTORS_NOMARKET)
        arr_factor_ret = np.array(df_factor_ret)

        # 采用协方差参数计算因子协方差矩阵
        cov_weight = Algo.ewma_weight(len(arr_factor_ret), ewma_param['cov_half_life'])
        cov_mat = np.cov(arr_factor_ret, rowvar=False, bias=True, aweights=cov_weight)

        # 采用方差参数计算因子方差向量, 并替换协方差矩阵的对角线数据
        vol_weight = Algo.ewma_weight(len(arr_factor_ret), ewma_param['vol_half_life'])
        avg = np.average(arr_factor_ret, axis=0, weights=vol_weight)
        X = arr_factor_ret - avg[None, :]
        arr_vol = np.sum(vol_weight[:, None] * X ** 2, axis=0)
        for k in np.arange(len(arr_vol)):
            cov_mat[k, k] = arr_vol[k]

        return df_factor_ret.columns.tolist(), cov_mat

    def _save_factor_covmat(self, date, cov_mat, factor_names, cov_type):
        """
        保存协方差矩阵数据
        Parameters:
        --------
        :param date: datetime-like, str
             日期, e.g: YYYY-MM-Dd, YYYYMMDD
        :param cov_mat: np.array
            风险因子协方差矩阵数据
        :param factor_names: list of str
            风险因子名称list, 协方差矩阵的header
        :param cov_type: str
            协方差矩阵类型, 'naive'=朴素协方差矩阵, 'cov'=最终协方差矩阵
        :return:
        """
        date = Utils.to_date(date)
        if cov_type == 'naive':
            cov_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.FACTOR_NAIVE_COVMAT_PATH, 'cov_{}.csv'.format(Utils.datetimelike_to_str(date, dash=False)))
        elif cov_type == 'cov':
            cov_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.FACTOR_COVMAT_PATH, 'cov_{}.csv'.format(Utils.datetimelike_to_str(date, dash=False)))
        else:
            raise ValueError("协方差矩阵类型错误.")

        df_covmat = pd.DataFrame(cov_mat, index=factor_names, columns=factor_names)
        df_covmat.to_csv(cov_path, index_label='factor')

    def _get_factor_covmat(self, cov_type, end, ndays=None, factors=None):
        """
        读取风险因子协方差矩阵数据
        Parameters:
        --------
        :param end: datetime-like, str
            结束日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param ndays: int
            天数, 默认为None
        :param cov_type: str
            协方差矩阵类型, 'naive'=朴素协方差矩阵, 'cov'=最终协方差矩阵
        :param factors: str, list of str
            指定需要返回的因子或因子列表
        :return: dict{str, np.array}
        --------
            风险因子协方差因子矩阵数据时间序列, dict{'YYYYMMDD': cov_mat}
        """
        if cov_type == 'naive':
            cov_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.FACTOR_NAIVE_COVMAT_PATH)
        elif cov_type == 'cov':
            cov_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.FACTOR_COVMAT_PATH)
        else:
            raise ValueError("协方差矩阵类型错误.")

        end = Utils.to_date(end)
        if ndays is None:
            trading_days_series = Utils.get_trading_days(end=end, ndays=1)
        else:
            trading_days_series = Utils.get_trading_days(end=end, ndays=ndays)

        factor_covmat = {}
        for calc_date in trading_days_series:
            covmat_path = os.path.join(cov_path, 'cov_{}.csv'.format(Utils.datetimelike_to_str(calc_date, dash=False)))
            df_factor_covmat = pd.read_csv(covmat_path, header=0, index_col=0)
            if factors is not None:
                df_factor_covmat = df_factor_covmat.loc[factors, factors]
            arr_factor_covmat = np.array(df_factor_covmat)
            factor_covmat[Utils.datetimelike_to_str(calc_date, dash=False)] = arr_factor_covmat

        if len(factor_covmat) == 0:
            return None
        else:
            return factor_covmat

    def get_factor_covmat(self, date, cov_type='cov', factors=None):
        """
        读取风险因子协方差矩阵数据
        Parameters:
        --------
        :param cov_type: str
            协方差矩阵类型, 'naive'=朴素协方差矩阵, 'cov'=最终协方差矩阵, 默认'cov'
        :param date: datetime-like, str
            日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param factors: str, list of str
            指定需要返回的因子或因子列表
        :return: np.array
        """
        dict_factor_covmat = self._get_factor_covmat(cov_type=cov_type, end=date, factors=factors)
        if dict_factor_covmat is not None:
            return dict_factor_covmat[list(dict_factor_covmat.keys())[0]]
        else:
            return None

    def _covmat_NeweyWest_adj(self, naive_covmat, date):
        """
        计算经newey_west调整后的风险因子协方差矩阵
        Parameters:
        --------
        :param naive_covmat: np.array
            风险因子朴素协方差矩阵,矩阵大小为: N×N
        :param date: datetime-like, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: np.array
        --------
            newey_west调整后的协方差矩阵
        """
        # 读取风险因子报酬序列数据
        nw_param = riskmodel_ct.FACTOR_COVMAT_PARAMS['Newey_West']
        date = Utils.to_date(date)
        df_factor_ret = self._get_factor_ret(end=date, ndays=nw_param['trailing'], factors=riskfactor_ct.RISK_FACTORS_NOMARKET)
        arr_factor_ret = np.array(df_factor_ret)

        if naive_covmat.shape[0] != naive_covmat.shape[1]:
            raise ValueError("风险因子朴素协方差矩阵(naive_covmat)行和列的长度不一致")
        if naive_covmat.shape[0] != arr_factor_ret.shape[1]:
            raise ValueError("风险因子朴素协方差矩阵(naive_covmat)的长度与风险因子报酬矩阵(arr_factor_ret)的列长度不一致")

        m, n = arr_factor_ret.shape
        covmat_NW_adj = np.zeros((n, n))
        D = nw_param['cov_lags']
        if D < 1:
            raise ValueError("参数cov_lags必须大于0")
        for delta in range(1, D+1):
            weight = Algo.ewma_weight(len(arr_factor_ret) - delta, nw_param['cov_half_life'])

            arr_factor_ret1 = arr_factor_ret[: -delta, :]
            avg1 = np.average(arr_factor_ret1, axis=0, weights=weight)
            arr_factor_ret1 = arr_factor_ret1 - avg1

            arr_factor_ret2 = arr_factor_ret[delta: , :]
            avg2 = np.average(arr_factor_ret2, axis=0, weights=weight)
            arr_factor_ret2 = arr_factor_ret2 - avg2

            # m, n = weight.shape
            weight = np.reshape(weight, (len(weight), 1))
            arr_factor_ret1 = arr_factor_ret1 * weight
            cov1 = np.dot(arr_factor_ret1.T, arr_factor_ret2)
            cov2 = np.dot(arr_factor_ret2.T, arr_factor_ret1)

            alpha = 1.0 - delta / (1.0 + D)
            covmat_NW_adj += alpha * (cov1 + cov2)

        covmat_NW_adj = 21.0 * (naive_covmat + covmat_NW_adj)
        return covmat_NW_adj

    def _vol_RegimeAdj_multiplier(self, date):
        """
        计算波动率偏误乘数
        Parameters:
        --------
        :param date: datetime-like, str
            计算日期
        :return:
        """
        regime_adj_param = riskmodel_ct.FACTOR_COVMAT_PARAMS['Vol_Regime_Adj']
        date = Utils.to_date(date)
        # 读取风险因子报酬数据序列
        df_factorret = self._get_factor_ret(end=date, ndays=regime_adj_param['trailing'], factors=riskfactor_ct.RISK_FACTORS_NOMARKET)
        if df_factorret.shape[0] != regime_adj_param['trailing']:
            raise ValueError("风险因子报酬数据序列长度不等于%d." % regime_adj_param['trailing'])
        if df_factorret.shape[1] != len(riskfactor_ct.RISK_FACTORS_NOMARKET):
            raise ValueError("风险因子报酬数据序列的列数与风险因子数量不一致.")
        arr_factorret = np.array(df_factorret)
        # 读取风险因子朴素协方差矩阵数据序列, 并转换为风险因子方差序列数据
        factor_covmat_series = self._get_factor_covmat(cov_type='naive', end=date-datetime.timedelta(days=1), ndays=regime_adj_param['trailing'], factors=riskfactor_ct.RISK_FACTORS_NOMARKET)
        arr_varmat = np.zeros(arr_factorret.shape)
        k = 0
        for str_date, cov_mat in factor_covmat_series.items():
            n = len(cov_mat)
            for i in range(n):
                arr_varmat[k][i] = cov_mat[i][i]
            k += 1
        if arr_varmat.shape[0] != regime_adj_param['trailing']:
            raise ValueError("风险因子方差序列长度不等于%d." % regime_adj_param['trailing'])
        if arr_factorret.shape != arr_varmat.shape:
            raise ValueError("风险因子报酬数据序列与风险因子方差序列的大小不一致.")
        arr_factorret_square = arr_factorret ** 2
        b_stat = arr_factorret_square / arr_varmat
        b_stat = np.sum(b_stat, axis=1) / arr_factorret_square.shape[1]

        weight = Algo.ewma_weight(regime_adj_param['trailing'], regime_adj_param['half_life'])
        multiplier = np.sum(weight * b_stat)
        return math.sqrt(multiplier)

    def _naive_spec_var(self, date):
        """
        计算特质收益率朴素方差向量, 采用EWMA算法计算
        Parameters:
        --------
        :param date: datetime-like, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
        --------
            返回个股特质收益率方差数据, pd.Series的index为个股代码
        """
        date = Utils.to_date(date)
        # 读取个股基本信息数据
        stock_basics = Utils.get_stock_basics(date)
        # 读取截止date日期的个股特质收益率数据
        ewma_param = riskmodel_ct.SPECIFICRISK_VARMAT_PARAMS['EWMA']
        df_residual_ret = self._get_residual_ret(end=date)
        # 遍历个股基本信息, 采用EWMA算法计算特质收益率方差
        symbols = [symbol for symbol in stock_basics['symbol'] if symbol in df_residual_ret.columns.tolist()]
        df_residual_ret = df_residual_ret[symbols]
        weight = Algo.ewma_weight(ewma_param['trailing'], ewma_param['half_life'])

        ser_spec_var = pd.Series(name='spec_var')
        for symbol in df_residual_ret.columns:
            arr_residual_ret = np.asarray(df_residual_ret[symbol].dropna().tail(ewma_param['trailing']))
            if len(arr_residual_ret) < ewma_param['trailing']/2:
                continue
            if len(arr_residual_ret) == ewma_param['trailing']:
                avg = np.average(arr_residual_ret, weights=weight)
                arr_residual_ret -= avg
                naive_var = np.sum(weight * arr_residual_ret ** 2)
            else:
                w = Algo.ewma_weight(len(arr_residual_ret), ewma_param['half_life'])
                avg = np.average(arr_residual_ret, weights=w)
                arr_residual_ret -= avg
                naive_var = np.sum(w * arr_residual_ret ** 2)
            ser_spec_var[symbol] = naive_var

        ser_spec_var.index.name = 'code'
        return ser_spec_var

    def _save_spec_var(self, date, spec_var, var_type):
        """
        保存个股特质收益率方差数据
        Parameters:
        --------
        :param date: datetime-like, str
            日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param spec_var: pd.Series
            个股特质收益率方差数据
        :param var_type: str
            方差数据类型, 'naive'=朴素方差数据, 'var'=最终特质收益率方差数据
        :return:
        """
        date = Utils.to_date(date)
        if var_type == 'naive':
            var_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.SPECIFICRISK_NAIVE_VARMAT_PATH, 'specvar_{}.csv'.format(Utils.datetimelike_to_str(date, dash=False)))
        elif var_type == 'var':
            var_path = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.SPECIFICRISK_VARMAT_PATH, 'specvar_{}.csv'.format(Utils.datetimelike_to_str(date, dash=False)))
        else:
            raise ValueError("特质收益率方差类型错误.")

        spec_var.to_csv(var_path, index=True, header=True)

    def _get_spec_var(self, var_type, end, ndays=None):
        """
        读取个股特质波动率方差数据
        Parameters:
        --------
        :param var_type: str
            特质波动率方差的数据类型, 'naive'=朴素方差数据, 'var'=最终特质波动率方差数据
        :param end: datetime-like, str
            结束日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param ndays: int
            天数, 默认为None
        :return: dict{str, pd.DataFrame}
        --------
            个股特质波动率方差数据时间序列, dict{'YYYYMMDD': spec_var}
        """
        if var_type == 'naive':
            specvar_basepath = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.SPECIFICRISK_NAIVE_VARMAT_PATH)
        elif var_type == 'var':
            specvar_basepath = os.path.join(SETTINGS.FACTOR_DB_PATH, riskmodel_ct.SPECIFICRISK_VARMAT_PATH)
        else:
            raise ValueError("特质波动率数据类型错误.")

        end = Utils.to_date(end)
        if ndays is None:
            trading_days_series = Utils.get_trading_days(end=end, ndays=1)
        else:
            trading_days_series = Utils.get_trading_days(end=end, ndays=ndays)

        spec_var = {}
        for calc_date in trading_days_series:
            specvar_path = os.path.join(specvar_basepath, 'specvar_{}.csv'.format(Utils.datetimelike_to_str(calc_date, dash=False)))
            if not os.path.isfile(specvar_path):
                raise FileExistsError("文件%s不存在." % specvar_path)
            df_specvar = pd.read_csv(specvar_path, header=0)
            spec_var[Utils.datetimelike_to_str(calc_date, dash=False)] = df_specvar

        if len(spec_var) == 0:
            return None
        else:
            return spec_var

    def get_spec_var(self, date, var_type='var'):
        """
        读取个股特质波动率方差数据
        Parameters:
        --------
        :param date: datetime-like,str
            日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param var_type: str
            特质波动率方差的数据类型, 'naive'=朴素方差数据, 'var'=最终特质波动率方差数据
        :return: pd.DataFrame
        --------
            个股特质波动率方差矩阵数据
        """
        dict_spec_var = self._get_spec_var(var_type=var_type, end=date)
        if dict_spec_var is not None:
            return dict_spec_var[list(dict_spec_var.keys())[0]]
        else:
            return None

    def _specvar_NeweyWest_adj(self, naive_specvar, date):
        """
        计算经newey_west调整后的特质波动率方差数据
        Parameters:
        --------
        :param naive_specvar: pd.Series
            特质波动率朴素方差向量数据
        :param date: datetime-like, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.Series
        --------
            newey_west调整后的特质波动率方差向量数据
        """
        nw_param = riskmodel_ct.SPECIFICRISK_VARMAT_PARAMS['Newey_West']
        date = Utils.to_date(date)
        # 读取特质收益率序列数据
        df_spec_ret = self._get_residual_ret(end=date)
        # 读取股票基本信息
        stock_basics = Utils.get_stock_basics(date)
        # 遍历特质收益率序列数据, 计算个股经newey_west调整后的特质波动率方差数据
        symbols = [symbol for symbol in stock_basics['symbol'] if symbol in df_spec_ret.columns.tolist()]
        df_spec_ret = df_spec_ret[symbols]
        codes = df_spec_ret.columns.tolist()
        D = nw_param['lags']    # 滞后时间长度
        if D < 1:
            raise ValueError("滞后时间长度参数lags的值必须大于0")
        # 计算权重向量组
        weights = []
        for delta in range(1, D+1):
            weights.append(Algo.ewma_weight(nw_param['trailing']-delta, nw_param['half_life']))

        nw_specvar = pd.Series(name='spec_var')     # newey_west调整后的特质收益率方差向量
        for code in codes:
            if code not in naive_specvar:
                continue

            arr_spec_ret = np.array(df_spec_ret[code].dropna())
            if len(arr_spec_ret) < int(nw_param['trailing']/2):
                continue
            if len(arr_spec_ret) > nw_param['trailing']:
                arr_spec_ret = arr_spec_ret[: nw_param['trailing']]

            nw_adj = 0
            for delta in range(1, D+1):
                if len(arr_spec_ret) < nw_param['trailing']:
                    w = Algo.ewma_weight(len(arr_spec_ret)-delta, nw_param['half_life'])
                else:
                    w = weights[delta-1]

                arr_spec_ret1 = arr_spec_ret[: -delta]
                avg1 = np.average(arr_spec_ret1, weights=w)
                arr_spec_ret1 = arr_spec_ret1 - avg1

                arr_spec_ret2 = arr_spec_ret[delta:]
                avg2 = np.average(arr_spec_ret2, weights=w)
                arr_spec_ret2 = arr_spec_ret2 - avg2

                cov = np.sum(w * arr_spec_ret1 * arr_spec_ret2)
                alpha = 1.0 - delta / (1.0 + D)
                nw_adj += alpha * cov * 2
            secu_nw_specvar = 21.0 * (naive_specvar[code] + nw_adj)
            nw_specvar[code] = secu_nw_specvar if secu_nw_specvar > 0 else 0    # 确保特质波动率方差大于等于0

        nw_specvar.index.name = 'code'
        return nw_specvar

    def risk_contribution(self, holding, date=datetime.date.today(), benchmark=None):
        """
        对给定的持仓数据进行风险归因
        Parameters:
        --------
        :param holding: WeightHolding类, 或PortHolding类
            持仓数据
        :param date: datetime-like, str
            计算日期
        :param benchmark: str
            比较基准的代码, e.g: SH000300
        :return: pd.Series
        --------
            持仓数据在风险因子上的风险归因值(index为风险因子代码)
        """
        # TODO 完善benchmark不为None的风险归因计算
        if (not isinstance(holding, CWeightHolding)) and (not isinstance(holding, CPortHolding)):
            raise ValueError("风险归因应提供WeightHolding类或PortHolding类的持仓数据.")
        date = Utils.to_date(date)
        if holding.count < 1:
            raise ValueError("持仓数据不能为空.")
        codes = holding.holding['code'].tolist()
        # 取得持仓数据
        df_holding = holding.holding
        # # 取得持仓个股权重列向量
        # w = np.array(holding.holding['weight']).reshape((holding.count, 1))
        # 取得持仓的风险因子载荷数据
        df_factorloading = self._get_factorloading_matrix(date)
        df_factorloading = pd.merge(left=df_holding, right=df_factorloading, how='inner', on='code')
        # 取得个股特质波动率方差矩阵数据
        df_specvar = self._get_spec_var('var', date)[Utils.datetimelike_to_str(date, dash=False)]
        df_factorloading = pd.merge(left=df_specvar, right=df_factorloading, how='inner', on='code')

        # df_factorloading.set_index('code', inplace=True)
        arr_factorloading = np.array(df_factorloading.loc[:, riskfactor_ct.RISK_FACTORS_NOMARKET])               # 因子暴露矩阵
        arr_weight = np.array(df_factorloading.loc[:, 'weight']).reshape((len(df_factorloading), 1))    # 个股权重向量
        arr_specvar =np.diagflat(df_factorloading['spec_var'].tolist())                                 # 特质波动率方差矩阵
        # 取得风险因子协方差矩阵
        arr_factor_covmat = self._get_factor_covmat('cov', date, factors=riskfactor_ct.RISK_FACTORS_NOMARKET)[Utils.datetimelike_to_str(date, dash=False)]

        # 计算组合预期波动率
        fsigma = np.sqrt(np.linalg.multi_dot([arr_weight.T, arr_factorloading, arr_factor_covmat, arr_factorloading.T, arr_weight]) + np.linalg.multi_dot([arr_weight.T, arr_specvar, arr_weight]))
        fsigma = float(fsigma)

        # 计算风险归因值
        Psi = np.dot(arr_weight.T, arr_factorloading).transpose()
        risk_contribution = 1.0 / fsigma * Psi * np.dot(arr_factor_covmat, Psi)
        risk_contribution = pd.Series(risk_contribution.flatten(), index=riskfactor_ct.RISK_FACTORS_NOMARKET)    # 风险因子的风险贡献数据
        fselection = fsigma - risk_contribution.sum()
        # fallocation = risk_contribution.sum() - risk_contribution['market']
        risk_contribution['sigma'] = fsigma                                                         # 组合预期波动率
        risk_contribution['selection'] = fselection                                                 # 选股带来的风险贡献
        # risk_contribution['allocation'] = fallocation                                               # 配置带来的风险贡献
        risk_contribution['industry'] = risk_contribution[riskfactor_ct.INDUSTRY_FACTORS].sum()     # 行业因子风险贡献
        risk_contribution['style'] = risk_contribution[riskfactor_ct.STYLE_RISK_FACTORS].sum()      # 风格因子风险贡献

        return risk_contribution


if __name__ == '__main__':
    BarraModel = Barra()
    # BarraModel.calc_factorloading('2014-01-01', '2014-12-31')
    # BarraModel._calc_secu_dailyret('2018-07-02')
    # BarraModel._get_cap_weight('2017-12-29')
    # BarraModel._calc_IndFactorloading('2017-12-29')
    # BarraModel._calc_StyleFactorloading('2017-12-29')
    # print(BarraModel._get_IndFactorloading_matrix('2017-12-29').head())
    # print(BarraModel._get_StyleFactorloading_matrix('2017-12-29').head())
    # print(BarraModel._get_secu_dailyret('2018-01-02').head())

    # BarraModel.estimate_factor_ret(start_date='2018-01-01', end_date='2018-06-30')
    # print(BarraModel._naive_factor_covmat('2018-06-29'))
    # BarraModel.calc_factor_covmat(start_date='2018-06-29', end_date='2018-06-29', calc_mode='cov')
    # BarraModel.calc_spec_varmat(start_date='2018-06-29', end_date='2018-06-30', calc_mode='var')

    # holding_data = load_holding_data('tmp', 'sh50')
    # risk_contribution = BarraModel.risk_contribution(holding_data, '2018-06-29')
    # print(risk_contribution)

    holding_data = CWeightHolding()
    mvpfp_filepath = os.path.join(SETTINGS.FACTOR_DB_PATH, 'AlphaFactor/SmartMoney/mvpfp/SmartMoney_20180629.csv')
    holding_data.from_file(mvpfp_filepath)
    risk_contribution = BarraModel.risk_contribution(holding_data, '2018-06-29')
    print(risk_contribution)