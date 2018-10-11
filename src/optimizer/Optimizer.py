#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 组合优化器
# @Filename: Optimizer
# @Date:   : 2018-09-14 14:15
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.util.utils import Utils
import src.alphamodel.AlphaModel as AlphaModel
from src.riskmodel.RiskModel import Barra
from src.portfolio.portfolio import CWeightHolding
import src.optimizer.cons as opt_ct
import src.riskmodel.riskfactors.cons as riskfactor_ct
import src.util.cons as util_ct
import src.settings as SETTINGS
import pandas as pd
import numpy as np
import cvxpy as cvx
import os
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def calc_optimized_portfolio(start_date, end_date=None, port_name=None, month_end=True, save=False):
    """
    计算最优化组合权重
    Parameters:
    --------
    :param start_date: datetime-like, str
        开始计算日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param end_date: datetime-like, str
        结束计算日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param port_name: list of str, str
        需优化的组合名称
        默认为None, 优化所有组合
    :param month_end: bool
        是否仅优化月末数据, 默认True
    :param save: bool
        是否保存优化组合权重数据, 默认为False
    :return: pd.DataFrame
        返回组合优化权重数据
    --------
        0.date: 日期
        1.code: 个股代码
        2.weight: 个股权重
    """
    start_date = Utils.to_date(start_date)
    if end_date is None:
        trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
    else:
        end_date = Utils.to_date(end_date)
        trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)

    # 遍历交易日, 计算最优化组合
    for calc_date in trading_days_series:
        if month_end and (not Utils.is_month_end(calc_date)):
            continue
        # 读取alpha模型数据
        df_alphafactor_loading, ser_alphafactor_ret = AlphaModel.get_alphamodel_data(calc_date)
        # 计算个股预期收益向量
        df_alphafactor_loading.set_index('code', inplace=True)
        ser_secu_ret = (df_alphafactor_loading * ser_alphafactor_ret).sum(axis=1)
        df_secu_ret = ser_secu_ret.to_frame(name='ret')
        df_secu_ret.index.name = 'code'
        df_secu_ret.reset_index(inplace=True)

        # 读取风险模型数据
        BarraModel = Barra()
        df_riskfactor_loading, arr_riskfactor_covmat, ser_spec_var = BarraModel.get_riskmodel_data(calc_date)
        df_riskfactor_loading.reset_index(inplace=True)
        df_spec_var = ser_spec_var.to_frame(name='spec_var')
        df_spec_var.index.name = 'code'
        df_spec_var.reset_index(inplace=True)

        # 合并个股预期收益向量、个股风险因子暴露矩阵、个股特质波动率向量数据
        df_data = pd.merge(left=df_secu_ret, right=df_riskfactor_loading, how='inner', on='code')
        df_data = pd.merge(left=df_data, right=df_spec_var, how='inner', on='code')
        # df_data.set_index('code', inplace=True)
        df_secu_ret = df_data[['ret']]                                      # 个股预期收益率矩阵
        df_IndFactor_loading = df_data[riskfactor_ct.INDUSTRY_FACTORS]      # 行业因子载荷矩阵
        df_StyleFactor_loading = df_data[riskfactor_ct.STYLE_RISK_FACTORS]  # 风格因子载荷矩阵
        df_riskfactor_loading = df_data[riskfactor_ct.RISK_FACTORS]         # 风险因子载荷矩阵(含市场因子、行业因子和风格因子)
        ser_spec_var = df_data['spec_var']                                  # 个股特质波动率向量

        # 设置优化目标
        n = len(df_data)                        # 个股数量
        mu = np.array(df_secu_ret)              # 个股预期收益率向量
        F = np.array(df_riskfactor_loading)     # 个股风险因子载荷矩阵
        F_i = np.array(df_IndFactor_loading)    # 个股行业因子载荷矩阵
        F_s = np.array(df_StyleFactor_loading)  # 个股风格因子载荷矩阵
        sigma = arr_riskfactor_covmat           # 风险因子协方差矩阵
        D = np.diag(ser_spec_var)               # 个股特质波动率方差矩阵(对角矩阵)
        w = cvx.Variable((n, 1))                # 个股权重向量
        f = F.T * w

        ret = mu.T * w                          # 组合预期收益率
        risk = cvx.quad_form(f, sigma) + cvx.quad_form(w, D)    # 组合预期波动率

        # 遍历投资组合名称, 计算最优化权重
        if port_name is None:
            port_names = opt_ct.portfolios
        else:
            port_names = [port_name]
        for portfolio_name in port_names:
            logging.info("[%s] Calc %s's optimized portfolio." % (Utils.datetimelike_to_str(calc_date, dash=True), portfolio_name))
            # 组合优化的约束条件
            opt_constraints = eval('opt_ct.'+portfolio_name)
            # 读取基准成份股权重数据
            df_ben_weight = Utils.get_index_weight(opt_constraints['benchmark'], calc_date)
            df_ben_weight.drop(columns='date', inplace=True)
            df_data = pd.merge(left=df_data, right=df_ben_weight, how='left', on='code')
            df_data.fillna(0, inplace=True)
            df_ben_weight = df_data[['weight']]
            wb = np.array(df_ben_weight)

            # 构建组合优化的约束条件
            # 1.风格因子中性约束
            arr_stylefactorloading_lower = []
            arr_stylefactorloading_upper = []
            for stylefactor in riskfactor_ct.STYLE_RISK_FACTORS:
                arr_stylefactorloading_lower += [opt_constraints['riskfactor_const'][stylefactor][0]]
                arr_stylefactorloading_upper += [opt_constraints['riskfactor_const'][stylefactor][1]]
            arr_stylefactorloading_lower = np.array(arr_stylefactorloading_lower).reshape((1, F_s.shape[1]))
            arr_stylefactorloading_upper = np.array(arr_stylefactorloading_upper).reshape((1, F_s.shape[1]))
            style_neutral_constraints = [(w-wb).T*F_s >= arr_stylefactorloading_lower,
                                         (w-wb).T*F_s <= arr_stylefactorloading_upper]
            # 2.行业中性
            ind_neutral_constraints = []
            if opt_constraints['industry_neutral']:
                ind_neutral_constraints += [(w-wb).T*F_i >= -0.01,
                                            (w-wb).T*F_i <= 0.01]

            # 3.权重下限
            weight_lower_constraints = [w >= opt_constraints['weight_bound'][0]]

            # 4.权重上限
            weight_upper_constraints = [w <= opt_constraints['weight_bound'][1]]

            # 5.权重之和
            weight_sum_constraints = [cvx.sum(w) == opt_constraints['weight_sum']]

            # 6.个股数量约束
            n_max_constraints = []
            if opt_constraints['secu_num_cons']:
                eta = cvx.Variable((n, 1), boolean=True)
                n_max_constraints = [w <= eta,
                                     cvx.sum(eta) <= opt_constraints['n_max']]

            constraints = style_neutral_constraints + ind_neutral_constraints + weight_lower_constraints + \
                           weight_upper_constraints + weight_sum_constraints + n_max_constraints

            # 构建优化目标
            opt_obj = cvx.Maximize(ret - opt_constraints['lambda'] * risk)

            # 构建优化问题
            opt_prob = cvx.Problem(opt_obj, constraints)

            # 组合优化计算
            opt_prob.solve(verbose=False)

            datelabel = Utils.datetimelike_to_str(calc_date, dash=False)
            df_holding = pd.DataFrame({'date': [datelabel]*n, 'code': df_data['code'].tolist(), 'weight': w.value.flatten()})
            df_holding.sort_values(by='weight', ascending=False, inplace=True)
            df_holding = df_holding[df_holding['weight'] > util_ct.TINY_ABS_VALUE]
            optimized_holding = CWeightHolding()
            optimized_holding.from_dataframe(df_holding)
            if save:
                holding_path = os.path.join(SETTINGS.FACTOR_DB_PATH, opt_constraints['holding_path'], '%s.csv' % datelabel)
                optimized_holding.save_data(holding_path)


if __name__ == '__main__':
    pass
    calc_optimized_portfolio('2012-12-31', '2018-09-28', port_name=None, month_end=True, save=True)

    # stocks = Utils.get_stock_basics(all=True)
    # for _, stock_info in stocks.iterrows():
    #     ipo_data = Utils.get_ipo_info(stock_info.symbol)
    #     if (ipo_data is None) or (ipo_data['发行价格'][:-1] == '--'):
    #         print('%s,%s' % (stock_info['symbol'], stock_info['name']))
