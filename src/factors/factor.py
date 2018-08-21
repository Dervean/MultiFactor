#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Filename: factor
# @Date:   : 2017-11-01 21:16
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.util.utils import Utils
import src.factors.cons as factor_ct
import src.riskmodel.riskfactors.cons as risk_ct
import pandas as pd
from pandas import Series
import numpy as np
import os
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class Factor(object):
    """因子基类"""
    _db_file = ''       # 因子对应的在因子数据库的数据文件名(包含数据)
    # def __init__(self):
    #     self._db_file = ''  # 因子对应的在因子数据库的数据文件名(包含数据)

    @classmethod
    def get_db_file(cls):
        return cls._db_file

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
            只计算月末时点的因子载荷，该参数只在end_date不为None时有效，并且不论end_date是否为None，都会计算第一天的因子载荷
        :param save: 是否保存至因子数据库，默认为False
        :return: 因子载荷，DataFrame
        --------
            因子载荷,DataFrame
            0: ID, 证券ID，为索引
            1: factorvalue, 因子载荷
        """
        pass

    @classmethod
    def calc_secu_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股的因子载荷
        :param code: 个股代码
        :param calc_date: 计算日期
        :return:
        """
        return cls._calc_factor_loading(code, calc_date)

    @classmethod
    def _calc_factor_loading(cls, code, calc_date):
        """
        计算指定日期、指定个股的因子载荷
        :param code: 个股代码
        :param calc_date: 计算日期
        :return:
        """
        pass

    @classmethod
    def _calc_factor_loading_proc(cls, code, calc_date, q):
        """
        用于并行计算因子载荷
        Parameters
        --------
        :param code: str
            个股代码，如果600000
        :param calc_date: datetime like or str
            计算日期，格式YYYY-MM-DD
        :param q: 队列，用于进程间通信
        :return: 添加因子载荷至队列q中
        """
        pass

    @classmethod
    def _calc_synthetic_factor_loading(cls, start_date, end_date=None, month_end=True, save=False, **kwargs):
        """
        计算指定日期的样本个股的合成因子的载荷，并保存至因子数据库
        Parameters
        --------
        :param start_date: datetime-like, str
            开始日期
        :param end_date: datetime-like, str，默认None
            结束日期，如果为None，则只计算start_date日期的因子载荷
        :param month_end: bool，默认True
            只计算月末时点的因子载荷，该参数只在end_date不为None时有效，并且不论end_date是否为None，都会计算第一天的因子载荷
        :param save: 是否保存至因子数据库，默认为False
        :param kwargs:
            'multi_proc': bool, True=采用多进程, False=采用单进程, 默认为False
            'com_factors': list, 成分因子的类实例list
        :return: 因子载荷，DataFrame
        --------
            因子载荷,DataFrame
            0: ID, 证券ID，为索引
            1: factorvalue, 因子载荷
        """
        # 取得交易日序列
        start_date = Utils.to_date(start_date)
        if end_date is not None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
        # 遍历交易日序列, 计算合成因子下各个成分因子的因子载荷
        if 'multi_proc' not in kwargs:
            kwargs['multi_proc'] = False
        for calc_date in trading_days_series:
            if month_end and (not Utils.is_month_end(calc_date)):
                continue
            logging.info('[{}] Calc {} factor loading.'.format(Utils.datetimelike_to_str(calc_date), cls.__name__))
            # 计算各成分因子的因子载荷
            # for com_factor in eval('risk_ct.' + cls.__name__.upper() + '_CT')['component']:
            #     factor = eval(com_factor + '()')
            #     factor.calc_factor_loading(start_date=calc_date, end_date=None, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'])
            for com_factor in kwargs['com_factors']:
                com_factor.calc_factor_loading(start_date=calc_date, end_date=None, month_end=month_end, save=save, multi_proc=kwargs['multi_proc'])
            # 计算合成因子
            synthetic_factor = pd.DataFrame()
            df_industry_classify = Utils.get_industry_classify(calc_date)    # 个股行业分类数据
            for com_factor in eval('risk_ct.' + cls.__name__.upper() + '_CT')['component']:
                factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, eval('risk_ct.' + com_factor + '_CT')['db_file'])
                factor_loading = Utils.read_factor_loading(factor_path, Utils.datetimelike_to_str(calc_date, dash=False))
                factor_loading.drop(columns='date', inplace=True)
                factor_loading.rename(columns={'factorvalue': com_factor}, inplace=True)
                # 添加行业分类数据
                factor_loading = pd.merge(left=factor_loading, right=df_industry_classify[['id', 'ind_code']], how='inner', on='id')
                # 取得含缺失值的因子载荷数据
                missingdata_factor = factor_loading[factor_loading[com_factor].isna()]
                # 删除factor_loading中的缺失值
                factor_loading.dropna(axis='index', how='any', inplace=True)
                # 对factor_loading去极值、标准化
                factor_loading = Utils.normalize_data(factor_loading, id='id', columns=com_factor, treat_outlier=True, weight='cap', calc_date=calc_date)
                # 把missingdata_factor中的缺失值替换为行业均值
                ind_codes = set(missingdata_factor['ind_code'])
                ind_mean_factor = {}
                for ind_code in ind_codes:
                    ind_mean_factor[ind_code] = factor_loading[factor_loading['ind_code'] == ind_code][com_factor].mean()
                for idx, missingdata in missingdata_factor.iterrows():
                    missingdata_factor.loc[idx, com_factor] = ind_mean_factor[missingdata['ind_code']]
                # 把missingdata_factor和factor_loading合并
                factor_loading = pd.concat([factor_loading, missingdata_factor])
                # 删除ind_code列
                factor_loading.drop(columns='ind_code', inplace=True)
                # merge成分因子
                if synthetic_factor.empty:
                    synthetic_factor = factor_loading
                else:
                    synthetic_factor = pd.merge(left=synthetic_factor, right=factor_loading, how='inner', on='id')

            # 合成因子
            synthetic_factor.set_index('id', inplace=True)
            weight = pd.Series(eval('risk_ct.' + cls.__name__.upper() + '_CT')['weight'])
            synthetic_factor = (synthetic_factor * weight).sum(axis=1)
            synthetic_factor.name = 'factorvalue'
            synthetic_factor.index.name = 'id'
            synthetic_factor = pd.DataFrame(synthetic_factor)
            synthetic_factor.reset_index(inplace=True)
            synthetic_factor['date'] = Utils.get_trading_days(start=calc_date, ndays=2)[1]
            # 保存synthetic_factor因子载荷
            if save:
                Utils.factor_loading_persistent(cls._db_file, Utils.datetimelike_to_str(calc_date, dash=False), synthetic_factor.to_dict('list'), ['date', 'id', 'factorvalue'])

    @classmethod
    def _save_factor_loading(cls, db_file, str_key, dict_factorloading, factor_name=None, factor_type=None, columns=None):
        """
        保存因子载荷数据
        Parameters:
        --------
        :param db_file: str
            因子载荷数据文件路径(绝对路径)
        :param str_key: str
            持久化因子载荷时用到的key，一般为日期，格式YYYYMMDD
        :param dict_factorloading: dict, pd.DataFrame
            因子载荷数据
        :param factor_name: str, 默认为None
            因子名称
        :param factor_type: str, 默认None
            因子类型, e.g: 'raw', 'standardized', 'orthogonalized'
        :param columns: sequence, 默认=None
            输出的列，并按指定顺序输出
        :return:
        """
        if factor_type is not None:
            db_file = os.path.join(db_file, factor_type, factor_name)
        Utils.factor_loading_persistent(db_file, str_key, dict_factorloading, columns)

    @classmethod
    def _get_factor_loading(cls, db_file, str_key, factor_name=None, factor_type=None, **kwargs):
        """
        读取因子载荷数据
        Parameters:
        --------
        :param db_file: str
            因子载荷数据文件路径(绝对路径)
        :param str_key: str
            键值, 一般为日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param factor_name: str, 默认为None
            因子名称
        :param factor_type: str, 默认为None
            因子类型, e.g: 'raw', 'standardized', 'orthogonalized'
        :param kwargs:
            kwargs['code']: str, 默认为None; 个股代码, e.g: SH600000, 600000
            kwargs['nan_value']: object, 默认为None; 如果不为None, 那么缺失值用nan_value替换
            kwargs['drop_na']: bool, 默认False; 是否删除含有NaN值的行
        :return: pd.DataFrame or pd.Series, 因子载荷
        --------
            pd.DataFrame(code==None) or pd.Series(code!=None)
            0. date
            1. id
            2. factorvalue
        """
        if factor_type is not None:
            db_file = os.path.join(db_file, factor_type, factor_name)
        if 'code' not in kwargs:
            kwargs['code'] = None
        if 'na_value' not in kwargs:
            kwargs['na_value'] = None
        if 'drop_na' not in kwargs:
            kwargs['drop_na'] = False
        return Utils.read_factor_loading(db_file, str_key, kwargs['code'], kwargs['na_value'], kwargs['drop_na'])

    @classmethod
    def get_dependent_factors(cls, date):
        """
        计算用于因子提纯的相关性因子值，包换行业、规模、价值、成长、短期动量、长期动量
        Parameters:
        --------
        :param date: datetime-like or str
            日期
        :return: pd.DataFrame
            index为个股代码, columns=[28个申万一级行业,规模(scale),价值(value),成长(growth),短期动量(short_momentum),长期动量(long_momentum)]
        """
        str_date = Utils.to_date(date).strftime('%Y%m%d')
        # 1. 行业因子
        # 1.1. 读取行业分类信息
        df_industry_calssify = Utils.get_industry_classify(str_date)
        df_industry_calssify = df_industry_calssify.set_index('id')
        # 1.2. 构建行业分裂哑变量
        df_industry_dummies = pd.get_dummies(df_industry_calssify['ind_code'])
        # 2. 规模因子
        # 2.1. 读取规模因子
        scale_factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.SCALE_CT.db_file)
        df_scale_raw = Utils.read_factor_loading(scale_factor_path, str_date, nan_value=0)
        # 2.2. 规模因子去极值、标准化
        scale_cleaned_arr = Utils.clean_extreme_value(np.array(df_scale_raw[['LnLiquidMktCap', 'LnTotalMktCap']]))
        scale_normalized_arr = Utils.normalize_data(scale_cleaned_arr)
        # 2.3. 规模因子降维
        scale_factor_arr = np.mean(scale_normalized_arr, axis=1)
        scale_factor = Series(scale_factor_arr, index=df_scale_raw['id'])
        # 3. 价值因子
        # 3.1. 读取价值因子
        value_factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.VALUE_CT.db_file)
        df_value_raw = Utils.read_factor_loading(value_factor_path, str_date, nan_value=0)
        # 3.2. 价值因子去极值、标准化
        value_cleaned_arr = Utils.clean_extreme_value(np.array(df_value_raw[['ep_ttm', 'bp_lr', 'ocf_ttm']]))
        value_normalized_arr = Utils.normalize_data(value_cleaned_arr)
        # 3.3. 价值因子降维
        value_factor_arr = np.mean(value_normalized_arr, axis=1)
        value_factor = Series(value_factor_arr, index=df_value_raw['id'])
        # 4. 成长因子
        # 4.1. 读取成长因子
        growth_factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.GROWTH_CT.db_file)
        df_growth_raw = Utils.read_factor_loading(growth_factor_path, str_date, nan_value=0)
        # 4.2. 成长因子去极值、标准化
        growth_cleaned_arr = Utils.clean_extreme_value(np.array(df_growth_raw[['npg_ttm', 'opg_ttm']]))
        growth_normalized_arr = Utils.normalize_data(growth_cleaned_arr)
        # 4.3. 成长因子降维
        growth_factor_arr = np.mean(growth_normalized_arr, axis=1)
        growth_factor = Series(growth_factor_arr, index=df_growth_raw['id'])
        # 5. 动量因子
        # 5.1. 读取动量因子
        mom_factor_path = os.path.join(factor_ct.FACTOR_DB.db_path, factor_ct.MOMENTUM_CT.db_file)
        df_mom_raw = Utils.read_factor_loading(mom_factor_path, str_date, nan_value=0)
        # 5.2. 动量因子去极值、标准化
        short_term_mom_header = ['short_term_' + d for d in factor_ct.MOMENTUM_CT.short_term_days.split('|')]
        short_mom_cleaned_arr = Utils.clean_extreme_value(np.array(df_mom_raw[short_term_mom_header]))
        short_mom_normalized_arr = Utils.normalize_data(short_mom_cleaned_arr)
        long_term_mom_header = ['long_term_' + d for d in factor_ct.MOMENTUM_CT.long_term_days.split('|')]
        long_mom_cleaned_arr = Utils.clean_extreme_value(np.array(df_mom_raw[long_term_mom_header]))
        long_mom_normalized_arr = Utils.normalize_data(long_mom_cleaned_arr)
        # 5.3. 动量因子降维
        short_mom_arr = np.mean(short_mom_normalized_arr, axis=1)
        short_mom = Series(short_mom_arr, index=df_mom_raw['id'])
        long_mom_arr = np.mean(long_mom_normalized_arr, axis=1)
        long_mom = Series(long_mom_arr, index=df_mom_raw['id'])

        # 拼接除行业因子外的因子
        df_style_factor = pd.concat([scale_factor, value_factor, growth_factor, short_mom, long_mom], axis=1, keys=['scale', 'value', 'growth', 'short_mom', 'long_mom'], join='inner')
        # 再拼接行业因子
        df_dependent_factor = pd.concat([df_industry_dummies, df_style_factor], axis=1, join='inner')
        return df_dependent_factor



if __name__ == '__main__':
    # pass
    Factor.get_dependent_factors('2016-02-29')
