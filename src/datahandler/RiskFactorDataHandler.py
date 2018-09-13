#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 导出指定时间区间的风险因子载荷数据
# @Filename: RiskFactorDataHandler
# @Date:   : 2018-09-05 18:05
# @Author  : YuJun
# @Email   : yujun_mail@163.com


import os, shutil
import src.settings as SETTINGS
from src.util.utils import Utils


src_riskfactor_path = os.path.join(SETTINGS.FACTOR_DB_PATH, 'RiskFactor')
dst_riskfactor_path = 'F:/share/project/Data/RiskFactor'


def export_riskfactor_loading(start_date, end_date=None):
    """
    把风险模型中的风险因子载荷数据导出
    Parameters:
    --------
    :param start_date: datetime-like, str
        开始日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param end_date: datetime-like, str
        结束日期, e.g: YYYY-MM-DD, YYYYMMDD
        默认为None
    :return:
    """
    start_date = Utils.datetimelike_to_str(start_date, dash=False)
    if end_date is None:
        end_date = start_date
    else:
        end_date = Utils.datetimelike_to_str(end_date, dash=False)

    copy_file(src_riskfactor_path, start_date, end_date)


def copy_file(file_fullpath, start_date, end_date):
    dst_fullpath = os.path.join(dst_riskfactor_path, file_fullpath.replace(src_riskfactor_path+'\\', ''))
    if os.path.isfile(file_fullpath):
        if os.path.splitext(file_fullpath)[1] == '.csv':
            str_date = file_date(file_fullpath)
            if str_date >= start_date and str_date <= end_date:
                shutil.copyfile(file_fullpath, dst_fullpath)
    elif os.path.isdir(file_fullpath):
        if not os.path.exists(dst_fullpath):
            os.mkdir(dst_fullpath)
        for file_name in os.listdir(file_fullpath):
            copy_file(os.path.join(file_fullpath, file_name), start_date, end_date)


def file_date(file_fullname):
    file_basename = os.path.basename(file_fullname)
    str_date = file_basename.split('.')[0].split('_')[1]
    return str_date


if __name__ == '__main__':
    pass
