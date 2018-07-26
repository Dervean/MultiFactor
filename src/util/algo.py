#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 算法相关类
# @Filename: algo
# @Date:   : 2018-07-11 20:38
# @Author  : YuJun
# @Email   : yujun_mail@163.com


import numpy as np


class Algo(object):
    """算法类"""

    @classmethod
    def ewma_weight(cls, h, tau):
        """
        计算半衰指数加权平均算法的权重向量
        Parameters:
        --------
        :param h: int
            样本时间长度
        :param tau: int
            半衰期长度
        :return: np.ndarray
        --------
            采用半衰指数平均算法计算的权重向量
        """
        flambda = 0.5**(1.0/tau)
        weight = flambda ** np.arange(h)[::-1]
        weight /= sum(weight)
        return weight


if __name__ == '__main__':
    pass
