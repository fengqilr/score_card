# -*- coding: utf-8 -*-
"""
此代码用于：
1. 提供逻辑回归模型的建模方法

数据来源：
1.无

注意事项：
1. 本篇代码中不包含模型上线所需要的代码。上线需要根据实际情况调整。

调用方法：
见projects/self_define_bad_model/main.py里面的调用示例
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm_api
import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import time
import datetime
import re


# 建模数据分割
def split_sample(dataset, dev_rate=0.7, max_rate_diff=0.01, target_var='target', verbose=True):
    """
    按比例随机分割建模数据
    :param dataset: 需要分割的建模数据集
    :param dev_rate: dev和val样本的分割比例
    :param max_rate_diff: 两个数据集target变量之间允许误差
    :param target_var: target变量名
    :param verbose: 输出中间过程
    :return: develop sample, validation sample
    """

    # 计算develop样本大小
    dev_sample_size = int(dataset.shape[0] * dev_rate)

    # 反复测试切割样本
    keep_split = True
    test_cnt = 0
    while keep_split:
        test_cnt += 1

        # 拷贝索引数据
        idx_values = dataset.index.values.copy()
        np.random.shuffle(idx_values)

        # 分割样本
        dev_sample = dataset.ix[idx_values[:dev_sample_size]]
        val_sample = dataset.ix[idx_values[dev_sample_size:]]

        # 分割完成，计算bad rate
        dev_bad = len(dev_sample[dev_sample[target_var] == 1])
        dev_good = len(dev_sample[dev_sample[target_var] == 0])
        dev_bad_rate = 1.0 * dev_bad / (dev_bad + dev_good)

        val_bad = len(val_sample[val_sample[target_var] == 1])
        val_good = len(val_sample[val_sample[target_var] == 0])
        val_bad_rate = 1.0 * val_bad / (val_bad + val_good)

        rate_diff = 1.0 * (dev_bad_rate - val_bad_rate) / val_bad_rate

        if verbose:
            print "in %d split test" % test_cnt
            print "dev: %d, %d, %f" % (dev_bad, dev_good, dev_bad_rate)
            print "val: %d, %d, %f" % (val_bad, val_good, val_bad_rate)
            print "diff: %f" % rate_diff

        if -1.0 * max_rate_diff <= rate_diff <= max_rate_diff:
            return dev_sample, val_sample


# grouping
# 根据用户自定义分组 来统计good/bad 个数
# def var_grouping(dataset, grp_var_name, tgt_var_name, verbose=True):
#     """
#     根据用户自定义分组来统计good/bad个数，以及对应的WOE，IV值
#     :param dataset: 进行分组计算的数据集名称
#     :param grp_var_name: 分组变量名
#     :param tgt_var_name: 目标变量名
#     :param verbose: 输出中间过程
#     :return:
#         group_name: 变量分组
#         count: 组内样本数量
#         total_dist: 组内样本数量/数据集样本数量
#         good: 组内tgt=0的样本数量
#         good_dist: 组内tgt=0的样本数量/tgt=0的样本数量
#         bad: 组内tgt=1的样本数量
#         bad_dist: 组内tgt=1的样本数量/tgt=1的样本数量
#         bad_rate: bad/count
#         woe: ln(good_dist/bad_dist)
#         iv: (good_dist - bad_dist)*woe
#         total_iv: sum(iv)
#     """
#
#     # 计算总体样本量，good/bad总数
#     total_good = len(dataset.ix[dataset[tgt_var_name] == 0])
#     total_bad = len(dataset.ix[dataset[tgt_var_name] == 1])
#     total_count = len(dataset)
#
#     # 记录每一个分组的统计结果
#     grouping_data = []
#     for grp_var, grp_data in dataset.groupby(grp_var_name):
#         d = dict()
#         d['group_name'] = grp_var
#         d['count'] = len(grp_data)
#         d['total_dist'] = 1.0 * d['count'] / total_count
#         d['good'] = len(grp_data.ix[grp_data[tgt_var_name] == 0])
#         d['good_dist'] = 1.0 * d['good'] / total_good
#         d['bad'] = len(grp_data.ix[grp_data[tgt_var_name] == 1])
#         d['bad_dist'] = 1.0 * d['bad'] / total_bad
#         d['bad_rate'] = 1.0 * d['bad'] / d['count']
#         if d['bad'] > 0 and d['good'] > 0:
#             d['woe'] = np.math.log(1.0 * d['good_dist'] / d['bad_dist'])
#             d['iv'] = 1.0 * (d['good_dist'] - d['bad_dist']) * d['woe']
#         grouping_data.append(d)
#
#     # 保存到数据集
#     grouping_cols = ['group_name', 'count', 'total_dist',
#                      'good', 'good_dist', 'bad', 'bad_dist',
#                      'bad_rate', 'woe', 'iv']
#     grouping_df = pd.DataFrame(grouping_data, columns=grouping_cols)
#     total_iv = grouping_df.iv.sum()
#     if verbose:
#         print grp_var_name, 'total_iv', total_iv
#         # print grouping_df
#     return grouping_df

def var_grouping(dataset, grp_var_name, tgt_var_name, verbose=True):
    """
    根据用户自定义分组来统计good/bad个数，以及对应的WOE，IV值
    :param dataset: 进行分组计算的数据集名称
    :param grp_var_name: 分组变量名
    :param tgt_var_name: 目标变量名
    :param verbose: 输出中间过程
    :return:
        group_name: 变量分组
        count: 组内样本数量
        total_dist: 组内样本数量/数据集样本数量
        good: 组内tgt=0的样本数量
        good_dist: 组内tgt=0的样本数量/tgt=0的样本数量
        bad: 组内tgt=1的样本数量
        bad_dist: 组内tgt=1的样本数量/tgt=1的样本数量
        bad_rate: bad/count
        woe: ln(good_dist/bad_dist)
        iv: (good_dist - bad_dist)*woe
        total_iv: sum(iv)
    """

    # dataset = dataset[[grp_var_name, tgt_var_name]].copy()
    dataset = pd.DataFrame(np.c_[dataset[grp_var_name].values, dataset[tgt_var_name].values], columns=[grp_var_name, tgt_var_name])

    # 计算总体样本量，good/bad总数
    total_bad = dataset[tgt_var_name].sum()
    total_count = len(dataset)
    total_good = total_count - total_bad
    # 记录每一个分组的统计结果x

    count = dataset.groupby(dataset[grp_var_name])[tgt_var_name].count()

    total_dist = count/float(total_count)

    # good_count = data_grouped.apply(lambda df: len(df[df[tgt_var_name] == 0]))

    bad_count = dataset.groupby(dataset[grp_var_name])[tgt_var_name].sum()

    bad_dist = bad_count/float(total_bad)
    good_count = count - bad_count
    good_dist = good_count/float(total_good)
    bad_rate = bad_count/count
    is_tgt = bad_dist>0
    odds = (good_dist[is_tgt]/bad_dist[is_tgt]).reindex(bad_dist.index)

    # odds[odds==np.inf] = np.nan

    woe = odds.map(lambda x: np.math.log(x) if x > 0 else np.nan)

    iv = (good_dist - bad_dist) * woe

    grouping_df = pd.concat([count, total_dist, good_count, good_dist, bad_count, bad_dist, bad_rate, woe, iv], axis=1)
    grouping_df.columns = ['count', 'total_dist',  'good_count', 'good_dist', 'bad_count', 'bad_dist', 'bad_rate', 'woe', 'iv']
    grouping_df.reset_index(inplace=True)
    return grouping_df


def var_grouping_v2(dataset, grp_var_name, tgt_var_name, verbose=True):
    """
    根据用户自定义分组来统计good/bad个数，已经对应的WOE，IV值,版本2
    :param dataset: 进行分组计算的数据集名称
    :param grp_var_name: 分组变量名
    :param tgt_var_name: 目标变量名
    :param verbose: 输出中间过程
    :return:
        group_name: 变量分组
        count: 组内样本数量
        total_dist: 组内样本数量/数据集样本数量
        good: 组内tgt=0的样本数量
        good_dist: 组内tgt=0的样本数量/tgt=0的样本数量
        bad: 组内tgt=1的样本数量
        bad_dist: 组内tgt=1的样本数量/tgt=1的样本数量
        bad_rate: bad/count
        woe: ln(good_dist/bad_dist)
        iv: (good_dist - bad_dist)*woe
        total_iv: sum(iv)
    """

    # 计算总体样本量，good/bad总数
    total_good = len(dataset.ix[dataset[tgt_var_name] == 0])
    total_bad = len(dataset.ix[dataset[tgt_var_name] == 1])
    total_count = len(dataset)

    # 记录每一个分组的统计结果
    grouping_data = []
    for grp_var, grp_data in dataset.groupby(grp_var_name):
        d = dict()
        d['group_name'] = grp_var
        d['count'] = len(grp_data)
        d['total_dist'] = 1.0 * d['count'] / total_count
        d['good'] = len(grp_data.ix[grp_data[tgt_var_name] == 0])
        d['good_dist'] = 1.0 * d['good'] / total_good
        d['bad'] = len(grp_data.ix[grp_data[tgt_var_name] == 1])
        d['bad_dist'] = 1.0 * d['bad'] / total_bad
        d['bad_rate'] = 1.0 * d['bad'] / d['count']
        if d['bad'] > 0 and d['good'] > 0:
            d['woe'] = np.math.log(1.0 * d['good_dist'] / d['bad_dist'])
        elif d["good"] == 0:
            d['woe'] = -1
        else:
            d["woe"] = 1
        d['iv'] = 1.0 * (d['good_dist'] - d['bad_dist']) * d['woe']
        grouping_data.append(d)

    # 保存到数据集
    grouping_cols = ['group_name', 'count', 'total_dist',
                     'good', 'good_dist', 'bad', 'bad_dist',
                     'bad_rate', 'woe', 'iv']
    grouping_df = pd.DataFrame(grouping_data, columns=grouping_cols)
    total_iv = grouping_df.iv.sum()
    if verbose:
        print 'total_iv', total_iv
    return grouping_df


# woe值替换
def get_woe_var(dataset, grp_dataset, grp_var_name, woe_var_name=None):
    """
    对数据集进行woe值替换
    :param dataset: 进行分组计算的数据集名称
    :param grp_dataset: 存放分组计算结果的数据集名称
    :param grp_var_name: 分组变量名
    :param woe_var_name: 存放woe值的变量名
    :return: 无
    """
    # dataset:
    # grp_dataset:
    # grp_var_name:
    # woe_var_name:

    if woe_var_name is None:
        woe_var_name = grp_var_name + '_woe'

    woe_dict = {}
    for row_idx, grp_data in grp_dataset.iterrows():
        grp_name = grp_data[grp_var_name]
        woe_value = grp_data['woe']
        woe_dict[grp_name] = woe_value
    dataset[woe_var_name] = dataset[grp_var_name].map(lambda x:
                                                      woe_dict.get(x, 0) if pd.notnull(woe_dict.get(x, 0)) else 0)
    # is_zero = dataset[woe_var_name].map(lambda x: True if x == 0 else False)
    # print grp_var_name, is_zero.sum()


# woe值替换
def get_woe_var_raw(raw_var, grp_dataset):
    """
    对原值进行woe值替换
    :param raw_var: 原值
    :param grp_dataset: 存放分组计算结果的数据集名称
    :return: woe值
    """
    # dataset:
    # grp_dataset:
    # grp_var_name:
    # woe_var_name:

    # print grp_dataset
    return float(grp_dataset.woe[grp_dataset['group_name'] == raw_var])
    # return grp_dataset.woe[grp_dataset['group_name'] == raw_var].ix[0]


# 开始建模
def gen_logistic_model(dataset, var_list, target_var='target', intercept_var='intercept', verbose=True):
    """
    对输入变量进行拟合
    :param dataset: 建模数据集名称
    :param var_list: 变量列表
    :param target_var: 目标变量名称
    :param intercept_var: 截距变量名称
    :param verbose: 输出中间过程
    :return: 逻辑回归模型，模型结果
    """
    # 拷贝建模数据
    ind_var_df = pd.DataFrame(dataset, columns=var_list).astype(float)
    # ind_var_df.astype(float)

    # 添加截距
    if intercept_var:
        ind_var_df[intercept_var] = 1

    # 模型拟合
    logit_model = sm_api.Logit(dataset[target_var].astype(float), ind_var_df)
    logit_res = logit_model.fit()
    if verbose:
        logit_res_summary = logit_res.summary()
        print logit_res_summary
        # for idx, table in enumerate(logit_res_summary.tables):
        #     print idx
        #     with open("%s_result.csv" % idx, "w") as fcsv:
        #         fcsv.write(table.as_csv())
    return logit_model, logit_res


def gen_regularized_logistic_model(dataset, var_list, target_var='target', alpha=0.0001, intercept_var='intercept', verbose=True):
    """
    对输入变量进行拟合
    :param dataset: 建模数据集名称
    :param var_list: 变量列表
    :param target_var: 目标变量名称
    :param intercept_var: 截距变量名称
    :param verbose: 输出中间过程
    :return: 逻辑回归模型，模型结果
    """
    # 拷贝建模数据
    ind_var_df = pd.DataFrame(dataset, columns=var_list)

    # 添加截距
    if intercept_var:
        ind_var_df[intercept_var] = 1

    # 模型拟合
    logit_model = sm_api.Logit(dataset[target_var], ind_var_df)
    logit_res = logit_model.fit_regularized(alpha=alpha)
    if verbose:
        print logit_res.summary()
    return logit_model, logit_res


# 计算VIF值
def gen_vif(dataset, var_list=None, verbose=True):
    """

    :param dataset: 建模数据集名称
    :param var_list: 变量列表
    :param verbose: 输出中间过程
    :return:
    """
    if var_list is None:
        var_list = dataset.columns

    vif_data = []
    for idx, var_name in enumerate(var_list):
        var_vif = variance_inflation_factor(dataset.ix[:, var_list].as_matrix(), idx)
        vif_data.append({'var_name': var_name, 'var_vif': var_vif})
        if verbose:
            print var_name, "\t\t", var_vif
    vif_df = pd.DataFrame(vif_data, columns=['var_name', 'var_vif'])
    return vif_df


# 打分
def gen_log_odds_score(dataset, var_list, scr_model, intercept_var='intercept'):
    """

    :param dataset: 打分数的数据集
    :param var_list: 变量名称列表
    :param scr_model: 存放打分公式的模型
    :return:
    """
    score_data = (dataset[var_list].values * scr_model.params[var_list].values).sum(axis=1) + scr_model.params[intercept_var]
    # for row_idx, row_data in dataset.iterrows():
    #     log_odds = scr_model.params[intercept_var]
    #     for var_name in var_list:
    #         log_odds += 1.0 * scr_model.params[var_name] * row_data[var_name]
    #     score_data.append(log_odds)
    return score_data


# 获得打分函数的参数
def gen_scoring_param(std_offset, std_odds, pdo):
    """
    获得打分函数的参数
    :param std_offset: 基准分
    :param std_odds: 基准分对应的好坏比值
    :param pdo: 好坏比值翻倍对应的分值
    :return: 基准偏差值，倍乘系数
    """
    factor = 1.0 * pdo / np.math.log(2)
    offset = std_offset + factor * np.math.log(std_odds)
    # offset = std_offset - factor * np.math.log(std_odds)
    return offset, factor


# 打分函数
def gen_score(offset, factor, log_odds):
    """
    计算标准评分的分数
    :param offset: 基准偏差值
    :param factor: 倍乘系数
    :param log_odds: 好坏比值的自然对数
    :return: 分数
    """
    return 1.0 * offset + factor * log_odds


# 变量评分表
def gen_attr_score(var_grp_dict, scr_model, factor, offset, num_vars, verbose=True):
    """
    对所有参与打分的变量，针对变量属性分配分数
    :param var_grp_dict: 变量以及对应的分组数据集字典
    :param scr_model: 模型结果，其params属性包含了模型参数
    :param factor: 基准偏差值
    :param offset: 倍乘系数
    :param num_vars: 参与打分的变量个数
    :param verbose: 输出中间过程
    :return:
    """

    neutral_score = 1.0 * scr_model.params['intercept'] / num_vars * factor + 1.0 * offset / num_vars
    if verbose:
        print "neutral_score:", neutral_score

    attr_score_dict = {}
    attr_score_dict.update({"neutral_score": neutral_score})
    for var_name, grp_df in var_grp_dict.items():
        group_name = "%s_bin" % var_name
        if verbose:
            print var_name

        score_data = []
        for row_idx, row_data in grp_df.iterrows():
            woe_to_score = 1.0 * (1.0 * row_data.woe * scr_model.params[var_name + "_woe"] +
                                  1.0 * scr_model.params['intercept'] / num_vars) * factor + 1.0 * offset / num_vars
            rel_score = woe_to_score - neutral_score
            if verbose:
                print row_data[group_name], "\t\t", woe_to_score, "\t", rel_score

            score_data.append({'group_name': row_data[group_name], 'score': woe_to_score, 'rel_score': rel_score})
        attr_score_dict[var_name] = pd.DataFrame(score_data, columns=['group_name', 'score', 'rel_score'])

        if verbose:
            print
    return attr_score_dict


# 生成gains table
def gen_gains(dataset, scr_var_name, tgt_var_name, group_num=10, verbose=True):
    """
    生成gains table
    :param dataset: 打分数的数据集
    :param scr_var_name: 保存分数的变量名
    :param tgt_var_name: 目标变量的变量名
    :param group_num: 分组数量
    :param verbose: 输出中间过程
    :return:
    """
    # 对分数变量和target变量单独拿出来
    sorted_df = dataset.ix[:, [tgt_var_name, scr_var_name]]
    sorted_df.sort(scr_var_name, inplace=True, ascending=True)

    # 对分数变量按从低到高的顺序分组
    sorted_df['grp_idx'] = pd.qcut(sorted_df[scr_var_name], group_num, labels=["%02d" % i for i in range(group_num)])

    # 计算总体样本量
    total_good = len(dataset.ix[dataset[tgt_var_name] == 0])
    total_bad = len(dataset.ix[dataset[tgt_var_name] == 1])
    total_count = len(dataset)

    # 用于计算cum_*的临时变量
    last_good = 0
    last_bad = 0
    last_good_dist = 0
    last_bad_dist = 0

    # 记录gains table的结果
    gains_data = []
    for grp_idx, grp_data in sorted_df.groupby('grp_idx', sort=False):
        d = dict()

        # 计算组内分数
        d['grp_idx'] = grp_idx
        d['min_scr'] = grp_data[scr_var_name].min()
        d['max_scr'] = grp_data[scr_var_name].max()
        d['avg_scr'] = grp_data[scr_var_name].mean()

        # 计算组内样本数
        d['count'] = len(grp_data)
        d['total_dist'] = 1.0 * d['count'] / total_count

        # 计算good分布
        d['good'] = len(grp_data.ix[grp_data[tgt_var_name] == 0])
        d['good_dist'] = 1.0 * d['good'] / total_good
        d['cum_good'] = last_good + d['good']
        last_good = d['cum_good']

        # 计算累计good分布
        d['cum_good_dist'] = last_good_dist + d['good_dist']
        last_good_dist = d['cum_good_dist']

        # 计算bad分布
        d['bad'] = len(grp_data.ix[grp_data[tgt_var_name] == 1])
        d['bad_dist'] = 1.0 * d['bad'] / total_bad
        d['cum_bad'] = last_bad + d['bad']
        last_bad= d['cum_bad']

        # 计算累计bad分布
        d['cum_bad_dist'] = last_bad_dist + d['bad_dist']
        last_bad_dist = d['cum_bad_dist']

        d['bad_rate'] = 1.0 * d['bad'] / d['count']
        d['cum_bad_rate'] = 1.0 * d['cum_bad'] / (d['cum_good'] + d['cum_bad'])

        d['KS'] = "%6.4f" % (d['cum_bad_dist'] - d['cum_good_dist'])

        gains_data.append(d)

    gains_df = pd.DataFrame(gains_data, columns=['grp_idx', 'min_scr', 'max_scr', 'avg_scr',
                                                 'count', 'total_dist',
                                                 'good', 'good_dist', 'cum_good_dist',
                                                 'bad', 'bad_dist', 'cum_bad_dist', 'bad_rate', 'KS'])
    return gains_df


# 绘制组分布和woe曲线
def draw_by_var_chart(grp_df, var_bin_name, y_range_low=-0.6, y_range_up=0.6, y2_range_low=-1, y2_range_up=1):
    """
    绘制组分布和woe曲线
    :param grp_df: 分组结果数据集
    :param y_range_low: y轴下限
    :param y_range_up: y轴上限
    :return:
    """
    # 创建新图
    # font = FontProperties(fname=r"c:\Windows\Fonts\simsun.ttc", size=14)
    grp_df["group_name_unicode"] = grp_df[var_bin_name].map(unicode)
    pattern = re.compile(u"\[(.+?), ")
    # grp_df['index_num'] = grp_df[var_bin_name].map(lambda x: float(pattern.findall(x)[0]) if x != u'无数据' else -1)
    # print grp_df[var_bin_name]
    # grp_df['index_num'] = grp_df[var_bin_name].map(lambda x: float(x.split("_")[0]) if x not in [u'无数据', 'no data'] else -1)
    # grp_df = grp_df.sort("index_num").reset_index(drop=True).drop('index_num', axis=1)
    grp_df = grp_df.set_index(keys="group_name_unicode", drop=True)

    fig, ax = plt.subplots(1, 1)

    # 设定左边y轴
    ax.yaxis.grid()
    ax.set_ylabel('group dist')
    ax.set_ylim((y_range_low, y_range_up))

    # 设定右边y轴
    ax2 = ax.twinx()
    ax2.set_ylabel('woe')
    ax2.set_ylim((y2_range_low, y2_range_up))

    # 画出基准线
    plt.axhline(0, color='k')
    # 画出组分布
    # ax.bar(grp_df.total_dist, style='b', alpha=0.9, label="group dist")
    grp_df.total_dist.plot(kind='bar', style='b', alpha=0.9, ax=ax, label="group dist")
    handles, labels = ax.get_legend_handles_labels()
    # 画出woe曲线
    ax2.plot(grp_df['woe'].values, linestyle='-', marker='o', color='g', label="woe")
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2)
    ax2.legend(handles2, labels2, loc=9)
    # 渲染
    # plt.legend()
    plt.show()


# 绘制组分布和woe曲线
def draw_score_chart(dataset, score_var, target_var='target', bins=20):
    # 创建新图
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # 分数分布
    ax[0].yaxis.grid()
    dataset.ix[:, score_var].hist(bins=bins, alpha=0.9, ax=ax[0])

    # 设定下方good和bad的分布图
    dataset.ix[dataset[target_var] == 0, score_var].hist(bins=bins, normed=True, color='g', ax=ax[1], histtype='step')
    dataset.ix[dataset[target_var] == 1, score_var].hist(bins=bins, normed=True, color='r', ax=ax[1], histtype='step')

    plt.show()


def draw_gains_chart(gains_df):
    # 创建新图
    fig, ax = plt.subplots(1, 1)

    # 加上0点
    plot_data = [{'cum_good_dist': 0, 'cum_bad_dist': 0}]
    for row_idx, row_data in gains_df.iterrows():
        plot_data.append({'cum_good_dist': row_data['cum_good_dist'], 'cum_bad_dist': row_data['cum_bad_dist']})

    # 作图
    plot_data_df = pd.DataFrame(plot_data)
    plot_data_df.plot(ax=ax)
    plt.show()


def draw_gains_compare_chart(dev_gain_df, val_gain_df):
    # 创建新图
    fig, ax = plt.subplots(1, 1)

    # 加上0点
    dev_data = [{'dev_cum_good_dist': 0, 'dev_cum_bad_dist': 0}]
    val_data = [{'val_cum_good_dist': 0, 'val_cum_bad_dist': 0}]
    for row_idx, row_data in dev_gain_df.iterrows():
        dev_data.append({'dev_cum_good_dist': row_data['cum_good_dist'], 'dev_cum_bad_dist': row_data['cum_bad_dist']})
    for row_idx, row_data in val_gain_df.iterrows():
        val_data.append({'val_cum_good_dist': row_data['cum_good_dist'], 'val_cum_bad_dist': row_data['cum_bad_dist']})
    # 作图
    dev_data_df = pd.DataFrame(dev_data)
    val_data_df = pd.DataFrame(val_data)
    dev_data_df.plot(ax=ax)
    val_data_df.plot(ax=ax)
    plt.show()


def gen_roc_curve(tgt, prob):
    fpr, tpr, thresholds = metrics.roc_curve(tgt, prob)
    area = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label="area = %0.2f" % area)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# 自动分组
def auto_grouping(dataset, group_var, target_var='target', group_num=5):
    # 取得一个临时数据集
    test_data = dataset.ix[:, [group_var, target_var]].copy()
    test_data.sort(group_var, inplace=True)

    # 记录分组变量的长度
    group_var_len = len(dataset[group_var])

    # 加上排序列
    test_data['rank'] = range(group_var_len)
    print test_data

    cut_points = sorted([0, 600, 200, 300, 400, group_var_len])
    print cut_points

    def grp_mapping(x):
        res = 0
        for idx, point_value in enumerate(cut_points[:-1]):
            if cut_points[idx] <= x < cut_points[idx+1]:
                res = cut_points[idx]
                break
        return res

# 绘制混淆矩阵
def plot_confusion_matrix(cm, class_name, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel(u'True class')
    plt.xlabel(u'Predict class')
    plt.xticks(range(len(class_name)), class_name)
    plt.yticks(range(len(class_name)), class_name)

# 计算混淆矩阵并选择是否绘图
def cal_confusion_matrix(y_true, y_predict, class_name=None, is_plot=False):
    cm = confusion_matrix(y_true, y_predict)
    print 'Confusion matrix done'
    if is_plot:
        if class_name is None:
            class_name = np.unique(np.append(y_true, y_predict))
        plot_confusion_matrix(cm, class_name)
    return cm

# 绘制grid search scores随参数的分布图，主要针对rbf核的svm，
# 可直观的看出交叉验证的分数与随参数(C, gamma)变化的情况,
# 便于确定如何进一步减小grid search的参数范围
def plot_grid_search_cv_2d(grid_search_cv_grid_scores_, x_param, y_param, other_param_value={},
                           interpolation='nearest', cmap=plt.cm.Blues):
    mean_score = []
    res = DataFrame()
    index = 0
    for v_grid_search_cv_grid_scores_ in grid_search_cv_grid_scores_:
        res = res.append(DataFrame(v_grid_search_cv_grid_scores_[0], index=[index]))
        index += 1
        mean_score.append(v_grid_search_cv_grid_scores_[1])
    res['mean_score'] = mean_score
    x = np.sort(res[x_param].unique())
    y = np.sort(res[y_param].unique())
    xv, yv = np.meshgrid(x, y)
    z = np.zeros(xv.shape)
    other_param_condition_list = np.ones(index, dtype=np.bool)
    for v_other_param_value in other_param_value.iteritems():
        other_param_condition_list.append(res[v_other_param_value[0]] == v_other_param_value[1])
    reduce(np.logical_and, other_param_condition_list)
    for i_x in range(len(x)):
        for i_y in range(len(y)):
            z[i_y, i_x] = res.ix[reduce(np.logical_and,
                                        [res[x_param] == xv[i_y, i_x],
                                         res[y_param] == yv[i_y, i_x],
                                         other_param_condition_list]),
                                 'mean_score'].values[0]
    plt.imshow(z, interpolation=interpolation, cmap=cmap)

    plt.xticks(range(len(x)), x, rotation=45)
    plt.yticks(range(len(y)), y, rotation=45)
    plt.colorbar()

# 观测模型的方差和偏差(训练分数和测试分数随数据集大小的变化)
def bias_variance(dataset, label, model, scoring=accuracy_score, size_variation=10, test_size_ratio=0.33, fit_params=None, is_plot=True):
    # dataset: 数据集的x， Type=ndarray, shape=(m, n)
    # label: 数据集的y， Type=ndarray, shape=(m, )
    # model: 训练模型, 需要有sklearn中的fit()和predict()方法
    # scoring: 评分函数， 默认为准确度, accuracy_score()
    # size_variation: 指定数据集如何变化
    #                   1. N: Type=int, 将数据集采样N次，每次的大小为: (1/N, 2/N, ..., N/N)*size_dataset
    #                   2. [n_1, n_2, ..., n_N]: Type=int,  将数据集采样len(size_variation)=N次， 每次的大小为: n_1, n_2, ..., n_N
    #                   3. [f_1, f_2, ..., f_N]; Type=float, 将数据集采样len(size_variation)=N次， 每次的大小为：(f_1, f_2, ..., f_N)*size_dataset
    # test_size_ratio: Type=float, 对于每一个dataset子集进行评估时测试集占比大小
    # fit_params: 执行fit时所需的参数，默认为None
    if not(hasattr(model, 'fit') and hasattr(model, 'predict')):
        print 'The model is wrong, there are no "fit" method or "predict" method'
        return -1
    if np.ndim(dataset) != 2:
        print 'The dimension of dataset is wrong!'
        return -1
    if np.ndim(label) != 1:
        print 'The dimension of label is wrong!'
        return -1
    if np.shape(dataset)[0] != np.shape(label)[0]:
        print 'The length of samples in dataset is not match the length of label!'
        return -1
    n_dataset = dataset.shape[0]
    size_list = []
    if type(size_variation) == int:
        for v_size in range(size_variation):
            size_list.append(float(v_size+1)/size_variation)
        size_list[-1] -= 0.000001
    else:
        size_list = size_variation.copy()
    fit_params = fit_params if fit_params is not None else {}
    train_score = []
    test_score = []
    start_time = time.time()
    for i_size, v_size in enumerate(size_list):
        _, dataset_estimation, _, label_estimation = train_test_split(dataset, label, test_size=v_size)
        dataset_train, dataset_test, label_train, label_test = train_test_split(dataset_estimation, label_estimation, test_size=test_size_ratio)
        model.fit(dataset_train, label_train, **fit_params)
        train_score.append(scoring(label_train, model.predict(dataset_train)))
        test_score.append(scoring(label_test, model.predict(dataset_test)))
        print 'Bias and variance calculating..., current calc: {0:f}%% of dataset, total progress: {1:f}%%, elapse: {2:f} minutes, now: {3:s}'.\
            format(v_size*100, 100.*v_size/len(size_list)*100, (time.time()-start_time)/60, datetime.datetime.now())
    print 'Bias and variance done, elapse: {0:f} minutes, now: {1:s}'.format((time.time()-start_time)/60, datetime.datetime.now())
    train_score = np.asarray(train_score)
    test_score = np.asarray(test_score)
    if is_plot:
        plt.plot(size_list, np.concatenate((train_score, test_score)).reshape(len(size_list), 2), '-o')
        if type(size_list) == float:
            plt.xlabel('Percent of dataset')
        else:
            plt.xlabel('N samples of dataset')
        plt.ylabel(scoring.__name__)
        plt.title('Bias and variance')
        plt.grid()
    return train_score, test_score

    # 第一次取得平均分布的中心点作为开始点

def find_best_split_point_based_on_f1score(y_true, y_pred_prob):
    y_df = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})
    y_df.sort(['y_pred_prob'], ascending=True, inplace=True)
    split_point_list = np.percentile(list(y_df.y_pred_prob), list(np.arange(0.01, 100, 0.01)))
    f1_score_list = []
    for split_point in split_point_list:
        y_df['y_pred'] = y_df['y_pred_prob'].map(lambda x: 1 if x > split_point else 0)
        f1_score = metrics.f1_score(y_df['y_true'], y_df['y_pred'])
        f1_score_list.append(f1_score)
    split_point_f1_score_df = pd.DataFrame({'split_point': split_point_list, 'f1_score': f1_score_list})
    best_split_point = split_point_f1_score_df['split_point'][split_point_f1_score_df['f1_score'].idxmax()]
    return best_split_point

if __name__ == "__main__":
    data = pd.read_pickle(u'D:/juxinli/experience_modeling/target_def/result_data.pkl')
    data['target'] = data.loan_call_cnt.map(lambda x: 1 if x > 10 else 0)

    auto_grouping(data, group_var='phone_cnt')