# -*- coding:utf-8-*-
__author__ = 'Max'

from pandas import DataFrame
import numpy as np
import os
import utils
import random

profiles_size = 225
day_size = 23


class Pusher:
    def __init__(self):
        self.logger = utils.initlog('Console', 'Colsole.log')
        self.logger.info('initializing pusher...')
        self.push_threshold = self.cal_init_threshold()    # 初始推送阈值，根据历史统计TopK的平均取值
        self.total_num_predicted = self.pred_total_related()   # 预测当天总相关推文数
        self.windows_threshold = [0.618 for _ in range(profiles_size)]    # 初始窗口阈值，黄金分割点
        self.received_count = [0 for _ in range(profiles_size)]  # 当前已接收相关推文数
        self.pushed_count = [0 for _ in range(profiles_size)]   # 当前已推送推文数
        self.adjust_count = [0 for _ in range(profiles_size)]   # 阈值调整次数
        self.left_coeff = np.arange(0.9, 0.96, 0.01)    # 未推满条数的惩罚系数
        self.logger.info('pusher initialized!')

    def push(self, score, pid):   # 判定为相关的推文，进一步决策是否推送
        is_pushed = False
        self.received_count[pid] += 1
        if self.pushed_count[pid] != 10:     # 未达到推送上限
            if round(score, 12) > self.push_threshold[pid]:   # 超过当前设定阈值，则推送
                is_pushed = True
                self.pushed_count[pid] += 1
            # received_ratio = self.received_count[pid] / self.total_num_predicted[pid]

            # if received_ratio > self.windows_threshold[pid]:  # 超过当前该profile窗口阈值，未推满，需调整阈值
                """阈值调整"""
            #     self.push_threshold[pid] *= 0.95  # 策略1：按接收比例调整
            #     self.windows_threshold[pid] *= 1.1   # 窗口阈值提升10%

        return is_pushed

    def pred_total_related(self):
        related_tweet_nums = self.read_history_data('related_tweet_nums')
        # predict_nums = related_tweet_nums[:, -3:].mean(axis=1)  # 最近3天的平均值
        predict_nums = self.exponent_predict(related_tweet_nums)    # 指数平滑预测
        return related_tweet_nums[13] # predict_nums

    def cal_init_threshold(self):
        rank_k_scores = self.read_history_data('rank_k_scores')
        # threshold = rank_k_scores[:, -3:].mean(axis=1)
        threshold = self.exponent_predict(rank_k_scores)
        return rank_k_scores[13] # threshold

    def exponent_predict(self, data_frame):
        alphas = 0.8 * np.ones((profiles_size, ))  # 平滑参数 波动较大序列取（0.6-0.8），较平稳序列取（0.1-0.3）
        cols = len(data_frame.columns)
        rows = len(data_frame.index)
        predicts = DataFrame(np.zeros((rows, cols+1)), index=data_frame.index)
        for row_index in range(rows):
            series = data_frame.ix[row_index]
            predicts.ix[row_index][0] = series[:2].mean()
            for i in range(cols):
                predicts.ix[row_index][i+1] = alphas[row_index] * series[i] + (1-alphas[row_index])*predicts.ix[row_index][i]
        return predicts[cols]

    def save_series(self, real_df, predict_df):
        for index in range(profiles_size):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(real_df.columns, real_df.ix[index], 'k-', predict_df.columns[:-1], predict_df.ix[index][:-1], 'r-.')
            fig.savefig('predict/profile_MB' + str(index+226) + '.png')

    def plot_series(self, real_df, predict_df):
        fig_nums = 9
        rows = cols = 5
        nums_per_fig = 25
        for i in range(fig_nums):
            fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
            for row_index in range(nums_per_fig):
                real_series = real_df.ix[row_index + i * nums_per_fig]
                predict_series = predict_df.ix[row_index + i * nums_per_fig]
                axes[row_index / rows][row_index % cols].\
                    plot(real_df.columns, real_series, 'k-', predict_df.columns[:-1], predict_series[:-1], 'r-.')
        plt.show()

    def save_relate_tweets(self, relate_tweets):
        for index, tweets in relate_tweets:
            tweet_sorted = sorted(tweets.iteritems(), key=lambda item: item[1], reverse=True)
            with open('topic_tweets_' + str(index), 'wa') as fw:
                for tid, score in tweet_sorted:
                    fw.write(tid + '\t' + score + '\n')

    def statistic_all(self, file_path):
        """统计各profile历史N天相关推文数情况及Rank_K的分值"""
        day_files = os.listdir(file_path)
        day_files.sort()
        days = len(day_files)
        all_data_num = np.zeros((profiles_size, days), dtype=np.int32)
        all_data_score = np.zeros((profiles_size, days))

        for index, data_file in enumerate(day_files):
            day_data_num, day_data_score = self.statistic_day(os.path.join(file_path, data_file))
            all_data_num[:, index] = day_data_num
            all_data_score[:, index] = day_data_score

        days = [df[-2:] for df in day_files]
        self.dump_history_data('related_tweet_nums', all_data_num, days)
        self.dump_history_data('rank_k_scores', all_data_score, days)

    def statistic_day(self, data_file):
        """统计各profile1天相关推文数情况"""
        records = [[] for _ in range(profiles_size)]
        print 'processing...', data_file

        with open(data_file, 'r') as fr:
            for line in fr.readlines():
                fields = line.split('\t')
                pid = int(fields[1][2:])-226
                score = float(fields[4])
                if score >= 0.15:
                    records[pid].append(score)
        records = map(lambda r: sorted(r, reverse=True), records)
        day_data_num = np.array(map(len, records), dtype=np.int32)
        day_data_score = np.array(map(lambda r: r[9] if len(r) > 10 else (r[-1] if len(r) > 0 else 0.15), records))

        return day_data_num, day_data_score

    def dump_history_data(self, filename, statistic_data, days):
        with open(filename, 'w') as fw:
            fw.write('\t'.join(days))
            fw.write('\n' + '='*160 + '\n')
            for sd in statistic_data:
                datas = [str(data) for data in sd]
                fw.write('\t'.join(datas) + '\n')

    def read_history_data(self, filename):
        if not os.path.exists(filename):
            self.statistic_all('history_data')
        history_data = np.zeros((profiles_size, day_size))
        profiles = map(lambda x: 'MB' + str(x+226), range(profiles_size))
        with open(filename, 'r') as fr:
            for index, line in enumerate(fr.readlines()[2:]):
                data = line[:-1].split('\t')
                history_data[index, :] = np.array(map(float, data))
        history_data_frame = DataFrame(history_data, index=profiles)
        return history_data_frame


if __name__ == '__main__':
    pusher = Pusher()
    # pusher.push()
