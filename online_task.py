# -*-encoding:utf-8-*-
__author__ = 'Max'

import threading
import time
import datetime
import sys
import json
import sched

from simhash import Simhash
from py4j.java_gateway import JavaGateway
from classify import Classifier
from push import Pusher
import utils

reload(sys)
sys.setdefaultencoding('utf-8')


class Controller(threading.Thread):
    def __init__(self, thread_name, event):
        super(Controller, self).__init__()
        self.name = thread_name
        self.threadEvent = event
        self.logger_info = utils.initlog('Console', 'Console.log')
        self.schedule = sched.scheduler(time.time, time.sleep)

        self.profiles_name, profiles = utils.load_profiles('profiles')
        self.related_tweets = [[] for _ in range(len(profiles))]
        self.pushed_tweets = [[] for _ in range(len(profiles))]
        self.pushed_tweets_ids = set([])
        self.related_tweets_hash = set([])

        self.classifier = Classifier()
        self.ranker = self.load_ranker()
        self.pusher = Pusher()

    def load_ranker(self):
        self.logger_info.info('loading ranker...')
        gateway = JavaGateway()
        ranker = gateway.entry_point
        self.logger_info.info('ranker loaded!')
        return ranker

    def run(self):
        self.logger_info.info('%s is starting...' % self.name)
        self.threadEvent.wait()
        self.logger_info.info('%s is running...' % self.name)
        self.schedule.enter(0, 0, self.dump_schedule, ())
        self.schedule.run()
        self.process()

    def process(self):
        start_day = time.gmtime(time.time()).tm_mday
        for line in sys.stdin:
            tweet_text, tid_origin, tid_retweet, timestamp, tweet_json = utils.extract_text(line)
            simhash_value = Simhash(tweet_text).value
            if simhash_value in self.related_tweets_hash or tid_origin in self.pushed_tweets_ids or tid_retweet in self.pushed_tweets_ids:
                continue

            topic_id, similarity = self.classifier.classify(tweet_text)
            if topic_id == '':
                continue

            tweet_json['similarity'] = similarity
            evaluate_score = self.ranker.predict(json.dumps(tweet_json))
            total_score = similarity * evaluate_score
            if total_score < 0.15:
                continue

            is_pushed = self.pusher.push(evaluate_score, topic_id)
            if is_pushed:
                delivery_time = time.time()
                self.pushed_tweets[topic_id].append([tid_origin, str(delivery_time)[:10], similarity, total_score, tweet_text])
                self.pushed_tweets_ids.add(tid_retweet)

            struct_time = time.gmtime(float(timestamp[:-3]))
            utc_time = time.strftime('%Y%m%d', struct_time)
            self.related_tweets[topic_id].append([utc_time, tid_origin, total_score, tweet_text])
            self.related_tweets_hash.add(simhash_value)

            if struct_time.tm_mday != start_day:
                self.dump_result(start_day)
                start_day = struct_time.tm_mday

    def dump_result(self, file_name):
        self.logger_info.info('saving result...')
        with open('submit/task-b/'+file_name, 'w') as fw:
            for index, records in enumerate(self.related_tweets):
                pid = str(index+226)
                sorted_records = sorted(records, key=lambda item: -item[2])
                for rank, record in enumerate(sorted_records):
                    if rank >= 100:
                        break
                    fw.write('%s\tMB%s\tQ0\t%s\t%d\t%f\t%s\n' % (record[0], pid, record[1], rank+1, record[2], 'CSSNA'))
        with open('submit/task-a/'+file_name, 'w') as fw:
            with open('submit/task-a_extr/'+file_name, 'w') as fw_extr:
                for index, records in enumerate(self.pushed_tweets):
                    pid = str(index+226)
                    for record in records:
                        fw.write('MB%s\t%s\t%s\tCSSNA\n' % (pid, record[0], record[1]))
                        fw_extr.write('MB%s\t%s\t%s\tCSSNA\t%s\t%s\t%s\n' % (pid, record[0], record[1], record[2], record[3], record[4]))

        self.related_tweets = [[] for _ in range(225)]
        self.pushed_tweets = [[] for _ in range(225)]


def test():
    signal = threading.Event()
    controller = Controller("online-task", signal)
    controller.start()
    is_ready = detect_tweet_stream(2015, 7, 20, 8, 0, 0, 0)
    if is_ready:
        signal.set()


def detect_tweet_stream(self, year, month, d, h, m, s, ms):
    start = datetime.datetime(year, month, d, h, m, s, ms)
    delta = (start - datetime.datetime.now()).seconds
    self.logger_info.info('waiting secondes: ' + str(delta))
    time.sleep(delta)
    self.logger_info.info('tweet stream is ready')
    is_ready = True
    return is_ready


if __name__ == '__main__':
    '''
    parser = optparse.OptionParser()
    parser.add_option("-m", dest="model", help="word2vec model")
    parser.add_option("-p", dest="profiles", help="user profiles")

    (options, args) = parser.parse_args()
    if not options.model: parser.error("need word2vec models (-m)")
    '''
    test()
