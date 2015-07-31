# -*-coding:utf-8-*-
__author__ = 'Max-Zhu'

import numpy as np
import ahocorasick
import nltk
import json
import re
import os
import logging
import collections
import xml.sax


def load_profiles(profile_file):
    """read num and title from original profile"""
    logging.info('loading profiles...' + profile_file)
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = XMLHandler()
    parser.setContentHandler(handler)
    parser.parse(profile_file)
    result = handler.getDict()
    return result['num'], result['title']


class XMLHandler(xml.sax.handler.ContentHandler):
    def __init__(self):
        self.mapping = {'num': [], 'title': []}
        self.content = ""
        self.tag = ""
        self.stopwords = nltk.corpus.stopwords.words('english')

    def startElement(self, tag, attributes):
        self.content = ""
        self.tag = tag

    def characters(self, content):
        if self.tag == 'num':
            self.content = content.strip()[-5:]
        elif self.tag == 'title':
            content = re.sub('\t+|\n+', '', content)
            self.content += content

    def endElement(self, tag):
        if tag in ['num', 'title']:
            self.mapping[tag].append(self.content)

    def getDict(self):
        return self.mapping


def read_profiles():
    """read expanded profiles"""
    exp_profiles = []
    with open('expanded_profiles', 'r') as fr:
        while True:
            line = fr.readline()
            if line == "":
                break
            line = line.replace('\n', '')
            profile = line.split(' ')
            exp_profiles.append(profile)
    return exp_profiles


def dump_result(results, file_names, day_name):
    """save classified result of each profile"""
    if not os.path.exists('./result/' + day_name):
        os.mkdir('./result/' + day_name)
        os.mkdir('./result/' + day_name + '/json')
        os.mkdir('./result/' + day_name + '/text')
    for index, result in enumerate(results):
        fw = open('./result/' + day_name + '/json/profile_' + file_names[index] + '.txt', 'w')
        fw2 = open('./result/' + day_name + '/text/profile_' + file_names[index] + '.txt', 'w')
        fw2.write('similarity\t' + 'tweet text\n')
        fw2.write('='*80 + '\n')
        result_sorted = sorted(result.iteritems(), key=lambda item: item[1], reverse=True)
        for key, value in result_sorted:
            fw.write(key + "\n")
            fw2.write(str(value) + "\t" + json.loads(key)['text'] + "\n")
        fw.close()
        fw2.close()


def extract_text(line):
    """extract text, timestamp and urls from tweet json"""
    tweet_text = ""
    tid_origin = ""
    tid_retweet = ""
    tweet_json = ""
    timestamp = ""
    if line[0] == '{':
        tweet_json = json.loads(line)
        if 'created_at' in tweet_json:
            text = tweet_json['retweeted_status']['text'] if 'retweeted_status' in tweet_json else tweet_json['text']
            tweet_text = tweet_json['user']['screen_name'] + ' ' + re.sub(r'\t+|\n+', ' ', text)
            tid_origin = tweet_json['id_str']
            tid_retweet = tweet_json['retweeted_status']['id_str'] if 'retweeted_status' in tweet_json else tid_origin
            timestamp = tweet_json['timestamp_ms']
    return tweet_text, tid_origin, tid_retweet, timestamp, tweet_json


def load_keywords():
    optional_keywords_list = []
    required_keywords_list = []
    multigram_keywords = []
    with open('profile_keywords') as fr:
        while True:
            line = fr.readline()
            if line == '':
                break

            keywords = json.loads(line, strict=False)['keyword']
            optional_keywords = set(map(lambda w: w.strip(), re.split('\t|\|\|', keywords['0'].lower())))
            required_keywords = parser_express(keywords['1'].lower())
            # required_keywords.append(optional_keywords)
            all_keywords = reduce(lambda x, y: x.union(y), required_keywords).copy()
            all_keywords.update(optional_keywords)

            multigram_keywords.append([word for word in all_keywords if ' ' in word])
            optional_keywords_list.append(optional_keywords)
            required_keywords_list.append(required_keywords)

    return optional_keywords_list, required_keywords_list, multigram_keywords


def parser_express(keywords_expression):
    required_sets = keywords_expression.split('&&')
    optional_sets = []

    for required_set in required_sets:
        optional_set = set([word.strip() for word in required_set.split('||')])
        optional_sets.append(optional_set)

    return optional_sets


def build_inverted_index(profiles_keywords):
    """build inverted index with required keywords of each profiles"""
    inverted_index = collections.defaultdict(list)
    for index, words_list in enumerate(profiles_keywords):
        for words in words_list:
            for word in words:
                inverted_index[word].append(index)

    return inverted_index


def build_actries(profiles_keywords):
    actries = ahocorasick.KeywordTree()
    for profile_keywords in profiles_keywords:
        for keyword in profile_keywords:
            actries.add(keyword)

    actries.make()
    return actries


def initlog(log_name, log_file):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='UTF-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    load_keywords()
    # names, profiles = load_profiles('profiles')
    # exp_profiles = weighted_profiles(profiles)
    # print exp_profiles[:5]
    # inverted_index = build_inverted_index(exp_profiles)
    # print inverted_index
