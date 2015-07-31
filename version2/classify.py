# -*-coding:utf-8-*-

__author__ = 'Max'

from math import sqrt
import re
import logging
import sys
import nltk
import gensim
import utils
from common import ldig

reload(sys)
sys.setdefaultencoding('utf-8')


class Classifier:
    def __init__(self):
        self.logger = utils.initlog('Console', 'Colsole.log')
        self.logger.info('initializing classifier...')
        self.count = 0
        self.w2v_model = self.load_model()
        self.profile_keywords, self.required_keywords, multigram_keywords = utils.load_keywords()
        self.inverted_index = utils.build_inverted_index(self.required_keywords)
        self.ac_tries = utils.build_actries(multigram_keywords)

        detector = ldig.LangDetector('./common/model.latin')
        self.param, self.labels, self.trie = detector.load_params()

        self.stopwords = nltk.corpus.stopwords.words('english')
        self.url_pattern = re.compile(r'(https?:/*)[^ ]+|#|@')
        self.tokenizer_pattern = r'''([a-z]\.)+[a-z]?|\w+(-\w+)*'''
        self.tags = ['JJ', 'NN', 'VB']
        self.logger.info('classifier initialized!')

    def load_model(self):
        self.logger.info('loading word2vec model ...')
        w2v_model = gensim.models.Word2Vec.load_word2vec_format('/home/git/wiki.en.text.vector', binary=False)
        # w2v_model = "test"
        self.logger.info('word2vec model loaded')
        return w2v_model

    def classify(self, tweet_text):
        if tweet_text == "":
            return '', ''

        lang = ldig.predict_lang(self.param, self.labels, self.trie, 'en\t'+tweet_text)
        if lang != 'en':
            return '', ''
        else:
            raw_len = len(list(tweet_text))
            filter_len = len(re.findall(r'[\w\d\s\.\?,@#]',tweet_text))
            ratio = float(filter_len)/raw_len
            if ratio < 0.7:
                return '', ''

        self.count += 1
        if self.count % 10000 == 0:
            print self.count

        text = self.url_pattern.sub('', tweet_text.lower())
        matches = self.ac_tries.findall(text)
        multigram_words = []
        for match in matches:
            multigram_words.append(text[match[0]:match[1]])

        tweet_keywords = filter(lambda w: w not in self.stopwords, nltk.regexp_tokenize(text, self.tokenizer_pattern))
        words_pos_taged = nltk.pos_tag(tweet_keywords)
        tweet_keywords = set([word for word, tag in words_pos_taged if tag[:2] in self.tags])

        tweet_keywords.update(multigram_words)
        if len(tweet_keywords) < 1:
            return '', ''

        profiles_indexed = set([])
        for keywords in tweet_keywords:
            profiles_indexed = profiles_indexed.union(self.inverted_index[keywords])

        profiles_index_related = []
        for index in profiles_indexed:
            flag = True
            for word_set in self.required_keywords[index]:
                if len(tweet_keywords & word_set) == 0:
                    flag = False
                    break
            if flag:
                profiles_index_related.append(index)

        if len(profiles_index_related) < 1:
            return '', ''

        related_profiles = [(index, self.profile_keywords[index]) for index in profiles_index_related]
        topic_id, similarity = self.clf(tweet_keywords, related_profiles)

        # print tweet_text, ' : ', topic_id, ' : ', similarity 
        return topic_id, similarity

    def clf(self, tweet, profiles):
        best_match = -1
        max_similarity = 0.0
        for index, profile in profiles:
            similarity = 1 if len(profile) == 0 else self.calc_sim(tweet, profile)/2.0 + 0.5    # base similarity score
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = index
        return best_match, max_similarity

    def calc_sim(self, tweet, profile):
        similarity = 0.0
        # count = 0
        for profile_keyword in profile:
            sim_values = []
            for token in tweet:
                if token == profile_keyword:
                    sim = 1
                elif token in self.w2v_model and profile_keyword in self.w2v_model:
                    sim = self.w2v_model.similarity(token, profile_keyword)
                else:
                    sim = 0
                sim_values.append(sim)
            similarity += max(sim_values)
        norm_length = len(profile)
        similarity /= norm_length
        return similarity


if __name__ == '__main__':
    clf = Classifier()
