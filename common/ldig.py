#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ldig : Language Detector with Infinite-Gram
# This code is available under the MIT License.
# (c)2011 Nakatani Shuyo / Cybozu Labs Inc.

import os
import sys
import re
import codecs
import json
import gzip
import htmlentitydefs

import numpy

from Microblog_Trec.common import da

reload(sys)
sys.setdefaultencoding('utf-8')

class LangDetector:
    def __init__(self, model_dir):
        self.features = os.path.join(model_dir, 'features')
        self.labels = os.path.join(model_dir, 'labels.json')
        self.param = os.path.join(model_dir, 'parameters.npy')
        self.doublearray = os.path.join(model_dir, 'doublearray.npz')

    def load_da(self):
        trie = da.DoubleArray()
        trie.load(self.doublearray)
        return trie

    def load_features(self):
        features = []
        with codecs.open(self.features, 'rb',  'utf-8') as f:
            pre_feature = ""
            for n, s in enumerate(f):
                m = re.match(r'(.+)\t([0-9]+)', s)
                if not m:
                    sys.exit("irregular feature : '%s' at %d" % (s, n + 1))
                if pre_feature >= m.groups(1):
                    sys.exit("unordered feature : '%s' at %d" % (s, n + 1))
                pre_feature = m.groups(1)
                features.append(m.groups())
        return features

    def load_labels(self):
        with open(self.labels, 'rb') as f:
            return json.load(f)

    def load_params(self):
        trie = self.load_da()
        param = numpy.load(self.param)
        labels = self.load_labels()
        return param, labels, trie

# from http://www.programming-magic.com/20080820002254/
reference_regex = re.compile(u'&(#x?[0-9a-f]+|[a-z]+);', re.IGNORECASE)
num16_regex = re.compile(u'#x\d+', re.IGNORECASE)
num10_regex = re.compile(u'#\d+', re.IGNORECASE)
def htmlentity2unicode(text):
    result = u''
    i = 0
    while True:
        match = reference_regex.search(text, i)
        if match is None:
            result += text[i:]
            break

        result += text[i:match.start()]
        i = match.end()
        name = match.group(1)

        if name in htmlentitydefs.name2codepoint.keys():
            result += unichr(htmlentitydefs.name2codepoint[name])
        elif num16_regex.match(name):
            result += unichr(int(u'0'+name[1:], 16))
        elif num10_regex.match(name):
            result += unichr(int(name[1:]))
    return result

def normalize_twitter(text):
    """normalization for twitter"""
    text = re.sub(r'(@|#|https?:\/\/)[^ ]+', '', text)
    text = re.sub(r'(^| )[:;x]-?[\(\)dop]($| )', ' ', text)  # facemark
    text = re.sub(r'(^| )(rt[ :]+)*', ' ', text)
    text = re.sub(r'([hj])+([aieo])+(\1+\2+){1,}', r'\1\2\1\2', text, re.IGNORECASE)  # laugh
    text = re.sub(r' +(via|live on) *$', '', text)
    return text


re_ignore_i = re.compile(r'[^I]')
re_turkish_alphabet = re.compile(u'[\u011e\u011f\u0130\u0131]')
vietnamese_norm = {
	u'\u0041\u0300':u'\u00C0', u'\u0045\u0300':u'\u00C8', u'\u0049\u0300':u'\u00CC', u'\u004F\u0300':u'\u00D2',
	u'\u0055\u0300':u'\u00D9', u'\u0059\u0300':u'\u1EF2', u'\u0061\u0300':u'\u00E0', u'\u0065\u0300':u'\u00E8',
	u'\u0069\u0300':u'\u00EC', u'\u006F\u0300':u'\u00F2', u'\u0075\u0300':u'\u00F9', u'\u0079\u0300':u'\u1EF3',
	u'\u00C2\u0300':u'\u1EA6', u'\u00CA\u0300':u'\u1EC0', u'\u00D4\u0300':u'\u1ED2', u'\u00E2\u0300':u'\u1EA7',
	u'\u00EA\u0300':u'\u1EC1', u'\u00F4\u0300':u'\u1ED3', u'\u0102\u0300':u'\u1EB0', u'\u0103\u0300':u'\u1EB1',
	u'\u01A0\u0300':u'\u1EDC', u'\u01A1\u0300':u'\u1EDD', u'\u01AF\u0300':u'\u1EEA', u'\u01B0\u0300':u'\u1EEB',

	u'\u0041\u0301':u'\u00C1', u'\u0045\u0301':u'\u00C9', u'\u0049\u0301':u'\u00CD', u'\u004F\u0301':u'\u00D3',
	u'\u0055\u0301':u'\u00DA', u'\u0059\u0301':u'\u00DD', u'\u0061\u0301':u'\u00E1', u'\u0065\u0301':u'\u00E9',
	u'\u0069\u0301':u'\u00ED', u'\u006F\u0301':u'\u00F3', u'\u0075\u0301':u'\u00FA', u'\u0079\u0301':u'\u00FD',
	u'\u00C2\u0301':u'\u1EA4', u'\u00CA\u0301':u'\u1EBE', u'\u00D4\u0301':u'\u1ED0', u'\u00E2\u0301':u'\u1EA5',
	u'\u00EA\u0301':u'\u1EBF', u'\u00F4\u0301':u'\u1ED1', u'\u0102\u0301':u'\u1EAE', u'\u0103\u0301':u'\u1EAF',
	u'\u01A0\u0301':u'\u1EDA', u'\u01A1\u0301':u'\u1EDB', u'\u01AF\u0301':u'\u1EE8', u'\u01B0\u0301':u'\u1EE9',

	u'\u0041\u0303':u'\u00C3', u'\u0045\u0303':u'\u1EBC', u'\u0049\u0303':u'\u0128', u'\u004F\u0303':u'\u00D5',
	u'\u0055\u0303':u'\u0168', u'\u0059\u0303':u'\u1EF8', u'\u0061\u0303':u'\u00E3', u'\u0065\u0303':u'\u1EBD',
	u'\u0069\u0303':u'\u0129', u'\u006F\u0303':u'\u00F5', u'\u0075\u0303':u'\u0169', u'\u0079\u0303':u'\u1EF9',
	u'\u00C2\u0303':u'\u1EAA', u'\u00CA\u0303':u'\u1EC4', u'\u00D4\u0303':u'\u1ED6', u'\u00E2\u0303':u'\u1EAB',
	u'\u00EA\u0303':u'\u1EC5', u'\u00F4\u0303':u'\u1ED7', u'\u0102\u0303':u'\u1EB4', u'\u0103\u0303':u'\u1EB5',
	u'\u01A0\u0303':u'\u1EE0', u'\u01A1\u0303':u'\u1EE1', u'\u01AF\u0303':u'\u1EEE', u'\u01B0\u0303':u'\u1EEF',

	u'\u0041\u0309':u'\u1EA2', u'\u0045\u0309':u'\u1EBA', u'\u0049\u0309':u'\u1EC8', u'\u004F\u0309':u'\u1ECE',
	u'\u0055\u0309':u'\u1EE6', u'\u0059\u0309':u'\u1EF6', u'\u0061\u0309':u'\u1EA3', u'\u0065\u0309':u'\u1EBB',
	u'\u0069\u0309':u'\u1EC9', u'\u006F\u0309':u'\u1ECF', u'\u0075\u0309':u'\u1EE7', u'\u0079\u0309':u'\u1EF7',
	u'\u00C2\u0309':u'\u1EA8', u'\u00CA\u0309':u'\u1EC2', u'\u00D4\u0309':u'\u1ED4', u'\u00E2\u0309':u'\u1EA9',
	u'\u00EA\u0309':u'\u1EC3', u'\u00F4\u0309':u'\u1ED5', u'\u0102\u0309':u'\u1EB2', u'\u0103\u0309':u'\u1EB3',
	u'\u01A0\u0309':u'\u1EDE', u'\u01A1\u0309':u'\u1EDF', u'\u01AF\u0309':u'\u1EEC', u'\u01B0\u0309':u'\u1EED',

	u'\u0041\u0323':u'\u1EA0', u'\u0045\u0323':u'\u1EB8', u'\u0049\u0323':u'\u1ECA', u'\u004F\u0323':u'\u1ECC',
	u'\u0055\u0323':u'\u1EE4', u'\u0059\u0323':u'\u1EF4', u'\u0061\u0323':u'\u1EA1', u'\u0065\u0323':u'\u1EB9',
	u'\u0069\u0323':u'\u1ECB', u'\u006F\u0323':u'\u1ECD', u'\u0075\u0323':u'\u1EE5', u'\u0079\u0323':u'\u1EF5',
	u'\u00C2\u0323':u'\u1EAC', u'\u00CA\u0323':u'\u1EC6', u'\u00D4\u0323':u'\u1ED8', u'\u00E2\u0323':u'\u1EAD',
	u'\u00EA\u0323':u'\u1EC7', u'\u00F4\u0323':u'\u1ED9', u'\u0102\u0323':u'\u1EB6', u'\u0103\u0323':u'\u1EB7',
	u'\u01A0\u0323':u'\u1EE2', u'\u01A1\u0323':u'\u1EE3', u'\u01AF\u0323':u'\u1EF0', u'\u01B0\u0323':u'\u1EF1',
}
re_vietnamese = re.compile(u'[AEIOUYaeiouy\u00C2\u00CA\u00D4\u00E2\u00EA\u00F4\u0102\u0103\u01A0\u01A1\u01AF\u01B0][\u0300\u0301\u0303\u0309\u0323]')
re_latin_cont = re.compile(u'([a-z\u00e0-\u024f])\\1{2,}')
re_symbol_cont = re.compile(u'([^a-z\u00e0-\u024f])\\1{1,}')
def normalize_text(org):
    m = re.match(r'([-A-Za-z]+)\t(.+)', org)
    if m:
        label, org = m.groups()
    else:
        label = ""
    m = re.search(r'\t([^\t]+)$', org)
    if m:
        s = m.group(0)
    else:
        s = org
    s = htmlentity2unicode(s)
    s = re.sub(u'[\u2010-\u2015]', '-', s)
    s = re.sub(u'[0-9]+', '0', s)
    s = re.sub(u'[^\u0020-\u007e\u00a1-\u024f\u0300-\u036f\u1e00-\u1eff]+', ' ', s)
    s = re.sub(u'  +', ' ', s)

    # vietnamese normalization
    s = re_vietnamese.sub(lambda x:vietnamese_norm[x.group(0)], s)

    # lower case with Turkish
    s = re_ignore_i.sub(lambda x:x.group(0).lower(), s)
    #if re_turkish_alphabet.search(s):
    #    s = s.replace(u'I', u'\u0131')
    #s = s.lower()

    # Romanian normalization
    s = s.replace(u'\u0219', u'\u015f').replace(u'\u021b', u'\u0163')

    s = normalize_twitter(s)
    s = re_latin_cont.sub(r'\1\1', s)
    s = re_symbol_cont.sub(r'\1', s)

    return label, s.strip(), org


# load courpus
def load_corpus(filelist, labels):
    idlist = dict((x, []) for x in labels)
    corpus = []
    for filename in filelist:
        f = codecs.open(filename, 'rb',  'utf-8')
        for i, s in enumerate(f):
            label, text, org_text = normalize_text(s)
            if label not in labels:
                sys.exit("unknown label '%s' at %d in %s " % (label, i+1, filename))
            idlist[label].append(len(corpus))
            corpus.append((label, text, org_text))
        f.close()
    return corpus, idlist

# prediction probability
def predict(param, events):
    sum_w = numpy.dot(param[events.keys(),].T, events.values())
    exp_w = numpy.exp(sum_w - sum_w.max())
    return exp_w / exp_w.sum()

def predict_lang(param, labels, trie, inText):
    K = len(labels)
    corrects = numpy.zeros(K, dtype=int)
    counts = numpy.zeros(K, dtype=int)

    label_map = dict((x, i) for i, x in enumerate(labels))

    n_available_data = 0
    log_likely = 0.0

    label, text, org_text = normalize_text(inText)

    if label not in label_map:
        sys.stderr.write("WARNING : unknown label '%s' in %s (ignore the later same labels)\n" % (label, text))
        label_map[label] = -1
    label_k = label_map[label]

    events = trie.extract_features(u"\u0001" + text + u"\u0001")
    y = predict(param, events)
    predict_k = y.argmax()

    if label_k >= 0:
        log_likely -= numpy.log(y[label_k])
        n_available_data += 1
        counts[label_k] += 1
        if label_k == predict_k and y[predict_k] >= 0.6:
            corrects[predict_k] += 1

    predict_lang = labels[predict_k]
    if y[predict_k] < 0.6: predict_lang = ""
    #print "%s\t%s\t%s" % (label, predict_lang, org_text)
    return predict_lang

def generate_doublearray(file, features):
    trie = da.DoubleArray()
    trie.initialize(features)
    trie.save(file)

re_filtUrl=re.compile(r'(#|@|https?:\/\/)[^ ]+')

def extract_text(line):
    line = line.decode('utf-8')
    origin_text=""; text = ""; id = ""
    if line[0] == '{':
        tweet = json.loads(line)
        if tweet.has_key('created_at') and tweet['user']['lang'] == 'en':
            origin_text = re.sub(r'\t+|\n+', ' ', tweet['text'])
            text = re_filtUrl.sub(' ', origin_text).strip()
            id = tweet['id_str']
    return origin_text, text, id

if __name__ == '__main__':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
    fw = open("originEn_tweets"+sys.argv[1][-11:-3]+".txt", 'w')

    model = os.path.join(sys.path[0], 'model.latin')
    detector = ldig(model)
    param, labels, trie = detector.load_params()

    count_en = 0; count_total = 0
#    for text in fileinput.input():
    for line in gzip.open(sys.argv[1], 'rb'):
        count_total += 1
        origin_text, text, id = extract_text(line)
        if text != "":
            lang = predict_lang(param, labels, trie, 'en\t'+text)
            if lang == 'en':
                count_en += 1
                #fw.write(id + '\t' + origin_text + "\n")
                fw.write(line)
        if count_en % 1000 == 0: print "count: ", count_en
    print "total count: ", count_total, " en count: ", count_en
		
