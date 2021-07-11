import os

class Lemmas(object):
    def __init__(self):
        self.lemmaDict = {}

        with open(str(os.getcwd()) + '/data/lemmatization-es.txt', 'rb') as f:
            data = f.read().decode('utf8').replace(u'\r', u'').split(u'\n')
            data = [a.split(u'\t') for a in data]

        for a in data:
            if len(a) > 1:
                self.lemmaDict[a[1]] = a[0]

    def Lemmatize(self, word):
        # busco la palabra en la lista de lemas si la encuentro devuelvo el lema
        # sino la misma palabra
        return self.lemmaDict.get(word, word)