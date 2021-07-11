import re
# import string
import collections.abc
from time import time
from datetime import datetime
# from unicodedata import category
from pathlib import Path

from nltk import word_tokenize
from pandas.core.dtypes.missing import notna
# from nltk.stem import SnowballStemmer
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np

from InputDataSetCSV import InputDataSetCSV


palabrasstopwords = ['a', 'e', 'al', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'del', 'desde', 'el', 'en', 'entre', 'hacia', 'hasta', 'la', 'las', 'lo', 'los', 'más', 'para', 'por', 'que', 'se', 'segun', 'sin', 'sobre', 'su', 'sus', 'te', 'tras', 'un', 'una', 'y', 'le', 'los']
# palabrasstopwords = sorted(palabrasstopwords)
# print(palabrasstopwords)
lemmaDict = {}


def CargarLemas():
    with open(str(Path.cwd()) + '/data/lemmatization-es.txt', 'rb') as f:
        data = f.read().decode('utf8').replace(u'\r', u'').split(u'\n')
        data = [a.split(u'\t') for a in data]

    for a in data:
        if len(a) > 1:
            lemmaDict[a[1]] = a[0]


def lemmatize(word):
    # busco la palabra en la lista de lemas si la encuentro devuelvo el lema
    # sino la misma palabra
    return lemmaDict.get(word, word)


def convertir_texto_a_tokens(text):
    """ Tokenize text and stem words removing punctuation """
    tokens = word_tokenize(text, language='spanish')

    for i in range(0, len(tokens)):
        if tokens[i] in palabrasstopwords:
            tokens[i] = ''

        tokens[i] = lemmatize(tokens[i])

        matchObj = re.match(r'^[0-9]+$', tokens[i])
        if matchObj:
            tokens[i] = ConvertirNumerosALetras(tokens[i])

    while '' in tokens:
        tokens.remove('')

    return tokens


def Generar_Clusters(texts):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=convertir_texto_a_tokens,
                                 max_df=1.0,
                                 min_df=0.05,
                                 strip_accents='unicode')

    #175x20
    tfidf_model = vectorizer.fit_transform(texts)

    # no need to normalize, since Vectorizer will return normalized tf-idf
    #175x175
    pairwise_similarity = tfidf_model * tfidf_model.T

    #pasamos la matriz de distancia a un objeto matriz
    #arreglo = pairwise_similarity.A


    db_model = DBSCAN(eps=0.3, min_samples=10)

    print("Iniciando Clustering con DBScan")
    t0 = time()

    db = db_model.fit(pairwise_similarity)

    print("Terminado en %0.3fs" % (time() - t0))
    print()

    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(tfidf_model, db_model.labels_, sample_size=1000))
    print()

    clustering = collections.defaultdict(list)

    for idx, label in enumerate(db_model.labels_):
        clustering[label].append(idx)

    return clustering


def GenerarCorpus(nombreArchivoEntrada):
    inicio = 0
    fin = None

    datasetCSV = InputDataSetCSV(nombreArchivoEntrada, inicio, fin)
    posteos = datasetCSV.dataset
    textos = [posteos[i][2] for i in range(datasetCSV.init, datasetCSV.end)]
    textos = [i.strip() if str(i) != 'nan' else '' for i in textos]
    textos = [i for i in textos if i]
    print('Cantidad de Posteos Corpus:' + str(len(textos)))
    return textos


def LimpiarCorpus(textos):
    textos = [LimpiarTextoPosteo(post) for post in textos]
    return textos


def LimpiarTextoPosteo(texto_post):

    texto_post = texto_post.replace('Diario Clarín shared a link.', '')
    texto_post = texto_post.replace('LA NACION shared a link.', '')

    texto_post = texto_post.replace('...', '')
    texto_post = re.sub(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', '', texto_post)

    #texto_post = texto_post.replace('\r\n', '')
    texto_post = texto_post.replace('<3', '')

    texto_post = texto_post.replace('.', ' ')
    texto_post = texto_post.replace(':', ' ')
    texto_post = texto_post.replace(',', ' ')
    texto_post = texto_post.replace('¨', ' ')
    texto_post = texto_post.replace('-', ' ')
    texto_post = texto_post.replace(';', ' ')
    texto_post = texto_post.replace('[','')
    texto_post = texto_post.replace(']','')
    texto_post = texto_post.replace('(','')
    texto_post = texto_post.replace(')','')
    texto_post = texto_post.replace('¿', ' ¿ ')
    texto_post = texto_post.replace('¿', ' ¿ ')
    
    texto_post = texto_post.replace('¡', '¡ ')
    #texto_post = texto_post.replace('!',' !')

    texto_post = texto_post.replace('¿','')
    texto_post = texto_post.replace('?','')
    texto_post = texto_post.replace('¡','')
    texto_post = texto_post.replace('!','')

    texto_post = texto_post.replace('\'', ' ')
    texto_post = texto_post.replace('"', ' ')
    texto_post = texto_post.replace('_', ' ')
    
    texto_post = texto_post.replace('#','')
    texto_post = texto_post.replace('|', ' ')
    texto_post = texto_post.replace('<', ' ')
    texto_post = texto_post.replace('>', ' ')

    texto_post = texto_post.replace('“', ' ')
    texto_post = texto_post.replace('”', ' ')

    texto_post = texto_post.strip()
    texto_post = texto_post.lower()

    #texto_post = ''.join(ch for ch in texto_post if category(ch)[0] != 'P')
    return texto_post


def ConvertirNumerosALetras(n):
    lista = list(str(n))
    inverse = lista[::-1]
    new = ['', '', '', '', '', '', '', '', '']
    con = 0
    for i in inverse:
        new[con]=int(i)
        con+=1
    a,b,c,d,e,f,g,h,i=new[::-1] # recorre y asigna los valores de la lista hacia atras
    if len(str(new[3]))>0:
        new.insert(3,'.')
    if len(str(new[7]))>0:
        new.insert(7,'.')
    numero=new[::-1]

    unidad={1:'un', 2:'dos', 3:'tres', 4:'cuatro', 5:'cinco', 6:'seis', 7:'siete', 8:'ocho', 9:'nueve',0:'','':''}
    unidadi={1:'uno', 2:'dos', 3:'tres', 4:'cuatro', 5:'cinco', 6:'seis', 7:'siete', 8:'ocho', 9:'nueve',0:'','':''}
    unidad2={10:'diez', 11:'once', 12:'doce', 13:'trece', 14:'catorce', 15:'quince',16:'dieciseis',17:'diecisiete', 18:'dieciocho', 19:'diecinueve'}
    decena={1:'diez', 2:'veinti', 3:'treinta', 4:'cuarenta', 5:'cincuenta', 6:'sesenta', 7:'setenta', 8:'ochenta', 9:'noventa','':'',0:''}
    centena={1:'ciento', 2:'dos cientos',3:'tres cientos',4:'cuatro cientos',5:'quinientos',6:'seis cientos',7:'setecientos',8:'ocho cientos',9:'novecientos','':'',0:''}

    a=centena[a]
    if b==1 and c<6:
        b,c=unidad2[int(str(b)+str(c))],'millones'
    elif c==1:
        c,b='un millon',decena[b]
    elif b==0:
        b,c='',(unidad[c]+len(str(c))*' millones')
    else:
        b=(decena[b]+len(str(b))*' y')
        c=(unidad[c]+len(str(c))*' millones')
    d=centena[d]
    if e==1 and f<6:
        e,f=unidad2[int(str(e)+str(f))],'mil'
    elif f==0:
        e,f=decena[e],'mil'
    elif e==0:
        e,f='',(unidad[f]+len(str(f))*' mil')
    else:
        e=(decena[e]+len(str(e))*' y')
        f=(unidad[f]+len(str(f))*' mil')
    g=centena[g]
    if h==1: #and i<6:
        h,i=unidad2[int(str(h)+str(i))],''
    elif h==0:
        h,i='',unidadi[i]
    else:
        if i==0:
            i,h='',decena[h]
        elif h==2:
            i =unidadi[i]
            h = decena[h]
        else:
            i,h=unidadi[i],decena[h]+len(str(h))*' y'
    orden= a + b +c+d+e+f+g+h+i
    return orden

def GuardarClusters(clusters, corpus, corpuslimpio):
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M")
    f = open("data/output/clusters_dbscan_" + str(date_time) + ".txt", "w") 
    cantposteos = 0

    for cluster in clusters.keys():
        f.writelines("----------------------------------------\r\n")
        f.write("Cluster " + str(cluster) + '\r\n')
        for id in enumerate(dict(clusters)[cluster]):
            a = convertir_texto_a_tokens(corpuslimpio[id[1]])
            texto = ''
            for k in range(1, len(a)):
                texto = texto + ' ' + a[k]
            f.writelines(corpus[id[1]] + ' (' + texto + ')\r\n')
            cantposteos = cantposteos + 1

    f.close()

    print('ENTRADA: '+str(len(corpus)))
    print('AGRUPADOS: '+ str(cantposteos))
    

if __name__ == "__main__":
    CargarLemas()
    nombreArchivoEntrada = str(Path.cwd()) + '/data/input/posteos_mensaje_titulos.csv'
    corpus = GenerarCorpus(nombreArchivoEntrada)
    corpus_limpio = LimpiarCorpus(corpus)
    clusters = Generar_Clusters(corpus_limpio)
    GuardarClusters(clusters, corpus, corpus_limpio)

    #for texto in corpus[1:175]:
    #texto = 'Siete playas para escaparse del frío Diario Clarín shared a link.'
    #    texto_limpio = LimpiarTextoPosteo(texto)
    #    texto_tokens = convertir_texto_a_tokens(texto_limpio)
    #    print(texto)
    #    print(texto_limpio)
    #    print(texto_tokens)
