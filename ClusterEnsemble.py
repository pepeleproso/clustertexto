import re
# import string
import collections.abc
from time import time
from datetime import datetime
# from unicodedata import category
from pathlib import Path
import seaborn as sns

from nltk import word_tokenize
from pandas.core.dtypes.missing import notna
# from nltk.stem import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from InputDataSetCSV import InputDataSetCSV


#palabrasstopwords = ['a', 'e', 'al', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'del', 'desde', 'el', 'en', 'entre', 'este', 'hacia', 'hasta', 'la', 'las', 'lo', 'los', 'más', 'para', 'por', 'que', 'se', 'segun', 'sin', 'sobre', 'su', 'sus', 'te', 'tras', 'un', 'una', 'y', 'le', 'los']
palabrasstopwords = ['a', 'e', 'o', 'al', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'del', 'desde', 'el', 'en', 'entre', 'hacia', 'hasta', 'la', 'las', 'lo', 'los', 'más', 'para', 'por', 'que', 'se', 'segun', 'sin', 'sobre', 'su', 'sus', 'te', 'tras', 'un', 'una', 'y', 'le', 'los']
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

def ObtenerMatrizDistancia(texts):
    vectorizer = TfidfVectorizer(tokenizer=convertir_texto_a_tokens,
                                 max_df=1.0,
                                 min_df=0.05,
                                 strip_accents='unicode')
    #175x20
    tfidf_model = vectorizer.fit_transform(texts)

    # no need to normalize, since Vectorizer will return normalized tf-idf
    #175x175
    pairwise_similarity = tfidf_model * tfidf_model.T
    return tfidf_model, pairwise_similarity

def GenerarClustersKMeansMultiple(data, k_list):
    ## Run clustering with different k and check the metrics
    tfidf_model, pairwise_similarity = ObtenerMatrizDistancia(data)

    labs=[]
    for r in range(1,20):
        for clusters in k_list:
            print(clusters)
            km_model = KMeans(n_clusters=clusters, init='k-means++')
            km_model.fit(pairwise_similarity)
            labs.append(km_model.labels_)
    
    return labs

def GenerarClustersDBSCANMultiple(data, eps_list):
    ## Run clustering with different k and check the metrics
    tfidf_model, pairwise_similarity = ObtenerMatrizDistancia(data)

    labs=[]
    for eps_i in eps_list:
        print(eps_i)
        db_model = DBSCAN(eps=eps_i, min_samples=10)
        db_model.fit(pairwise_similarity)
        labs.append(db_model.labels_)

    return labs

def ConstruirMatrixConsenso(labels):
    C = np.zeros([labels.shape[1],labels.shape[1]], np.int64)
    for label in labels:
        for i, val1 in enumerate(label):
            for j, val2 in enumerate(label):
                #filling C_ij
                
                if val1 == val2 :
                    C[i,j] += 1 
                    
    return pd.DataFrame(C)


def GenerarCorpus(nombreArchivoEntrada):
    inicio = 0
    fin = None

    datasetCSV = InputDataSetCSV(nombreArchivoEntrada, inicio, fin)
    posteos = datasetCSV.dataset
    textos = [posteos[i][1] for i in range(datasetCSV.init, datasetCSV.end)]
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
    texto_post = texto_post.replace('$', 'pesos ')

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

    texto_post = texto_post.replace('conversacionesln', 'conversaciones ln')
    texto_post = texto_post.replace('elecciones2015', 'elecciones 2015')



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
    f = open("data/output/clusters_ensamble_" + str(date_time) + ".txt", "w") 
    cantposteos = 0

    vectorizer = TfidfVectorizer(tokenizer=convertir_texto_a_tokens,
                                 max_df=1.0,
                                 min_df=0.05,
                                 strip_accents='unicode')

    for cluster in clusters.keys():
        f.writelines("----------------------------------------\r\n")
        f.write("Cluster " + str(cluster) + '\r\n')
        f.write("Palabras Comunes\r\n")
        texts = [corpuslimpio[int(id)] for id in clusters[cluster]]

        #175x20
        tfidf_model = vectorizer.fit_transform(texts)

        feature_names = vectorizer.get_feature_names()

        for word in feature_names:
            f.write(word + ", ")
        
        f.write("\r\nPosteos\r\n")
        for id in set(clusters[cluster]):
            a = convertir_texto_a_tokens(corpuslimpio[int(id)])
            if not a:
                continue
            texto = ''
            for k in range(0, len(a)):
                texto = texto + ' ' + a[k]
            f.writelines(corpus[int(id)] + ' (' + texto + ')\r\n')
            cantposteos = cantposteos + 1

    f.close()

    print('ENTRADA: '+str(len(corpus)))
    print('AGRUPADOS: '+ str(cantposteos))
    

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == "__main__":
    CargarLemas()
    #nombreArchivoEntrada = str(Path.cwd()) + '/data/input/posteos_mensaje_titulos.csv'
    nombreArchivoEntrada = str(Path.cwd()) + '/data/input/posteos_otros_completos.xlsx'
    corpus = GenerarCorpus(nombreArchivoEntrada)
    corpus_limpio = LimpiarCorpus(corpus)

    klist=range(2,14)
    
    cl_data = GenerarClustersKMeansMultiple(corpus_limpio, klist)
    cl_data = np.array(cl_data)

    #cl_data2 = GenerarClustersDBSCANMultiple(corpus_limpio, [0.1,0.2,0.3,0.4])
    #cl_todo = [cl_data.append(i) for i in cl_data2]
    #cl_data = np.array(cl_data)

    #cl_data= GenerarClustersDBSCANMultiple(corpus_limpio, [0.1,0.2,0.3,0.4])
    #cl_data = np.array(cl_data)

    MatrixConsenso = ConstruirMatrixConsenso(cl_data)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(MatrixConsenso.values)

    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    model = AgglomerativeClustering(n_clusters=7)
    model = model.fit(MatrixConsenso.values)

    cluster_idxs = collections.defaultdict(list)

    for idx, label in enumerate(model.labels_):
        cluster_idxs[label].append(idx)

    #Z = hierarchy.linkage(MatrixConsenso.values, 'average')
    #plt.figure()

    #den = hierarchy.dendrogram(Z)
    #plt.show()                              
    #print()

    #cluster_idxs = collections.defaultdict(list)
    #for c, pi in zip(den['color_list'], den['icoord']):
    #    for leg in pi[1:3]:
    #        i = (leg - 5.0) / 10.0
    #        if abs(i - int(i)) < 1e-5:
    #            cluster_idxs[c].append(den['ivl'][int(i)])
    #cluster_idxs

    

    GuardarClusters(cluster_idxs, corpus, corpus_limpio)


#Top palabras por cluster
#http://jonathansoma.com/lede/algorithms-2017/classes/clustering/k-means-clustering-with-scikit-learn/