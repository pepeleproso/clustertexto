import os
from Corpus import Corpus
from datetime import datetime
from LemmaTokenizer import convert_text_to_tokens

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

def ObtenerMatrizDistancia(texts):
    vectorizer = TfidfVectorizer(tokenizer=convert_text_to_tokens,
                                 max_df=1.0,
                                 min_df=0.05,
                                 strip_accents='unicode')
    #175x20
    tfidf_model = vectorizer.fit_transform(texts)

    # no need to normalize, since Vectorizer will return normalized tf-idf
    #175x175
    pairwise_similarity = tfidf_model * tfidf_model.T
    return tfidf_model, pairwise_similarity


def GenerarClustersKMeans(df, columnname, k_list):
    ## Run clustering with different k and check the metrics
    tfidf_model, pairwise_similarity = ObtenerMatrizDistancia(df[columnname])

    labs=[]
    for clusters in k_list:
        print ("K= %d", clusters)
        km_model = KMeans(n_clusters=clusters, init='k-means++')
        km_model.fit(pairwise_similarity)
        labs.append(km_model.labels_)

        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(tfidf_model, km_model.labels_, sample_size=1000))
        print()
    
        #print('Palabras Comunes de cada Cluster')
        #order_centroids = km_model.cluster_centers_.argsort()[:, ::-1]
        #terms = vectorizer.get_feature_names()
        #for i in range(1, len(dict(clustering))):
        #    print("Cluster %d:" % i, end='')
            #for ind in order_centroids[i, :5]:
            #    print(' %s' % terms[ind], end='')
        #    print()

    return labs

def GuardarClusters(clusters, corpus, corpuslimpio):
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M")
    f = open("data/output/clusters_" + str(date_time) + ".txt", "w") 
    cantposteos = 0

    for cluster in range(1, len(dict(clusters))):
        f.writelines("----------------------------------------\r\n")
        f.write("Cluster " + str(cluster) + '\r\n')
        for id in enumerate(dict(clusters)[cluster]):
            a = convert_text_to_tokens(corpuslimpio[id[1]])
            texto = ''
            for k in range(1, len(a)):
                texto = texto + ' ' + a[k]
            f.writelines(corpus[id[1]] + ' (' + texto + ')\r\n')
            cantposteos = cantposteos + 1

    f.close()

    print('ENTRADA: '+str(len(corpus)))
    print('AGRUPADOS: '+ str(cantposteos))
    

if __name__ == "__main__":
    nombreArchivoEntrada = str(os.getcwd()) + '/data/input/vista_minable_2010_2017_otros_20210710_202512.xlsx'
    df = Corpus(nombreArchivoEntrada, 'Mensajes').get_corpus()

    klist=range(2,14)
    cl_data = GenerarClustersKMeans(df, 'post_message_limpio', klist)

    df['cluster_nro'] = df.apply(lambda row: model.labels_[row.name], axis=1)

    #klist=range(2,14)
    #cl_data = GenerarClustersKMeans(df, 'titulo_facebook_limpio', klist)

    #GuardarClusters(clusters, corpus, corpus_limpio)
