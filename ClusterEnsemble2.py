from Corpus import Corpus
from datetime import datetime
from LemmaTokenizer import convert_text_to_tokens
from pathlib import Path
import seaborn as sns

from pandas.core.dtypes.missing import notna
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def GenerarClustersKMeansMultiple(df, columnname, k_list, nruns):
    ## Run clustering with different k and check the metrics
    tfidf_model, pairwise_similarity = ObtenerMatrizDistancia(df[columnname])

    labs=[]
    for r in range(1,nruns):
        for clusters in k_list:
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

def GuardarClusters(df, ncluster):
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M")
    f = open("data/output/clusters_ensamble_" + str(ncluster) + "_" + str(date_time) + ".txt", "w") 
    cantposteos = 0

    vectorizer = TfidfVectorizer(tokenizer=convert_text_to_tokens,
                                 max_df=1.0,
                                 min_df=0.05,
                                 strip_accents='unicode')

    
    clusters = np.sort(df['cluster_nro'].unique())

    for cluster in clusters:
        f.writelines("----------------------------------------\r\n")
        f.write("Cluster " + str(cluster) + '\r\n')
    #    f.write("Palabras Comunes\r\n")
    #    texts = [corpuslimpio[int(id)] for id in clusters[cluster]]

        #175x20
    #    tfidf_model = vectorizer.fit_transform(texts)

    #    feature_names = vectorizer.get_feature_names()

    #    for word in feature_names:
    #        f.write(word + ", ")
        
        f.write("\r\nPosteos\r\n")
        df_cl_post = df[df['cluster_nro'] == cluster]
        a = 0

        for index, row in df_cl_post.iterrows():
            f.write(str(row.name) + ' ')
            f.write(str(row['post_message']).strip() if str(row['post_message']) != 'nan' else '')
            f.write(' (' + str(row['post_message_limpio']) + ') - ')
            f.write(str(row['titulo_facebook']).strip() if str(row['titulo_facebook']) != 'nan' else '')
            f.write(' (' + str(row['titulo_facebook_limpio']) + ') - ')
            f.write(row['post_link'] + ' - ')
            f.write(str(row.name) + ' - ' + row['post_id'] + '\r\n')
            cantposteos = cantposteos + 1
    f.close()

    df.to_excel("data/output/clusters_ensamble_" + str(ncluster) + "_" + str(date_time) + ".xlsx") 
    print('ENTRADA: '+str(len(df.index)))
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
    nombreArchivoEntrada = str(Path.cwd()) + '/data/input/posteos_otros_completos.xlsx'
    df = Corpus(nombreArchivoEntrada, 'Mensajes').get_corpus()

    klist=range(2,14)
    cl_data = GenerarClustersKMeansMultiple(df, 'post_message_limpio', klist, 20)
    #cl_data = np.array(cl_data)

    klist=range(2,4)
    cl_data2 = GenerarClustersKMeansMultiple(df, 'titulo_facebook_limpio', klist, 20)
    #cl_data2 = np.array(cl_data2)
    #cl_todo = [cl_data.append(i) for i in cl_data2]
    for i in cl_data2:
        cl_data.append(i) 

    df_paratextuales = df[["tiene_corchete", "tiene_hashtag", "tiene_signo_interrogacion", "tiene_emoji", "tiene_mencion", "link_tipo_destino"]]

    klist=range(2,14)
    labs=[]
    for k in klist:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(df_paratextuales)
        cl_data.append(kmeans.labels_)

    cl_todo = np.array(cl_data)

    df_variables = df[["texto_propio","localizacion","tema","temporalidad"]]

    klist=range(2,14)
    labs=[]
    for k in klist:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(df_variables)
        cl_data.append(kmeans.labels_)

    cl_todo = np.array(cl_data)

    MatrixConsenso = ConstruirMatrixConsenso(cl_todo)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(MatrixConsenso.values)

    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    n_cl=8
    model = AgglomerativeClustering(n_clusters=n_cl)
    model = model.fit(MatrixConsenso.values)

    df['cluster_nro'] = df.apply(lambda row: model.labels_[row.name], axis=1)

    GuardarClusters(df, n_cl)


#Top palabras por cluster
#http://jonathansoma.com/lede/algorithms-2017/classes/clustering/k-means-clustering-with-scikit-learn/