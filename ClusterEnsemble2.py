from LemmaTokenizer import LemmaTokenizer
import os
from Corpus import Corpus
from datetime import datetime
from pathlib import Path
#import seaborn as sns

from pandas.core.dtypes.missing import notna
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, AgglomerativeClustering
#from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric

def ObtenerMatrizDistancia(texts):
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer().convert_text_to_tokens,
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

def GuardarClusters(df, ncluster, prefijo):
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M")

    os.mkdir("data/output/clusters_" + prefijo + "_" + str(date_time))
    df.to_excel("data/output/clusters_" + prefijo + "_" + str(date_time) + "/" + str(ncluster) + ".xlsx") 
    

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


def gower_distance(X):
    """
    This function expects a pandas dataframe as input
    The data frame is to contain the features along the columns. Based on these features a
    distance matrix will be returned which will contain the pairwise gower distance between the rows
    All variables of object type will be treated as nominal variables and the others will be treated as 
    numeric variables.
    Distance metrics used for:
    Nominal variables: Dice distance (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    Numeric variables: Manhattan distance normalized by the range of the variable (https://en.wikipedia.org/wiki/Taxicab_geometry)
    """
    individual_variable_distances = []

    for i in range(X.shape[1]):
        feature = X.iloc[:,[i]]
        if feature.dtypes[0] == np.object:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
        else:
            feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / np.ptp(feature.values)

        individual_variable_distances.append(feature_dist)

    return np.array(individual_variable_distances).mean(0)

if __name__ == "__main__":
    nombreArchivoEntrada = str(os.getcwd()) + '/data/input/vista_minable_2010_2017_otros_20210710_202512.xlsx'
    df = Corpus(nombreArchivoEntrada, 'Mensajes').get_corpus()

    klist=range(2,14)
    cl_data = GenerarClustersKMeansMultiple(df, 'post_message_limpio', klist, 20)
    #cl_data = np.array(cl_data)

    klist=range(2,4)
    cl_data2 = GenerarClustersKMeansMultiple(df, 'titulo_facebook_limpio', klist, 20)
 
    #cl_data2 = np.array(cl_data2)
    #cl_todo = [cl_data.append(i) for i in cl_data2]
#    for i in cl_data2:
#        cl_data.append(i) 

    #df_paratextuales = df[["tiene_corchete", "tiene_hashtag", "tiene_signo_interrogacion", "tiene_emoji", "tiene_mencion", "link_tipo_destino", "tiene_numero"]]
    df_paratextuales = df[["tiene_corchete", "tiene_hashtag", "tiene_signo_interrogacion", "tiene_emoji", "tiene_mencion", "link_tipo_destino", "tiene_numero"]]

    Y = gower_distance(df_paratextuales)

    klist=range(2,14)
    labs=[]
    for k in klist:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(Y)
        cl_data.append(kmeans.labels_)

    cl_todo = np.array(cl_data)

    df_variables = df[["texto_propio","localizacion","tema","temporalidad"]]
    Y = gower_distance(df_variables)
    klist=range(2,14)
    labs=[]
    for k in klist:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(Y)
        cl_data.append(kmeans.labels_)

    cl_todo = np.array(cl_data)

    MatrixConsenso = ConstruirMatrixConsenso(cl_todo)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(MatrixConsenso.values)

    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Número de posteos en el nodo")
    plt.show()

 #   n_cl=10
 #   model = AgglomerativeClustering(n_clusters=n_cl)
 #   model = model.fit(MatrixConsenso.values)

 #   df['cluster_nro'] = df.apply(lambda row: model.labels_[row.name], axis=1)

 #   GuardarClusters(df, n_cl, "ensamble_todo")


#Top palabras por cluster
#http://jonathansoma.com/lede/algorithms-2017/classes/clustering/k-means-clustering-with-scikit-learn/
