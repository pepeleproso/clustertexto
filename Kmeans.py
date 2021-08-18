import os
from Corpus import Corpus
from datetime import datetime
from LemmaTokenizer import LemmaTokenizer

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

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


def GenerarClustersKMeans(df, columnname, k_list):
    ## Run clustering with different k and check the metrics
    tfidf_model, pairwise_similarity = ObtenerMatrizDistancia(df[columnname])

    labs=[]
    for n_clusters in k_list:
        print ("K= %d", n_clusters)
        km_model = KMeans(n_clusters=n_clusters, init='k-means++')
        km_model.fit(pairwise_similarity)
        labs.append(km_model.labels_)
        print("- La varianza o inercia es: %0.3f"
            %  km_model.inertia_)
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

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0,len(pairwise_similarity.toarray()) + (n_clusters + 1) * 10])

        # Compute the silhouette scores for each sample
        cluster_labels = km_model.labels_
        sample_silhouette_values = metrics.silhouette_samples(pairwise_similarity, cluster_labels)
        silhouette_avg = metrics.silhouette_score(pairwise_similarity, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[km_model.labels_ == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Gráfico de Silueta para los distintos clusters.")
        ax1.set_xlabel("Coeficiente Silueta")
        ax1.set_ylabel("Cluster")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(km_model.labels_.astype(float) / n_clusters)
        ax2.scatter(pairwise_similarity.toarray() [:, 0], pairwise_similarity.toarray() [:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = km_model.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')
        plt.show()
    return labs

def GuardarClusters(df, klist, labels, prefijo):
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M")

    os.mkdir("data/output/clusters_" + prefijo + "_" + str(date_time))

    #Para cada k genero un excel
    i = 0
    for cluster_labels in labels:
        #el valor de K es el primero de la lista de klist
        df['cluster_nro'] = df.apply(lambda row: cluster_labels[row.name], axis=1)
        df.to_excel("data/output/clusters_" + prefijo + "_" + str(date_time) + "/" + str(klist[i]) + ".xlsx") 
        i = i + 1
    

if __name__ == "__main__":
    nombreArchivoEntrada = str(os.getcwd()) + '/data/input/vista_minable_2010_2017_otros_20210710_202512.xlsx'
    df = Corpus(nombreArchivoEntrada, 'Mensajes').get_corpus()

    #klist=range(2,14)
    klist=range(2,5)
    #cl_data = GenerarClustersKMeans(df, 'post_message_limpio', klist)

    #GuardarClusters(df, klist, cl_data, "kmeans_post_message")

#ordeno por el campo
#tengo que eliminar las filas sin nada
#hago el clustering
#parseo los resultados al original
    #df_cl_post = df[df['post_message_limpio'] != '']
    df_cl_post = df[df['titulo_facebook_limpio'] != '']
    df_cl_post.reset_index(inplace=True, drop=True)
    #cl_data = GenerarClustersKMeans(df_cl_post, 'post_message_limpio', klist)
    cl_data = GenerarClustersKMeans(df_cl_post, 'titulo_facebook_limpio', klist)
    GuardarClusters(df_cl_post, klist, cl_data, "kmeans_titulo_facebook")
