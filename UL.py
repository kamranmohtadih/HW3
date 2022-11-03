import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, silhouette_samples, silhouette_score, mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.preprocessing import StandardScaler, minmax_scale
from mpl_toolkits.mplot3d import Axes3D
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import VarianceThreshold
from plot_learning_curve import plot_learning_curve
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore",category=Warning)
####################################################################
# CLUSTERING #######################################################
####################################################################
# Read the data and prpare the plot

def clustering(dataset , n_init):
    df1 = pd.read_csv(dataset, index_col=0, header=0)
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1.dropna(inplace=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    names = df1.columns.values
    indexes = df1.index.values
    x = df1.iloc[:,:-1]
    y = df1.iloc[:,-1]
    x = preprocessing.OneHotEncoder().fit_transform(x).toarray()


    ############################
    # K-Means & EM #############
    ############################
    cl_size = [2,5,10,20,50,100]
    times_km = []
    homog_km = []
    sils_km = []
    times_em = []
    homog_em = []
    sils_em = []
    mutinfo_km =[]
    mutinfo_em =[]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    fig.set_size_inches(18, 7)

    for n_clusters in cl_size:

        start = time.time()
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(x)
        silhouette_avg = silhouette_score(x, cluster_labels,metric='manhattan')
        sils_km.append(silhouette_avg)
        times_km.append(time.time()-start)
        homog_km.append(homogeneity_score(cluster_labels,y))
        mutinfo_km.append(mutual_info_score(cluster_labels, y))

        start = time.time()
        clusterer = GaussianMixture(n_components =n_clusters, random_state=10,  n_init=n_init)
        cluster_labels = clusterer.fit_predict(x)
        silhouette_avg = silhouette_score(x, cluster_labels, metric='manhattan')
        sils_em.append(silhouette_avg)
        times_em.append(time.time() - start)
        homog_em.append(homogeneity_score(cluster_labels, y))
        mutinfo_em.append(mutual_info_score(cluster_labels, y))




    ax1.set_title("The silhouette score")
    ax1.set_xlabel("The silhouette score")
    ax1.set_ylabel("Number of clusters")
    ax1.plot(sils_km,cl_size, label="k-means", color="red", linestyle="--")
    ax1.plot(sils_em,cl_size, label="EM", color="blue", linestyle="--")

    ax2.set_title("The homogenity score")
    ax2.set_xlabel("The homogentiy score")
    ax2.set_ylabel("Number of clusters")
    ax2.plot(homog_km,cl_size, label="k-means", color="red", linestyle="--")
    ax2.plot(homog_em,cl_size, label="EM", color="blue", linestyle="--")

    ax4.set_title("The mutual info score")
    ax4.set_xlabel("The mutual info score")
    ax4.set_ylabel("Number of clusters")
    ax4.plot(mutinfo_km,cl_size, label="k-means", color="red", linestyle="--")
    ax4.plot(mutinfo_em,cl_size, label="EM", color="blue", linestyle="--")

    ax3.set_title("Time")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Number of clusters")
    ax3.plot(times_km,cl_size, label="k-means", color="red", linestyle="--")
    ax3.plot(times_em,cl_size, label="EM", color="blue", linestyle="--")

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')

    plt.suptitle(
        "KMeans vs EM clustering - How metrics change with different clusters",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig('images/UL_'+str(dataset)+'_n_init_'+str(n_init)+'.png')
    plt.clf()
    plt.close('images/UL_'+str(dataset)+'_n_init_'+str(n_init)+'.png')

    # ######################################################################
    # Silhouette analysis ##################################################
    # ######################################################################
    for n_clusters in cl_size:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([0, len(x) + (n_clusters + 1) * 10])
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(x)
        silhouette_avg = silhouette_score(x, cluster_labels)
        sample_silhouette_values = silhouette_samples(x, cluster_labels)

        clusterer = GaussianMixture(n_components=n_clusters, random_state=10, n_init=n_init)
        cluster_labels_em = clusterer.fit_predict(x)
        silhouette_avg_em = silhouette_score(x, cluster_labels_em)
        sample_silhouette_values_em = silhouette_samples(x, cluster_labels_em)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            ax2.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax1.set_title("The silhouette plot for the various clusters - K-Means.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2,
                        0, 0.2, 0.4, 0.6, 0.8, 1])

        ax2.set_title("The silhouette plot for the various clusters - EM")
        ax2.set_xlabel("The silhouette coefficient values")
        ax2.set_ylabel("Cluster label")
        ax2.axvline(x=silhouette_avg_em, color="red", linestyle="--")
        ax2.set_yticks([])  # Clear the yaxis labels / ticks
        ax2.set_xticks([ -0.2,
                        0, 0.2, 0.4, 0.6, 0.8, 1])


        plt.suptitle(
            "Silhouette analysis for KMeans and EM clustering with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig('images/Silhouette analysis for KMeans and EM clustering with n_clusters = '+ str(n_clusters) + '_' + str(dataset)+'_n_init_'+str(n_init)+'.png')
        plt.clf()
        plt.close('images/Silhouette analysis for KMeans and EM clustering with n_clusters = '+ str(n_clusters) + '_' + str(dataset) +'_n_init_'+str(n_init)+'.png')

def dimensionality_reduction(dataset):
    df1 = pd.read_csv(dataset, index_col=0, header=0)
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1.dropna(inplace=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    names = df1.columns.values
    indexes = df1.index.values
    x = df1.iloc[:, :-1]
    y = df1.iloc[:, -1]
    x = preprocessing.OneHotEncoder().fit_transform(x).toarray()
    PCA_df =[]
    ICA_df = []
    RP_df = []
    LDA_df = []

    # # PCA ####################################################
    pca2 = PCA(n_components=2)
    principalComponents = pca2.fit(x).transform(x)

    principalDf = pd.DataFrame(data=principalComponents, columns=['component 1', 'component 2'],index=df1.index)
    finalDf = pd.concat([principalDf,y],axis=1,ignore_index=True)
    PCA_df = finalDf
    plt.figure(figsize=(7, 7))
    fig, ax = plt.subplots()
    colors = ["r", "g", "b", "k", "c", "m"]
    for i, g in enumerate(np.unique(finalDf.iloc[:, -1])):
        if i > len(colors)-1:
            continue
        ix = np.where(finalDf.iloc[:, -1] == g)
        ax.scatter(finalDf.iloc[ix][0], finalDf.iloc[ix][1], c=colors[i], label=g, s=10)
    ax.legend()
    plt.xlabel('c1')
    plt.ylabel('c2')
    plt.legend()
    plt.savefig(
        'images/PCA_' + str(
            dataset) + '.png')
    plt.clf()
    plt.close(
         'images/PCA_' + str(
            dataset) + '.png')

    # # ICA ####################################################
    ica2 = FastICA(n_components=2)
    components = ica2.fit(x).transform(x)

    Df = pd.DataFrame(data=components, columns=['component 1', 'component 2'], index=df1.index)
    finalDf = pd.concat([Df, y], axis=1, ignore_index=True)
    ICA_df = finalDf
    plt.figure(figsize=(7, 7))
    fig, ax = plt.subplots()
    colors = ["r", "g", "b", "k", "c", "m"]
    for i, g in enumerate(np.unique(finalDf.iloc[:, -1])):
        if i > len(colors)-1:
            continue
        ix = np.where(finalDf.iloc[:, -1] == g)
        ax.scatter(finalDf.iloc[ix][0], finalDf.iloc[ix][1], c=colors[i], label=g, s=10)
    ax.legend()
    plt.xlabel('c1')
    plt.ylabel('c2')
    plt.savefig(
        'images/ICA_' + str(
            dataset) + '.png')
    plt.clf()
    plt.close(
        'images/ICA_' + str(
            dataset) + '.png')
    #
    # # Randomized projection ####################################################
    rp2 = GaussianRandomProjection(n_components=2)
    components = rp2.fit(x).transform(x)

    Df = pd.DataFrame(data=components, columns=['component 1', 'component 2'], index=df1.index)
    finalDf = pd.concat([Df, y], axis=1, ignore_index=True)
    RP_df = finalDf
    plt.figure(figsize=(7, 7))
    fig, ax = plt.subplots()
    colors = ["r", "g", "b", "k", "c", "m"]
    for i, g in enumerate(np.unique(finalDf.iloc[:, -1])):
        if i > len(colors)-1:
            continue
        ix = np.where(finalDf.iloc[:, -1] == g)
        ax.scatter(finalDf.iloc[ix][0], finalDf.iloc[ix][1], c=colors[i], label=g, s=10)
    ax.legend()
    plt.xlabel('c1')
    plt.ylabel('c2')
    plt.legend()
    plt.savefig(
        'images/RandomProj_' + str(
            dataset) + '.png')

    plt.clf()
    plt.close(
        'images/RandomProj_' + str(
            dataset) + '.png')

    # # LDA ####################################################
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    components = lda.fit_transform(x, y)

    Df = pd.DataFrame(data=components, columns=['component 1', 'component 2'], index=df1.index)
    finalDf = pd.concat([Df, y], axis=1, ignore_index=True)
    LDA_df = finalDf
    plt.figure(figsize=(7, 7))
    fig, ax = plt.subplots()
    colors = ["r","g","b","k","c","m"]
    for i,g in enumerate( np.unique(finalDf.iloc[:, -1])):
        if i > len(colors)-1:
            continue
        ix = np.where(finalDf.iloc[:, -1] == g)
        ax.scatter(finalDf.iloc[ix][0], finalDf.iloc[ix][1], c=colors[i], label=g, s=10)
    ax.legend()
    plt.xlabel('c1')
    plt.ylabel('c2')
    plt.legend()
    plt.savefig(
        'images/lda_' + str(
            dataset) + '.png')
    plt.clf()
    plt.close(
        'images/lda_' + str(
            dataset) + '.png')
    return PCA_df,ICA_df,RP_df,LDA_df


def clustering_for_reduced_features(name, df1 , n_init):

    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1.dropna(inplace=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    names = df1.columns.values
    indexes = df1.index.values
    x = df1.iloc[:,:-1]
    y = df1.iloc[:,-1]
    x = preprocessing.OneHotEncoder().fit_transform(x).toarray()


    ############################
    # K-Means & EM #############
    ############################
    cl_size = [2,5,10,20,50,100]
    times_km = []
    homog_km = []
    sils_km = []
    times_em = []
    homog_em = []
    sils_em = []
    mutinfo_km =[]
    mutinfo_em =[]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    fig.set_size_inches(18, 7)

    for n_clusters in cl_size:

        start = time.time()
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(x)
        silhouette_avg = silhouette_score(x, cluster_labels,metric='manhattan')
        sils_km.append(silhouette_avg)
        times_km.append(time.time()-start)
        homog_km.append(homogeneity_score(cluster_labels,y))
        mutinfo_km.append(mutual_info_score(cluster_labels, y))

        start = time.time()
        clusterer = GaussianMixture(n_components =n_clusters, random_state=10,  n_init=n_init)
        cluster_labels = clusterer.fit_predict(x)
        silhouette_avg = silhouette_score(x, cluster_labels, metric='manhattan')
        sils_em.append(silhouette_avg)
        times_em.append(time.time() - start)
        homog_em.append(homogeneity_score(cluster_labels, y))
        mutinfo_em.append(mutual_info_score(cluster_labels, y))




    ax1.set_title("The silhouette score")
    ax1.set_xlabel("The silhouette score")
    ax1.set_ylabel("Number of clusters")
    ax1.plot(sils_km,cl_size, label="k-means", color="red", linestyle="--")
    ax1.plot(sils_em,cl_size, label="EM", color="blue", linestyle="--")

    ax2.set_title("The homogenity score")
    ax2.set_xlabel("The homogentiy score")
    ax2.set_ylabel("Number of clusters")
    ax2.plot(homog_km,cl_size, label="k-means", color="red", linestyle="--")
    ax2.plot(homog_em,cl_size, label="EM", color="blue", linestyle="--")

    ax4.set_title("The mutual info score")
    ax4.set_xlabel("The mutual info score")
    ax4.set_ylabel("Number of clusters")
    ax4.plot(mutinfo_km,cl_size, label="k-means", color="red", linestyle="--")
    ax4.plot(mutinfo_em,cl_size, label="EM", color="blue", linestyle="--")

    ax3.set_title("Time")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Number of clusters")
    ax3.plot(times_km,cl_size, label="k-means", color="red", linestyle="--")
    ax3.plot(times_em,cl_size, label="EM", color="blue", linestyle="--")

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')

    plt.suptitle(
        "KMeans vs EM clustering - How metrics change with different clusters",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig('images/UL_reduced_features_with_'+name+'_'+str(dataset)+'_n_init_'+str(n_init)+'.png')
    plt.clf()
    plt.close('images/UL_reduced_features_with_'+name+'_'+str(dataset)+'_n_init_'+str(n_init)+'.png')

    # ######################################################################
    # Silhouette analysis ##################################################
    # ######################################################################
    # K means **************************************************************
    for n_clusters in cl_size:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        X = df1.iloc[:,0:2]

        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):

            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

        ax2.scatter(
            X.iloc[:, 0], X.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig('images/Reduced_dim_with_'+name+'_'+'_Silhouette analysis for KMeans clustering with n_clusters = '+ str(n_clusters) + '_' + str(dataset)+'_n_init_'+str(n_init)+'.png')
        plt.clf()
        plt.close('images/Reduced_dim_with_'+name+'_'+'_Silhouette analysis for KMeans clustering with n_clusters = '+ str(n_clusters) + '_' + str(dataset) +'_n_init_'+str(n_init)+'.png')


    for n_clusters in cl_size:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        X = df1.iloc[:,0:2]

        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = GaussianMixture(n_components
                                    =n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):

            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

        ax2.scatter(
            X.iloc[:, 0], X.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.means_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for EM clustering with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig('images/Reduced_dim_with_'+name+'_'+'_Silhouette analysis for EM clustering with n_clusters = '+ str(n_clusters) + '_' + str(dataset)+'_n_init_'+str(n_init)+'.png')
        plt.clf()
        plt.close('images/Reduced_dim_with_'+name+'_'+'_Silhouette analysis for EM clustering with n_clusters = '+ str(n_clusters) + '_' + str(dataset) +'_n_init_'+str(n_init)+'.png')

def neural_network(dataset):
    df1 = pd.read_csv(dataset, index_col=0, header=0)
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1.dropna(inplace=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    names = df1.columns.values
    indexes = df1.index.values
    x = df1.iloc[:, :-1]
    y = df1.iloc[:, -1]
    x = preprocessing.OneHotEncoder().fit_transform(x).toarray()

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 5), random_state=34, activation='relu')

    plot_learning_curve(
        clf,
        "Learning curve NLP - before applying dimensionality reduction",
        x,
        y,
        n_jobs=-1,
        scoring="accuracy",
    )
    plt.savefig('images/MLP_before_Learning_curve.png')
    plt.clf()
    plt.close('images/MLP_before_Learning_curve.png')


    pca = PCA(n_components=2)
    x_pca = pca.fit(x).transform(x)
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 5), random_state=34, activation='relu')

    plot_learning_curve(
        clf,
        "Learning curve NLP - after applying PCA",
        x_pca,
        y,
        n_jobs=-1,
        scoring="accuracy",
    )
    plt.savefig('images/MLP_after_PCA_Learning_curve.png')
    plt.clf()
    plt.close('images/MLP_after_PCA_Learning_curve.png')

    ica = FastICA(n_components=2)
    x_ica = ica.fit(x).transform(x)
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 5), random_state=34, activation='relu')

    plot_learning_curve(
        clf,
        "Learning curve NLP - after applying ICA",
        x_ica,
        y,
        n_jobs=-1,
        scoring="accuracy",
    )
    plt.savefig('images/MLP_after_ICA_Learning_curve.png')
    plt.clf()
    plt.close('images/MLP_after_ICA_Learning_curve.png')

    rp = GaussianRandomProjection(n_components=2)
    x_rp = rp.fit(x).transform(x)
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 5), random_state=34, activation='relu')

    plot_learning_curve(
        clf,
        "Learning curve NLP - after applying RP",
        x_rp,
        y,
        n_jobs=-1,
        scoring="accuracy",
    )
    plt.savefig('images/MLP_after_RP_Learning_curve.png')
    plt.clf()
    plt.close('images/MLP_after_RP_Learning_curve.png')

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    x_lda = lda.fit_transform(x, y)
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 5), random_state=34, activation='relu')

    plot_learning_curve(
        clf,
        "Learning curve NLP - after applying LDA",
        x_lda,
        y,
        n_jobs=-1,
        scoring="accuracy",
    )

    plt.savefig('images/MLP_after_LDA_Learning_curve.png')
    plt.clf()
    plt.close('images/MLP_after_LDA_Learning_curve.png')

    clusterer = KMeans(n_clusters=10, random_state=10)
    x_cl = clusterer.fit_transform(x_pca)
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 5), random_state=34, activation='relu')

    plot_learning_curve(
        clf,
        "Learning curve NLP - after applying K Means and PCA",
        x_cl,
        y,
        n_jobs=-1,
        scoring="accuracy",
    )
    plt.savefig('images/MLP_after_KMEANS and PCA_Learning_curve.png')
    plt.clf()
    plt.close('images/MLP_after_KMEANS and PCA_Learning_curve.png')

    x_cl = clusterer.fit_transform(x_ica)
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 5), random_state=34, activation='relu')

    plot_learning_curve(
        clf,
        "Learning curve NLP - after applying K Means and ICA",
        x_cl,
        y,
        n_jobs=-1,
        scoring="accuracy",
    )
    plt.savefig('images/MLP_after_KMEANS and ICA_Learning_curve.png')
    plt.clf()
    plt.close('images/MLP_after_KMEANS and ICA_Learning_curve.png')

    x_cl = clusterer.fit_transform(x_rp)
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 5), random_state=34, activation='relu')

    plot_learning_curve(
        clf,
        "Learning curve NLP - after applying K Means and RP",
        x_cl,
        y,
        n_jobs=-1,
        scoring="accuracy",
    )
    plt.savefig('images/MLP_after_KMEANS and RP_Learning_curve.png')
    plt.clf()
    plt.close('images/MLP_after_KMEANS and RP_Learning_curve.png')

    x_cl = clusterer.fit_transform(x_lda)
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 5), random_state=34, activation='relu')

    plot_learning_curve(
        clf,
        "Learning curve NLP - after applying K Means and LDA",
        x_cl,
        y,
        n_jobs=-1,
        scoring="accuracy",
    )
    plt.savefig('images/MLP_after_KMEANS and LDA_Learning_curve.png')
    plt.clf()
    plt.close('images/MLP_after_KMEANS and LDA_Learning_curve.png')

def kurtosis_ana(dataset):
    import seaborn as sns
    from scipy.stats import kurtosis
    df1 = pd.read_csv(dataset, index_col=0, header=0)
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1.dropna(inplace=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    names = df1.columns.values
    indexes = df1.index.values
    x = df1.iloc[:, :-1]
    y = df1.iloc[:, -1]
    X = preprocessing.OneHotEncoder().fit_transform(x).toarray()


    x_ica = FastICA(n_components=100).fit_transform(X)
    print(kurtosis(np.mean(x_ica)))


def PCA_eigen(dataset):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df1 = pd.read_csv(dataset, index_col=0, header=0)
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1.dropna(inplace=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    names = df1.columns.values
    indexes = df1.index.values
    x = df1.iloc[:, :-1]
    y = df1.iloc[:, -1]
    x = preprocessing.OneHotEncoder().fit_transform(x).toarray()
    n_samples = x.shape[0]

    pca = PCA()
    pca.fit_transform(x)
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('images/PCA_variance.png')
    plt.clf()
    plt.close('images/PCA_variance.png')

    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('images/PCA_cumsum.png')
    plt.clf()
    plt.close('images/PCA_cumsum.png')

if __name__ == '__main__':
    # dataset = 'data2.csv'
    # n_init = 1
    # # clustering(dataset,n_init)
    # # dimensionality_reduction(dataset)
    # PCA_df_d2, ICA_df_d2, RP_df_d2, LDA_df_d2 = dimensionality_reduction(dataset)
    # clustering_for_reduced_features("PCA_data2_", PCA_df_d2, n_init)
    # clustering_for_reduced_features("ICA_data2_", ICA_df_d2, n_init)
    # clustering_for_reduced_features("RP_data2_", RP_df_d2, n_init)
    # clustering_for_reduced_features("LDA_data2_", LDA_df_d2, n_init)

    # dataset = 'data3.csv'
    # n_init = 1
    # # clustering(dataset, n_init)
    # # dimensionality_reduction(dataset)
    # PCA_df_d3, ICA_df_d3, RP_df_d3, LDA_df_d3 = dimensionality_reduction(dataset)
    # clustering_for_reduced_features("PCA_data3_", PCA_df_d3, n_init)
    # clustering_for_reduced_features("ICA_data3_", ICA_df_d3, n_init)
    # clustering_for_reduced_features("RP_data3_", RP_df_d3, n_init)
    # clustering_for_reduced_features("LDA_data3_", LDA_df_d3, n_init)

    # dataset = 'data2.csv'
    # n_init = 10
    # # clustering(dataset, n_init)
    # PCA_df_d2,ICA_df_d2,RP_df_d2,LDA_df_d2= dimensionality_reduction(dataset)
    # clustering_for_reduced_features("PCA_data2_", PCA_df_d2, n_init)
    # clustering_for_reduced_features("ICA_data2_",ICA_df_d2,n_init)
    # clustering_for_reduced_features("RP_data2_", RP_df_d2, n_init)
    # clustering_for_reduced_features("LDA_data2_", LDA_df_d2, n_init)

    # dataset = 'data3.csv'
    # n_init = 10
    # # clustering(dataset, n_init)
    # # dimensionality_reduction(dataset)
    # PCA_df_d3, ICA_df_d3, RP_df_d3, LDA_df_d3 = dimensionality_reduction(dataset)
    # clustering_for_reduced_features("PCA_data3_", PCA_df_d3, n_init)
    # clustering_for_reduced_features("ICA_data3_", ICA_df_d3, n_init)
    # clustering_for_reduced_features("RP_data3_", RP_df_d3, n_init)
    # clustering_for_reduced_features("LDA_data3_", LDA_df_d3, n_init)

    dataset = 'data2.csv'
    kurtosis_ana(dataset)
    # PCA_eigen(dataset)

    # dataset = 'data2.csv'
    # neural_network(dataset)