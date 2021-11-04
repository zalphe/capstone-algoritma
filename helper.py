import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.decomposition import PCA
from numpy.linalg import eig
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import importlib

plt.style.use('seaborn')

def biplot_pca (data):
    """
    Function to perform plots with first 2 PC from PCA on selected data
    Data needs to be scales for the better result
    
    data = selected dataframe
    """
    
    pca = PCA()
    pca.fit(data.values)
    eig_vec_pc1 = pca.components_[0]
    eig_vec_pc2 = pca.components_[1]
    value_pc1 = pca.transform(data)[:,0]
    value_pc2 = pca.transform(data)[:,1]
    for i in range(len(eig_vec_pc1)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
        plt.arrow(0, 0, eig_vec_pc1[i]*max(value_pc1), eig_vec_pc2[i]*max(value_pc2),
                  color='yellow', width=0.0005, head_width=0.0025)
        plt.text(eig_vec_pc1[i]*max(value_pc1)*1.2, eig_vec_pc2[i]*max(value_pc2)*1.2,
                 list(data.columns.values)[i], color='magenta')

    for i in range(len(value_pc1)):
    # circles project documents (ie rows from csv) as points onto PC axes
        plt.scatter(value_pc1[i], value_pc2[i], c='grey')
        plt.text(value_pc1[i]*1.2, value_pc2[i]*1.2, list(data.index)[i], color='brown')
    plt.title('Biplot PCA', fontsize=20)
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    
    return plt.show()

def plot_silscore(data, kmax=10):
    """
    function for plotting silhouette score result,
    perform better with scaled data;
    
    data = dataframe selected;
    kmax = int, default=10;
    """
    np.random.seed(2102)
    sil = []
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2,kmax+1):
        kmeansx = KMeans(n_clusters = k).fit(data)
        labels = kmeansx.labels_
        sil.append(silhouette_score(data, labels, metric = 'euclidean', random_state=0))
    
    plt.plot(list(range(2,kmax+1)), sil)
    plt.title('Silhouette Score', fontsize=20)
    plt.xlabel("Number of cluster (K)", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)
    
    return plt.show()

def plot_elbow(data, kmax=10):
    """
    function for plotting silhouette score result,
    perform better with scaled data;
    
    data = dataframe selected;
    kmax = int, default=10;
    """
    np.random.seed(121)
    wss = []
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2,kmax+1):
        kmeansx = KMeans(n_clusters = k).fit(data)
        wss_iter = kmeansx.inertia_
        wss.append(wss_iter)
    
    plt.plot(list(range(2,kmax+1)), wss)
    plt.title('Elbow Method with WSS', fontsize=20)
    plt.xlabel("Number of cluster (K)", fontsize=14)
    plt.ylabel("Total Within Sum of Square", fontsize=14)
    
    return plt.show()

def biplot_kmeans(data, k, feature_name=False):
    """
    Function to perform biplots for kmeans;
    
    data = selected dataframe, pandas.dataframe or numpy.ndarray;
    K = number of cluster, int;
    feature_name = option to show feature names and its arrow, bool, default=False;
    """
    
    x = np.arange(k)
    ys = [i+x+(i*x)**2 for i in range(k)]
    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    rainbow = [colors.rgb2hex(i) for i in colors_array]
    
    pca = PCA()
    try:
        pca.fit(data.values)
    except:
        pca.fit(data)
    eig_vec_pc1 = pca.components_[0]
    eig_vec_pc2 = pca.components_[1]
    transformed_data = pca.transform(data)
    value_pc1 = transformed_data[:,0]
    value_pc2 = transformed_data[:,1]
    kmeansx = KMeans(n_clusters = k).fit(transformed_data)
    label = list(kmeansx.labels_)
    u_labels = np.unique(label)
    if feature_name:        
        for i in range(len(eig_vec_pc1)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
            plt.arrow(0, 0, eig_vec_pc1[i]*max(value_pc1), eig_vec_pc2[i]*max(value_pc2),
                      color='yellow', width=0.0005, head_width=0.0025)
            plt.text(eig_vec_pc1[i]*max(value_pc1)*1.2, eig_vec_pc2[i]*max(value_pc2)*1.2,
                     list(data.columns.values)[i], color='magenta')

    #for i in range(len(value_pc1)):
    # circles project documents (ie rows from csv) as points onto PC axes
        #plt.scatter(value_pc1[i], value_pc2[i], c=rainbow[label[i]], label=label[i])
        #plt.text(value_pc1[i]*1.2, value_pc2[i]*1.2, list(data.index)[i], color='brown')
    for i in u_labels:
       plt.scatter(transformed_data[label == i , 0] , transformed_data[label == i , 1] , c = rainbow[i], label=i)
  
    plt.title(f'Biplot KMeans, number of cluster = {k}', fontsize=20)
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.legend()
    return plt.show()

def plot_cluster_radar(data, cluster_label, features=None, center=np.mean, ax=None, figsize=(8,8), legend_loc=(1.3,0.9), labels=None): 
    if data.__class__.__name__ == 'DataFrame':
        hasClusterInDF = cluster_label.__class__.__name__ == 'str'
        noFeaturesGiven = features.__class__.__name__ == 'NoneType'
        
        if noFeaturesGiven:
            features = data.columns.drop(cluster_label).tolist() if hasClusterInDF else data.columns.tolist()
                
        if hasClusterInDF:
            cluster_label = data[cluster_label]
    
        data = data[features].values
    
    if ax.__class__.__name__ == 'NoneType':
        fig, ax = plt.subplots(1, figsize=figsize, subplot_kw={'projection': 'polar'})
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False) # Set the angle
    angles = np.concatenate((angles,[angles[0]]))
    ax.grid(True)
    ax.set_thetagrids(angles * 180/np.pi, features)
    
    for i, group in enumerate(np.unique(cluster_label)):
        indices = np.where(cluster_label==group)
        
        if labels.__class__.__name__ != 'NoneType':
            group = labels[i]
        
        stats = center(data[indices], axis=0)
        stats = np.concatenate((stats, [stats[0]]))
        ax.plot(angles, stats, linewidth=1, linestyle='solid', label='{}: {}'.format(group, indices[0].shape[0]))
        ax.fill(angles, stats, alpha=0.1)

    n_clusters = np.unique(cluster_label)
    ax.set_title('{} clusters'.format(n_clusters[n_clusters!=-1].shape[0]))
    ax.legend(loc='upper right', bbox_to_anchor=legend_loc)
    return plt.show()

def BCSS(X, kmeans):
    _, label_counts = np.unique(kmeans.labels_, return_counts = True)
    diff_cluster_sq = np.linalg.norm(kmeans.cluster_centers_ - np.mean(X, axis = 0), axis = 1)**2
    return sum(label_counts * diff_cluster_sq)