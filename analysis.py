import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn.cluster
from sklearn.cluster import KMeans


def plot_pca(sp, demo, pca_comp, kmeans_comp):
    # arrange demographic
    d = pd.DataFrame(columns=['Subject number', demo.columns[1]])
    for row in range(len(sp)):
        sn = sp.iloc[row].at['Subject number']
        dem = int((demo.iloc[demo.index[demo['Subject number']==sn],1]).values)
        d.loc[len(d.index)] = [sn, dem]

    # Standardize the Data
    features = []
    for col in sp.columns: features.append(col)
    features.remove(features[len(features)-1])
    features.remove(features[0])
    x = sp.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    if pca_comp == 2:
        # PCA Projection to 2D
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, d[[demo.columns[1]]]], axis=1)
        #
        # # Visualize 2D Projection
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(1, 1, 1)
        # ax.set_xlabel('Principal Component 1', fontsize=15)
        # ax.set_ylabel('Principal Component 2', fontsize=15)
        # ax.set_title('2 Component PCA', fontsize=20)
        #
        # targets = [0, 1]
        # colors = ['r', 'b']
        # for target, color in zip(targets, colors):
        #     indicesToKeep = finalDf[demo.columns[1]] == target
        #     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
        #                , finalDf.loc[indicesToKeep, 'principal component 2']
        #                , c=color
        #                , s=50)
        # ax.legend(targets)
        # ax.grid()
        # plt.show()
        print(pca.explained_variance_ratio_)

        if kmeans_comp==2:
            model = KMeans(n_clusters=2)
            df = principalDf.to_numpy()
            label = model.fit(df)
            labels = pd.DataFrame(data=label.labels_, columns=['label'])
            df2 = pd.concat([principalDf, labels], axis=1)

            # Visualize 2D Projection
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            ax.set_title('2 Component PCA, 2 Component KMeans', fontsize=20)

            targets = [0, 1]
            colors = ['r', 'b']
            for target, color in zip(targets, colors):
                indicesToKeep = df2['label'] == target
                ax.scatter(df2.loc[indicesToKeep, 'principal component 1']
                           , df2.loc[indicesToKeep, 'principal component 2']
                           , c=color
                           , s=50)
            ax.legend(targets)
            ax.grid()
            plt.show()

        if kmeans_comp==3:
            model = KMeans(n_clusters=3)
            df = principalDf.to_numpy()
            label = model.fit(df)
            labels = pd.DataFrame(data=label.labels_, columns=['label'])
            df2 = pd.concat([principalDf, labels], axis=1)

            # Visualize 2D Projection
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            ax.set_title('2 Component PCA, 3 Component KMeans', fontsize=20)

            targets = [0, 1, 2]
            colors = ['r', 'b', 'g']
            for target, color in zip(targets, colors):
                indicesToKeep = df2['label'] == target
                ax.scatter(df2.loc[indicesToKeep, 'principal component 1']
                           , df2.loc[indicesToKeep, 'principal component 2']
                           , c=color
                           , s=50)
            ax.legend(targets)
            ax.grid()
            plt.show()

    if pca_comp == 3:
        # PCA Projection to 3D
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2','principal component 3'])
        finalDf = pd.concat([principalDf, d[[demo.columns[1]]]], axis=1)

        # # Visualize 3D Projection
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(projection='3d')
        # ax.set_xlabel('Principal Component 1', fontsize=15)
        # ax.set_ylabel('Principal Component 2', fontsize=15)
        # ax.set_ylabel('Principal Component 3', fontsize=15)
        # ax.set_title('3 Component PCA', fontsize=20)
        #
        # targets = [0, 1]
        # colors = ['r', 'b']
        # for target, color in zip(targets, colors):
        #     indicesToKeep = finalDf[demo.columns[1]] == target
        #     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
        #                , finalDf.loc[indicesToKeep, 'principal component 2']
        #                , finalDf.loc[indicesToKeep, 'principal component 3']
        #                , c=color)
        # ax.legend(targets)
        # ax.grid()
        # plt.show()
        print(pca.explained_variance_ratio_)

        if kmeans_comp==2:
            model = KMeans(n_clusters=2)
            df = principalDf.to_numpy()
            label = model.fit(df)
            labels = pd.DataFrame(data=label.labels_, columns=['label'])
            df2 = pd.concat([principalDf, labels], axis=1)

            # Visualize 3D Projection
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            ax.set_ylabel('Principal Component 3', fontsize=15)
            ax.set_title('3 Component PCA, 2 Component KMeans', fontsize=20)

            targets = [0, 1]
            colors = ['r', 'b']
            for target, color in zip(targets, colors):
                indicesToKeep = df2['label'] == target
                ax.scatter(df2.loc[indicesToKeep, 'principal component 1']
                           , df2.loc[indicesToKeep, 'principal component 2']
                           , df2.loc[indicesToKeep, 'principal component 3']
                           , c=color
                           , s=50)
            ax.legend(targets)
            ax.grid()
            plt.show()

        if kmeans_comp==3:
            model = KMeans(n_clusters=3)
            df = principalDf.to_numpy()
            label = model.fit(df)
            labels = pd.DataFrame(data=label.labels_, columns=['label'])
            df2 = pd.concat([principalDf, labels], axis=1)

            # Visualize 3D Projection
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            ax.set_ylabel('Principal Component 3', fontsize=15)
            ax.set_title('3 Component PCA, 2 Component KMeans', fontsize=20)

            targets = [0, 1, 2]
            colors = ['r', 'b', 'g']
            for target, color in zip(targets, colors):
                indicesToKeep = df2['label'] == target
                ax.scatter(df2.loc[indicesToKeep, 'principal component 1']
                           , df2.loc[indicesToKeep, 'principal component 2']
                           , df2.loc[indicesToKeep, 'principal component 3']
                           , c=color
                           , s=50)
            ax.legend(targets)
            ax.grid()
            plt.show()