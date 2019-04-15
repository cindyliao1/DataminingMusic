import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stat
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer, StandardScaler, scale
from sklearn.decomposition import PCA as sklearnPCA
from scipy.spatial import distance
from collections import defaultdict
from data_reader import DataReader


class Song:
    def __init__(self, loudness, mode, key, temp, time, tit):
        self.loudness = loudness
        self.mode = mode
        self.key = key
        self.tempo = temp
        self.time_signature = time
        self.title = tit


class Library:
    def __init__(self, s, c, l_file):
        # read csv file
        self.all_songs_csv = pd.read_csv(l_file)
        # make library of size s
        self.size = s
        self.library = {}
        self.titles = self.populate_library()
        # print self.titles
        self.num_cluster = c
        self.kmeans = KMeans(init='k-means++', n_clusters=c)
        self.song_matrix = self.create_song_matrix()
        # print self.song_matrix
        # print 'after'
        self.transformed_data = self.preprocessing()
        # print self.song_matrix
        self.clusters = defaultdict(list)
        # self.cluster_centers = self.k_cluster(c, True)

    def create_song_matrix(self):
        song_matrix = [None] * len(self.library)
        for index, song in self.library.iteritems():
            song_a = []
            song_a.append(song.loudness)
            song_a.append(song.mode)
            song_a.append(song.key)
            song_a.append(song.tempo)
            song_a.append(song.time_signature)
            # song_a[5] = song.title
            song_matrix[index] = song_a
        return song_matrix

    # def get_song_info(self, song):
    #     return song.loudness, song.max_loudness, song.timbre, song.tempo, song.time_signature

    def k_cluster(self, c, graph):
        # kmeans = KMeans(init='k-means++', n_clusters=c)
        # make copy of library
        center_belong = self.kmeans.fit_predict(self.song_matrix)
        # populate dictionary: key = center index, value = song values
        for i in range(len(center_belong)):
            self.clusters[center_belong[i]].append(self.song_matrix[i])  # key is cluster index, value is list
            # of songs
        # print "cluster centers:", self.kmeans.cluster_centers_
        # print "self.cluster:", self.clusters

        if graph:
            data = scale(self.song_matrix)
            reduced_copy = sklearnPCA(n_components=2).fit_transform(data)
            self.kmeans.fit(reduced_copy)

            h = 0.2
            x_min, x_max = reduced_copy[:, 0].min() - 1, reduced_copy[:, 0].max() + 1
            y_min, y_max = reduced_copy[:, 1].min() - 1, reduced_copy[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            # Obtain labels for each point in mesh. Use last trained model.
            Z = self.kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure(1)
            plt.clf()
            plt.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       cmap=plt.cm.Paired,
                       aspect='auto', origin='lower')

            plt.plot(reduced_copy[:, 0], reduced_copy[:, 1], 'k.', markersize=2)
            # Plot the centroids as a white X
            centroids = self.kmeans.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker='x', s=169, linewidths=3,
                        color='w', zorder=10)
            plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                      'Centroids are marked with white cross')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            plt.show()

        return self.kmeans.cluster_centers_

    def populate_library(self):
        titles = []
        for i in range(self.size):
            song, title = self.get_random_song()
            self.library[i] = song
            titles.append(title)
        return titles

    def make_playlist(self, size, index, same_cluster=True):
        start_song = self.song_matrix[index]
        song_title = self.library[index].title
        titles = []
        playlist_array = []
        playlist_array.append(start_song)
        titles.append(song_title)
        playlist = {}
        playlist[0] = start_song
        for i in range(size):
            if same_cluster:
                curr_cluster = self.current_cluster(start_song)
                print curr_cluster
                suggestions, s_titles = self.suggest_song(playlist_array, same_cluster, size)
                for song, title in zip(suggestions, s_titles):
                    playlist_array.append(song)
                    titles.append(title)
                return playlist_array, titles
            else:
                suggestion, title = self.suggest_song(playlist_array, same_cluster, size)
                titles.append(title)
            return playlist, titles

    def current_cluster(self, song_data):
        for cluster, songs in self.clusters.iteritems():
            if song_data in songs:
                return cluster

    def get_random_song(self):
        # print "LIB FILE TYPE: ", (self.all_songs_csv)
        song_index = random.randint(0, len(self.all_songs_csv))
        loudness, mode, key, tempo, timesignature, title = self.all_songs_csv.iloc[song_index]
        song = Song(loudness=loudness, key=key, mode=mode,
                    temp=tempo, time=timesignature, tit=title)
        while song in self.library:
            song_index = random.randint(len(self.all_songs_csv))
            loudness, mode, key, tempo, timesignature, title = self.all_songs_csv[song_index]
            song = Song(loudness=loudness, key=key, mode=mode,
                        temp=tempo, time=timesignature, tit=title)
        title = song.title
        return song, title

    def calculate_center(self, playlist_matrix, song_count):
        if song_count == 1:
            return playlist_matrix
        values = np.zeros(5)
        for category in playlist_matrix:
            # for j in range(len(playlist_matrix)):
            for i in range(len(category)):
                values[i] += category[i]

        for i in range(len(values)):
            values[i] = values[i] / song_count

        return values

    def suggest_song(self, playlist, samecluster, size):
        playlist_center = self.calculate_center(playlist, len(playlist))
        # playlist_center = playlist_center.reshape(1, -1)
        if samecluster:
            index = self.kmeans.predict(playlist_center)  # get the cluster center index playlist belongs to
            cluster_songs = self.clusters[index[0]]  # get the indexes of songs in the cluster
            closest_songs_dist, s_dict = self.find_closest_songs(playlist_center, cluster_songs, size, samecluster)
            new_playlist = []
            titles = []
            for dist in closest_songs_dist:
                s = s_dict[dist]
                new_playlist.append(s)
                titles.append(self.get_song_title(s))
            return new_playlist, titles
        else:
            index = self.kmeans.predict(playlist_center)  # get the cluster center index playlist belongs to
            cluster_songs = self.clusters[index[0]]
            closest_songs_dist, s_dict = self.find_closest_songs(playlist_center, cluster_songs, size, samecluster)
            suggested_dist = closest_songs_dist[0]  # get closest distance
            suggested_song = s_dict[suggested_dist]  # get song from closest distance
            i = 1
            while suggested_song in playlist:
                suggested_song = closest_songs_dist[i]
                i += 1

            return suggested_song, self.get_song_title(suggested_song)

    def find_closest_songs(self, playlist_center, cluster_songs, size, samecluster):
        songs = []
        distances = []
        s_distance = {}
        for song in cluster_songs:
            sd = distance.euclidean(playlist_center, song)
            if sd == 0:
                continue
            distances.append(sd)
            songs.append(song)

        for i in range(len(distances)):
            s_distance[distances[i]] = songs[i]  # assigning distances to their songs

        np.sort(distances)  # sort distance in ascending order
        if samecluster & size <= len(distances):
            return distances[0:size], s_distance

        return distances, s_distance
        # closest = s_distance[distances[0]]
        # return closest, self.get_song_title(closest)

    def get_song_title(self, song):
        # title = self.all_songs_csv[(self.all_songs_csv['Loudness'] == song.loudness) &
        #                            (self.all_songs_csv['Mode'] == song.mode) &
        #                            (self.all_songs_csv['Key'] == song.key) &
        #                            (self.all_songs_csv['Tempo'] == song.tempo) &
        #                            (self.all_songs_csv['timeSignature'] == song.time_signature), ['Title']]
        s = self.song_matrix.index(song)
        title = self.titles[s]

        return title

    def pca_plot(self, show):
        pca = sklearnPCA(n_components=2)
        # print self.song_matrix
        transformed = pd.DataFrame(self.transformed_data)
        # print transformed
        # print transformed[y == 1][0], transformed[y == 1][1]
        if show:
            for point in transformed.values:
                # print point
                plt.scatter(point[0], point[1])

            plt.legend()
            plt.show()

            # display
            plt.show()

        return transformed

    def find_num_cluster(self):
        pca = sklearnPCA().fit(self.song_matrix)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

    def preprocessing(self):
        copy = self.song_matrix
        x = Normalizer().fit_transform(copy)
        return x
