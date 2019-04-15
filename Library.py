import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stat
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler
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
        # preprocess data
        # x, self.all_titles = self.preprocessing()
        # make library of size s
        self.size = s
        self.library = {}
        self.titles = self.populate_library()
        # print self.titles
        # self.num_cluster = c
        # self.kmeans = KMeans(init='k-means++', n_clusters=c)
        self.song_matrix = self.create_song_matrix(self.library)
        # print self.song_matrix
        # print 'after'
        self.song_matrix = self.preprocessing()
        # print self.song_matrix
        # self.clusters = defaultdict(list)
        # self.cluster_centers = self.k_cluster()

    def create_song_matrix(self, songs):
        song_matrix = []
        for song in songs.values():
            song_a = []
            song_a.append(song.loudness)
            song_a.append(song.mode)
            song_a.append(song.key)
            song_a.append(song.tempo)
            song_a.append(song.time_signature)
            # song_a[5] = song.title
            song_matrix.append(song_a)
        return song_matrix

    # def get_song_info(self, song):
    #     return song.loudness, song.max_loudness, song.timbre, song.tempo, song.time_signature

    def k_cluster(self):
        # make copy of library
        lib_copy = self.song_matrix
        center_belong = self.kmeans.fit_predict(lib_copy)
        # populate dictionary: key = center index, value = song values
        for i in range(len(center_belong)):
            self.clusters[center_belong[i]].append(i)
        print "cluster centers:", self.kmeans.cluster_centers_
        return self.kmeans.cluster_centers_

    def populate_library(self):
        titles = []
        for i in range(self.size):
            song, title = self.get_random_song()
            self.library[i] = song
            titles.append(title)
        return titles

    def make_playlist(self, size):
        playlist = {}
        for i in range(size):
            rand = random.randint(1, size + 1)
            song = self.library[rand]
            # check if song is already in playlist. If it is, pick a new one.
            while (song in playlist.values()):
                rand = random.randint(1, size + 1)
                song = self.library[rand]

            playlist[i] = song

        return playlist

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
        values = np.zeros(5)
        for category in playlist_matrix:
            # for j in range(len(playlist_matrix)):
            for i in range(len(category)):
                values[i] += category[i]

        for i in range(len(values)):
            values[i] = values[i] / song_count

        return values

    def suggest_song(self, playlist):
        playlist_center = self.calculate_center(playlist, len(playlist))
        playlist_center = playlist_center.reshape(1, -1)
        index = self.kmeans.predict(playlist_center)  # get the cluster center index playlist belongs to
        # center = self.cluster_centers[index]  # get the point of cluster center
        cluster_songs = self.clusters[index[0]]  # get the indexes of songs in the cluster
        closest_songs_dist, s_dict = self.find_closest_songs(playlist_center, cluster_songs)
        suggested_dist = closest_songs_dist[0]  # get closest distance
        suggested_song = s_dict[suggested_dist]  # get song from closest distance
        i = 1
        while suggested_song in playlist:
            suggested_song = closest_songs_dist[i]
            i += 1

        return suggested_song

    def find_closest_songs(self, playlist_center, cluster_songs):
        songs = []
        distances = []
        s_distance = {}
        for index in cluster_songs:
            song = self.library[index]
            song_a = [song.loudness, song.mode, song.key,
                      song.tempo, song.time_signature]
            sd = distance.euclidean(playlist_center, song_a)
            distances.append(sd)
            songs.append(song)

        for i in range(len(distances)):
            s_distance[distances[i]] = songs[i]  # assigning distances to their songs

        np.sort(distances)  # sort distance in ascending order
        return distances, s_distance

    def get_song_title(self, song):
        title = self.all_songs_csv[(self.all_songs_csv['Loudness'] == song.loudness) &
                                   (self.all_songs_csv['Mode'] == song.mode) &
                                   (self.all_songs_csv['Key'] == song.key) &
                                   (self.all_songs_csv['Tempo'] == song.tempo) &
                                   (self.all_songs_csv['timeSignature'] == song.time_signature), ['Title']]

        return title

    def pca_plot(self, show):
        pca = sklearnPCA(n_components=2)
        # print self.song_matrix
        transformed = pd.DataFrame(pca.fit_transform(self.song_matrix))
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
        x = StandardScaler().fit_transform(self.song_matrix)
        return x
