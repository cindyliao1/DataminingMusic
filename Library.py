import random
import numpy as np
import pandas as pd
import stat
from sklearn.cluster import KMeans
from scipy.spatial import distance
from collections import defaultdict
from data_reader import DataReader


class Song:
    def __init__(self, loudness, dance, energy, temp, time, tit):
        self.loudness = loudness
        self.danceability = dance
        self.energy = energy
        self.tempo = temp
        self.time_signature = time
        self.title = tit


class Library:
    def __init__(self, s, c, l_file):
        self.all_songs_csv = pd.read_csv(l_file)
        self.size = s
        self.library = {}
        self.populate_library()
        self.num_cluster = c
        self.kmeans = KMeans(init='k-means++', n_clusters=c)
        self.song_matrix = self.create_song_matrix(self.library)
        self.clusters = defaultdict(list)
        self.cluster_centers = self.k_cluster()

    def create_song_matrix(self, songs):
        # song_matrix = np.array()
        # loudness = []
        # max_l = []
        # timbre = []
        # tempo = []
        # time_signature = []
        #
        # for i in range(len(songs)):
        #     song = songs[i]
        #     l, ml, timb, temp, ts = self.get_song_info(song)
        #     loudness.append(l);
        #     max_l.append(ml)
        #     timbre.append(timb)
        #     tempo.append(temp)
        #     time_signature.append(ts)
        #
        # song_matrix = np.concatenate(loudness, max_l, timbre, tempo, time_signature)
        song_matrix = []
        for song in songs.values():
            song_a = []
            song_a.append(song.loudness)
            song_a.append(song.danceability)
            song_a.append(song.energy)
            song_a.append(song.tempo)
            song_a.append(song.time_signature)
            # song_a[5] = song.title
            song_matrix.append(song_a)
        return song_matrix

    def get_song_info(self, song):
        return song.loudness, song.max_loudness, song.timbre, song.tempo, song.time_signature

    def k_cluster(self):
        # make copy of library
        lib_copy = self.song_matrix
        # transform song names into dummies
        # lib_copy = pd.get_dummies(lib_copy, column=['Title'])
        # standardize
        # columns = ['Loudness', 'Danceability', 'Energy', 'Tempo', 'timeSignature', 'Title']
        # lib_copy_std = stat.zscore(lib_copy[columns])

        center_belong = self.kmeans.fit_predict(lib_copy)
        # populate dictionary: key = center index, value = song values
        for i in range(len(center_belong)):
            self.clusters[center_belong[i]].append(i)
        print self.kmeans.cluster_centers_
        return self.kmeans.cluster_centers_

    def populate_library(self):
        for i in range(self.size):
            song = self.get_random_song()
            self.library[i] = song

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
        print "LIB FILE TYPE: ", (self.all_songs_csv)
        song_index = random.randint(0, len(self.all_songs_csv))
        loudness, danceability, energy, tempo, timesignature, title = self.all_songs_csv.iloc[song_index]
        song = Song(loudness=loudness, dance=danceability, energy=energy,
                    temp=tempo, time=timesignature, tit=title)
        while song in self.library:
            song_index = random.randint(len(self.all_songs_csv))
            loudness, danceability, energy, tempo, timesignature, title = self.all_songs_csv[song_index]
            song = Song(loudness=loudness, dance=danceability, energy=energy,
                        temp=tempo, time=timesignature, tit=title)
        return song

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
        i = 1;
        while suggested_song in playlist:
            suggested_song = closest_songs_dist[i];
            i += 1

        return suggested_song

    def find_closest_songs(self, playlist_center, cluster_songs):
        songs = []
        distances = []
        s_distance = {}
        for index in cluster_songs:
            song = self.library[index]
            song_a = [song.loudness, song.danceability, song.energy,
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
                                   (self.all_songs_csv['Danceability'] == song.danceability) &
                                   (self.all_songs_csv['Energy'] == song.energy) &
                                   (self.all_songs_csv['Tempo'] == song.tempo) &
                                   (self.all_songs_csv['timeSignature'] == song.time_signature), ['Title']]

        return title
