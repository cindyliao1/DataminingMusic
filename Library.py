import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# commit testing with a comment

class Song:
    def __init__(self, loudness, max_loudness, timb, temp, time, tit):
        self.loudness = loudness
        self.max_loudness = max_loudness
        self.timbre = timb
        self.tempo = temp
        self.time_signature = time
        self.title = tit

class Library:
    def __init__(self, s, c, k):
        self.size = s
        self.library = {}
        self.populate_library()
        self.num_cluster = c
        self.song_matrix = self.create_song_matrix()
        self.clusters = self.k_cluster(k)



    def create_song_matrix(self):
        song_matrix = np.array()
        loudness = []
        max_l = []
        timbre = []
        tempo = []
        time_signature = []

        for i in range(self.size):
            song = self.library[i]
            l, ml, timb, temp, ts = self.get_song_info(song)
            loudness.append(l);
            max_l.append(ml)
            timbre.append(timb)
            tempo.append(temp)
            time_signature.append(ts)

        song_matrix = np.concatenate(loudness, max_l, timbre, tempo, time_signature)
        return song_matrix

    def get_song_info(self, song):
        return song.loudness, song.max_loudness, song.timbre, song.tempo, song.time_signature

    def k_cluster(self, clust):
        kmeans = KMeans(init='k-means++', n_clusters=clust)
        kmeans.fit(self.song_matrix)
        print(kmeans.labels_)


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
            while(song in playlist.values()):
                rand = random.randint(1, size + 1)
                song = self.library[rand]

            playlist[i] = song

        return playlist

    def get_random_song(self):
        song = 0
        return song
