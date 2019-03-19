import random
import numpy as np

class Song:
    def __init__(self, loudness, max_loudness, timb, temp, time, tit):
        self.loudness = loudness
        self.max_loudness = max_loudness
        self.timbre = timb
        self.tempo = temp
        self.time_signature = time
        self.title = tit

class Library:
    def __init__(self, s, c):
        self.size = s
        self.library = {}
        self.populate_library()
        self.num_cluster = c
        self.song_matrix = self.create_song_matrix()
        self.clusters = self.cluster()



    def create_song_matrix(self):
        song_matrix = [][]

    def cluster(self):
        asdlf


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

