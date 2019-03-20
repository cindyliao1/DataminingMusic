from Library import Library
import numpy as np


def main():
    library = Library(10, 4)
    playlist = library.make_playlist(3)
    playlist_matrix = library.create_song_matrix(playlist)
    suggestion = library.suggest_song(playlist_matrix)


