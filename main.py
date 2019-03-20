from Library import Library
import numpy as np
# import data_reader as DataReader
from data_reader import DataReader


def main():
    print "in main"
    dr = DataReader()
    dr.read(path="Data")
    library = Library(10, 4)
    playlist = library.make_playlist(3)
    playlist_matrix = library.create_song_matrix(playlist)
    suggestion = library.suggest_song(playlist_matrix)

if __name__ == "__main__":
	main()

