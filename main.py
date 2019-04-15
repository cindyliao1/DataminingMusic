from Library import Library
import numpy as np
# import data_reader as DataReader
from data_reader import DataReader

# path = "C:\\tonga\\Documents\\School\\Spring 2019\\Data mining\\millionsongsubset_full\\MillionSongSubset\\data"

def main():
    print "in main"
    # dr = DataReader()
    # dr.root_retrieve(path="Data")
    # dr.read(path="Data")
    library = Library(9000, 3, "library_csv.csv")
    library.pca_plot(False)
    library.find_num_cluster()
    # playlist = library.make_playlist(3)
    # # print playlist
    # playlist_matrix = library.create_song_matrix(playlist)
    # suggestion = library.suggest_song(playlist_matrix)
    # print suggestion.title


if __name__ == "__main__":
    main()

