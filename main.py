from Library import Library
import numpy as np
# import data_reader as DataReader
from data_reader import DataReader

# path = "C:\\tonga\\Documents\\School\\Spring 2019\\Data mining\\millionsongsubset_full\\MillionSongSubset\\data"
need_files = False
def main():
    print "begin"
    if need_files:
        print 'reading files'
        dr = DataReader()
        dr.root_retrieve(path="Data")
        dr.read(path="Data")

    print 'finding number of clusters'
    library = Library(10, 3, "library_csv.csv")
    library.pca_plot(False)
    library.find_num_cluster()

    print 'make k-grams'
    library.k_cluster(3, False)

    print "suggest song"
    playlist, titles = library.make_playlist(3, 1)
    print titles
    # playlist_matr ix = library.create_song_matrix(playlist)
    # suggestion = library.suggest_song(playlist_matrix)
    # print suggestion.title


if __name__ == "__main__":
    main()

