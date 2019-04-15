import hdf5_getters
from pathlib import Path
# class DataReader:
import os
import sys
import hdf5_getters as hdf5
import numpy as np


class DataReader:
    def __init__(self):
        hi = ""

    def die_with_usage(self):
        """ HELP MENU """
        print 'display_song.py'
        print 'T. Bertin-Mahieux (2010) tb2332@columbia.edu'
        print 'to quickly display all we know about a song'
        print 'usage:'
        print '   python display_song.py [FLAGS] <HDF5 file> <OPT: song idx> <OPT: getter>'
        print 'example:'
        print '   python display_song.py mysong.h5 0 danceability'
        print 'INPUTS'
        print '   <HDF5 file>  - any song / aggregate /summary file'
        print '   <song idx>   - if file contains many songs, specify one'
        print '                  starting at 0 (OPTIONAL)'
        print '   <getter>     - if you want only one field, you can specify it'
        print '                  e.g. "get_artist_name" or "artist_name" (OPTIONAL)'
        print 'FLAGS'
        print '   -summary     - if you use a file that does not have all fields,'
        print '                  use this flag. If not, you might get an error!'
        print '                  Specifically desgin to display summary files'
        sys.exit(0)

    # if __name__ == '__main__':
    #     """ MAIN """

    # help menu
    # if len(sys.argv) < 2:
    #     die_with_usage()
    # @staticmethod
    def read(self, path):
        files = os.listdir(path)
        import csv
        with open('library_csv.csv', 'w') as library_csv:
            writer = csv.writer(library_csv)
            writer.writerow(['Loudness', 'Danceability', 'Energy', 'Tempo', 'timeSignature', 'Title'])
            # get params
            for filename in files:
                # hdf5path = filename
                hdf5path = "Data/" + filename
                # hdf5path.replace("'","",2)
                # sanity check
                if not os.path.isfile(hdf5path):
                    print 'ERROR: file', hdf5path, 'does not exist.'
                    sys.exit(0)
                h5 = hdf5_getters.open_h5_file_read(hdf5path)
                # get all getters
                loudness = hdf5.get_loudness(h5)
                dance = hdf5.get_danceability(h5)
                energy = hdf5.get_energy(h5)
                tempo = hdf5.get_tempo(h5)
                ts = hdf5.get_time_signature(h5)
                title = hdf5.get_title(h5)

                writer.writerow([loudness, dance, energy, tempo, ts, title])
                # print them
                h5.close()
        library_csv.close()

    def root_retrieve(self, path):
        #r=root, d=directory, f=file
        import csv
        with open('library_csv.csv', 'w') as library_csv:
            writer = csv.writer(library_csv)
            writer.writerow(['Loudness', 'Key', 'Mode', 'Tempo', 'timeSignature', 'Title'])
            for r, d, f in os.walk(path):
                for song in f:
                    hdf5path = os.path.join(r, song)
                    h5 = hdf5_getters.open_h5_file_read(hdf5path)
                    # get all getters
                    loudness = hdf5.get_loudness(h5)
                    key = hdf5.get_key(h5)
                    mode = hdf5.get_mode(h5)
                    tempo = hdf5.get_tempo(h5)
                    ts = hdf5.get_time_signature(h5)
                    title = hdf5.get_title(h5)

                    writer.writerow([loudness, key, mode, tempo, ts, title])
                    # print them
                    h5.close()
            library_csv.close()
        print ("done")
