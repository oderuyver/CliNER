######################################################################
#  CliNER - train.py                                                 #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Build model for given training data.                     #
######################################################################


__author__ = 'Willie Boag'
__date__   = 'Oct. 5, 2014'


import os
import glob
import argparse
import sys

import tools
from model import Model
from note import Note


# base directory
CLINER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def main():

    # Parse arguments
    parser = argparse.ArgumentParser(prog='cliner train')
    parser.add_argument("--txt",
        dest = "txt",
        help = ".txt files of discharge summaries"
    )
    parser.add_argument("--annotations",
        dest = "con",
        help = "concept files for annotations of the .txt files",
    )
    parser.add_argument("--model",
        dest = "model",
        help = "Path to the model that should be stored",
    )
    parser.add_argument("--log",
        dest = "log",
        help = "Path to the log file for training info",
        default = os.path.join(CLINER_DIR, 'models', 'train.log')
    )
    parser.add_argument("--format",
        dest = "format",
        help = "Data format ( i2b2 )"
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Error check: Ensure that file paths are specified
    if not args.txt:
        print >>sys.stderr, '\n\tError: Must provide text files'
        print >>sys.stderr,  ''
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)
    if not args.con:
        print >>sys.stderr, '\n\tError: Must provide annotations for text files'
        print >>sys.stderr,  ''
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)
    if not args.model:
        print >>sys.stderr, '\n\tError: Must provide valid path to store model'
        print >>sys.stderr,  ''
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)
    modeldir = os.path.dirname(args.model)
    if (not os.path.exists(modeldir)) and (modeldir != ''):
        print >>sys.stderr, '\n\tError: Model dir does not exist: %s' % modeldir
        print >>sys.stderr,  ''
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)


    # A list of text    file paths
    # A list of concept file paths
    txt_files = glob.glob(args.txt)
    con_files = glob.glob(args.con)


    # data format
    if not args.format:
        print '\n\tERROR: must provide "format" argument\n'
        exit()


    # Must specify output format
    if args.format not in ['i2b2']:
        print >>sys.stderr, '\n\tError: Must specify output format'
        print >>sys.stderr,   '\tAvailable formats: i2b2'
        print >>sys.stderr, ''
        exit(1)


    # Collect training data file paths
    txt_files_map = tools.map_files(txt_files)
    con_files_map = tools.map_files(con_files)

    training_list = []
    for k in txt_files_map:
        if k in con_files_map:
            training_list.append((txt_files_map[k], con_files_map[k]))

    # Train the model
    train(training_list, args.model, args.format, logfile=args.log)



def train(training_list, model_path, format, logfile=None):

    # Read the data into a Note object
    notes = []
    for txt, con in training_list:
        #try:
            note_tmp = Note(txt, con)    # Create Note
            notes.append(note_tmp)    # Add the Note to the list
        #except Exception, e:
        #    exit( '\n\tWARNING: Note Exception - %s\n\n' % str(e) )

    # file names
    if not notes:
        print 'Error: Cannot train on 0 files. Terminating train.'
        return 1

    # Create a Machine Learning model
    model = Model()

    # Train the model using the Note's data
    model.train(notes)

    # Pickle dump
    print '\nserializing model to %s\n' % model_path
    model.serialize(model_path, logfile)




if __name__ == '__main__':
    main()
