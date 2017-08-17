######################################################################
#  CliNER - examine.py                                               #
#                                                                    #
#  Willie Boag                                      wboag@mit.edu    #
#                                                                    #
#  Purpose: Use trained model to predict concept labels for data.    #
######################################################################

__author__ = 'Willie Boag'
__date__   = 'Aug. 17, 2017'

import os,sys
import argparse
import cPickle as pickle


def main():

    # Get command line arguments
    parser = argparse.ArgumentParser(prog='cliner examine')
    parser.add_argument("--model", dest = "model", help = "The model whose log to print")
    args = parser.parse_args()

    # Error check: Ensure that file paths are specified
    if not args.model or not os.path.exists(args.model):
        print >>sys.stderr, '\n\tError: Must provide path to model\n'
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)

    # Load model
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # Print contents of model training log
    model.log(sys.stdout)


if __name__ == '__main__':
    main()
