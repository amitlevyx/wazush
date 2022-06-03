import argparse
import sys


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', dest='train_set', help="training set path")
    parser.add_argument('--test', dest='test_set', help="test set path")
    parser.add_argument('-d', dest='dates', help="dates path")
    parser.add_argument('-o', dest='out_dir', help="out directory", default="./")
    parser.add_argument('-b1', dest='baseline1', help="use baseline alg for task 1", action='store_true')
    # parser.add_argument('-b2', dest='baseline2', help="use baseline alg for task 2", action='store_true')

    args = parser.parse_args()
    return vars(args)


def use_args():
    args = get_arguments()
    if not args['train_set'] or not args['test_set'] or not args['dates']:
        print("use command -h for help, or use the format -t train_set -d dates -test test_set")
        print("you can also input the data in the following format:")
        print("<train_set> <test_set> <dates>")
        print("if this is not one of the ways you inputted, please kill and try again")
        train_set, test_set, dates = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        train_set, test_set, dates = args['train_set'], args['test_set'], args['dates']
    if args['baseline1']:
        task1_classifier = "baseline1"  # todo change this to baseline1
    output_path = args['out_dir'] + "/predictions.csv"
