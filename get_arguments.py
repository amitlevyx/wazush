import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', dest='itrk_path', help="itrk path", required=True)
    parser.add_argument('-o', dest='out_dir', help="out directory", required=True)
    parser.add_argument('--vd', dest='vd_pixels', help="add vd pixels information", action='store_true')
    parser.add_argument('--lanes', dest='lane_pixels', help="add lane pixels information", action='store_true')
    parser.add_argument('--lca', dest='lca_pixels', help="add lca pixels information", action='store_true')
    parser.add_argument('--cam', help="desired cameras with with space separator (for the moment only cameras from"
                                      "same side (i.e rear))", default='main', required=False)

    args = parser.parse_args()
    return vars(args)
