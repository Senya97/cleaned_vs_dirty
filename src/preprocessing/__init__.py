import argparse


def create_argument_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--', type=str, default='', help='')
    return parser


def main(args):
    pass


if __name__ == '__main__':
    argument_parser = create_argument_parser()
    arguments = argument_parser.parse_args()
    main(arguments)