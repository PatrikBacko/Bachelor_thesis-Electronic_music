import argparse

def build_arguments():
    description = ""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-h", "--help", help="show this help message and exit", action="store_true")

    