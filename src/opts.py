import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--configuration_file_path', type=str, default='config.yaml',
                    help='the path to the configuration file')
