import json
import argparse
import shutil
import subprocess
import os

from prettytable import PrettyTable

module_path = os.path.dirname(__file__)


def increment_version(path, version_type='MICRO'):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(json.dumps({'MICRO': 0, 'MINOR': 0, 'MAJOR': 0}))
    data = json.load(open(path))
    data[version_type.upper()] += 1
    json.dump(data, open(path, "w"), )


def print_cli_args(cli_args: argparse.Namespace):
    print("Using the following config:")
    t = PrettyTable(['Config Key', 'Specified Value'])
    t.align["Config Key"] = "r"
    t.align["Specified Value"] = "l"
    for k, v in cli_args.__dict__.items():
        t.add_row([k, v])
    print(t)


def build_wheel(inc_version, create_wheel):
    if inc_version:
        increment_version(os.path.join(module_path, 'version.json'), inc_version)
    setup_path = os.path.join(module_path, "setup.py")
    if create_wheel:
        import sys
        clean_cmd = [sys.executable, setup_path, "clean", "--all"]
        print("Cleaning before build...")
        subprocess.Popen(clean_cmd).communicate()
        print("Removing dist folder")
        shutil.rmtree("dist", ignore_errors=True)
        print("building wheel")
        build_cmd = [sys.executable, setup_path, "bdist_wheel"]
        subprocess.Popen(build_cmd).communicate()
        print("Cleaning after build")
        subprocess.Popen(clean_cmd).communicate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the distribution')

    parser.add_argument(
        '-i',
        '--inc',
        type=str,
        dest='increment_type',
        choices=['major', 'minor', 'micro'],
        help=f'If the argument is not specified, it will ...',
        required=False,
        default='micro'
    )
    parser.add_argument(
        '-w',
        '--build-wheel',
        type=bool,
        dest='build_wheel',
        help='If wheel has to be built',
        required=False,
        default=True,
    )

    args: argparse.Namespace = parser.parse_args()
    print_cli_args(cli_args=args)

    build_wheel(inc_version=args.increment_type, create_wheel=args.build_wheel)
