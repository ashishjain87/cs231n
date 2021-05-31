import argparse
import os
import sys


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--configs-dir', type=str, help="Directory containing a bunch of config files to use in data generation")
    parser.add_argument('--outputs-top-level-dir', type=str, help="Directory where all outputs should be located")
    parser.add_argument('--target-config-path', type=str, help="place config files should be individually copied to")

    return parser.parse_args()


def generate_dataset(args, configpath):
    os.system("python CreateSyntheticData.py")


def main():
    args = get_args()
    configs = os.listdir(args.configs_dir)

    if not os.exists("CreateSyntheticData.py"):
        print("Change to a directory containing 'CreateSyntheticData.py' please.")
        return

    print(f"{len(configs)} config files found. Generating {len(configs)} datasets.")

    for config in configs:
        configpath = f"{args.config_dir}/{config}"

        os.system(f"cp {configpath} {args.target_config_path}")
        try:
            generate_dataset(args, configpath)
        except:
            print(f"Encountered some error while generating dataset using {configpath}, moving on to next config",
                  file=sys.stderr)
