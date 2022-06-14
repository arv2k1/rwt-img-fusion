import os
import argparse
from fusion import Fusion

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def clear_directory(folder_path):
    for folder_name in os.listdir(folder_path):
        inner_folder_path = os.path.join(folder_path, folder_name)
        for file in os.listdir(inner_folder_path):
            os.remove(os.path.join(inner_folder_path, file))
        os.rmdir(inner_folder_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Name of the folder containing patient data set images')
    args = parser.parse_args()

    INPUT_FOLDER = os.path.join(APP_ROOT, args.folder)
    OUTPUT_FOLDER = os.path.join(APP_ROOT, 'Output Images')

    # Delete previous output images if any from the output directory
    clear_directory(OUTPUT_FOLDER)

    folders = os.listdir(INPUT_FOLDER)

    # sort based on the number on the folder name
    folders.sort(key=lambda f: int(f.replace('p', '')))
    print('Found %d folders...' % len(folders))

    for folder in folders:
        print('\nFolder : %s' % folder)

        input_path = os.path.join(INPUT_FOLDER, folder)
        output_path = os.path.join(OUTPUT_FOLDER, folder)

        os.makedirs(output_path, exist_ok=True)

        print('Performing fusion...')
        Fusion(input_path, output_path).fuse()

        # print('Computing performance parameters...\n')
        # params = p.get_performance_params(output_path)

if __name__ == '__main__':
    main()
