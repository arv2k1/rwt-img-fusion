import os
import argparse
from fusion import Fusion, ImageUtils
from xlwt import Workbook, easyxf
from performance import PCA, DwtMaxima, Performance, PixelAvg, RwtMaxima

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

    wb = Workbook()
    compObj = wb.add_sheet('Comparative objective analysis')
    header_style = easyxf('font: bold 1')

    compObj.write(0, 0, 'Study Set', header_style)
    compObj.write(0, 1, 'Algorithm', header_style)
    compObj.write(0, 2, 'En', header_style)
    compObj.write(0, 3, 'RMSE', header_style)
    compObj.write(0, 4, 'FusFac', header_style)
    compObj.write(0, 5, 'EQ', header_style)
    compObj.write(0, 6, 'mSSIM', header_style)

    row = 1

    for folder in folders:
        print('\nFolder : %s' % folder)

        input_path = os.path.join(INPUT_FOLDER, folder)
        output_path = os.path.join(OUTPUT_FOLDER, folder)

        os.makedirs(output_path, exist_ok=True)

        print('Performing fusion...')
        Fusion(input_path, output_path).fuse()

        print('Computing performance parameters...\n')
        
        ct = ImageUtils.read(os.path.join(input_path, 'ct.jpg'))
        mri = ImageUtils.read(os.path.join(input_path, 'mri.jpg'))
        fused = ImageUtils.read(os.path.join(output_path, 'FUSED.jpg'))

        compObj.write(row, 1, 'Pixel Average')
        pixAvg = PixelAvg.getParams(ct, mri, fused)

        for i in range(5):
            compObj.write(row, 2 + i, pixAvg[i])

        row += 1

        compObj.write(row, 1, 'PCA')
        pca = PCA.getParams(ct, mri, fused)

        for i in range(5):
            compObj.write(row, 2 + i, pca[i])

        row += 1

        compObj.write(row, 0, folder)

        compObj.write(row, 1, 'DWT maxima')
        dwtMaxima = DwtMaxima.getParams(ct, mri, fused)

        for i in range(5):
            compObj.write(row, 2 + i, dwtMaxima[i])

        row += 1

        compObj.write(row, 1, 'RWT maxima')
        rwtMaxima = RwtMaxima.getParams(ct, mri, fused)

        for i in range(5):
            compObj.write(row, 2 + i, rwtMaxima[i])

        row += 1

        compObj.write(row, 1, 'Proposed')
        prop = Performance.getParams(ct, mri, fused)

        for i in range(5):
            compObj.write(row, 2 + i, prop[i])

        row += 2

    try:
        wb.save('performance.xls')
    except:
        print('Please close previously opened excel sheet window')
        ignore = input("Press any key to continue...")
        wb.save('performance.xls')

    opt = input('Do you want to open the excel file (Y/N) ? ')
    if opt in ['Y', 'y']:
        os.startfile(os.path.join(APP_ROOT, 'performance.xls'))


if __name__ == '__main__':
    main()
