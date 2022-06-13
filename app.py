import os
from xlwt import Workbook, easyxf
import argparse

import main as m
import performance as p

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def clear_directory(folder_path):
    for folder_name in os.listdir(folder_path):
        inner_folder_path = os.path.join(folder_path, folder_name)
        for file in os.listdir(inner_folder_path):
            os.remove(os.path.join(inner_folder_path, file))
        os.rmdir(inner_folder_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True,
                        help='Name of the folder containing patient data set images')
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

    sub = wb.add_sheet('Subjective performance results')
    obj = wb.add_sheet('Objective performance results')
    ent = wb.add_sheet('Entropy results')
    api = wb.add_sheet('Average pixel intensity results')

    comp_sub = wb.add_sheet('Subjective perf comp res avg')
    comp_obj = wb.add_sheet('Objective perf comp res avg')

    header_style = easyxf('font: bold 1')

    sub.write(0, 0, 'SSIM value', header_style)
    sub.write(0, 1, 'FMI value', header_style)
    sub.write(0, 2, 'FS value', header_style)
    sub.write(0, 3, 'CC value', header_style)
    sub.write(0, 4, 'H value', header_style)
    sub.write(0, 5, 'SF value', header_style)
    sub.write(0, 6, 'AG value', header_style)
    sub.write(0, 7, 'API value', header_style)

    obj.write(0, 0, 'Q (AB/F) value', header_style)
    obj.write(0, 1, 'L (AB/F) value', header_style)
    obj.write(0, 2, 'N (AB/F) value', header_style)

    ent.write(0, 0, 'H (CT) value', header_style)
    ent.write(0, 1, 'H (MRI) value', header_style)
    ent.write(0, 2, 'H (F) value', header_style)

    api.write(0, 0, 'API (CT) value', header_style)
    api.write(0, 1, 'API (MRI) value', header_style)
    api.write(0, 2, 'API (F) value', header_style)

    comp_sub.write(0, 0, 'Method', header_style)
    comp_sub.write(0, 1, 'FMI value', header_style)
    comp_sub.write(0, 2, 'H value', header_style)
    comp_sub.write(0, 3, 'SF value', header_style)
    comp_sub.write(0, 4, 'AG value', header_style)
    comp_sub.write(0, 5, 'CC value', header_style)
    comp_sub.write(0, 6, 'FS value', header_style)
    comp_sub.write(0, 7, 'API value', header_style)

    comp_obj.write(0, 0, 'Method', header_style)
    comp_obj.write(0, 1, 'Q (AB/F) value', header_style)
    comp_obj.write(0, 2, 'L (AB/F), N (AB/F) %', header_style)

    row = 1

    # Sum of all the values, it used to compute average, which is displayed in comparative analysis table
    total = {
        'count': 0,
        'fmi': 0,
        'h': 0,
        'sf': 0,
        'ag': 0,
        'cc': 0,
        'fs': 0,
        'api': 0,
        'q': 0
    }

    for folder in folders:
        print('\nFolder : %s' % folder)

        input_path = os.path.join(INPUT_FOLDER, folder)
        output_path = os.path.join(OUTPUT_FOLDER, folder)

        os.makedirs(output_path, exist_ok=True)

        print('Performing fusion...')
        m.generate_output_images(
            input_path, output_path, show_output_image=True)

        print('Computing performance parameters...\n')
        params = p.get_performance_params(output_path)

        sub.write(row, 0, round(
            (params['ssim']['ct'] + params['ssim']['mri']) / 2, 2))
        sub.write(row, 1, round(params['fmi'], 2))
        sub.write(row, 2, round(
            (params['fs']['ct'] + params['fs']['mri']) / 2, 2))
        sub.write(row, 3, round(params['cc'], 2))
        sub.write(row, 7, round(params['h']['fused'], 2))
        sub.write(row, 4, round(params['sf'], 2))
        sub.write(row, 5, round(params['ag'], 2))
        sub.write(row, 6, round(params['api']['fused'], 2))

        obj.write(row, 0, round(params['obj']['q'], 2))
        obj.write(row, 1, round(params['obj']['l'], 2))
        obj.write(row, 2, round(params['obj']['n'], 2))

        ent.write(row, 0, round(params['h']['ct'], 2))
        ent.write(row, 1, round(params['h']['mri'], 2))
        ent.write(row, 2, round(params['h']['fused'], 2))

        api.write(row, 0, round(params['api']['ct'], 2))
        api.write(row, 1, round(params['api']['mri'], 2))
        api.write(row, 2, round(params['api']['fused'], 2))

        row += 1

        total['count'] += 1
        total['fmi'] += round(params['fmi'], 2)
        total['h'] += round(params['h']['fused'], 2)
        total['sf'] += round(params['sf'], 2)
        total['ag'] += round(params['ag'], 2)
        total['cc'] += round(params['cc'], 2)
        total['fs'] += round((params['fs']['ct'] + params['fs']['mri']) / 2, 2)
        total['api'] += round(params['api']['fused'], 2)
        total['q'] += round(params['obj']['q'], 2)

    comp_sub_params = p.get_subjective_analysis_performance_comparison_params()
    comp_obj_params = p.get_objective_analysis_performance_comparison_params()

    methods = list(comp_obj_params.keys())
    for i in range(len(methods)):
        sub_values = comp_sub_params[methods[i]]
        comp_sub.write(i+1, 0, methods[i])

        for j in range(len(sub_values)):
            comp_sub.write(i+1, j+1, sub_values[j])

        obj_values = comp_obj_params[methods[i]]
        comp_obj.write(i+1, 0, methods[i])

        for j in range(len(obj_values)):
            comp_obj.write(i+1, j+1, obj_values[j])

    comp_sub.write(6, 0, 'Proposed')
    comp_sub.write(6, 1, total['fmi'] / total['count'])
    comp_sub.write(6, 2, total['h'] / total['count'])
    comp_sub.write(6, 3, total['sf'] / total['count'])
    comp_sub.write(6, 4, total['ag'] / total['count'])
    comp_sub.write(6, 5, total['cc'] / total['count'])
    comp_sub.write(6, 6, total['fs'] / total['count'])
    comp_sub.write(6, 7, total['api'] / total['count'])

    comp_obj.write(6, 0, 'Proposed')
    comp_obj.write(6, 1, total['q'] / total['count'])
    comp_obj.write(6, 2, (1.0 - (total['q'] / total['count'])) * 100)

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
