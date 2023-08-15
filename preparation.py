import glob
import os
import numpy as np
from esperity.cough_analysis.utils import rhz_file

# dir_cough = glob.glob(r"C:\Users\mahmo\PycharmProjects\pythonProject\esperity\cough_analysis\data\ventolin\h*.wav")
# for i, filename in enumerate(dir_cough):
#     # os.rename(filename, filename[:-8] + 'NC' + filename[-7:])
#     os.rename(filename, filename[:-6] + '08_VE_' + str(i+1).zfill(2) + filename[-4:])

# change the name of audio data to the pattern as "patient No. + cough condition + cough No."
# example: the name of the first audio of the patient 1 coughing in the condition base = "01_B_01"
# Abbreviations for cough conditions: BA = base, NC = NaCl
def modify_audio_name():
    patient_nums = np.linspace(1, 15, 15, True, dtype=int)
    cough_cond = ['_BA_', '_C1_', '_C2_', '_C3_', '_C4_', '_C5_', '_C6_', '_NC_']
    dir_cough = glob.glob("esperity/cough_analysis/data/c6/h*.wav")
    for i, filename in enumerate(dir_cough):
        os.rename(filename, 'esperity/cough_analysis/data/c6/' + str(patient_nums[11]).zfill(2) + cough_cond[6] +
                  str(i + 1).zfill(2) + '.wav')


# convert all "RHz.xls" files to ".csv" equivalents together with some modifications on the original file
def produce_csv_file(xls_files_dir_abs_path, patient_numbers: list):
    """ xls_files_dir_abs_path: the absolute path of the folder containing xls files
        patient_numbers: the number of the patients of which the RHz/xls files are modified and converted """

    for i, xls_file in enumerate(glob.glob(xls_files_dir_abs_path)):
        rhz_file(xls_file, patient_numbers[i])


if __name__ == "__main__":
    absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    p = os.path.join(absolute_dirpath, *['data', 'RHz_files', 'RHz08.xls'])
    produce_csv_file(xls_files_dir_abs_path=p, patient_numbers=[8])
    # df = pd.read_csv(os.path.join(os.getcwd(), *['data', 'RHz_files', 'rhz_0001.csv']))
    # #arr = df.to_numpy(df)
    # df["r"] = df["FEV1"] + df["FVC"]
    # df.to_csv(os.path.join(os.getcwd(), *['data', 'RHz_files', 'rhz_0001.csv']), encoding='utf-8', index=False)
