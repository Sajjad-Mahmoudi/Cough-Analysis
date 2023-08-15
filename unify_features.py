# import pandas as pd
import numpy as np

from esperity.cough_analysis.utils import *


class UnifyFeaturesInDataframe:
    def __init__(self, dics: list, name_of_dics: list, spi: dict):
        """

        :param dics - A list of dictionaries containing the features.
        :param name_of_dics - A list of names of dictionaries (so it is a list of strings), e.g., if one of the
        dictionaries of dics is amplitude_envelopes, one of the elements of the name_of_dics should be the string
        "amplitude_envelopes". Note that the order of dictionaries in both lists, dics and name_of_dics, should be
        the same.
        :param spi - A dictionary of spirometry readings (values are Pandas dataframes)

        Note: The name of dictionaries should be exactly the same as the list below:
        [processed_signals, amplitude_envelopes, filtered_squared_cumsums, filtered_squared_waveforms, log_spectrograms,
         log_melspectrograms, mfccs, rms_energies, spectral_centroids, spectral_bandwidths, spectral_rolloffs,
         zero_crossing_rates, power_spectral_densities]
        """

        self.list_of_dicts = dics
        self.list_of_dict_names = name_of_dics
        self.dict_of_spirometries = spi

    def dicts_to_df(self):
        """
        :return: Dataframe consisting of the coughs as rows and the features as columns.
        """

        feature_abrreviation = ['proSig', 'spec', 'melSpec', 'mfcc', 'sc', 'sb', 'sr', 'zcr', 'ae', 'rms', 'fsw',
                                'fsc', 'psd']

        dfs = []
        for i, dict in enumerate(self.list_of_dicts):
            len_feature = len(list(dict.values())[0])
            match self.list_of_dict_names[i]:
                case 'processed_signals':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[0]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'amplitude_envelopes':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[8]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'filtered_squared_cumsums':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[11]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'filtered_squared_waveforms':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[10]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'log_spectrograms':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[1]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'log_melspectrograms':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[2]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'mfccs':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[3]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'rms_energies':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[9]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'spectral_centroids':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[4]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'spectral_bandwidths':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[5]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'spectral_rolloffs':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[6]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'zero_crossing_rates':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[7]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)
                case 'power_spectral_densities':
                    temp_df = pd.DataFrame.from_dict(dict, orient='index', dtype=np.float32,
                                                     columns=[f'{feature_abrreviation[12]}_{j}' for j in
                                                              range(1, len_feature + 1)])
                    dfs.append(temp_df)

        return pd.concat(dfs, join='outer', axis=1)

    def add_spirometry(self, df):
        """

        :return: A dataframe updated by two new columns including spirometry readings, Frez and R5Hz, added at the end
         of the given dataframe
        """

        df['R5Hz'] = np.float32(0)
        df['Frez'] = np.float32(0)

        for i, row in enumerate(list(df.index)):
            if '01_' in row:
                df_spi = self.dict_of_spirometries['spi_01']
                if '01_BA' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[0, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[0, 'Frez']
                elif '01_NC' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[1, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[1, 'Frez']
                elif '01_C1' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[2, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[2, 'Frez']
                elif '01_C2' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[3, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[3, 'Frez']
                elif '01_C3' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[4, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[4, 'Frez']
                elif '01_C4' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[5, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[5, 'Frez']
                elif '01_C5' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[6, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[6, 'Frez']
            elif '02_' in row:
                df_spi = self.dict_of_spirometries['spi_02']
                if '02_BA' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[0, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[0, 'Frez']
                elif '02_NC' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[1, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[1, 'Frez']
                elif '02_C1' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[2, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[2, 'Frez']
                elif '02_C2' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[3, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[3, 'Frez']
                elif '02_C3' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[4, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[4, 'Frez']
                elif '02_C4' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[5, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[5, 'Frez']
                elif '02_C5' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[6, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[6, 'Frez']
            elif '03_' in row:
                df_spi = self.dict_of_spirometries['spi_03']
                if '03_BA' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[0, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[0, 'Frez']
                elif '03_NC' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[1, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[1, 'Frez']
                elif '03_C1' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[2, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[2, 'Frez']
                elif '03_C2' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[3, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[3, 'Frez']
                elif '03_C3' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[4, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[4, 'Frez']
                elif '03_C4' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[5, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[5, 'Frez']
                elif '03_C5' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[6, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[6, 'Frez']
            elif '05_' in row:
                df_spi = self.dict_of_spirometries['spi_05']
                if '05_BA' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[0, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[0, 'Frez']
                elif '05_NC' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[1, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[1, 'Frez']
                elif '05_C1' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[2, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[2, 'Frez']
                elif '05_C2' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[3, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[3, 'Frez']
                elif '05_C3' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[4, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[4, 'Frez']
                elif '05_C4' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[5, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[5, 'Frez']
                elif '05_C5' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[6, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[6, 'Frez']
            elif '08_' in row:
                df_spi = self.dict_of_spirometries['spi_08']
                if '08_BA' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[0, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[0, 'Frez']
                elif '08_NC' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[1, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[1, 'Frez']
                elif '08_C1' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[2, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[2, 'Frez']
                elif '08_C2' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[3, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[3, 'Frez']
                elif '08_C3' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[4, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[4, 'Frez']
                elif '08_C4' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[5, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[5, 'Frez']
                elif '08_C5' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[6, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[6, 'Frez']
            elif '09_' in row:
                df_spi = self.dict_of_spirometries['spi_09']
                if '09_BA' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[0, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[0, 'Frez']
                elif '09_NC' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[1, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[1, 'Frez']
                elif '09_C1' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[2, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[2, 'Frez']
                elif '09_C2' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[3, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[3, 'Frez']
                elif '09_C3' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[4, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[4, 'Frez']
            elif '10_' in row:
                df_spi = self.dict_of_spirometries['spi_10']
                if '10_BA' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[0, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[0, 'Frez']
                elif '10_NC' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[1, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[1, 'Frez']
                elif '10_C1' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[2, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[2, 'Frez']
                elif '10_C2' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[3, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[3, 'Frez']
                elif '10_C3' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[4, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[4, 'Frez']
                elif '10_C4' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[5, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[5, 'Frez']
                elif '10_C5' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[6, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[6, 'Frez']
            elif '11_' in row:
                df_spi = self.dict_of_spirometries['spi_11']
                if '11_BA' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[0, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[0, 'Frez']
                elif '11_NC' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[1, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[1, 'Frez']
                elif '11_C1' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[2, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[2, 'Frez']
                elif '11_C2' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[3, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[3, 'Frez']
                elif '11_C3' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[4, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[4, 'Frez']
            elif '12_' in row:
                df_spi = self.dict_of_spirometries['spi_12']
                if '12_BA' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[0, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[0, 'Frez']
                elif '12_NC' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[1, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[1, 'Frez']
                elif '12_C1' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[2, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[2, 'Frez']
                elif '12_C2' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[3, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[3, 'Frez']
                elif '12_C3' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[4, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[4, 'Frez']
                elif '12_C4' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[5, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[5, 'Frez']
                elif '12_C5' in row:
                    df.loc[row, 'R5Hz'] = df_spi.loc[6, 'R5Hz']
                    df.loc[row, 'Frez'] = df_spi.loc[6, 'Frez']


if __name__ == "__main__":
    import os
    from esperity.cough_analysis.utils import multi_load

    absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    rms_energies = multi_load(os.path.join(absolute_dirpath, 'all\\features\\rms\\'))
    spectral_centroids = multi_load(os.path.join(absolute_dirpath, 'all\\features\\sc\\'))
    spi_readings = multi_load_spirometry(os.path.join(absolute_dirpath, 'all\\spi\\'), allow_pickle=True,
                                         as_dataframe=True)

    list_feature_dict = [rms_energies, spectral_centroids]
    list_feature_name = ['rms_energies', 'spectral_centroids']
    unify_features = UnifyFeaturesInDataframe(list_feature_dict, list_feature_name, spi_readings)
    df_dataset = unify_features.dicts_to_df()
    unify_features.add_spirometry(df_dataset)

    print(df_dataset)
