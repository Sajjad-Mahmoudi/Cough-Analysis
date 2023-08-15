from esperity.cough_analysis.utils import *
from esperity.cough_analysis.unify_features import *


if __name__ == "__main__":
    absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    # read the original signals and the extracted features
    # original_signals = multi_load(os.path.join(absolute_dirpath, 'all\\data\\audios\\'))
    # processed_signals = multi_load(os.path.join(absolute_dirpath, 'all\\features\\processedSig\\'))  # length = 12789
    amplitude_envelopes = multi_load(os.path.join(absolute_dirpath, 'all\\features\\ae\\'))  # length = 50
    log_spectrograms = multi_load(os.path.join(absolute_dirpath, 'all\\features\\logSpec\\'))  # length = (256, 50)
    log_melspectrograms = multi_load(os.path.join(absolute_dirpath, 'all\\features\\logMelSpec\\'))  # length = (20, 50)
    # MFCC: length = (39, 50) (there are 50 windows, each containing 39 coefficients)
    mfccs = multi_load(os.path.join(absolute_dirpath, 'all\\features\\mfccs\\'))
    rms_energies = multi_load(os.path.join(absolute_dirpath, 'all\\features\\rms\\'))  # length = 50
    spectral_centroids = multi_load(os.path.join(absolute_dirpath, 'all\\features\\sc\\'))  # length = 50
    spectral_bandwidths = multi_load(os.path.join(absolute_dirpath, 'all\\features\\sb\\'))  # length = 50
    spectral_rolloffs = multi_load(os.path.join(absolute_dirpath, 'all\\features\\sr\\'))  # length = 50
    zero_crossing_rates = multi_load(os.path.join(absolute_dirpath, 'all\\features\\zcr\\'))  # length = 50
    power_spectral_densities = multi_load(os.path.join(absolute_dirpath, 'all\\features\\psd\\'))  # length = 257
    #
    # # read the min max values
    # min_max_proSig = read_dic(os.path.join(absolute_dirpath, 'features\\processedSig\\min_max_values.pkl'))
    # min_max_sc = read_dic(os.path.join(absolute_dirpath, 'features\\sc\\min_max_values.pkl'))
    # min_max_ae = read_dic(os.path.join(absolute_dirpath, 'features\\ae\\min_max_values.pkl'))
    # min_max_rms = read_dic(os.path.join(absolute_dirpath, 'features\\rms\\min_max_values.pkl'))
    # min_max_zcr = read_dic(os.path.join(absolute_dirpath, 'features\\zcr\\min_max_values.pkl'))
    # min_max_sr = read_dic(os.path.join(absolute_dirpath, 'features\\sr\\min_max_values.pkl'))
    # min_max_sb = read_dic(os.path.join(absolute_dirpath, 'features\\sb\\min_max_values.pkl'))
    # min_max_logSpec = read_dic(os.path.join(absolute_dirpath, 'features\\logSpec\\min_max_values.pkl'))
    # min_max_logMelSpec = read_dic(os.path.join(absolute_dirpath, 'features\\logMelSpec\\min_max_values.pkl'))
    # min_max_mfcc = read_dic(os.path.join(absolute_dirpath, 'features\\mfccs\\min_max_values.pkl'))
    # min_max_psd = read_dic(os.path.join(absolute_dirpath, 'features\\psd\\min_max_values.pkl'))
    #
    # # obtain the denormalised feature dictionaries
    # denorm_processed_signals = multi_denorm(processed_signals, min_max_proSig)
    # denorm_amplitude_envelopes = multi_denorm(amplitude_envelopes, min_max_ae)
    # denorm_rms_energies = multi_denorm(rms_energies, min_max_rms)
    # denorm_zero_crossing_rates = multi_denorm(zero_crossing_rates, min_max_zcr)
    # denorm_log_spectrograms = multi_denorm(log_spectrograms, min_max_logSpec)
    # denorm_log_melspectrograms = multi_denorm(log_melspectrograms, min_max_logMelSpec)
    # denorm_mfccs = multi_denorm(mfccs, min_max_mfcc)
    # denorm_spectral_centroids = multi_denorm(spectral_centroids, min_max_sc)
    # denorm_spectral_bandwidths = multi_denorm(spectral_bandwidths, min_max_sb)
    # denorm_spectral_rolloffs = multi_denorm(spectral_rolloffs, min_max_sr)
    # denorm_power_spectral_densities = multi_denorm(power_spectral_densities, min_max_psd)

    # read the spirometry readings
    spi_readings = multi_load_spirometry(os.path.join(absolute_dirpath, 'all\\spi\\'), allow_pickle=True,
                                         as_dataframe=True)

    # instantiate the class of unify_feature to use the dic_to_df attribute
    list_feature_dict = [amplitude_envelopes, rms_energies, spectral_centroids, spectral_bandwidths, spectral_rolloffs,
                         zero_crossing_rates, power_spectral_densities]
    list_feature_name = ['amplitude_envelopes', 'rms_energies', 'spectral_centroids', 'spectral_bandwidths',
                         'spectral_rolloffs', 'zero_crossing_rates', 'power_spectral_densities']
    unify_features = UnifyFeaturesInDataframe(list_feature_dict, list_feature_name, spi_readings)
    df_dataset = unify_features.dicts_to_df()
    unify_features.add_spirometry(df_dataset)

    print(df_dataset)
    df_dataset.to_csv(os.path.join(absolute_dirpath, 'all\\all_dataset.csv'))

else:
    absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    log_spectrograms = multi_load(os.path.join(absolute_dirpath, 'all\\features\\logSpec\\'))  # length = (256, 50)
    log_melspectrograms = multi_load(os.path.join(absolute_dirpath, 'all\\features\\logMelSpec\\'))  # length = (20, 50)
    # MFCC: length = (39, 50) (there are 50 windows, each containing 39 coefficients)
    mfccs = multi_load(os.path.join(absolute_dirpath, 'all\\features\\mfccs\\'))

    spi_readings = multi_load_spirometry(os.path.join(absolute_dirpath, 'all\\spi\\'), allow_pickle=True,
                                         as_dataframe=True)

