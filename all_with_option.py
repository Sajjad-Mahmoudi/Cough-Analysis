import os
import pickle
import librosa
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
from esperity.cough_analysis.utils import stat_dur
import noisereduce as nr


class Loader:
    """ Load the audio file """

    def __init__(self, sample_rate, duration, mono=True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path, sr=self.sample_rate, duration=self.duration, mono=self.mono)[0]
        return signal


class Padder:
    """ Pad the audio file
        if mode = constant, pad the file with a constant value defaulted on 0 """

    def __init__(self, mode='constant'):
        self.mode = mode

    def left_pad(self, array, num_missing_items, constant_value=0):
        if self.mode == 'constant':
            padded_array = np.pad(array, (num_missing_items, 0), constant_value=constant_value, mode=self.mode)
            return padded_array
        else:
            padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
            return padded_array

    def right_pad(self, array, num_missing_items, constant_value=0):
        if self.mode == 'constant':
            padded_array = np.pad(array, (0, num_missing_items), constant_values=constant_value, mode=self.mode)
            return padded_array
        else:
            padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
            return padded_array


class Normaliser:
    """MinMaxNormaliser applies min max normalisation to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        # normalise between o and 1
        norm_array = (array - array.min()) / (array.max() - array.min())
        # normalise between a given range, in case for normalising between -1 and 1
        # so, for normalising between -1 and 1, self.max and self.min should be 1 and -1, respectively
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


# class ExtractFeatures:
#     pass

'''class SpirometryExtractor:
    """ Extract log spectrograms (in dB) from the signal """

    def __init__(self, csv_file_path):
        self.file_path = csv_file_path

    def spi_values(self, signal):
        df = pd.read_csv(os.path.join(os.getcwd(), *['data', 'RHz_files', 'rhz_01.csv']))
        return df.values'''


class ExtractFeatures:
    """ This class contains all features to be extracted

        def log_melspec(): number of mels (n_mels) is defaulted in 128 based on the paper below:
         https://www.frontiersin.org/articles/10.3389/frobt.2021.580080/full, but it needs to study literatures
         much more and experiments (it can be considered as an hyperparameter)

        def mfccs(): number of mfccs (n_mfccs) is defaulted in 13, so together with their 1st and 2nd derivatives,
         we will have 39 coefficients in total. It also needs more investigations to find the optimal n_mfccs.
         """

    def __init__(self, sample_rate, frame_size, hop_length, n_mels=128, n_mfcc=13, savgol_win_length=501,
                 savgol_polyorder=3):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.savgol_win_length = savgol_win_length
        self.savgol_polyorder = savgol_polyorder

    @staticmethod
    def processed_sig(signal):
        return signal

    def log_spec(self, signal):
        stft = librosa.stft(y=signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]  # the size of stft would
        # be (1 + frame_size/2, num_frames), for example for frame_size = 1024, it will be 513, so to have an even
        # number of frequency bins, the last element is dropped, that is why [:-1] is used
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)  # try spectrogram**2
        return log_spectrogram

    def log_melspec(self, signal):
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate, n_fft=self.frame_size,
                                                         hop_length=self.hop_length, n_mels=self.n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

    def mfccs(self, signal):  # ??? which one is better? mfccs, delta_mfccs, delta2_mfccs, or concatenation ???
        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=self.n_mfcc, sr=self.sample_rate, n_fft=self.frame_size,
                                     hop_length=self.hop_length)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        return np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

    def spectral_centroid(self, signal):
        sc = librosa.feature.spectral_centroid(y=signal, sr=self.sample_rate, n_fft=self.frame_size,
                                               hop_length=self.hop_length)[0]
        return sc

    def spectral_bandwidth(self, signal):
        sb = librosa.feature.spectral_bandwidth(y=signal, sr=self.sample_rate, n_fft=self.frame_size,
                                                hop_length=self.hop_length, norm=True)[0]
        return sb

    def spectral_rolloff(self, signal):  # ??? what roll_percent should be used? is it a hyperparameter? ???
        sr = librosa.feature.spectral_rolloff(y=signal, sr=self.sample_rate, n_fft=self.frame_size,
                                              hop_length=self.hop_length, roll_percent=0.85)[0]
        return sr

    def zero_crossing_rate(self, signal):
        zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=self.frame_size, hop_length=self.hop_length)[0]
        return zcr

    def amplitude_envelope(self, signal):
        ae = np.array([max(signal[i: i + self.frame_size]) for i in range(0, len(signal), self.hop_length)])
        return ae

    def rms_energy(self, signal):
        rms = librosa.feature.rms(y=signal, frame_length=self.frame_size, hop_length=self.hop_length)[0]
        return rms

    def filtered_squared_waveform(self, signal):
        fsw = savgol_filter(signal ** 2, window_length=self.savgol_win_length, polyorder=self.savgol_polyorder)
        return fsw

    def filtered_squared_cumsum(self, signal):
        fsc = np.cumsum(savgol_filter(signal ** 2, window_length=self.savgol_win_length,
                                      polyorder=self.savgol_polyorder))
        return fsc

    def power_spectral_density(self, signal):
        _, powers = welch(signal, fs=self.sample_rate, nperseg=self.frame_size, noverlap=self.hop_length,
                          scaling='density')
        return powers


class Saver:
    """ Save the features as well as min max values """

    def __init__(self, proSig_save_dir, logSpec_save_dir, logMelSpec_save_dir, mfccs_save_dir, sc_save_dir, sb_save_dir,
                 sr_save_dir, zcr_save_dir, ae_save_dir, rms_save_dir, fsw_save_dir, fsc_save_dir, psd_save_dir,
                 spi_save_dir):
        self.processed_sig_save_dir = proSig_save_dir
        self.log_spec_save_dir = logSpec_save_dir
        self.log_melspec_save_dir = logMelSpec_save_dir
        self.mfccs_save_dir = mfccs_save_dir
        self.spectral_centroid_save_dir = sc_save_dir
        self.spectral_bandwidth_save_dir = sb_save_dir
        self.spectral_rolloff_save_dir = sr_save_dir
        self.zero_crossing_rate_save_dir = zcr_save_dir
        self.amplitude_envelope_save_dir = ae_save_dir
        self.rms_energy_save_dir = rms_save_dir
        self.filtered_squared_waveform_save_dir = fsw_save_dir
        self.filtered_squared_cumsum_save_dir = fsc_save_dir
        self.power_spectral_density_save_dir = psd_save_dir
        self.spi_save_dir = spi_save_dir

    def save_feature(self, feature, file_path, feature_name):
        save_path = self._generate_save_path(file_path, feature_name)
        np.save(save_path, feature)

    def save_sig_min_max_values(self, min_max_values):
        save_path = os.path.join(self.processed_sig_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_spec_min_max_values(self, min_max_values):
        save_path = os.path.join(self.log_spec_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_melspec_min_max_values(self, min_max_values):
        save_path = os.path.join(self.log_melspec_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_mfccs_min_max_values(self, min_max_values):
        save_path = os.path.join(self.mfccs_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_sc_min_max_values(self, min_max_values):
        save_path = os.path.join(self.spectral_centroid_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_sb_min_max_values(self, min_max_values):
        save_path = os.path.join(self.spectral_bandwidth_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_sr_min_max_values(self, min_max_values):
        save_path = os.path.join(self.spectral_rolloff_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_zcr_min_max_values(self, min_max_values):
        save_path = os.path.join(self.zero_crossing_rate_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_ae_min_max_values(self, min_max_values):
        save_path = os.path.join(self.amplitude_envelope_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_rms_min_max_values(self, min_max_values):
        save_path = os.path.join(self.rms_energy_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_fsw_min_max_values(self, min_max_values):
        save_path = os.path.join(self.filtered_squared_waveform_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_fsc_min_max_values(self, min_max_values):
        save_path = os.path.join(self.filtered_squared_cumsum_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    def save_psd_min_max_values(self, min_max_values):
        save_path = os.path.join(self.power_spectral_density_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path, feature_name):
        file_name = os.path.split(file_path)[1]
        # based on the feature name, take the appropriate save directory from __init__ method
        feature_dir = getattr(self, feature_name + '_save_dir')
        dir_name = os.path.basename(os.path.dirname(feature_dir))
        save_path = os.path.join(feature_dir, dir_name + '_' + file_name[:-4] + ".npy")
        return save_path

    def save_spi(self, array, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.spi_save_dir, 'spi_' + file_name[:-4] + ".npy")
        np.save(save_path, array)


class PreprocessPipeline:
    """ PreprocessingPipeline processes audio files in a directory, applying
        the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting features from signal
        4- normalise features
        5- save the normalised features
        6- storing the min max values for all the log spectrograms/mel spectrograms

        features_list: features_list should be filled with the features required.
        The features can be one or more of the list below:
        ['logSpec', 'logMelSpec', ]
        Note that the name of wanted features should be exactly like the ones in the list
        """

    def __init__(self, features_list=None, normalised_signal=False, noise_reduction=False):
        if features_list is None:
            features_list = ['processed_sig', 'log_spec', 'log_melspec', 'mfccs', 'spectral_centroid',
                             'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate', 'amplitude_envelope',
                             'rms_energy', 'filtered_squared_waveform', 'filtered_squared_cumsum',
                             'power_spectral_density']
        self.features_list = features_list
        self.noise_reduction = noise_reduction
        self.normalised_signal = normalised_signal
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.sig_min_max_values = {}
        self.spec_min_max_values = {}
        self.melspec_min_max_values = {}
        self.mfccs_min_max_values = {}
        self.sc_min_max_values = {}
        self.sb_min_max_values = {}
        self.sr_min_max_values = {}
        self.zcr_min_max_values = {}
        self.ae_min_max_values = {}
        self.rms_min_max_values = {}
        self.fsw_min_max_values = {}
        self.fsc_min_max_values = {}
        self.psd_min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir, csv_files_dir):
        counter_audio = 0
        counter_csv = 0
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                # print(f"Processed file {file_path}")
                counter_audio += 1
        for r, _, f in os.walk(csv_files_dir):
            for csv_file in f:  # ??? should I normalize spirometry data as well as other features? ???
                p = os.path.join(r, csv_file)
                df = pd.read_csv(p)
                array_df = np.vstack((df.columns.values, df.values))
                self.saver.save_spi(array_df, p)
                counter_csv += 1
        print(f'{counter_audio} audio files are processed')
        print(f'{counter_csv} csv (spirometry results) files are processed')
        self.saver.save_sig_min_max_values(self.sig_min_max_values)
        self.saver.save_spec_min_max_values(self.spec_min_max_values)
        self.saver.save_melspec_min_max_values(self.melspec_min_max_values)
        self.saver.save_mfccs_min_max_values(self.mfccs_min_max_values)
        self.saver.save_sc_min_max_values(self.sc_min_max_values)
        self.saver.save_sb_min_max_values(self.sb_min_max_values)
        self.saver.save_sr_min_max_values(self.sr_min_max_values)
        self.saver.save_zcr_min_max_values(self.zcr_min_max_values)
        self.saver.save_ae_min_max_values(self.ae_min_max_values)
        self.saver.save_rms_min_max_values(self.rms_min_max_values)
        self.saver.save_fsw_min_max_values(self.fsw_min_max_values)
        self.saver.save_fsc_min_max_values(self.fsc_min_max_values)
        self.saver.save_psd_min_max_values(self.psd_min_max_values)

    def _process_file(self, file_path):
        signal = self._loader.load(file_path)
        if self.noise_reduction:
            signal = nr.reduce_noise(y=signal, sr=self.extractor.sample_rate, n_fft=self.extractor.frame_size,
                                     hop_length=self.extractor.hop_length)
        if self.normalised_signal:
            signal = self.normaliser.normalise(signal)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        for feature_name in self.features_list:
            # based on feature name in features_list, take the corresponding method from ExtractFeatures class
            feature = getattr(self.extractor, feature_name)(signal)
            norm_feature = self.normaliser.normalise(feature)
            self.saver.save_feature(norm_feature, file_path, feature_name)
            self._store_min_max_value(os.path.split(file_path)[1][:-4], feature.min(), feature.max(), feature_name)

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    # key_name is the name of audio file being processed like "01_B_01"
    def _store_min_max_value(self, key_name, min_val, max_val, feature_name):
        if feature_name == 'processed_sig':
            self.sig_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'log_spec':
            self.spec_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'log_melspec':
            self.melspec_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'mfccs':
            self.mfccs_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'spectral_centroid':
            self.sc_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'spectral_bandwidth':
            self.sb_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'spectral_rolloff':
            self.sr_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'zero_crossing_rate':
            self.zcr_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'amplitude_envelope':
            self.ae_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'rms_energy':
            self.rms_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'filtered_squared_waveform':
            self.fsw_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'filtered_squared_cumsum':
            self.fsc_min_max_values[key_name] = {"min": min_val, "max": max_val}
        elif feature_name == 'power_spectral_density':
            self.psd_min_max_values[key_name] = {"min": min_val, "max": max_val}
        else:
            pass


if __name__ == "__main__":
    my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    AUDIO_FILES_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\data\\audios\\')
    CSV_FILES_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\data\\csv_spi\\')
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    SAMPLE_RATE = 22050
    LIST_FEATURES = ['processed_sig', 'log_spec', 'log_melspec', 'mfccs', 'spectral_centroid', 'spectral_bandwidth',
                     'spectral_rolloff', 'zero_crossing_rate', 'amplitude_envelope', 'rms_energy',
                     'power_spectral_density']

    # Features that can be added: spectral_flatness(librosa) - spectral_flux(?) - spectral_kurtosis(scipy) -
    #                              spectral_skewness(scipy) -

    MONO = True
    N_MELS = 20
    N_MFCCS = 13
    SAVGOL_WINDOW_LENGTH = 501
    SAVGOL_POLYORDER = 3

    q25, q75, q90, mean_arithmetic, max_durs, min_durs = stat_dur(AUDIO_FILES_DIR, SAMPLE_RATE, FRAME_SIZE, HOP_LENGTH)
    # round to 2 because if not, there will be one more sample for some signals
    DURATION = round(q75, 2)  # in seconds

    PROCESSED_SIG_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\processedSig\\')
    SPECTROGRAMS_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\logSpec\\')
    MELSPECTROGRAMS_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\logMelSpec\\')
    MFCCS_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\mfccs\\')
    SC_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\sc\\')
    SB_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\sb\\')
    SR_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\sr\\')
    ZCR_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\zcr\\')
    AE_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\ae\\')
    RMS_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\rms\\')
    FSW_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\fsw\\')
    FSC_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\fsc\\')
    PSD_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\features\\psd\\')
    SPI_SAVE_DIR = os.path.join(my_absolute_dirpath, 'all_with_nor_nr\\spi')

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    features_extractor = ExtractFeatures(SAMPLE_RATE, FRAME_SIZE, HOP_LENGTH, N_MELS, N_MFCCS, SAVGOL_WINDOW_LENGTH,
                                         SAVGOL_POLYORDER)
    min_max_normaliser = Normaliser(0, 1)
    saver = Saver(PROCESSED_SIG_SAVE_DIR, SPECTROGRAMS_SAVE_DIR, MELSPECTROGRAMS_SAVE_DIR, MFCCS_SAVE_DIR, SC_SAVE_DIR,
                  SB_SAVE_DIR, SR_SAVE_DIR, ZCR_SAVE_DIR, AE_SAVE_DIR, RMS_SAVE_DIR, FSW_SAVE_DIR, FSC_SAVE_DIR,
                  PSD_SAVE_DIR, SPI_SAVE_DIR)

    preprocessing_pipeline = PreprocessPipeline(LIST_FEATURES, normalised_signal=True, noise_reduction=True)
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = features_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(AUDIO_FILES_DIR, CSV_FILES_DIR)
