import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import pickle as pk
import matplotlib
from numpy.polynomial import Polynomial
from sklearn.metrics import r2_score
from itertools import zip_longest
import scipy as sp

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import FormatStrFormatter


# from cycler import cycler
# import IPython.display as ipd


# function to read RHz/xls files and convert them to csv files
# save csv file with such a name as rhz_patient No., e.g. for patient 1, "rhz_01.csv"
def rhz_file(xls_abs_path, patient_number):
    # read the xls file skipping the first row
    df_xls = pd.read_excel(xls_abs_path, 'Sheet1', skiprows=1)
    # name the first column "intervention"
    df_xls.rename(columns={df_xls.columns[0]: 'intervention'}, inplace=True)
    # drop the unnamed columns
    df_xls.drop(df_xls.columns[df_xls.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    # if FEV1 and FVC values are in the xls file, add a column "FEV1/FVC" to calculate the ratio
    if 'FEV1' and 'FVC' in df_xls.columns:
        df_xls["FEV1/FVC"] = round(df_xls["FEV1"] / df_xls["FVC"], 3)
    # save the edited xls file as a csv file
    path, _ = os.path.split(xls_abs_path)
    df_xls.to_csv(os.path.join(path, 'rhz_' + str(patient_number).zfill(2) + '.csv'), encoding='utf-8', index=False)


# read a pkl file
def read_dic(dic_path):
    with open(dic_path, 'rb') as f:
        read_name = pk.load(f)
    return read_name


# get the duration of audio file(s) in second or millisecond
def get_duration(file_array, frame_size, hop_length, sample_rate=22050, unit='second', n_decimal=5):
    """ file_array is a list of arrays containing the audio file arrays
        unit can be 'second' or 'millisecond'
        n_decimal is used for rounding the duration by number of decimals, defaulted in 5 """

    durs = []
    for audio in file_array:
        dur = librosa.get_duration(y=audio, sr=sample_rate, n_fft=frame_size, hop_length=hop_length)
        durs.append(round(dur, n_decimal))
    if unit == 'second':
        return np.array(durs)
    return np.array(durs) * 1000


# plot histogram of an array + get the quantile mean, arithmetic mean, maximum, and minimum
def plot_hist_get_mean(array, bins=None):
    """ array is a 1-D numpy array from which the histogram is plotted
        If bins not given, it is calculated based on https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule """

    max_array = array.max()
    min_array = array.min()
    q25, q75, q90 = np.percentile(array, [25, 75, 90])
    if bins is None:
        bin_width = 2 * (q75 - q25) * len(array) ** (-1 / 3)
        bins = round((max_array - min_array) / bin_width)
    plt.hist(array, density=False, bins=bins)  # density=False would make counts, True for probabilities
    return q25, q75, q90, np.mean(array), max_array, min_array


# function to visualize audio signal(s) in time domain (plot waveform)
def multi_plot_waveform(signals: list, titles: list, num_row: int = 1, num_col: int = 1, fig_size=(10, 5),
                        sample_rate=22050, suptitle: str = None, x_suptitle=0.5, y_suptitle=0.98, label_outer=True):
    """ Signals should be a list of arrays (arrays are signals)
        titles should be list of strings containing the title of each plot """

    assert num_row * num_col >= len(signals), f'Number of signals to be plotted can not be larger than' \
                                              f' number of row multiplied by number of columns'

    length = len(signals)
    signals = np.array(signals)
    titles = np.array(titles, dtype=object)
    temp = num_row * num_col - len(signals)
    if temp > 0:  # pad the arrays in case to reshape them to (num_row, num_col)
        signals = np.pad(signals, (0, temp), 'edge')
        titles = np.pad(titles, (0, temp), 'edge')

    if num_row == 1 and num_col == 1:
        plt.figure(figsize=fig_size)
        plt.suptitle(suptitle)
        plt.title(titles[0])
        plt.ylabel('Amplitude')
        librosa.display.waveshow(signals[0], sr=sample_rate)

    else:
        if num_col == 1:
            fig, axs = plt.subplots(num_row, 1, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                librosa.display.waveshow(signals[i], sr=sample_rate, ax=axs[i])
                axs[i].set_title(titles[i])

        elif num_row == 1:
            fig, axs = plt.subplots(1, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for j in range(num_col):
                librosa.display.waveshow(signals[j], sr=sample_rate, ax=axs[j])
                axs[j].set_title(titles[j])

        else:
            signals_reshaped = signals.reshape((num_row, num_col, -1))
            titles_reshaped = titles.reshape((num_row, num_col))
            fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                for j in range(num_col):
                    librosa.display.waveshow(signals_reshaped[i, j], sr=sample_rate, ax=axs[i, j])
                    axs[i, j].set_title(titles_reshaped[i, j])
            for ax in axs.flat[length:]:
                ax.set_visible(False)

        for ax in axs.flat:
            ax.set(ylabel='Amplitude')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if label_outer:
            for ax in axs.flat:
                ax.label_outer()
        else:
            for ax in axs.flat[: -num_col]:
                ax.set(xlabel='')

        fig.tight_layout()


def multi_plot_spectrogram(D_signals: list, titles: list, plot: str, num_row: int = 1, num_col: int = 1,
                           fig_size=(10, 5), suptitle: str = None, x_suptitle=0.5, y_suptitle=0.92, label_outer=True,
                           sample_rate=22050, frame_size=None, hop_length=256, x_axis='time', y_axis=None, cax=None,
                           orientation='vertical'):
    """ Signals should be a list of arrays (arrays are signals). Signals can be spectrogram, melspectrogram, or mfccs
        titles should be list of strings containing the title of each plot
        plot determines which type of diagram should be plotted. It can be 'spec', 'mel_spec', 'mfcc'
        y_axis can take different ranges, defaulted in None
        (see https://librosa.org/doc/main/generated/librosa.display.specshow.html#librosa.display.specshow)
        cax determines the location of color bar
        orientation determines the orientation of the color bar, either vertical or horizontal

        Note: If you get an error about "C dimension is not incompatible with X and Y (pcolormesh arguments)",
              you probably give "frame_size" a number, if so, you can simply avoid the error by passing "None"
              (which is default) to the "frame_size". The "frame_size" is automatically calculated.
               """

    global m
    if cax is None:
        cax = [0.95, 0.15, 0.03, 0.7]

    assert num_row * num_col >= len(D_signals), f'Number of signals to be plotted can not be larger than' \
                                                f' number of row multiplied by number of columns'

    length = len(D_signals)
    signals = np.array(D_signals)
    titles = np.array(titles, dtype=object)
    temp = num_row * num_col - len(signals)
    if temp > 0:  # pad the arrays in case to reshape them to (num_row, num_col)
        signals = np.pad(signals, (0, temp), 'edge')
        titles = np.pad(titles, (0, temp), 'edge')

    if num_row == 1 and num_col == 1:
        plt.figure(figsize=fig_size)
        plt.suptitle(suptitle)
        plt.title(titles[0])
        # plt.ylabel('Hz')
        librosa.display.specshow(signals[0], sr=sample_rate, n_fft=frame_size, hop_length=hop_length,
                                 x_axis=x_axis, y_axis=y_axis)
        if plot == 'mfcc':
            plt.colorbar(format="%+2.f", orientation=orientation)
        else:
            plt.colorbar(format="%+2.f dB", orientation=orientation)

    else:
        if num_col == 1:
            fig, axs = plt.subplots(num_row, 1, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                m = librosa.display.specshow(signals[i], ax=axs[i], sr=sample_rate, n_fft=frame_size,
                                             hop_length=hop_length, x_axis=x_axis, y_axis=y_axis)
                axs[i].set_title(titles[i])

        elif num_row == 1:
            fig, axs = plt.subplots(1, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for j in range(num_col):
                m = librosa.display.specshow(signals[j], ax=axs[j], sr=sample_rate, n_fft=frame_size,
                                             hop_length=hop_length, x_axis=x_axis, y_axis=y_axis)
                axs[j].set_title(titles[j])

        else:
            _, freq_bins, frames = signals.shape
            signals_reshaped = signals.reshape((num_row, num_col, freq_bins, -1))
            titles_reshaped = titles.reshape((num_row, num_col))
            fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                for j in range(num_col):
                    m = librosa.display.specshow(signals_reshaped[i, j], ax=axs[i, j], sr=sample_rate, n_fft=frame_size,
                                                 hop_length=hop_length, x_axis=x_axis, y_axis=y_axis)
                    axs[i, j].set_title(titles_reshaped[i, j])
            for ax in axs.flat[length:]:
                ax.set_visible(False)

        if plot == 'mfcc':
            fig.colorbar(m, cax=plt.axes(cax), format="%+2.f", orientation=orientation)
            for ax in axs.flat:
                ax.set(ylabel='Coefficients')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            fig.colorbar(m, cax=plt.axes(cax), format="%+2.f dB", orientation=orientation)
            for ax in axs.flat:
                ax.set(ylabel='Hz')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if label_outer:
            for ax in axs.flat:
                ax.label_outer()
        else:
            for ax in axs.flat[: -num_col]:
                ax.set(xlabel='')


def multi_plot_uneven(signals: list, titles: list, labels: list, num_plot=1, num_row: int = 1, num_col: int = 1,
                      fig_size=(10, 5), suptitle: str = None, x_suptitle=0.5, y_suptitle=0.92, y_label: list = None,
                      label_outer=True, legend_location: str = 'lower right', legend_fontsize=10, sample_rate=22050,
                      frame_size=None, hop_length=256):
    """
    This function can plot multiple signals particularly when they are intended to be grouped in a same color
        signals should be a list that potentially can takes lists or lists of list as elements
        titles should be a list corresponded to the signal list
        labels should be a list corresponded to the signal list
    """

    len_signals = len(signals)
    len_titles = len(titles)

    if not isinstance(signals[0], list):
        assert len_titles == 1, f'According to the given signals, titles can only have one element/string'
        assert len_signals == len(labels), f'According to the given signals, labels and signals should have' \
                                           f' the same length'
        assert num_plot == 1 and num_row == 1 and num_col == 1, f'According to the given signals, the number of plot,' \
                                                                f' row, and col should be equal to 1'
        plt.figure(figsize=fig_size)
        plt.title(titles[0])
        plt.xlabel('Time')
        plt.ylabel(y_label)
        color = iter(cm.rainbow(np.linspace(0, 1, len_signals)))
        for i in range(len_signals):
            c = next(color)
            times = librosa.times_like(signals[i], sr=sample_rate, hop_length=hop_length, n_fft=frame_size)
            plt.plot(times, signals[i], color=c, label=labels[i])
        plt.legend(loc=legend_location, fontsize=legend_fontsize)

    elif not isinstance(signals[0][0], list):
        assert len_titles == 1, f'According to the given signals, titles can have only one element/string'
        assert len_signals == len(labels), f'According to the given signals, labels and signals should have' \
                                           f' the same length'
        assert num_plot == 1 and num_row == 1 and num_col == 1, f'According to the given signals, the number of plot,' \
                                                                f' row, and col should be equal to 1'
        plt.figure(figsize=fig_size)
        plt.title(titles[0])
        plt.xlabel('Time')
        plt.ylabel(y_label)
        color = iter(cm.rainbow(np.linspace(0, 1, len_signals)))
        if len(labels[0]) == 1:
            for i in range(len_signals):
                c = next(color)
                for j in range(len(signals[i])):
                    times = librosa.times_like(signals[i][j], sr=sample_rate, hop_length=hop_length, n_fft=frame_size)
                    plt.plot(times, signals[i][j], color=c)
                plt.plot(times, signals[i][j], color=c, label=labels[i][0])
            plt.legend(loc=legend_location, fontsize=legend_fontsize)
        else:
            for i in range(len_signals):
                c = next(color)
                for j in range(len(signals[i])):
                    times = librosa.times_like(signals[i][j], sr=sample_rate, hop_length=hop_length, n_fft=frame_size)
                    plt.plot(times, signals[i][j], color=c, label=labels[i][j])
            plt.legend(loc=legend_location, fontsize=legend_fontsize)

    else:
        assert len_signals == num_plot, f'The number of plots and the length of signal list are not consistent'
        assert len_signals == len_titles, f'According to the given signals, labels and signals should have' \
                                          f' the same length'
        fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
        fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)

        if num_row == 1 or num_col == 1:
            assert num_row * num_col == len_signals, f'Number of signals to be plotted should be equal to' \
                                                     f' the number of rows multiplied by number of columns'
            if len(labels[0][0]) == 1:
                for i in range(len_signals):
                    color = iter(cm.rainbow(np.linspace(0, 1, len(signals[i]))))
                    for j in range(len(signals[i])):
                        c = next(color)
                        for k in range(len(signals[i][j])):
                            times = librosa.times_like(signals[i][j][k], sr=sample_rate, hop_length=hop_length,
                                                       n_fft=frame_size)
                            axs[i].plot(times, signals[i][j][k], color=c)
                        axs[i].plot(times, signals[i][j][0], color=c, label=labels[i][j][0])
                    axs[i].set_title(titles[i])
                    axs[i].legend(loc=legend_location, fontsize=legend_fontsize)
            else:
                for i in range(len_signals):
                    color = iter(cm.rainbow(np.linspace(0, 1, len(signals[i]))))
                    for j in range(len(signals[i])):
                        c = next(color)
                        for k in range(len(signals[i][j])):
                            times = librosa.times_like(signals[i][j][k], sr=sample_rate, hop_length=hop_length,
                                                       n_fft=frame_size)
                            axs[i].plot(times, signals[i][j][k], color=c, label=labels[i][j][k])
                    axs[i].set_title(titles[i])
                    axs[i].legend(loc=legend_location, fontsize=legend_fontsize)
        else:
            assert num_row * num_col >= len_signals, f'Number of signals to be plotted can not be larger than' \
                                                     f' number of rows multiplied by number of columns'
            if len(labels[0][0]) == 1:
                for i in range(len_signals):
                    color = iter(cm.rainbow(np.linspace(0, 1, len(signals[i]))))
                    for j in range(len(signals[i])):
                        c = next(color)
                        for k in range(len(signals[i][j])):
                            times = librosa.times_like(signals[i][j][k], sr=sample_rate, hop_length=hop_length,
                                                       n_fft=frame_size)
                            axs.flat[i].plot(times, signals[i][j][k], color=c)
                        axs.flat[i].plot(times, signals[i][j][0], color=c, label=labels[i][j][0])
                    axs.flat[i].set_title(titles[i])
                    axs.flat[i].legend(loc=legend_location, fontsize=legend_fontsize)
            else:
                for i in range(len_signals):
                    color = iter(cm.rainbow(np.linspace(0, 1, len(signals[i]))))
                    for j in range(len(signals[i])):
                        c = next(color)
                        for k in range(len(signals[i][j])):
                            times = librosa.times_like(signals[i][j][k], sr=sample_rate, hop_length=hop_length,
                                                       n_fft=frame_size)
                            axs.flat[i].plot(times, signals[i][j][k], color=c, label=labels[i][j][k])
                    axs.flat[i].set_title(titles[i])
                    axs.flat[i].legend(loc=legend_location, fontsize=legend_fontsize)

        for ax in axs.flat[len_signals:]:
            ax.set_visible(False)

        for ax in axs.flat:
            ax.set(xlabel='Time', ylabel=y_label)
            # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if label_outer:
            for ax in axs.flat:
                ax.label_outer()
        else:
            for ax in axs.flat[: -num_col]:
                ax.set(xlabel='')

        fig.tight_layout()


def multi_plot_spectral(signals: list, titles: list, num_row: int = 1, num_col: int = 1, fig_size=(10, 5),
                        suptitle: str = None, x_suptitle=0.5, y_suptitle=0.92, y_label: str = None, label_outer=True,
                        sample_rate=22050, frame_size=None, hop_length=256):
    """ Signals should be a list of arrays (arrays are signals). Signals can be 1 dimensional features like
        spectral centroid. This function also can plot grouped features, but to do so, signals should be
        a list of lists of arrays for which each list in the signal list is plotted in a same color.
        Titles should be list of strings containing the title of each plot. For grouped features, titles should be
        a list of lists of strings"""

    assert len(set(map(len, signals))) == 1, f'All elements of the signal list should have the same length'
    assert len(set(map(len, titles))) == 1, f'All elements of the titles list should have the same length'

    length = len(signals)
    signals = np.array(signals)
    titles = np.array(titles, dtype=object)
    temp = num_row * num_col - length
    if temp > 0:  # pad the arrays in case to reshape them to (num_row, num_col)
        signals = np.pad(signals, (0, temp), 'edge')
        titles = np.pad(titles, (0, temp), 'edge')

    if num_row == 1 and num_col == 1:
        if length == 1:
            plt.figure(figsize=fig_size)
            plt.suptitle(suptitle)
            plt.title(titles[0])
            plt.xlabel('Time')
            plt.ylabel(y_label)
            times = librosa.times_like(signals[0], sr=sample_rate, hop_length=hop_length, n_fft=frame_size)
            plt.plot(times, signals[0])
        else:
            plt.figure(figsize=fig_size)
            plt.suptitle(suptitle)
            plt.xlabel('Time')
            plt.ylabel(y_label)
            color = iter(cm.rainbow(np.linspace(0, 1, length)))
            if len(signals.shape) < 3:
                for i in range(length):
                    c = next(color)
                    times = librosa.times_like(signals[i], sr=sample_rate, hop_length=hop_length, n_fft=frame_size)
                    plt.plot(times, signals[i], color=c, label=titles[i])
                    plt.legend()
            else:
                for i in range(length):
                    c = next(color)
                    for j in range(signals.shape[1]):
                        times = librosa.times_like(signals[i, j], sr=sample_rate, hop_length=hop_length,
                                                   n_fft=frame_size)
                        plt.plot(times, signals[i, j], color=c, label=titles[i, j])
                        plt.legend()

    else:
        if num_col == 1:
            fig, axs = plt.subplots(num_row, 1, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                times = librosa.times_like(signals[i], sr=sample_rate, hop_length=hop_length, n_fft=frame_size)
                axs[i].plot(times, signals[i])
                axs[i].set_title(titles[i])

        elif num_row == 1:
            fig, axs = plt.subplots(1, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_col):
                times = librosa.times_like(signals[i], sr=sample_rate, hop_length=hop_length, n_fft=frame_size)
                axs[i].plot(times, signals[i])
                axs[i].set_title(titles[i])

        else:
            signals_reshaped = signals.reshape((num_row, num_col, -1))
            titles_reshaped = titles.reshape((num_row, num_col))
            fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                for j in range(num_col):
                    times = librosa.times_like(signals_reshaped[i, j], sr=sample_rate, hop_length=hop_length,
                                               n_fft=frame_size)
                    axs[i, j].plot(times, signals_reshaped[i, j])
                    axs[i, j].set_title(titles_reshaped[i, j])
            for ax in axs.flat[length:]:
                ax.set_visible(False)

        for ax in axs.flat:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if label_outer:
            for ax in axs.flat:
                ax.label_outer()
            for ax in axs.flat[:: num_col]:
                ax.set(ylabel=y_label)
            for ax in axs.flat[-num_col:]:
                ax.set(xlabel='Time')
        else:
            for ax in axs.flat[:: num_col]:
                ax.set(ylabel=y_label)
            for ax in axs.flat[:-num_col]:
                ax.set_xticklabels([])
            for ax in axs.flat[-num_col:]:
                ax.set(xlabel='Time')

        fig.tight_layout()


def multi_plot_overlap_specshow(background_signals: list, foreground_signals: list, titles: list,
                                labels_foreground_signals: list, plot: str, num_row: int = 1, num_col: int = 1,
                                legend_location: str = 'lower right', legend_fontsize=10, fig_size=(10, 5),
                                suptitle: str = None, x_suptitle=0.5, y_suptitle=0.92, label_outer=True,
                                sample_rate=22050, frame_size=None, hop_length=256, x_axis='time', y_axis=None,
                                cax=None, orientation='vertical'):
    """ background_signals should be a list of arrays (arrays are signals). Signals can be spectrogram, melspectrogram,
        or mfccs
        foreground_signals should be a list of lists containing array of signals to be plotted overlapping with
        background plot
        titles should be list of strings containing the title of each plot
        labels_foreground_signals is a lost of strings containing the label of each foreground signal
        plot determines which type of diagram should be plotted. It can be 'spec', 'mel_spec', 'mfcc'
        y_axis can take such ranges as 'linear', 'log', 'mel', defaulted in None
        (see https://librosa.org/doc/main/generated/librosa.display.specshow.html#librosa.display.specshow)
        cax determines the location of color bar
        orientation determines the orientation of color bar, either vertical or horizontal

        Note: If you get an error about "C dimension is not incompatible with X and Y (pcolormesh arguments)",
              you probably give "frame_size" a number, if so, you can simply avoid the error by passing "None"
              (which is default) to the "frame_size". The "frame_size" is automatically calculated."""

    global m
    if cax is None:
        cax = [0.95, 0.15, 0.03, 0.7]

    length_back = len(background_signals)
    length_fore = len(foreground_signals)
    length_fore_per_element = len(foreground_signals[0])

    assert num_row * num_col >= len(background_signals), f'Number of signals to be plotted can not be larger than' \
                                                         f' number of row multiplied by number of columns'

    assert len(set(map(len, foreground_signals))) == 1, f'All elements of foreground signal list should have' \
                                                        f' the same length'

    assert length_back == length_fore, f'Number of background signals and foreground signals are not consistent'

    assert len(labels_foreground_signals) == length_fore_per_element, f'Foreground signals and the corresponding' \
                                                                      f' labels are not consistent'

    signals_back = np.array(background_signals)
    signals_fore = np.array(foreground_signals)
    titles = np.array(titles, dtype=object)
    labels_foreground_signals = np.array(labels_foreground_signals, dtype=object)

    temp = num_row * num_col - length_back
    if temp > 0:  # pad the arrays in case to reshape them to (num_row, num_col)
        signals_back = np.pad(signals_back, (0, temp), 'edge')
        # signals_fore = np.vstack((signals_fore, signals_fore[-1].reshape((1, signals_fore.shape[1], -1))))
        signals_fore = np.repeat(signals_fore, [1 for i in range(length_fore - 1)] + [temp + 1], axis=0)
        titles = np.pad(titles, (0, temp), 'edge')

    if num_row == 1 and num_col == 1:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        fig.suptitle(suptitle)
        m = librosa.display.specshow(signals_back[0], ax=ax, sr=sample_rate, n_fft=frame_size, hop_length=hop_length,
                                     x_axis=x_axis, y_axis=y_axis)
        color = iter(cm.rainbow(np.linspace(0, 1, length_fore_per_element)))
        for i in range(length_fore_per_element):
            times = librosa.times_like(signals_fore[0, i], sr=sample_rate, hop_length=hop_length, n_fft=frame_size)
            c = next(color)
            ax.plot(times, signals_fore[0, i], label=labels_foreground_signals[i], color=c)
        ax.legend(loc=legend_location, fontsize=legend_fontsize)
        ax.set(title=titles[0])
        if plot == 'mfcc':
            fig.colorbar(m, cax=plt.axes(cax), format="%+2.f", orientation=orientation)
        else:
            fig.colorbar(m, cax=plt.axes(cax), format="%+2.f dB", orientation=orientation)
            ax.set(ylabel='Hz')

    else:
        if num_col == 1:
            fig, axs = plt.subplots(num_row, 1, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                m = librosa.display.specshow(signals_back[i], ax=axs[i], sr=sample_rate, n_fft=frame_size,
                                             hop_length=hop_length, x_axis=x_axis, y_axis=y_axis)
                color = iter(cm.rainbow(np.linspace(0, 1, length_fore_per_element)))
                for j in range(length_fore_per_element):
                    times = librosa.times_like(signals_fore[i, j], sr=sample_rate, hop_length=hop_length,
                                               n_fft=frame_size)
                    c = next(color)
                    axs[i].plot(times, signals_fore[i, j], label=labels_foreground_signals[j], color=c)
                axs[i].legend(loc=legend_location, fontsize=legend_fontsize)
                axs[i].set_title(titles[i])

        elif num_row == 1:
            fig, axs = plt.subplots(1, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for j in range(num_col):
                m = librosa.display.specshow(signals_back[j], ax=axs[j], sr=sample_rate, n_fft=frame_size,
                                             hop_length=hop_length, x_axis=x_axis, y_axis=y_axis)
                color = iter(cm.rainbow(np.linspace(0, 1, length_fore_per_element)))
                for i in range(length_fore_per_element):
                    times = librosa.times_like(signals_fore[j, i], sr=sample_rate, hop_length=hop_length,
                                               n_fft=frame_size)
                    c = next(color)
                    axs[j].plot(times, signals_fore[j, i], label=labels_foreground_signals[i], color=c)
                axs[j].legend(loc=legend_location, fontsize=legend_fontsize)
                axs[j].set_title(titles[j])

        else:
            signals_back_reshaped = signals_back.reshape((num_row, num_col, signals_back.shape[1], -1))
            signals_fore_reshaped = signals_fore.reshape((num_row, num_col, signals_fore.shape[1], -1))
            titles_reshaped = titles.reshape((num_row, num_col))
            fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                for j in range(num_col):
                    m = librosa.display.specshow(signals_back_reshaped[i, j], ax=axs[i, j], sr=sample_rate,
                                                 n_fft=frame_size, hop_length=hop_length, x_axis=x_axis, y_axis=y_axis)
                    color = iter(cm.rainbow(np.linspace(0, 1, length_fore_per_element)))
                    for k in range(length_fore_per_element):
                        times = librosa.times_like(signals_fore_reshaped[i, j, k], sr=sample_rate,
                                                   hop_length=hop_length, n_fft=frame_size)
                        c = next(color)
                        axs[i, j].plot(times, signals_fore_reshaped[i, j, k], label=labels_foreground_signals[k],
                                       color=c)
                    axs[i, j].legend(loc=legend_location, fontsize=legend_fontsize)
                    axs[i, j].set_title(titles_reshaped[i, j])
            for ax in axs.flat[length_back:]:
                ax.set_visible(False)

        if plot == 'mfcc':
            fig.colorbar(m, cax=plt.axes(cax), format="%+2.f", orientation=orientation)
            for ax in axs.flat[:: num_col]:
                ax.set(ylabel='Coefficients')
        else:
            fig.colorbar(m, cax=plt.axes(cax), format="%+2.f dB", orientation=orientation)
            for ax in axs.flat[-1:: -num_col]:
                ax.set(ylabel='')

        if label_outer:
            for ax in axs.flat:
                ax.label_outer()
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            for ax in axs.flat:
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            for ax in axs.flat[: -num_col]:
                ax.set(xlabel='')
                ax.set_xticklabels([])


def multi_plot_overlap_waveshow(background_signals: list, foreground_signals: list, titles: list,
                                labels_foreground_signals: list, num_row: int = 1, num_col: int = 1,
                                legend_location: str = 'lower right', legend_fontsize=10, fig_size=(10, 5),
                                suptitle: str = None, x_suptitle=0.5, y_suptitle=0.92, label_outer=True,
                                sample_rate=22050, frame_size=None, hop_length=256, color_back_sig='b',
                                alpha_back_sig=None):
    """ background_signals should be a list of arrays (arrays are signals). Signals are plotted in time domain
        foreground_signals should be a list of lists containing array of signals to be plotted overlapping
        with background plot
        titles should be a list of strings containing the title of each plot
        labels_foreground_signals is a list of strings containing the label of each foreground signal """

    length_back = len(background_signals)
    length_fore = len(foreground_signals)
    length_fore_per_element = len(foreground_signals[0])

    assert num_row * num_col >= len(background_signals), f'Number of signals to be plotted can not be larger than' \
                                                         f' number of row multiplied by number of columns'

    assert len(set(map(len, foreground_signals))) == 1, f'All elements of foreground signal list should have' \
                                                        f' the same length'

    assert length_back == length_fore, f'Number of background signals and foreground signals are not consistent'

    assert len(
        labels_foreground_signals) == length_fore_per_element, f'Foreground signals and the corresponding labels' \
                                                               f' are not consistent'

    signals_back = np.array(background_signals)
    signals_fore = np.array(foreground_signals)
    titles = np.array(titles, dtype=object)
    labels_foreground_signals = np.array(labels_foreground_signals, dtype=object)

    temp = num_row * num_col - length_back
    if temp > 0:  # pad the arrays in case to reshape them to (num_row, num_col)
        signals_back = np.pad(signals_back, (0, temp), 'edge')
        # signals_fore = np.vstack((signals_fore, signals_fore[-1].reshape((1, signals_fore.shape[1], -1))))
        signals_fore = np.repeat(signals_fore, [1 for i in range(length_fore - 1)] + [temp + 1], axis=0)
        titles = np.pad(titles, (0, temp), 'edge')

    if num_row == 1 and num_col == 1:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        fig.suptitle(suptitle)
        librosa.display.waveshow(y=signals_back[0], ax=ax, sr=sample_rate, color=color_back_sig, alpha=alpha_back_sig)
        color = iter(cm.rainbow(np.linspace(0, 1, length_fore_per_element)))
        for i in range(length_fore_per_element):
            times = librosa.times_like(signals_fore[0, i], sr=sample_rate, hop_length=hop_length, n_fft=frame_size)
            c = next(color)
            ax.plot(times, signals_fore[0, i], label=labels_foreground_signals[i], color=c)
        ax.legend(loc=legend_location, fontsize=legend_fontsize)
        ax.set(title=titles[0])
        ax.set(ylabel='Amplitude')

    else:
        if num_col == 1:
            fig, axs = plt.subplots(num_row, 1, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                librosa.display.waveshow(y=signals_back[i], ax=axs[i], sr=sample_rate, color=color_back_sig,
                                         alpha=alpha_back_sig)
                color = iter(cm.rainbow(np.linspace(0, 1, length_fore_per_element)))
                for j in range(length_fore_per_element):
                    times = librosa.times_like(signals_fore[i, j], sr=sample_rate, hop_length=hop_length,
                                               n_fft=frame_size)
                    c = next(color)
                    axs[i].plot(times, signals_fore[i, j], label=labels_foreground_signals[j], color=c)
                axs[i].legend(loc=legend_location, fontsize=legend_fontsize)
                axs[i].set(ylabel='Amplitude')
                axs[i].set_title(titles[i])
                axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        elif num_row == 1:
            fig, axs = plt.subplots(1, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for j in range(num_col):
                librosa.display.waveshow(y=signals_back[j], ax=axs[j], sr=sample_rate, color=color_back_sig,
                                         alpha=alpha_back_sig)
                color = iter(cm.rainbow(np.linspace(0, 1, length_fore_per_element)))
                for i in range(length_fore_per_element):
                    times = librosa.times_like(signals_fore[j, i], sr=sample_rate, hop_length=hop_length,
                                               n_fft=frame_size)
                    c = next(color)
                    axs[j].plot(times, signals_fore[j, i], label=labels_foreground_signals[i], color=c)
                axs[j].legend(loc=legend_location, fontsize=legend_fontsize)
                axs[j].set(ylabel='Amplitude')
                axs[j].set_title(titles[j])
                axs[j].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        else:
            signals_back_reshaped = signals_back.reshape((num_row, num_col, -1))
            signals_fore_reshaped = signals_fore.reshape((num_row, num_col, signals_fore.shape[1], -1))
            titles_reshaped = titles.reshape((num_row, num_col))
            fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
            fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
            for i in range(num_row):
                for j in range(num_col):
                    librosa.display.waveshow(signals_back_reshaped[i, j], ax=axs[i, j], sr=sample_rate,
                                             color=color_back_sig, alpha=alpha_back_sig)
                    color = iter(cm.rainbow(np.linspace(0, 1, length_fore_per_element)))
                    for k in range(length_fore_per_element):
                        times = librosa.times_like(signals_fore_reshaped[i, j, k], sr=sample_rate,
                                                   hop_length=hop_length, n_fft=frame_size)
                        c = next(color)
                        axs[i, j].plot(times, signals_fore_reshaped[i, j, k], label=labels_foreground_signals[k],
                                       color=c)
                    axs[i, j].legend(loc=legend_location, fontsize=legend_fontsize)
                    axs[i, j].set_title(titles_reshaped[i, j])

                for ax in axs.flat[length_back:]:
                    ax.set_visible(False)

            if label_outer:
                for ax in axs.flat:
                    ax.label_outer()
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            else:
                for ax in axs.flat:
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                for ax in axs.flat[: -num_col]:
                    ax.set(xlabel='')
                    ax.set_xticklabels([])
                for ax in axs.flat[:: num_col]:
                    ax.set(ylabel='Amplitude')


def multi_plot_box(signals: list, titles: list, labels: list, num_row: int = 1, num_col: int = 1,
                   fig_size=(10, 5), suptitle: str = None, x_suptitle=0.5, y_suptitle=0.92, y_label: str = None,
                   label_outer=True, vert=True, patch_artist=True, notch=True, median_color=None):
    """
        This function can plot one or more box plots
        signals should be a list that potentially can takes lists or lists of list as elements
        titles should be a list corresponded to the signal list
        labels should be a list corresponded to the signal list

        Note: when plotting more than one plots, if the labels list has lists with different length, the label_outer
        should be set to False
    """

    len_signals = len(signals)
    len_titles = len(titles)

    if not isinstance(signals[0], list):
        assert len_titles == 1, f'According to the given signals, titles can only have one element/string'
        assert len_signals == len(labels), f'According to the given signals, labels and signals should have' \
                                           f' the same length'
        plt.figure(figsize=fig_size)
        plt.title(titles[0])
        plt.ylabel(y_label)
        color = iter(cm.rainbow(np.linspace(0, 1, len_signals)))
        box = plt.boxplot(signals, labels=labels, patch_artist=patch_artist, vert=vert, notch=notch)
        if patch_artist:
            for patch in box['boxes']:
                c = next(color)
                patch.set_facecolor(c)
        if median_color is not None:
            for median in box['medians']:
                median.set(color=median_color, linewidth=3)

    elif not isinstance(signals[0][0], list):
        assert len_titles == 1, f'According to the given signals, titles can have only one element/string'
        assert len_signals == len(labels), f'According to the given signals, labels and signals should have' \
                                           f' the same length'
        assert num_row == 1 and num_col == 1, f'According to the given signals, the number of row and col should be' \
                                              f' equal to 1'
        plt.figure(figsize=fig_size)
        plt.title(titles[0])
        plt.ylabel(y_label)
        if isinstance(signals[0][0], np.ndarray):
            data = [np.concatenate(signal) for signal in signals]
        else:
            data = signals
        color = iter(cm.rainbow(np.linspace(0, 1, len_signals)))
        box = plt.boxplot(data, labels=labels, patch_artist=patch_artist, vert=vert, notch=notch)
        if patch_artist:
            for patch in box['boxes']:
                c = next(color)
                patch.set_facecolor(c)
        if median_color is not None:
            for median in box['medians']:
                median.set(color=median_color, linewidth=3)
    else:
        assert len_signals == len_titles, f'According to the given signals, labels and signals should have' \
                                          f' the same length'
        assert num_row * num_col >= len_signals, f'Number of signals to be plotted can not be larger than' \
                                                 f' number of rows multiplied by number of columns'
        fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
        fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
        for i in range(len_signals):
            data = [np.concatenate(signal) for signal in signals[i]]
            color = iter(cm.rainbow(np.linspace(0, 1, len(signals[i]))))
            box = axs.flat[i].boxplot(data, labels=labels[i], patch_artist=patch_artist, vert=vert, notch=notch)
            axs.flat[i].set_title(titles[i])
            if patch_artist:
                for patch in box['boxes']:
                    c = next(color)
                    patch.set_facecolor(c)
            if median_color is not None:
                for median in box['medians']:
                    median.set(color=median_color, linewidth=3)

        for ax in axs.flat[len_signals:]:
            ax.set_visible(False)
        for ax in axs.flat:
            ax.set(ylabel=y_label)

        if label_outer:
            for ax in axs.flat:
                ax.label_outer()
        fig.tight_layout()


def multi_plot_scatter(x: list, y: list, titles: list, labels: list = None, num_plot=1, num_row: int = 1,
                       num_col: int = 1, polynomial=True, poly_deg=1, label_fontsize=10, label_cols=1,
                       label_loc='upper left', points_alpha=1, x_text=0.65, y_text=0.25, fig_size=(10, 5),
                       suptitle: str = None, x_suptitle=0.5, y_suptitle=0.92, x_label: str = None, y_label: str = None,
                       label_outer=True):
    """
        x and y should be a list of array(s) when num_plot is 1, and should be a list of lists of array(s) when multiple
        plots requested.
        Titles should be a list of strings containing the title for each plot.
        Labels should be a list of string(s) or a list of lists of string(s)
        If polynomial is True (default), the polynomial of poly_deg between x and y is illustrated on the plot using
        Polynomial package of Numpy
        (for more info, see https://numpy.org/doc/stable/reference/routines.polynomials.html)
        poly_deg determines the degree(s) of the fitting polynomials (only available if polynomial is True)
    """

    len_x = len(x)
    len_y = len(y)
    len_titles = len(titles)
    assert len_x == len_y, f'The length of x and y lists are not consistent'

    if num_plot == 1:
        assert len_titles == 1, f'According to the given x/y, titles can only have one element/string'
        assert num_row == num_col == 1, f'When num_plot = 1, num_row and num_col should be equal to 1'

        if len_x == 1:
            assert len(labels) == 1, f'According to the given x/y, labels can only have one element/string'
            plt.figure(figsize=fig_size)
            plt.title(titles[0])
            plt.ylabel(y_label)
            plt.xlabel(x_label)
            s = plt.scatter(x[0], y[0], alpha=points_alpha)
            plt.legend((s,), (f'{labels[0]} = {np.corrcoef(x[0], y[0])[0, 1]:0.3f}',), loc=label_loc,
                       ncol=label_cols, fontsize=label_fontsize)
            if polynomial is True:
                coeffs = Polynomial.fit(x[0], y[0], poly_deg).convert().coef
                corr = Polynomial(coeffs)
                y_hat = corr(x[0])
                plt.plot(x[0], y_hat, color='r')
                text = f"$y={coeffs[1]:0.3f}\;x{coeffs[0]:+0.3f}$\n$R^2 = {r2_score(y[0], y_hat):0.3f}$"
                plt.gca().text(x_text, y_text, text, transform=plt.gca().transAxes, fontsize=10,
                               verticalalignment='top')

        else:
            assert polynomial is False, f'According to the given x, y, and number of plot, polynomial cannot be True'
            assert len_x == len_y == len(labels), f'The length of x/y lists and label list should be equal'
            plt.figure(figsize=fig_size)
            plt.title(titles[0])
            plt.ylabel(y_label)
            plt.xlabel(x_label)
            colors = iter(cm.rainbow(np.linspace(0, 1, len_x)))
            markers = iter(['o', '+', 'x', '*', 'p', '.'] * (len_x // 6 + 1))
            scatters = []
            ccs = []
            for i in range(len_x):
                s = plt.scatter(x[i], y[i], color=next(colors), marker=next(markers), alpha=points_alpha)
                scatters.append(s)
                cc = f'{labels[i]} = {np.corrcoef(x[i], y[i])[0, 1]:0.3f}'
                ccs.append(cc)
            plt.legend(scatters, ccs, loc=label_loc, ncol=label_cols, fontsize=label_fontsize)

    else:
        assert len_x == len_y == len_titles == len(labels), f'x, y, and num_plots are not consistent' \
                                                            f' (they all should have an equivalent length)'
        assert num_row * num_col >= len_x, f'Number of signals to be plotted can not be larger than number of rows' \
                                           f' multiplied by number of columns'

        fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
        fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)
        if len(x[0]) == 1:
            for i in range(len_x):
                s = axs.flat[i].scatter(x[i][0], y[i][0], alpha=points_alpha)
                axs.flat[i].set_title(titles[i])
                axs.flat[i].legend((s,), (f'{labels[i]} = {np.corrcoef(x[i][0], y[i][0])[0, 1]:0.3f}',),
                                   loc=label_loc, ncol=label_cols, fontsize=label_fontsize)
                if polynomial is True:
                    coeffs = Polynomial.fit(x[i][0], y[i][0], poly_deg).convert().coef
                    corr = Polynomial(coeffs)
                    y_hat = corr(x[i][0])
                    axs.flat[i].plot(x[i][0], y_hat, color='r')
                    text = f"$y={coeffs[1]:0.3f}\;x{coeffs[0]:+0.3f}$\n$R^2 = {r2_score(y[i][0], y_hat):0.3f}$"
                    axs.flat[i].text(x_text, y_text, text, transform=axs.flat[i].transAxes, fontsize=label_fontsize,
                                     verticalalignment='top')
        else:
            assert polynomial is False, f'According to the given x, y, and number of plot, polynomial cannot be True'
            for i in range(len_x):
                colors = iter(cm.rainbow(np.linspace(0, 1, len(x[i]))))
                markers = iter(['o', '+', 'x', '*', 'p', '.'] * (len(x[i]) // 6 + 1))
                scatters = []
                ccs = []
                for j in range(len(x[i])):
                    s = axs.flat[i].scatter(x[i][j], y[i][j], color=next(colors), marker=next(markers),
                                            alpha=points_alpha)
                    scatters.append(s)
                    cc = f'{labels[i][j]} = {np.corrcoef(x[i][j], y[i][j])[0, 1]:0.3f}'
                    ccs.append(cc)
                axs.flat[i].legend(scatters, ccs, scatterpoints=1, loc=label_loc, ncol=label_cols,
                                   fontsize=label_fontsize)
                axs.flat[i].set_title(titles[i])

        for ax in axs.flat[len_x:]:
            ax.set_visible(False)

        if label_outer:
            for ax in axs.flat:
                ax.label_outer()
            for ax in axs.flat[:: num_col]:
                ax.set(ylabel=y_label)
            for ax in axs.flat[-num_col:]:
                ax.set(xlabel=x_label)
        else:
            for ax in axs.flat[:: num_col]:
                ax.set(ylabel=y_label)
            for ax in axs.flat[:-num_col]:
                ax.set_xticklabels([])
            for ax in axs.flat[-num_col:]:
                ax.set(xlabel=x_label)


def multi_plot_bar(heights: list, bars: list, titles: list, num_plot=1, num_row: int = 1,
                   num_col: int = 1, bar_width: int = None, intra_width: int = None, edge_color='black',
                   fig_size=(10, 5), suptitle: str = None, x_suptitle=0.5, y_suptitle=0.92, x_label: str = None,
                   y_label: str = None, label_fontweight='normal', label_fontsize=10, label_outer=True):
    """
        heights (the values of each bar) can be a list of array(s)/list(s) or list of lists of array(s).
        bars (the label of each bar) can be a list of string(s) or lists of string(s)
        Titles (the label of each plot) should be a list of string(s) containing the title for each plot.
    """

    len_h = len(heights)
    len_titles = len(titles)

    if num_plot == 1:
        assert len_h == len(bars), f'The length of heights and bars lists are not consistent'
        assert len_titles == 1, f'According to the given heights/bars, titles can only have one element/string'
        assert num_row == num_col == 1, f'When num_plot = 1, num_row and num_col should be equal to 1'
        assert len_titles == 1, f'According to the given num_plot, titles can only have one element/string'

        if not isinstance(heights[0], list):
            bars_pos = np.arange(len(bars))
            colors = iter(cm.rainbow(np.linspace(0, 1, len_h)))
            plt.figure(figsize=fig_size)
            plt.xlabel(x_label, fontweight=label_fontweight, fontsize=label_fontsize)
            plt.ylabel(y_label, fontweight=label_fontweight, fontsize=label_fontsize)
            plt.title(titles[0])
            plt.bar(bars_pos, heights, width=bar_width, color=[next(colors) for _ in range(len_h)],
                    edgecolor=edge_color)
            plt.xticks(bars_pos, bars)
        else:
            # transpose the list of heights and if necessary fill the no data with 0 to have equal length for each list
            # in heights
            transposed_heights = list(map(list, zip_longest(*heights, fillvalue=0)))
            len_longest_list = max(list(map(len, transposed_heights)))
            position = np.arange(len_longest_list)
            colors = iter(cm.rainbow(np.linspace(0, 1, len(transposed_heights))))
            plt.figure(figsize=fig_size)
            plt.xlabel(x_label, fontweight=label_fontweight, fontsize=label_fontsize)
            plt.ylabel(y_label, fontweight=label_fontweight, fontsize=label_fontsize)
            plt.title(titles[0])
            for i in range(len(transposed_heights)):
                plt.bar(position + (i * intra_width), transposed_heights[i], width=bar_width, color=next(colors),
                        edgecolor=edge_color)
            plt.xticks([r + (len_h // 2 * intra_width) for r in range(len_longest_list)], bars)

    else:
        assert len_h == len_titles, f'The length of heights and titles lists are not consistent'
        assert num_row * num_col >= len_h, f'Number of bar graphs to be plotted can not be larger than number of' \
                                           f' rows multiplied by number of columns'
        fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
        fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)

        if not isinstance(heights[0][0], list):
            assert bars is None, f'According to the given heights and num_plot, the bars argument should be "None"'
            for i in range(len_h):
                l = len(heights[i])
                bars_pos = np.arange(l)
                colors = iter(cm.rainbow(np.linspace(0, 1, l)))
                axs.flat[i].set_title(titles[i])
                axs.flat[i].bar(bars_pos, heights[i], width=bar_width, color=[next(colors) for _ in range(l)],
                                edgecolor=edge_color)
                axs.flat[i].set_xticks(bars_pos, [num_cough for num_cough in range(1, l + 1)])
        else:
            for i in range(len_h):
                transposed_heights = list(map(list, zip_longest(*heights[i], fillvalue=0)))
                l = len(transposed_heights)
                len_longest_list = max(list(map(len, transposed_heights)))
                position = np.arange(len_longest_list)
                colors = iter(cm.rainbow(np.linspace(0, 1, l)))
                axs.flat[i].set_title(titles[i])
                for j in range(l):
                    axs.flat[i].bar(position + (j * intra_width), transposed_heights[j], width=bar_width,
                                    color=next(colors), edgecolor=edge_color)
                    axs.flat[i].set_xticks([r + (l // 2 * intra_width) for r in range(len_longest_list)], bars[i])

        for ax in axs.flat[len_h:]:
            ax.set_visible(False)

        if label_outer:
            for ax in axs.flat:
                ax.label_outer()
            for ax in axs.flat[:: num_col]:
                ax.set_ylabel(ylabel=y_label, fontweight=label_fontweight, fontsize=label_fontsize)
            for ax in axs.flat[-num_col:]:
                ax.set_xlabel(xlabel=x_label, fontweight=label_fontweight, fontsize=label_fontsize)
        else:
            for ax in axs.flat[:: num_col]:
                ax.set_ylabel(ylabel=y_label, fontweight=label_fontweight, fontsize=label_fontsize)
            for ax in axs.flat[-num_col:]:
                ax.set_xlabel(xlabel=x_label, fontweight=label_fontweight, fontsize=label_fontsize)


def multi_plot_simple(x: list, y: list, titles: list, labels: list = None, label_loc='upper right',
                      log_axis: str = None,
                      num_plot=1, num_row: int = 1, num_col: int = 1, fig_size=(10, 5), legend_fontsize=8,
                      suptitle: str = None, x_suptitle=0.5, y_suptitle=0.92, x_label: str = None, y_label: str = None,
                      label_outer=True):
    """
        This function plots y vs. x in linear or log scales.
        x and y should be a list of array(s) when num_plot is 1, and should be a list of arrays or a list of lists of
        array(s) when multiple plots requested.
        Titles should be a list of strings containing the title for each plot.
        Labels should be a list of string(s) or a list of lists of string(s) according to the x and y
        log_axis can be 'x', 'y', or 'both' to determine which axis should be logarithmically plotted. If None
        (default), both axes will be linearly plotted
    """

    len_x = len(x)
    len_y = len(y)
    len_titles = len(titles)
    assert len_x == len_y, f'The length of x and y lists are not consistent'

    if num_plot == 1:
        assert len_titles == 1, f'According to the given x/y, titles can only have one element/string'
        assert num_row == num_col == 1, f'When num_plot = 1, num_row and num_col should be equal to 1'

        plt.figure(figsize=fig_size)
        plt.title(titles[0])
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if len_x == 1:
            match log_axis:
                case 'y':
                    plt.semilogy(x[0], y[0])
                case 'x':
                    plt.semilogx(x[0], y[0])
                case 'both':
                    plt.loglog(x[0], y[0])
                case _:
                    plt.plot(x[0], y[0])
        else:
            colors = iter(cm.rainbow(np.linspace(0, 1, len_x)))
            # markers = iter(['o', '+', 'x', '*', 'p', '.'] * (len_x // 6 + 1))
            alphas = iter(np.flip(np.linspace(0.3, 0.7, len_x)))
            for i in range(len_x):
                match log_axis:
                    case 'y':
                        plt.semilogy(x[i], y[i], color=next(colors), alpha=next(alphas), label=labels[i])
                    case 'x':
                        plt.semilogx(x[i], y[i], color=next(colors), alpha=next(alphas), label=labels[i])
                    case 'both':
                        plt.loglog(x[i], y[i], color=next(colors), alpha=next(alphas), label=labels[i])
                    case _:
                        plt.plot(x[i], y[i], color=next(colors), alpha=next(alphas), label=labels[i])
            plt.legend(loc=label_loc, fontsize=legend_fontsize)

    else:
        assert len_x == len_y == len_titles, f'x, y, and num_plots are not consistent' \
                                             f'(they all should have an equivalent length)'
        assert num_plot >= len_x, f'The length of x/y lists can not be larger than the number of plots'

        fig, axs = plt.subplots(num_row, num_col, figsize=fig_size)
        fig.suptitle(suptitle, x=x_suptitle, y=y_suptitle)

        if not isinstance(x[0], list):
            for i in range(len_x):
                match log_axis:
                    case 'y':
                        plt.semilogy(x[i], y[i])
                    case 'x':
                        plt.semilogx(x[i], y[i])
                    case 'both':
                        axs.flat[i].loglog(x[i], y[i])
                    case _:
                        axs.flat[i].plot(x[i], y[i])
                axs.flat[i].set_title(titles[i])
        else:
            for i in range(len_x):
                colors = iter(cm.rainbow(np.linspace(0, 1, len(x[i]))))
                # markers = iter(['o', '+', 'x', '*', 'p', '.'] * (len(x[i]) // 6 + 1))
                alphas = iter(np.flip(np.linspace(0.3, 0.7, len(x[i]))))
                for j in range(len(x[i])):
                    match log_axis:
                        case 'y':
                            axs.flat[i].semilogy(x[i][j], y[i][j], color=next(colors), alpha=next(alphas),
                                                 label=labels[i][j])
                        case 'x':
                            axs.flat[i].semilogx(x[i][j], y[i][j], color=next(colors), alpha=next(alphas),
                                                 label=labels[i][j])
                        case 'both':
                            axs.flat[i].loglog(x[i][j], y[i][j], color=next(colors), alpha=next(alphas),
                                               label=labels[i][j])
                        case _:
                            axs.flat[i].plot(x[i][j], y[i][j], color=next(colors), alpha=next(alphas),
                                             label=labels[i][j])
                axs.flat[i].legend(loc=label_loc, fontsize=legend_fontsize)
                axs.flat[i].set_title(titles[i])

        for ax in axs.flat[len_x:]:
            ax.set_visible(False)

        if label_outer:
            for ax in axs.flat:
                ax.label_outer()
            for ax in axs.flat[:: num_col]:
                ax.set(ylabel=y_label)
            for ax in axs.flat[-num_col:]:
                ax.set(xlabel=x_label)
        else:
            for ax in axs.flat[:: num_col]:
                ax.set(ylabel=y_label)
            for ax in axs.flat[:-num_col]:
                ax.set_xticklabels([])
            for ax in axs.flat[-num_col:]:
                ax.set(xlabel=x_label)


def advanced_multi_spectrum(signals, sr=22050, f_ratio=0.5, equal_length=True):
    """
    This function calculates fourier transform (FT) of the given signals, and returns the magnitude of FTs together with
    frequency bins as a list of sets or a dictionary of sets
    signals can be a list of arrays or a dictionary of which the values should be the arrays
    sr: sampling rate
    f_ratio: the ratio that determines the number of frequency bins (if 0.5 (default), the bins will be considered up to
    Nyquist frequency (sr/2))
    equal_length: determines whether or not all signals/arrays are equal in length (i.e., the number of samples of
    each signal are the same or not)
    """

    if equal_length:
        if isinstance(signals, dict):
            n_samples = len(list(signals.values())[0])
            f = np.linspace(0, sr, n_samples)
            f_bins = int(n_samples * f_ratio)

            output_dic = dict()
            for key, value in signals.items():
                mag_ft = np.absolute(np.fft.fft(value))
                output_dic[key] = (f[:f_bins], mag_ft[:f_bins])

            return output_dic
        else:
            n_samples = len(signals[0])
            f = np.linspace(0, sr, n_samples)
            f_bins = int(n_samples * f_ratio)

            output_list = list()
            for i in range(len(signals)):
                mag_ft = np.absolute(np.fft.fft(signals[i]))
                output_list.append((f[:f_bins], mag_ft[:f_bins]))

            return output_list

    else:
        if isinstance(signals, dict):
            output_dic = dict()
            for key, value in signals.items():
                mag_ft = np.absolute(np.fft.fft(value))
                n_samples = len(value)
                f = np.linspace(0, sr, n_samples)
                f_bins = int(n_samples * f_ratio)
                output_dic[key] = (f[:f_bins], mag_ft[:f_bins])

            return output_dic
        else:
            output_list = list()
            for i in range(len(signals)):
                mag_ft = np.absolute(np.fft.fft(signals[i]))
                n_samples = len(signals[i])
                f = np.linspace(0, sr, n_samples)
                f_bins = int(n_samples * f_ratio)
                output_list.append((f[:f_bins], mag_ft[:f_bins]))

            return output_list


def stat_dur(path_audios, sample_rate, frame_size, hop_length, unit='second', n_decimal=5):
    audios_array = []
    for root, _, files in os.walk(path_audios):
        for file in files:
            f = os.path.join(root, file)
            audios_array.append(librosa.load(f, sr=sample_rate)[0])
    durations = get_duration(audios_array, frame_size=frame_size, hop_length=hop_length, sample_rate=sample_rate,
                             unit=unit, n_decimal=n_decimal)
    q25, q75, q90, mean_arithmetic, max_durations, min_durations = plot_hist_get_mean(durations)
    return q25, q75, q90, mean_arithmetic, max_durations, min_durations


def multi_load(files_dir, feature_name: str = None, allow_pickle=False):
    """
    The function returns a dictionary of loaded files. Files can be either in .wav format or in .npy format.
    files_dir is the path of the folder containing the files to be read.
    if feature_name is 'spec', the keys of dictionary preceded by an 'spec'
    Set allow_pickle True when loading object arrays

    Note: To use this function, the hierarchy of files should be the same as below:
    """

    if feature_name is None:
        feature_name = os.path.basename(os.path.dirname(files_dir))

    loads = {}

    if feature_name == 'audio' or feature_name == 'audios':
        for file in os.listdir(files_dir):
            # n = 'orgSig_' + file[-12: -4]
            n = file[-12: -4]
            loads[n], _ = librosa.load(os.path.join(files_dir, file))

    else:
        for file in os.listdir(files_dir):
            if file.endswith(".npy"):
                # n = feature_name + file[-12: -4]
                n = file[-12: -4]
                loads[n] = np.load(os.path.join(files_dir, file), allow_pickle=allow_pickle)

    return loads


def multi_load_spirometry(files_dir, allow_pickle=True, as_dataframe=False):
    """
    The function returns a dictionary of loaded files. Files should be in .npy format.
    files_dir is the path of the folder containing the files to be read.
    allow_pickle argument is set to True by default because of loading object arrays
    If as_dataframe is True, the returned dictionary will have pandas dataframe as keys rather than arrays
    """

    loads = dict()
    for file in os.listdir(files_dir):
        n = file[:4] + file[-6:-4]
        if as_dataframe:
            load = np.load(os.path.join(files_dir, file), allow_pickle=allow_pickle)
            loads[n] = pd.DataFrame(data=load[1:, :], columns=load[0, :])
        else:
            loads[n] = np.load(os.path.join(files_dir, file), allow_pickle=allow_pickle)

    return loads


def denormalise(norm_array, original_min, original_max):
    array = (norm_array - 0) / (1 - 0)
    array = array * (original_max - original_min) + original_min
    return array


def multi_denorm(feature_dic, min_max_dic):
    """ This function takes a dictionary containing the feature to be denormalised (feature_dic) and the dictionary
        containing the corresponding min and max values of the feature (min_max_dic), and returns a dictionary just
        like feature_dic, but with denormalised feature values """

    denorm_feature = {}
    for key, value in feature_dic.items():
        denorm_array = denormalise(value, min_max_dic[key]['min'], min_max_dic[key]['max'])
        denorm_feature[key] = denorm_array

    return denorm_feature


def mean_percentile(array, low_percentile=10, high_percentile=90):
    """ This function returns the average of the values of an array which are between the given low_percentile and
        high percentile
        array can be a numpy array or a list """

    low = np.percentile(array, low_percentile)
    high = np.percentile(array, high_percentile)
    reduced_array = [value for value in array if low < value < high]

    return np.mean(reduced_array)


def simple_multi_spectrum(signals, sr=22050, f_ratio=0.5):
    """
    This function calculates fourier transform (FT) of the given signals, and returns the magnitude of FTs together with
    frequency bins
    signals should be list of arrays
    sr: sampling rate
    f_ratio: the ratio that determines the number of frequency bins (if 0.5, the bins will be considered up to
    Nyquist frequency (sr/2))
    """
    bins = []
    mag_fts = []

    for i in range(len(signals)):
        sig = signals[i]
        Y = np.fft.fft(sig)
        Y_mag = np.absolute(Y)
        f = np.linspace(0, sr, len(Y_mag))
        f_bins = int(len(Y_mag) * f_ratio)

        mag_fts.append(Y_mag[:f_bins])
        bins.append(f[:f_bins])

    return bins, mag_fts


def multi_psd(signals, sr=22050, nperseg=512, noverlap=256, nfft=512, scaling='density', **kwargs):
    """
    This function calculates power spectral density or power spectrum of the given signals, and returns the powers
    together with frequency bins as a list of sets or a dictionary of sets (for more details see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
    signals can be a list of arrays or a dictionary of which the values should be the arrays.
    """

    if isinstance(signals, dict):
        output_dic = dict()
        for key, value in signals.items():
            frequencies, powers = sp.signal.welch(value, fs=sr, nperseg=nperseg, noverlap=noverlap,
                                                  nfft=nfft, scaling=scaling, **kwargs)
            output_dic[key] = (frequencies, powers)

        return output_dic
    else:
        output_list = list()
        for i in range(len(signals)):
            frequencies, powers = sp.signal.welch(signals[i], fs=sr, nperseg=nperseg, noverlap=noverlap,
                                                  nfft=nfft, scaling=scaling, **kwargs)
            output_list.append((frequencies, powers))

        return output_list


def compute_mean_by_substring(df, column: str, substrings: list):
    """
    Computes the mean of the rows in a Pandas dataframe that contain a given list of substrings in a specified column.

    Parameters:
    - df: Pandas dataframe
    - column: name of the column to be queried
    - substrings: list of substrings that are used for query

    Returns:
    - Pandas dataframe with the mean of the rows that contain the substrings in the specified column.
    """

    dfs_mean = []
    # df_mean = pd.DataFrame()

    for s in substrings:
        # select the rows that contain the substring in the specified column
        df_filtered = df[df[column].str.contains(s)]

        # compute the mean of the rows
        df_temp = df_filtered.mean(numeric_only=True).to_frame().T

        # rename the index column to the substring
        df_temp = df_temp.reset_index(drop=True)
        df_temp.insert(0, column, s)

        dfs_mean.append(df_temp)
        # df_mean = df_mean.append(df_temp, ignore_index=True)

    df_means = pd.concat(dfs_mean, ignore_index=True)
    return df_means


def average_arrays(arrays):
    """
    Calculate the average of a list of NumPy arrays.

    Parameters:
    arrays (list): A list of NumPy arrays. The arrays can be either 1D or 2D.

    Returns:
    numpy.ndarray: A NumPy array that is the average of the input arrays. The output
    array will have the same shape as the input arrays.
    """

    # Check if the arrays are 1D or 2D
    if all(arr.ndim == 1 for arr in arrays):
        # Stack the arrays along the second axis
        stacked_arr = np.stack(arrays, axis=1)
        # Take the mean along the second axis
        avg_arr = np.mean(stacked_arr, axis=1)
    else:
        # Stack the arrays along the third axis
        stacked_arr = np.stack(arrays, axis=2)
        # Take the mean along the third axis
        avg_arr = np.mean(stacked_arr, axis=2)
    return avg_arr


def avg_2d_features(feature_dict):
    """
    Calculate the average of a 2-dimensional feature (e.g. spectrogram) of an entity (an entity represents all the
    features for a certain patient and intervention (XX_YY), e.g., "01_BA" is an entity which has 6 spectrograms
    for its 6 coughs from "01_BA_01" to "01_BA_06").

    Parameters:
    feature_dict (dict): A dictionary where the keys are strings of the form "XX_YY_ZZ", where "XX" is the patient
    number, "YY" is the intervention, and "ZZ" is the 2D feature number. The values of the dictionary are 2-dimensional
    NumPy arrays.

    Returns:
    dict: A dictionary where the keys are strings/entities of the form "XX_YY", and the values are NumPy arrays that are
    the average of the feature(s) for each entity.
    """

    # Initialize an empty dictionary to store the average arrays
    avg_dict = {}
    # Get a list of all the keys in the dictionary
    keys = list(feature_dict.keys())
    # Sort the keys alphabetically (not necessary)
    keys.sort()

    # Iterate over the keys
    for key in keys:
        # Split the key into parts
        parts = key.split('_')
        # Get the patient number and the intervention
        patient_number = parts[0]
        parameter = parts[1]
        # Construct the entity key
        entity_key = f"{patient_number}_{parameter}"
        # Check if the entity key is in the dictionary
        if entity_key in avg_dict:
            # If the key is already in the dictionary, append the array to the list of arrays
            avg_dict[entity_key].append(feature_dict[key])
        else:
            # If the key is not in the dictionary, create a new list with the array
            avg_dict[entity_key] = [feature_dict[key]]

    # Iterate over the keys in the average dictionary
    for key in avg_dict.keys():
        # Get the list of arrays for the key
        cough_arrays = avg_dict[key]
        # Calculate the average array using the average_arrays function
        avg_array = average_arrays(cough_arrays)
        # Update the value in the dictionary with the average array
        avg_dict[key] = avg_array

    return avg_dict


if __name__ == '__main__':
    abs_dir_path = os.path.abspath(os.path.dirname(__file__))
    audios_array = []
    path_audios = os.path.join(abs_dir_path, *['p1-2-5__B-C5', 'data', 'audio'])

    for root, _, files in os.walk(path_audios):
        for file in files:
            f = os.path.join(root, file)
            audios_array.append(librosa.load(f, sr=22050)[0])

    durations = get_duration(audios_array, 32, 16)
    q25, q75, q90, mean_arithmetic, max_durations, min_durations = plot_hist_get_mean(durations)
