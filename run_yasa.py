# import packages
# from DataObj.DataObj import DataObj
import numpy as np
import yasa
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from datetime import datetime
import os
import scipy as scipy
import preprocessing


def look4spindles(data, sample_rate, desired_ch, max_amp, hypno):
    data[data == 0] = 0.001  # Be careful from dead channels
    # data[data > max_amp] = max_amp  # Cut high amplitudes
    # data[data < -max_amp] = -max_amp
    sp = yasa.spindles_detect(data, sf=sample_rate, ch_names=[desired_ch], hypno=hypno, include=(2, 3),
                              freq_sp=(11, 16), freq_broad=(1, 35), duration=(0.5, 2), min_distance=500,
                              thresh={'corr': 0.45, 'rel_pow': 0.2, 'rms': 1.5},
                              multi_only=False, remove_outliers=False, verbose=True)
    if sp is None:
        result = None
    else:
        result = sp.summary()
        result_agg = sp.summary(grp_stage=True)
    return result,result_agg


def look4sw(data, sample_rate, desired_ch, hypno):
    sw = yasa.sw_detect(data, sf=sample_rate, ch_names=[desired_ch], hypno=hypno, include=3,
                        freq_sw=(0.3, 1.5), dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(40, 200),
                        amp_pos=(10, 150), amp_ptp=(75, 350), coupling=False,
                        coupling_params={'freq_sp': (12, 16), 'p': 0.05, 'time': 1}, remove_outliers=False,
                        verbose=False)
    result = sw.summary()
    return result


def plot_spindles(data, result, sample_rate):
    # plot figure per spindle
    for i in range(len(result.index)):
        print(i)
        start_ind = round((result.loc[i].at["Start"]) * sample_rate)
        end_ind = round((result.loc[i].at["End"]) * sample_rate)
        first_spindle = data[0, int(start_ind):int(end_ind)]
        epoch = data[0, int(start_ind) - 10 * sample_rate:int(end_ind) + 10 * sample_rate]

        first_spindle_ind = (np.arange(start_ind, end_ind))
        epoch_ind = (np.arange(start_ind - 10 * sample_rate, end_ind + 10 * sample_rate))

        figure(figsize=(18, 1))
        plt.plot(epoch_ind, epoch, lw=0.5, color='k')
        plt.plot(first_spindle_ind, first_spindle, lw=0.7, color='r')
        plt.show()


def plot_spindles_comparison(subject, sample_rate,timeshift):  # Time shift > 0 if XTR starts after PSG
    # Load data
    psg_spindles_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\Spindles_summary',
                                     str(subject) + ' -PSG- Spindles summary.csv')
    psg_spindles = pd.read_csv(psg_spindles_path)

    psg_edf_path = preprocessing.get_psg_path(subject)
    psg_edf = preprocessing.prepare_raw_signal(psg_edf_path)

    xtr_spindles_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\Spindles_summary',
                                     str(subject) + ' -XTR- Spindles summary.csv')
    xtr_spindles = pd.read_csv(xtr_spindles_path)
    xtr_edf_path = preprocessing.get_xtr_path(subject)
    xtr_edf = preprocessing.prepare_raw_signal(xtr_edf_path)

    # Create mask
    psg_edf_mask = np.copy(psg_edf)
    for i in range(len(psg_spindles)):
        if i == 0:
            zero_start_ind = 0
            zero_stop_ind = int((psg_spindles.loc[i].at["Start"]) * sample_rate)
            psg_edf_mask[0, zero_start_ind:zero_stop_ind] = 0
        if 0 < i < len(psg_spindles)-1:
            zero_start_ind = int((psg_spindles.loc[i].at["End"]) * sample_rate)
            zero_stop_ind = int((psg_spindles.loc[i + 1].at["Start"]) * sample_rate)
            psg_edf_mask[0, zero_start_ind:zero_stop_ind] = 0
        if i == len(psg_spindles):
            zero_start_ind = int((psg_spindles.loc[i].at["End"]) * sample_rate)
            zero_stop_ind = len(psg_edf[0])
            psg_edf_mask[0, zero_start_ind:zero_stop_ind] = 0

    xtr_edf_mask = np.copy(xtr_edf)
    for i in range(len(xtr_spindles)):
        if i == 0:
            zero_start_ind = 0
            zero_stop_ind = int((xtr_spindles.loc[i].at["Start"]) * sample_rate)
            xtr_edf_mask[0, zero_start_ind:zero_stop_ind] = 0
        if 0 < i < len(xtr_spindles)-1:
            zero_start_ind = int((xtr_spindles.loc[i].at["End"]) * sample_rate)
            zero_stop_ind = int((xtr_spindles.loc[i + 1].at["Start"]) * sample_rate)
            xtr_edf_mask[0, zero_start_ind:zero_stop_ind] = 0
        if i == len(xtr_spindles):
            zero_start_ind = int((xtr_spindles.loc[i].at["End"]) * sample_rate)
            zero_stop_ind = len(xtr_edf[0])
            xtr_edf_mask[0, zero_start_ind:zero_stop_ind] = 0

        # Align in time
    zeros = np.zeros(shape=(1, timeshift*sample_rate), dtype=float)
    xtr_edf = np.concatenate((zeros, xtr_edf), axis=1)
    xtr_edf_mask = np.concatenate((zeros, xtr_edf_mask), axis=1)


    # plot
    # psg_edf_mask_bol = psg_edf_mask != 0
    # psg_edf_mask = psg_edf_mask[psg_edf_mask_bol]

    ind = 0
    while ind+30*sample_rate < min(len(xtr_edf[0]), len(psg_edf[0])):
        psg_roi = psg_edf[0, ind:ind+30*sample_rate]
        psg_roi_mask = psg_edf_mask[0, ind:ind+30*sample_rate]
        # try
        psg_roi_mask_bol = psg_roi_mask != 0
        psg_roi_mask = psg_roi_mask[psg_roi_mask_bol]
        psg_times_mask = np.arange(ind, ind+30*sample_rate)
        psg_times_mask = psg_times_mask[psg_roi_mask_bol]

        xtr_roi = xtr_edf[0, ind:ind+30*sample_rate]
        xtr_roi_mask = xtr_edf_mask[0, ind:ind+30*sample_rate]
        if np.sum(abs(psg_roi_mask)) > 0 or np.sum(abs(xtr_roi_mask)) > 0:  # Drop if there is no detections here
            fig, axs = plt.subplots(2)
            fig.set_size_inches(17, 1)
            axs[0].plot((np.arange(ind, ind+30*sample_rate)), psg_roi, lw=0.7, color='k')
            axs[0].plot(psg_times_mask, psg_roi_mask, lw=1, color='r')
            axs[1].plot((np.arange(ind, ind+30*sample_rate)), xtr_roi, lw=0.7, color='k')
            axs[1].plot((np.arange(ind, ind + 30 * sample_rate)), xtr_roi_mask, lw=1, color='c')
            plt.savefig(f"Spindles_pics/{ind}- subject {subject}-spindles.jpg")
            plt.close(fig)
        ind = ind+30*sample_rate


def calc_spindles_comparison(subjects_numbers, sample_rate, element):
    for i in range(len(subjects_numbers)):
        edf_psg_info_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF_info',
                                         str(subjects_numbers[i]) + '_PSG_EDF_info.csv')
        raw_psg_edf_info = pd.read_csv(edf_psg_info_path)

        edf_xtr_info_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF_info',
                                         str(subjects_numbers[i]) + '_XTR_EDF_info.csv')
        raw_xtr_edf_info = pd.read_csv(edf_xtr_info_path)
        if element == 'spindles':
            psg_summary = pd.read_csv(os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject'
                                                   r'\Spindles_summary',
                                                   str(subjects_numbers[i]) + ' -PSG- Spindles summary.csv'))
            xtr_summary = pd.read_csv(os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject'
                                                   r'\Spindles_summary',
                                                   str(subjects_numbers[i]) + ' -XTR- Spindles summary.csv'))
        if element == 'SW':
            psg_summary = pd.read_csv(os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject'
                                                   r'\SW_summary',
                                                   str(subjects_numbers[i]) + ' -PSG- SW summary.csv'))
            xtr_summary = pd.read_csv(os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject'
                                                   r'\SW_summary',
                                                   str(subjects_numbers[i]) + ' -XTR- SW summary.csv'))

        psg_start_time = raw_psg_edf_info.loc[0].at['Start time'][0:19]
        psg_start_time = datetime.strptime(psg_start_time, '%Y-%m-%d %H:%M:%S')
        xtr_start_time = raw_xtr_edf_info.loc[0].at['Start time'][0:19]
        xtr_start_time = datetime.strptime(xtr_start_time, '%Y-%m-%d %H:%M:%S')
        time_shift = (xtr_start_time - psg_start_time).seconds
        if time_shift < 0: print('Error: XTR starts before PSG')

        psg_ind = (psg_summary['Start'] * sample_rate).astype(int)
        xtr_ind = ((xtr_summary['Start'] + time_shift) * sample_rate).astype(int)

        compare = yasa.compare_detection(xtr_ind, psg_ind, max_distance=5 * sample_rate)
        tp = (compare['tp']).size
        tp_ind = compare['tp']
        fp = (compare['fp']).size
        fp_ind = compare['fp']
        fn = (compare['fn']).size
        fn_ind = compare['fn']
        precision = compare['precision']
        recall = compare['recall']
        f1 = compare['f1']
        statistics = {'tp': [tp], 'tp_ind': [tp_ind.tolist()], 'fp': [fp], 'fp_ind': [fp_ind.tolist()], 'fn': [fn],
                      'fn_ind': [fn_ind.tolist()],
                      'precision': [precision], 'recall': [recall], 'f1': [f1]}
        statistics = pd.DataFrame(data=statistics)
        if element == 'spindles':
            statistics.to_csv(
                path_or_buf=os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\Spindles_summary',
                                         f'{subjects_numbers[i]} -XTR vs PSG- Spindles detections.csv'))

        if element == 'SW':
            statistics.to_csv(
                path_or_buf=os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\SW_summary',
                                         f'{subjects_numbers[i]} -XTR vs PSG- SW detections.csv'))
    return time_shift


def calc_scorer_sp_params(signal, sp_ind, sp_dur, sample_rate):
    sp_amp = list()
    sp_rms = list()
    sp_abs = list()
    sp_freq = list()
    for row in range(len(sp_ind)):
        roi = signal[0, int(sp_ind[row]):int((sp_ind[row]+sp_dur[row]))]
        sp_x = np.arange(roi.size, dtype=np.float64)
        roi_det = yasa.numba._detrend(sp_x, roi)
        pp = np.ptp(roi_det)
        sp_amp.append(pp)

        rms = yasa.numba._rms(roi_det)
        sp_rms.append(rms)

        signal_sigma = yasa.detection.filter_data(signal, sample_rate, 11, 16, l_trans_bandwidth=1.5, h_trans_bandwidth=1.5, method="fir", verbose=0)

        # Hilbert power (to define the instantaneous frequency / power)
        nfast = scipy.fft.next_fast_len(len(signal[0, :]))
        analytic = scipy.signal.hilbert(signal_sigma, N=nfast)[:, :len(signal[0, :])]
        inst_phase = np.angle(analytic)
        inst_pow = np.square(np.abs(analytic))
        inst_freq = sample_rate / (2 * np.pi) * np.diff(inst_phase, axis=-1)
        # Hilbert-based instantaneous properties
        sp_inst_freq = inst_freq[0, int(sp_ind[row]):int((sp_ind[row]+sp_dur[row]))]
        sp_inst_pow = inst_pow[0, int(sp_ind[row]):int((sp_ind[row]+sp_dur[row]))]
        sp_abs.append(np.median(np.log10(sp_inst_pow[sp_inst_pow > 0])))
        sp_freq.append(np.median(sp_inst_freq[sp_inst_freq > 0]))

    return sp_amp, sp_rms, sp_abs, sp_freq
