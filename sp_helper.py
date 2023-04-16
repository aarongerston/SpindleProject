import numpy as np
import scipy.stats as sta
import matplotlib.pyplot as plt
import pandas as pd
from os import walk
import os as os

def choose_ttest(group1, group2):
    var1 = np.var(group1)
    var2 = np.var(group2)
    if len(group1) > len (group2):
        ratio = var1/var2
    else:
        ratio = var2/var1
    if ratio < 4:
        return True
    else:
        return False

def plot_all_sp_features_yasa_vs_scorer(sample_rate, sc_sp, yasa_sp_all):
    sc_duration = np.array(sc_sp.loc[:, 'Duration']) / sample_rate
    yasa_duration = np.array(yasa_sp_all.loc[:, 'Duration'])
    b = choose_ttest(sc_duration, yasa_duration)
    duration_statistics, duration_pvalue = sta.ttest_ind(a=sc_duration, b=yasa_duration, equal_var=b)

    sc_amp = np.array(sc_sp.loc[:, 'Amplitude'])
    yasa_amp = np.array(yasa_sp_all.loc[:, 'Amplitude'])
    b = choose_ttest(sc_amp, yasa_amp)
    amp_statistics, amp_pvalue = sta.ttest_ind(a=sc_amp, b=yasa_amp, equal_var=b)

    sc_rms = np.array(sc_sp.loc[:, 'RMS'])
    yasa_rms = np.array(yasa_sp_all.loc[:, 'RMS'])
    b = choose_ttest(sc_rms, yasa_rms)
    rms_statistics, rms_pvalue = sta.ttest_ind(a=sc_rms, b=yasa_rms, equal_var=b)

    sc_abs = np.array(sc_sp.loc[:, 'AbsPower'])
    yasa_abs = np.array(yasa_sp_all.loc[:, 'AbsPower'])
    b = choose_ttest(sc_abs, yasa_abs)
    abs_statistics, abs_pvalue = sta.ttest_ind(a=sc_abs, b=yasa_abs, equal_var=b)

    sc_freq = np.array(sc_sp.loc[:, 'Frequency'])
    yasa_freq = np.array(yasa_sp_all.loc[:, 'Frequency'])
    b = choose_ttest(sc_freq, yasa_freq)
    freq_statistics, freq_pvalue = sta.ttest_ind(a=sc_freq, b=yasa_freq, equal_var=b)

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(17, 8)
    fig.suptitle('All spindles: Yasa vs Scorer', fontsize=14)
    fig.tight_layout(pad=3.0)
    axs[0, 0].boxplot([sc_duration, yasa_duration])
    axs[0, 0].set_xticks([1, 2])
    axs[0, 0].set_xticklabels(['Scorer', 'YASA'], fontsize=10)
    axs[0, 0].set_ylabel('Duration [sec]')
    axs[0, 0].set_title(f'P_value ={round(duration_pvalue,7)}')

    axs[0, 1].boxplot([sc_amp, yasa_amp])
    axs[0, 1].set_xticks([1, 2])
    axs[0, 1].set_xticklabels(['Scorer', 'YASA'], fontsize=10)
    axs[0, 1].set_ylabel('Amplitude [uV]')
    axs[0, 1].set_title(f'P_value ={round(amp_pvalue,7)}')

    axs[0, 2].boxplot([sc_rms, yasa_rms])
    axs[0, 2].set_xticks([1, 2])
    axs[0, 2].set_xticklabels(['Scorer', 'YASA'], fontsize=10)
    axs[0, 2].set_ylabel('RMS [uV]')
    axs[0, 2].set_title(f'P_value ={round(rms_pvalue,7)}')

    axs[1, 0].boxplot([sc_abs, yasa_abs])
    axs[1, 0].set_xticks([1, 2])
    axs[1, 0].set_xticklabels(['Scorer', 'YASA'], fontsize=10)
    axs[1, 0].set_ylabel('AbsPower [uV]')  # Median absolute power (in log10 µV^2), calculated from the Hilbert-transform of the freq_sp filtered signal.
    axs[1, 0].set_title(f'P_value ={round(abs_pvalue,7)}')

    axs[1, 1].boxplot([sc_freq, yasa_freq])
    axs[1, 1].set_xticks([1, 2])
    axs[1, 1].set_xticklabels(['Scorer', 'YASA'], fontsize=10)
    axs[1, 1].set_ylabel('Frequency [Hz]')
    axs[1, 1].set_title(f'P_value ={round(freq_pvalue,7)}')

    plt.show()

def plot_TP_vs_FP_sp_features_yasa(sample_rate, yasa_sp_all, sc_sp, TP_yasa_ind, FP_yasa_ind, FN_sc_ind):
    files = np.unique(yasa_sp_all['file'])
    yasa_sp_TP = yasa_sp_all.copy()
    for row in range(len(yasa_sp_all)):
        f = yasa_sp_all.loc[row, 'file']
        TP_ind_f = np.array((TP_yasa_ind.loc[TP_yasa_ind['file'] == f]['ind']).values[0])/sample_rate
        if yasa_sp_all.loc[row, 'Start'] not in TP_ind_f:
            yasa_sp_TP.drop([row], axis=0, inplace=True)

    yasa_sp_FP = yasa_sp_all.copy()
    for row in range(len(yasa_sp_all)):
        f = yasa_sp_all.loc[row, 'file']
        FP_ind_f = np.array((FP_yasa_ind.loc[FP_yasa_ind['file'] == f]['ind']).values[0])/sample_rate
        if yasa_sp_all.loc[row, 'Start'] not in FP_ind_f:
            yasa_sp_FP.drop([row], axis=0, inplace=True)

    yasa_sp_FN = sc_sp.copy()
    for row in range(len(yasa_sp_all)):
        f = sc_sp.loc[row, 'file']
        FN_ind_f = np.array((FN_sc_ind.loc[FN_sc_ind['file'] == f]['ind']).values[0])
        if sc_sp.loc[row, 'Start'] not in FN_ind_f:
            yasa_sp_FN.drop([row], axis=0, inplace=True)

    TP_duration = np.array(yasa_sp_TP.loc[:, 'Duration'])
    FP_duration = np.array(yasa_sp_FP.loc[:, 'Duration'])
    FN_duration = np.array(yasa_sp_FN.loc[:, 'Duration'])/sample_rate
    b = choose_ttest(TP_duration, FP_duration)
    duration_statistics, duration_pvalue = sta.ttest_ind(a=TP_duration, b=FP_duration, equal_var=b)

    TP_amp = np.array(yasa_sp_TP.loc[:, 'Amplitude'])
    FP_amp = np.array(yasa_sp_FP.loc[:, 'Amplitude'])
    FN_amp = np.array(yasa_sp_FN.loc[:, 'Amplitude'])
    b = choose_ttest(TP_amp, FP_amp)
    amp_statistics, amp_pvalue = sta.ttest_ind(a=TP_amp, b=FP_amp, equal_var=b)

    TP_rms = np.array(yasa_sp_TP.loc[:, 'RMS'])
    FP_rms = np.array(yasa_sp_FP.loc[:, 'RMS'])
    FN_rms = np.array(yasa_sp_FN.loc[:, 'RMS'])
    b = choose_ttest(TP_rms, FP_rms)
    rms_statistics, rms_pvalue = sta.ttest_ind(a=TP_rms, b=FP_rms, equal_var=b)

    TP_abs = np.array(yasa_sp_TP.loc[:, 'AbsPower'])
    FP_abs = np.array(yasa_sp_FP.loc[:, 'AbsPower'])
    FN_abs = np.array(yasa_sp_FN.loc[:, 'AbsPower'])
    b = choose_ttest(TP_abs, FP_abs)
    abs_statistics, abs_pvalue = sta.ttest_ind(a=TP_abs, b=FP_abs, equal_var=b)

    TP_freq = np.array(yasa_sp_TP.loc[:, 'Frequency'])
    FP_freq = np.array(yasa_sp_FP.loc[:, 'Frequency'])
    FN_freq = np.array(yasa_sp_FN.loc[:, 'Frequency'])
    b = choose_ttest(TP_freq, FP_freq)
    freq_statistics, freq_pvalue = sta.ttest_ind(a=TP_freq, b=FP_freq, equal_var=b)

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(17, 8)
    fig.suptitle('TP, FP, FN spindles features', fontsize=14)
    fig.tight_layout(pad=3.0)
    axs[0, 0].boxplot([TP_duration, FP_duration, FN_duration])
    axs[0, 0].set_xticks([1, 2, 3])
    axs[0, 0].set_xticklabels(['TP', 'FP', 'FN'], fontsize=10)
    axs[0, 0].set_ylabel('Duration [sec]')
    axs[0, 0].set_title(f'P_value of TP, FP ={round(duration_pvalue,7)}')

    axs[0, 1].boxplot([TP_amp, FP_amp, FN_amp])
    axs[0, 1].set_xticks([1, 2, 3])
    axs[0, 1].set_xticklabels(['TP', 'FP', 'FN'], fontsize=10)
    axs[0, 1].set_ylabel('Amplitude [uV]')
    axs[0, 1].set_title(f'P_value of TP, FP ={round(amp_pvalue,7)}')

    axs[0, 2].boxplot([TP_rms, FP_rms, FN_rms])
    axs[0, 2].set_xticks([1, 2, 3])
    axs[0, 2].set_xticklabels(['TP', 'FP', 'FN'], fontsize=10)
    axs[0, 2].set_ylabel('RMS [uV]')
    axs[0, 2].set_title(f'P_value of TP, FP ={round(rms_pvalue,7)}')

    axs[1, 0].boxplot([TP_abs, FP_abs, FN_abs])
    axs[1, 0].set_xticks([1, 2, 3])
    axs[1, 0].set_xticklabels(['TP', 'FP', 'FN'], fontsize=10)
    axs[1, 0].set_ylabel('AbsPower [uV]')  # Median absolute power (in log10 µV^2), calculated from the Hilbert-transform of the freq_sp filtered signal.
    axs[1, 0].set_title(f'P_value of TP, FP ={round(abs_pvalue,7)}')

    axs[1, 1].boxplot([TP_freq, FP_freq, FN_freq])
    axs[1, 1].set_xticks([1, 2, 3])
    axs[1, 1].set_xticklabels(['TP', 'FP', 'FN'], fontsize=10)
    axs[1, 1].set_ylabel('Frequency [Hz]')
    axs[1, 1].set_title(f'P_value of TP, FP ={round(freq_pvalue,7)}')

    plt.show()

def calc_avg_sp(sp_summary):
    avg = pd.DataFrame(columns=['file', 'ind'])
    return avg

def get_features(path):
    features = None
    filenames = next(walk(path), (None, None, []))[2]
    for file in filenames:
        file_path = os.path.join(path, file)
        f = pd.read_csv(file_path)
        f.drop([1], axis=0, inplace=True)
        f.drop(columns=['Stage'], inplace=True)
        f['Subject number'] = int(file[0:3])
        if features is None:
            features = f
        else:
            features = pd.concat([features, f], ignore_index=True)
    return features

def get_demographic(path, param):
    demographic = None
    filenames = next(walk(path), (None, None, []))[2]
    for file in filenames:
        file_path = os.path.join(path, file)
        f = pd.read_csv(file_path)
    demographic = f[['Subject number', param]]
    return demographic






