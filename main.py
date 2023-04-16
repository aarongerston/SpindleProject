# import packages
# import boto3
# import matplotlib.pyplot as plt
import sp_helper
import run_yasa
import preprocessing
import plots
import matplotlib
# import yasa
# import pandas as pd
# from datetime import datetime
# import csv
import os
# import XtrViz
import yasa
import pandas as pd
import numpy as np
from datetime import datetime
# import scipy.stats as sta
import analysis

matplotlib.use('Qt5Agg')
# s3_client = boto3.client('s3')
#####################################################################################################
# Initiating parameters:
subjects_numbers = [14, 20, 21, 22, 25, 26, 27, 28, 30, 31, 35, 36, 38, 39, 41, 44, 45, 49, 55, 56, 57, 58, 60, 67, 69, 74, 76, 78, 81, 82, 83, 87, 88, 91, 92, 93, 96, 101, 104]
# 52, 53, 61, 103, 105, 108, 112,114, 116, 117 - missing scoring files
# 37 - 0808 vs 01/08 edfs. need to delete wrong file from s3
desired_ch_name = 'EEG C3 - M2'
desired_ch_psg = ['C3', 'M2']
desired_ch_xtr = ['EEG Fpz-M2']
sample_rate = 250

validation_single_record = False
validation_all_records = False
multiple_optimizing = False

need_to_download = False
Spindles_detection = False
Spindles_detection_plot = False
Spindles_detection_analysis = True

Slow_waves_detection = False
Slow_waves_detection_plot = False
Slow_waves_detection_analysis = False

#####################################################################################################
if validation_single_record:
    # Validation
    # Compare yasa to scorer
    # get XTR signal:
    file_path = r'C:\Users\Administrator\PycharmProjects\SpindleProject\Validation\files\sp1.edf'
    signal = preprocessing.prepare_raw_signal(file_path)

    # Find spindles using YASA:
    sp = yasa.spindles_detect(signal, sf=sample_rate,
                              freq_sp=(11, 16), freq_broad=(1, 35), duration=(0.5, 2), min_distance=500,
                              thresh={'corr': 0.588, 'rel_pow': 0.2, 'rms': 1.4},
                              multi_only=False, remove_outliers=False, verbose=True)

    # Find YASA start indices and calc params:
    yasa_sp = sp.summary(grp_stage=False)
    yasa_sp_ind = ((yasa_sp['Start']) * sample_rate).astype(int)

    # Load scorer's spindles annotation:
    scorer_sp = pd.read_csv(
        r'C:\Users\Administrator\PycharmProjects\SpindleProject\Validation\scorer\sp1_annotations.csv')

    # Find scorer's start and duration indices and calc params:
    scorer_sp_ind = (scorer_sp['Onset'])
    st = datetime.strptime('00:11:11', '%H:%M:%S')
    st = ((st.minute * 60 + st.second) * sample_rate)
    for i in range(len(scorer_sp_ind)):
        scorer_sp_ind[i] = datetime.strptime(scorer_sp_ind[i], '%H:%M:%S')
        scorer_sp_ind[i] = ((scorer_sp_ind[i].minute * 60 + scorer_sp_ind[i].second) * sample_rate)
        scorer_sp_ind[i] = scorer_sp_ind[i] - st
    scorer_sp_dur = ((scorer_sp['Duration']) * sample_rate)

    # Compare Yasa and scorer: confusion and f1 score
    stats = yasa.compare_detection(yasa_sp_ind, scorer_sp_ind, sample_rate)

    # Plot
    plots.plot_comp(signal=signal, sample_rate=sample_rate, sc_sp=scorer_sp, yasa_sp=yasa_sp, stats=stats)

#####################################################################################################
if validation_all_records:
    TP_yasa_ind = pd.DataFrame(columns=['file', 'ind'])
    FP_yasa_ind = pd.DataFrame(columns=['file', 'ind'])
    FN_sc_ind = pd.DataFrame(columns=['file', 'ind'])
    positive_sc_ind = pd.DataFrame(columns=['file', 'ind'])
    sc_sp = pd.DataFrame(columns=['file', 'Start', 'Duration', 'Amplitude', 'RMS', 'AbsPower', 'Frequency'])
    yasa_sp_all = None
    st = ['0', '00:11:11', '01:32:53', '22:47:03', '23:26:21', '02:47:28', '03:11:37', '02:48:27', '03:13:06',
          '02:04:02', '00:13:20', '22:54:37', '22:09:25']
    # files = [5, 6, 11]
    files = [1, 3, 4, 7, 8, 9, 10]
    for file in files:  # sp_file_number
        file_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\Validation\files',
                                 f'sp{file}.edf')
        signal = preprocessing.prepare_raw_signal(file_path)
        sp = yasa.spindles_detect(signal, sf=sample_rate,
                                  freq_sp=(11, 16), freq_broad=(1, 35), duration=(0.5, 2), min_distance=500,
                                  thresh={'corr': 0.588, 'rel_pow': 0.2, 'rms': 1.4},
                                  multi_only=False, remove_outliers=False, verbose=True)
        if sp is not None:
            yasa_sp = sp.summary()
            yasa_sp_ind = ((yasa_sp['Start']) * sample_rate).astype(int)
            annotation_path = os.path.join(
                r'C:\Users\Administrator\PycharmProjects\SpindleProject\Validation\scorer',
                f'sp{file}_annotations.csv')
            scorer_sp = pd.read_csv(annotation_path)
            scorer_sp_ind = (scorer_sp['Onset'])
            start = datetime.strptime(st[file], '%H:%M:%S')
            start = ((start.minute * 60 + start.second) * sample_rate)
            for i in range(len(scorer_sp_ind)):
                scorer_sp_ind[i] = datetime.strptime(scorer_sp_ind[i], '%H:%M:%S')
                scorer_sp_ind[i] = ((scorer_sp_ind[i].minute * 60 + scorer_sp_ind[i].second) * sample_rate)
                scorer_sp_ind[i] = scorer_sp_ind[i] - start
            scorer_sp_dur = ((scorer_sp['Duration']) * sample_rate)
            stats = yasa.compare_detection(yasa_sp_ind, scorer_sp_ind, sample_rate)

            # Track statistics:
            TP_yasa_ind.loc[len(TP_yasa_ind)] = [file, list(stats['tp'])]
            FP_yasa_ind.loc[len(FP_yasa_ind)] = [file, list(stats['fp'])]
            FN_sc_ind.loc[len(FN_sc_ind)] = [file, list(stats['fn'])]
            positive_sc_ind.loc[len(positive_sc_ind)] = [file, list(scorer_sp_ind)]

            # Track YASA spindles params:
            yasa_sp['file'] = [file for x in range(len(yasa_sp))]
            if yasa_sp_all is None:
                yasa_sp_all = yasa_sp
            else:
                yasa_sp_all = pd.concat([yasa_sp_all, yasa_sp], ignore_index=True)

            # Track sc spindles params:
            scorer_sp_amp, scorer_sp_rms, scorer_sp_abs, scorer_sp_freq = run_yasa.calc_scorer_sp_params(signal=signal,
                                                                                                         sp_ind=scorer_sp_ind,
                                                                                                         sp_dur=scorer_sp_dur,
                                                                                                         sample_rate=sample_rate)

            for j in range(len(scorer_sp_ind)):
                sc_sp.loc[len(sc_sp)] = [file, scorer_sp_ind[j], scorer_sp_dur[j], scorer_sp_amp[j], scorer_sp_rms[j],
                                         scorer_sp_abs[j], scorer_sp_freq[j]]

    for i in range(len(files) - 1):
        file = files[i]
        sens = round((len(TP_yasa_ind.loc[i].at['ind']) / len(positive_sc_ind.loc[i].at['ind'])), 3)
        f1 = round((len(TP_yasa_ind.loc[i].at['ind']) / (len(TP_yasa_ind.loc[i].at['ind']) + 0.5 * (
                len(FP_yasa_ind.loc[i].at['ind']) + len(FN_sc_ind.loc[i].at['ind'])))), 3)
        print(f'file {file} reached: sensitivity= {sens}, f1 score= {f1}')
    TP = list(TP_yasa_ind.loc[:, 'ind'])
    TP = len([item for sublist in TP for item in sublist])
    FP = list(FP_yasa_ind.loc[:, 'ind'])
    FP = len([item for sublist in FP for item in sublist])
    FN = list(FN_sc_ind.loc[:, 'ind'])
    FN = len([item for sublist in FN for item in sublist])
    P = list(positive_sc_ind.loc[:, 'ind'])
    P = len([item for sublist in P for item in sublist])
    print(f'All records reached sensitivity= {round(TP / P, 3)}, f1 score= {round(TP / (TP + 0.5 * (FP + FN)), 3)}')
    print(f'Total TP= {TP}, Total FP= {FP}, Total FN= {FN}, Total P= {P}')

    sp_helper.plot_all_sp_features_yasa_vs_scorer(sample_rate, sc_sp, yasa_sp_all)
    sp_helper.plot_TP_vs_FP_sp_features_yasa(sample_rate, yasa_sp_all, sc_sp, TP_yasa_ind, FP_yasa_ind, FN_sc_ind)

    # yasa_avg = sp_helper.calc_avg_sp(yasa_sp)
    # analysis.plot_pca(yasa_sp_all)

#####################################################################################################

if multiple_optimizing:
    st = ['0', '00:11:11', '01:32:53', '22:47:03', '23:26:21', '02:47:28', '03:11:37', '02:48:27', '03:13:06',
          '02:04:02', '00:13:20', '22:54:37', '22:09:25']
    result = pd.DataFrame(columns=['corr', 'rel_pow', 'rms', 'tp', 'fp', 'fn', 'p'])
    for corr in np.linspace(2, 9, 10):
        for rel_pow in np.linspace(2, 10, 5):
            for rms in np.linspace(2, 14, 7):
                optimize = [0, 0, 0, 0]
                for file in [1, 3, 4, 7, 8, 9, 10]:
                    file_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\Validation\files',
                                             f'sp{file}.edf')
                    signal = preprocessing.prepare_raw_signal(file_path)
                    sp = yasa.spindles_detect(signal, sf=sample_rate,
                                              freq_sp=(11, 16), freq_broad=(1, 35), duration=(0.5, 2), min_distance=500,
                                              thresh={'corr': corr / 10, 'rel_pow': rel_pow / 10, 'rms': rms / 10},
                                              multi_only=False, remove_outliers=False, verbose=True)
                    if sp is not None:
                        yasa_sp = sp.summary()
                        yasa_sp_ind = ((yasa_sp['Start']) * sample_rate).astype(int)
                        annotation_path = os.path.join(
                            r'C:\Users\Administrator\PycharmProjects\SpindleProject\Validation\scorer',
                            f'sp{file}_annotations.csv')
                        scorer_sp = pd.read_csv(annotation_path)
                        scorer_sp_ind = (scorer_sp['Onset'])
                        start = datetime.strptime(st[file], '%H:%M:%S')
                        start = ((start.minute * 60 + start.second) * sample_rate)
                        for i in range(len(scorer_sp_ind)):
                            scorer_sp_ind[i] = datetime.strptime(scorer_sp_ind[i], '%H:%M:%S')
                            scorer_sp_ind[i] = ((scorer_sp_ind[i].minute * 60 + scorer_sp_ind[i].second) * sample_rate)
                            scorer_sp_ind[i] = scorer_sp_ind[i] - start
                        scorer_sp_dur = ((scorer_sp['Duration']) * sample_rate)
                        stats = yasa.compare_detection(yasa_sp_ind, scorer_sp_ind, sample_rate)
                        optimize = [sum(x) for x in zip(optimize, [len(stats['tp']), len(stats['fp']), len(stats['fn']),
                                                                   len(scorer_sp_ind)])]
                result.loc[len(result)] = [corr / 10, rel_pow / 10, rms / 10] + optimize
    result.to_csv(
        path_or_buf=os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\Validation\Optimizing.csv'))
#####################################################################################################
if need_to_download:
    # Download: 1. raw EDF files 2. Dropped down-sampled files 3. edf info csv
    preprocessing.download_xtr_data(subjects_numbers, desired_ch_xtr, sample_rate)

# Let's use YASA for spindles detection
if Spindles_detection:

    # Hypnograms preprocessing:
    raw_xtr_hypno_info = preprocessing.download_xtr_hypnogram(subjects_numbers=subjects_numbers, scorer='Shani')
    edited_xtr_hypnograms = preprocessing.xtr_adjust_hypno_to_edf(subjects_numbers=subjects_numbers,
                                                                  raw_hypno_info=raw_xtr_hypno_info,
                                                                  edf_sample_rate=sample_rate)
    for i in range(len(subjects_numbers)):
        file_path = preprocessing.get_xtr_path(subjects_numbers[i])
        signal = preprocessing.prepare_raw_signal(file_path)
        art_std, zscores_std = yasa.art_detect(signal, sample_rate, window=5, hypno=edited_xtr_hypnograms[i], include=(1, 2, 3, 4),
                                               method='std', threshold=3, verbose='info')

        # art_std = np.repeat(art_std, 30 * sample_rate)
        art_std = yasa.hypno_upsample_to_data(art_std, 0.2, signal, sample_rate)
        hypno_with_art = edited_xtr_hypnograms[i].copy()
        hypno_with_art[art_std] = -1
        spindles_summary, spindles_summary_agg = run_yasa.look4spindles(data=signal, sample_rate=sample_rate,
                                                  desired_ch=desired_ch_name,
                                                  max_amp=500, hypno=hypno_with_art)
        if spindles_summary is None:
            print(f'no spindles was found in subject {subjects_numbers[i]}')
        else:
            spindles_summary.to_csv(
                path_or_buf=os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\Spindles_summary',
                                         f'{subjects_numbers[i]} -XTR- Spindles summary.csv'))
            spindles_summary_agg.to_csv(
                path_or_buf=os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\Spindles_summary_agg',
                                         f'{subjects_numbers[i]} -XTR- Spindles summary agg.csv'))

        if Spindles_detection_plot:plots.plot_xtr_sp(signal=signal, sample_rate=sample_rate, yasa_sp=spindles_summary, hypno =hypno_with_art)

if Spindles_detection_analysis:
    Spindles_summary_agg_path =r'C:\Users\Administrator\PycharmProjects\SpindleProject\Spindles_summary_agg'
    features = sp_helper.get_features(Spindles_summary_agg_path)
    demographic_path = r'C:\Users\Administrator\PycharmProjects\SpindleProject\Demographic'
    demographic = sp_helper.get_demographic(path=demographic_path, param='Group')
    analysis.plot_pca(features, demographic, pca_comp=2, kmeans_comp=2)


#####################################################################################################
# Let's use YASA for SW detection
if Slow_waves_detection:
    for i in range(len(subjects_numbers)):
        file_path = preprocessing.get_xtr_path(subjects_numbers[i])
        signal = preprocessing.prepare_raw_signal(file_path)
        sw_summary = run_yasa.look4sw(data=signal, sample_rate=sample_rate, desired_ch=desired_ch_name,
                                      hypno=edited_xtr_hypnograms[i])
        sw_summary.to_csv(
            path_or_buf=os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\SW_summary',
                                     f'{subjects_numbers[i]} -XTR- SW summary.csv'))
