# import packages
import boto3
import XtrS3Utils.s3_utils as S3
from DataObj.DataObj import DataObj
import os
import pyedflib as pyedflib
from pyedflib import highlevel
import numpy as np
import ExternalEDFHandler.external_edf_handler as Exedf
import pandas as pd
from datetime import datetime


def download_psg_data(subjects_numbers, desired_ch, sample_rate):
    s3_client = boto3.client('s3')

    for i in range(len(subjects_numbers)):  # Iterate over all desired subjects
        raw_edf_info = pd.DataFrame(
            columns=['URI', 'Start time', 'Samples number', 'DownSampled URI'])
        prefix_temp = 'SleepScoringData/DOD/0' + str(subjects_numbers[i]) + str('/')
        PSG_URI = list(S3.get_paths(s3_client, bucket_name='sleep-staging-data', prefix=prefix_temp,
                                    pattern='*/ClinicalPSGData/RawEDFFiles/*_F.EDF'))
        filename = os.path.basename(PSG_URI[0])

        EDF_local = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF', filename)
        if not os.path.exists(EDF_local):
            S3.download(s3_client, PSG_URI[0], EDF_local)

        EDF_local_new = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF',
                                     f'{subjects_numbers[i]} -PSG_Dropped_Downsampled.edf')
        if not os.path.exists(EDF_local_new):
            obj_EDF = DataObj(EDF_local)
            num_of_samples = len(obj_EDF.data_exg) / obj_EDF.fs_exg * sample_rate
            obj_EDF_ds = Exedf.resample(obj=obj_EDF, desired_fs=sample_rate)
            all_channels = obj_EDF.ch_names_exg
            desired_ch_ind = [all_channels.index(desired_ch[0])]
            desired_ch_ind.append(all_channels.index(desired_ch[1]))
            signal_edited = np.atleast_2d(obj_EDF_ds.data_exg[:, desired_ch_ind])
            signal_edited = np.atleast_2d(signal_edited[:, 0] - signal_edited[:, 1])
            signals, signal_header, header = pyedflib.highlevel.read_edf(EDF_local, ch_names=desired_ch[0])
            signal_header[0]['label'] = 'EEG ' + str(desired_ch[0]) + ' - ' + str(desired_ch[1])
            signal_header[0]['sample_rate'] = sample_rate
            signal_header[0]['sample_frequency'] = sample_rate
            highlevel.write_edf(edf_file=EDF_local_new, signals=signal_edited, signal_headers=signal_header,
                                header=header)
            print(f'{EDF_local_new} created')
            raw_edf_info.loc[i] = [EDF_local, obj_EDF.start_time, num_of_samples, EDF_local_new]
            raw_edf_info.to_csv(
                path_or_buf=os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF_info',
                                         f'{subjects_numbers[i]}_PSG_EDF_info.csv'))
            print(f"subject {subjects_numbers[i]} - PSG file downloaded successfully")


def download_xtr_data(subjects_numbers, desired_ch, sample_rate):
    s3_client = boto3.client('s3')

    for i in range(len(subjects_numbers)-1):  # Iterate over all desired subjects
        raw_edf_info = pd.DataFrame(
            columns=['URI', 'Start time', 'Samples number', 'DownSampled URI'])  # prepare for return
        if subjects_numbers[i] < 100: prefix_temp = 'SleepScoringData/DOD/0' + str(subjects_numbers[i]) + str('/')
        if subjects_numbers[i] > 99: prefix_temp = 'SleepScoringData/DOD/' + str(subjects_numbers[i]) + str('/')
        xtr_URI = list(S3.get_paths(s3_client, bucket_name='sleep-staging-data', prefix=prefix_temp,
                                    pattern='*_Lab/XtrodesData/RawEDFFiles/*_AASM.EDF'))
        filename = os.path.basename(xtr_URI[0])
        EDF_local = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF', filename)
        if not os.path.exists(EDF_local):
            S3.download(s3_client, xtr_URI[0], EDF_local)

        EDF_local_new = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF',
                                     f'{subjects_numbers[i]} -XTR_Dropped_Downsampled.edf')
        if not os.path.exists(EDF_local_new):
            obj_EDF = DataObj(EDF_local)
            num_of_samples = len(obj_EDF.data_exg) / obj_EDF.fs_exg * sample_rate
            obj_EDF_ds = Exedf.resample(obj=obj_EDF, desired_fs=sample_rate)
            all_channels = obj_EDF.ch_names_exg
            if 'Fpz-M2' in all_channels: desired_ch_ind = all_channels.index((desired_ch[0])[4:10])
            if 'EEG Fpz-M2' in all_channels: desired_ch_ind = all_channels.index(desired_ch[0])
            if not ('Fpz-M2' in all_channels or 'EEG Fpz-M2' in all_channels): print('Channel was not found')
            signal_edited = np.atleast_2d(obj_EDF_ds.data_exg[:, desired_ch_ind])
            signals, signal_header, header = pyedflib.highlevel.read_edf(EDF_local, ch_names=desired_ch)
            signal_header[0]['sample_rate'] = sample_rate
            signal_header[0]['sample_frequency'] = sample_rate
            highlevel.write_edf(edf_file=EDF_local_new, signals=signal_edited, signal_headers=signal_header,
                                header=header)
            print(f'{EDF_local_new} created')
            raw_edf_info.loc[i] = [EDF_local, obj_EDF.start_time, num_of_samples, EDF_local_new]
            raw_edf_info.to_csv(
                path_or_buf=os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF_info',
                                         f'{subjects_numbers[i]}_XTR_EDF_info.csv'))
            print(f"subject {subjects_numbers[i]} - XTR file downloaded successfully")


def download_psg_hypnogram(subjects_numbers, scorer=str):
    #  This function takes the raw scoring file and converts it the accepted hypnogram input by YASA

    s3_client = boto3.client('s3')
    raw_hypno_info = pd.DataFrame(columns=['URI'])  # prepare for return
    for i in range(len(subjects_numbers)):  # Iterate over all desired subjects
        prefix_temp = 'SleepScoringData/DOD/0' + str(subjects_numbers[i]) + str('/')
        if scorer == 'Danielle':
            PSG_URI_csv = list(S3.get_paths(s3_client, bucket_name='sleep-staging-data', prefix=prefix_temp,
                                            pattern='*/ClinicalPSGData/ScoredData/Ichilov - Danielle Wasserman/*.csv'))
        else:
            print('Only Danielle was implemented in this code')
        filename = os.path.basename(PSG_URI_csv[0])
        new_filename = str(subjects_numbers[i]) + '_psg_hypno.csv'
        hypno_local = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawHypnograms', new_filename)

        if not os.path.exists(hypno_local):
            S3.download(s3_client, PSG_URI_csv[0], hypno_local)
        raw_hypno_info.loc[i] = [hypno_local]
    return raw_hypno_info


def download_xtr_hypnogram(subjects_numbers, scorer=str):
    #  This function takes the raw scoring file and converts it the accepted hypnogram input by YASA

    s3_client = boto3.client('s3')
    raw_hypno_info = pd.DataFrame(columns=['URI'])  # prepare for return
    for i in range(len(subjects_numbers)):  # Iterate over all desired subjects
        if subjects_numbers[i] < 100: prefix_temp = 'SleepScoringData/DOD/0' + str(subjects_numbers[i]) + str('/')
        if subjects_numbers[i] > 99: prefix_temp = 'SleepScoringData/DOD/' + str(subjects_numbers[i]) + str('/')
        if scorer == 'Shani':
            xtr_URI_csv = list(S3.get_paths(s3_client, bucket_name='sleep-staging-data', prefix=prefix_temp,
                                            pattern='*_Lab/XtrodesData/ScoredData/DOD - Shani OZ/*_SO_fixed.csv'))
        else:
            print('Only Shani was implemented in this code')
        # filename = os.path.basename(xtr_URI_csv[0])
        new_filename = str(subjects_numbers[i]) + '_xtr_hypno.csv'
        hypno_local = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawHypnograms', new_filename)

        if not os.path.exists(hypno_local):
            S3.download(s3_client, xtr_URI_csv[0], hypno_local)
        raw_hypno_info.loc[i] = [hypno_local]
    return raw_hypno_info


def psg_adjust_hypno_to_edf(subjects_numbers, raw_hypno_info, edf_sample_rate):
    hypno_stages_upsample_all = list()
    for i in range(len(subjects_numbers)):  # Iterate over all desired subjects
        edf_info_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF_info',
                                     str(subjects_numbers[i]) + '_PSG_EDF_info.csv')
        raw_edf_info = pd.read_csv(edf_info_path)
        edf_start = raw_edf_info.loc[0].at['Start time'][0:19]
        edf_start = datetime.strptime(edf_start, '%Y-%m-%d %H:%M:%S')
        hypno = pd.read_csv(raw_hypno_info.loc[i].at['URI'])
        hypno_start = str(hypno.loc[0].at['Date [DD-MM-YYYY]']) + ' ' + str(hypno.loc[0].at['Time [hh:mm:ss]'])
        hypno_start = datetime.strptime(hypno_start, '%d-%m-%Y %H:%M:%S')
        hypno_stages = (hypno.loc[:, 'Stage']).values
        hypno_stages_upsample = np.repeat(hypno_stages, 30 * edf_sample_rate)
        # Fix start
        if not hypno_start == edf_start:
            if edf_start < hypno_start: hypno_stages_upsample = ["W" for x in range(
                ((hypno_start - edf_start).seconds) * edf_sample_rate)] + list(hypno_stages_upsample)
            if edf_start > hypno_start: hypno_stages_upsample = hypno_stages_upsample[((
                                                                                               edf_start - hypno_start).seconds) * edf_sample_rate: hypno_stages_upsample.size]
        # Fix end
        num_of_samples = int(raw_edf_info.loc[0].at['Samples number'])
        if len(hypno_stages_upsample) < num_of_samples:  # add wake for few seconds at the end
            hypno_stages_upsample = list(hypno_stages_upsample) + ["W" for x in
                                                                   range(num_of_samples - len(hypno_stages_upsample))]
        if len(hypno_stages_upsample) > num_of_samples:
            hypno_stages_upsample = hypno_stages_upsample[0:num_of_samples]
        hypno_stages_upsample = np.array(hypno_stages_upsample)
        hypno_stages_upsample[hypno_stages_upsample == 'W'] = 0
        hypno_stages_upsample[hypno_stages_upsample == 'N1'] = 1
        hypno_stages_upsample[hypno_stages_upsample == 'N2'] = 2
        hypno_stages_upsample[hypno_stages_upsample == 'N3'] = 3
        hypno_stages_upsample[hypno_stages_upsample == 'R'] = 4
        hypno_stages_upsample_all.append(hypno_stages_upsample)
    return hypno_stages_upsample_all


def xtr_adjust_hypno_to_edf(subjects_numbers, raw_hypno_info, edf_sample_rate):
    hypno_stages_upsample_all = list()
    for i in range(len(subjects_numbers)):  # Iterate over all desired subjects
        edf_info_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF_info',
                                     str(subjects_numbers[i]) + '_XTR_EDF_info.csv')
        raw_edf_info = pd.read_csv(edf_info_path)
        edf_start = raw_edf_info.loc[0].at['Start time'][0:19]
        edf_start = datetime.strptime(edf_start, '%Y-%m-%d %H:%M:%S')
        hypno = pd.read_csv(raw_hypno_info.loc[i].at['URI'])
        hypno_start = hypno.loc[0].at['Datetime']
        hypno_start = datetime.strptime(hypno_start, '%d-%m-%Y %H:%M:%S')
        hypno_stages = (hypno.loc[:, 'Stage']).values
        hypno_stages_upsample = np.repeat(hypno_stages, 30 * edf_sample_rate)
        # Fix start
        if not hypno_start == edf_start:
            if edf_start < hypno_start: hypno_stages_upsample = ['W' for x in range(
                ((hypno_start - edf_start).seconds) * edf_sample_rate)] + list(hypno_stages_upsample)
            if edf_start > hypno_start: hypno_stages_upsample = hypno_stages_upsample[((edf_start - hypno_start).seconds) * edf_sample_rate: hypno_stages_upsample.size]
        # Fix end
        num_of_samples = int(raw_edf_info.loc[0].at['Samples number'])
        if len(hypno_stages_upsample) < num_of_samples:  # add wake for few seconds at the end
            hypno_stages_upsample = list(hypno_stages_upsample) + ['W' for x in
                                                                   range(num_of_samples - len(hypno_stages_upsample))]
        if len(hypno_stages_upsample) > num_of_samples:
            hypno_stages_upsample = hypno_stages_upsample[0:num_of_samples]

        hypno_stages_upsample = np.array(hypno_stages_upsample)
        hypno_stages_upsample[hypno_stages_upsample == 'W'] = 0
        hypno_stages_upsample[hypno_stages_upsample == 'N1'] = 1
        hypno_stages_upsample[hypno_stages_upsample == 'N2'] = 2
        hypno_stages_upsample[hypno_stages_upsample == 'N3'] = 3
        hypno_stages_upsample[hypno_stages_upsample == 'R'] = 4

        hypno_stages_upsample_all.append(hypno_stages_upsample)
    return hypno_stages_upsample_all


def prepare_raw_signal(file_path):
    obj_EDF = DataObj(file_path)
    data = np.atleast_2d(obj_EDF.data_exg)
    data = np.transpose(data)
    return data


def get_psg_path(subject):
    edf_info_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF_info',
                                 str(subject) + '_PSG_EDF_info.csv')
    raw_edf_info = pd.read_csv(edf_info_path)
    file_path = raw_edf_info.loc[0].at['DownSampled URI']
    return file_path


def get_xtr_path(subject):
    edf_info_path = os.path.join(r'C:\Users\Administrator\PycharmProjects\SpindleProject\RawEDF_info',
                                 str(subject) + '_XTR_EDF_info.csv')
    raw_edf_info = pd.read_csv(edf_info_path)
    file_path = raw_edf_info.loc[0].at['DownSampled URI']
    return file_path


