import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from datetime import datetime


def plot_comp(signal, sample_rate, sc_sp, yasa_sp, stats):
    sc_no_spindles = signal[0].copy()
    sc_start_time = (sc_sp['Onset']) / sample_rate
    sc_duration_time = (sc_sp['Duration'])
    sc_stop_time = sc_start_time + sc_duration_time
    for srt, stp in zip(sc_start_time, sc_stop_time):
        sc_no_spindles[int(srt * sample_rate):int(stp * sample_rate)] = np.nan

    ya_no_spindles = signal[0].copy()
    ya_start_time = yasa_sp['Start']
    ya_stop_time = yasa_sp['End']
    for srt, stp in zip(ya_start_time, ya_stop_time):
        ya_no_spindles[int(srt * sample_rate):int(stp * sample_rate)] = np.nan

    Plot, Axis = plt.subplots(figsize=(15, 4))
    plt.subplots_adjust(bottom=0.25)
    time_vec = np.arange(0, len(signal[0]))
    plt.plot(time_vec, signal[0], lw=0.5, color='r', label='yasa')
    plt.plot(time_vec, ya_no_spindles, color='k', lw=0.8)
    plt.plot(time_vec, signal[0] + 200, lw=0.5, color='c', label='scorer')
    plt.plot(time_vec, sc_no_spindles + 200, color='k', lw=0.8)
    plt.legend(bbox_to_anchor=(1, 0), loc="lower right")
    plt.ylabel('uV')
    plt.xlabel('Sample')

    if stats is not None:  # Add stats:
        tp = len(stats['tp'])
        fp = len(stats['fp'])
        fn = len(stats['fn'])
        f1 = round(stats['f1'], 3)
        stats_str = 'TP: ' + str(tp) + '  || FP: ' + str(fp) + ' ||  FN: ' + str(fn) + ' ||  F1 Score: ' + str(f1)

    slider_color = 'White'
    axis_position = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=slider_color)
    slider_position = Slider(axis_position, 'Sample', 0.1, len(signal[0]))

    def update(val):
        pos = slider_position.val
        Axis.axis([pos, pos + 30 * 250, -100, 300])
        Plot.canvas.draw_idle()

    slider_position.on_changed(update)

    plt.title(stats_str, fontsize=10, y=25.8)
    plt.suptitle('Spindles detection: YASA vs Scorer', fontsize=16)
    plt.show()


def plot_sw(signal, sample_rate, yasa_sw):
    ya_no_sw = signal[0].copy()
    ya_start_time = yasa_sw['Start']
    ya_stop_time = yasa_sw['End']
    for srt, stp in zip(ya_start_time, ya_stop_time):
        ya_no_sw[int(srt * sample_rate):int(stp * sample_rate)] = np.nan

    Plot, Axis = plt.subplots(figsize=(15, 4))
    plt.subplots_adjust(bottom=0.25)
    time_vec = np.arange(0, len(signal[0]))
    plt.plot(time_vec, signal[0], lw=0.5, color='c')
    plt.plot(time_vec, ya_no_sw, color='k', lw=0.8)
    plt.ylabel('uV')
    plt.xlabel('Sample')
    slider_color = 'White'
    axis_position = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=slider_color)
    slider_position = Slider(axis_position, 'Sample', 0.1, len(signal[0]))

    def update(val):
        pos = slider_position.val
        Axis.axis([pos, pos + 30 * 250, -100, 300])
        Plot.canvas.draw_idle()

    slider_position.on_changed(update)
    plt.suptitle('SW detections by YASA', fontsize=16)
    plt.show()


def plot_xtr_sp(signal, sample_rate, yasa_sp, hypno):
    signal[signal > 500] = 500  # Cut high amplitudes
    signal[signal < -500] = -500
    ya_no_spindles = signal[0].copy()
    ya_start_time = yasa_sp['Start']
    ya_stop_time = yasa_sp['End']
    for srt, stp in zip(ya_start_time, ya_stop_time):
        ya_no_spindles[int(srt * sample_rate):int(stp * sample_rate)] = np.nan

    box = round(len(signal[0])/(1200*sample_rate))
    for box_num in range(box):
        st = box_num*1200*250
        if box_num < box:
            signal_box = signal[0, st:st+1200*sample_rate]
        else:
            signal_box = signal[0, st:len(signal[0])]
        Plot, Axis = plt.subplots(figsize=(15, 4))
        plt.subplots_adjust(bottom=0.25)
        time_vec = np.arange(0, len(signal_box))
        plt.plot(time_vec, signal_box, lw=0.5, color='r', label='yasa')
        plt.plot(time_vec, ya_no_spindles[st:st+1200*sample_rate], color='k', lw=0.8)

        plt.ylabel('uV')
        plt.xlabel('Sample')

        slider_color = 'White'
        axis_position = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=slider_color)
        slider_position = Slider(axis_position, 'Sample', 0.1, len(signal_box))

