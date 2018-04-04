#!/usr/bin/env python

"""
Parse training log

Evolved from parse_log.sh
"""

import os
import re
import extract_seconds
import argparse
import csv
#import pdb
from collections import OrderedDict
import pandas as pd
from matplotlib import *
from matplotlib.pyplot import *


def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, test_dict_list)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows
    """

    regex_iteration = re.compile('Iteration (\d+)')
    regex_train_output = re.compile('Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile('Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_learning_rate = re.compile('lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')

    # Pick out lines of interest
    iteration = -1
    learning_rate = float('NaN')
    train_dict_list = [[] for i in range(4)]
    test_dict_list = [[] for i in range(4)]
    train_row = None
    test_row = None
    stage_num = 0

    logfile_year = extract_seconds.get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        start_time = extract_seconds.get_start_time(f, logfile_year)

        for line in f:
            iteration_match = regex_iteration.search(line)
            line_s = line.strip()
            if len(line_s) == 0:
                continue
            if line.find('Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model') != -1:
                stage_num = 1
            if line.find('Stage 2 RPN, init from stage 1 Fast R-CNN model') != -1:
                stage_num = 2
            if line.find('Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model') != -1:
                stage_num = 3
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue

            try:
                time = extract_seconds.extract_datetime_from_line(line,
                                                                  logfile_year)
            except ValueError:
                # Skip lines with bad formatting, for example when resuming solver
                continue

            seconds = (time - start_time).total_seconds()

            learning_rate_match = regex_learning_rate.search(line)
            if learning_rate_match:
                learning_rate = float(learning_rate_match.group(1))

            train_dict_list[stage_num], train_row = parse_line_for_net_output(
                regex_train_output, train_row, train_dict_list[stage_num],
                line, iteration, seconds, learning_rate
            )
            test_dict_list[stage_num], test_row = parse_line_for_net_output(
                regex_test_output, test_row, test_dict_list[stage_num],
                line, iteration, seconds, learning_rate
            )

    for i in range(4):
        fix_initial_nan_learning_rate(train_dict_list[i])
        fix_initial_nan_learning_rate(test_dict_list[i])

    return train_dict_list, test_dict_list


def parse_line_for_net_output(regex_obj, row, row_dict_list,
                              line, iteration, seconds, learning_rate):
    """Parse a single line for training or test output

    Returns a a tuple with (row_dict_list, row)
    row: may be either a new row or an augmented version of the current row
    row_dict_list: may be either the current row_dict_list or an augmented
    version of the current row_dict_list
    """

    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one
            if row:
                # If we're on a new iteration, push the last row
                # This will probably only happen for the first row; otherwise
                # the full row checking logic below will push and clear full
                # rows
                row_dict_list.append(row)

            row = OrderedDict([
                ('NumIters', iteration),
                ('Seconds', seconds),
                ('LearningRate', learning_rate)
            ])

        # output_num is not used; may be used in the future
        # output_num = output_match.group(1)
        output_name = output_match.group(2)
        output_val = output_match.group(3)
        row[output_name] = float(output_val)

    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        # The row is full, based on the fact that it has the same number of
        # columns as the first row; append it to the list
        row_dict_list.append(row)
        row = None

    return row_dict_list, row


def fix_initial_nan_learning_rate(dict_list):
    """Correct initial value of learning rate

    Learning rate is normally not printed until after the initial test and
    training step, which means the initial testing and training rows have
    LearningRate = NaN. Fix this by copying over the LearningRate from the
    second row, if it exists.
    """

    if len(dict_list) > 1:
        dict_list[0]['LearningRate'] = dict_list[1]['LearningRate']


def save_csv_files(logfile_path, output_dir, train_dict_list, test_dict_list,
                   delimiter=',', verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """
    for i in range(4):
        log_basename = os.path.basename(logfile_path)
        train_filename = os.path.join(output_dir, log_basename + '.train' + str(i))
        write_csv(train_filename, train_dict_list[i], delimiter, verbose)
        test_filename = os.path.join(output_dir, log_basename + '.test' + str(i))
        write_csv(test_filename, test_dict_list[i], delimiter, verbose)


def write_csv(output_filename, dict_list, delimiter, verbose=False):
    """Write a CSV file
    """

    if not dict_list:
        if verbose:
            print('Not writing %s; no lines to write' % output_filename)
        return

    dialect = csv.excel
    dialect.delimiter = delimiter

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=dict_list[0].keys(),
                                     dialect=dialect)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print 'Wrote %s' % output_filename


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('output_dir',
                        help='Directory in which to place output CSV files')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    parser.add_argument('--delimiter',
                        default=',',
                        help=('Column delimiter in output files '
                              '(default: \'%(default)s\')'))

    args = parser.parse_args()
    return args

def show():
    train_log = pd.read_csv("/work/dev/experiments/py-mask-rcnn/caffe-fast-rcnn/tools/extra/faster_rcnn_end2end_VGG_CNN_M_1024_.txt.2017-12-07_11-05-11.log.train0")
    #test_log = pd.read_csv("./lenet_train.log.test")
    _, ax1 = subplots(figsize=(15, 10))
    ax2 = ax1.twinx()
    ax1.plot(train_log["NumIters"], train_log["loss_mask"], alpha=0.4)
    #ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
    #ax2.plot(test_log["NumIters"], test_log["acc"], 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    savefig("./train_test_image.png")  # save image as png
def main():
    # args = parse_args()
    logfile_path = '/work/dev/experiments/py-mask-rcnn/experiments/logs/faster_rcnn_end2end_VGG_CNN_M_1024_.txt.2017-12-07_11-05-11.log'
    train_dict_list, test_dict_list = parse_log(logfile_path)
    output_dir = './'
    save_csv_files(logfile_path, output_dir, train_dict_list,
                   test_dict_list, delimiter=',')

    show()

if __name__ == '__main__':
    main()
