#!/usr/bin/env python3
#####################################################################
#
#  VAP Pipeline
#                                                                   
#  Copyright 2025
#
#  Bing Wang
#  Christopher M. Sullivan
#
#  Center for Quantitative Life Science (CEOAS)
#  College of Earth, Ocean, and Atmospheric Sciences
#  Oregon State University
#  Corvallis, OR 97331
#
#  email: wangb5@oregonstate.edu
#  email: chris.sullivan@oregonstate.edu
#
# This program is not free software; you can not redistribute it
# and/or modify it at all.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#                                                                   
#                                                                   
#####################################################################

from datetime import datetime
import time
import glob
import os
import sys
import getopt
import re
import subprocess

#####################################################################
#                                                                   #
#                         Start of Program                          #
#                                                                   #
#####################################################################

#####################################################################
# Setup verbose mode as no to we has a silent run by default        #
#####################################################################
verbose = "no"
output_dir = ""
processing_dir = ""

#####################################################################
# datetime object containing current date and time                  #
#####################################################################
current_time = datetime.now()
current_unix = (time.mktime(current_time.timetuple()))

#####################################################################
# Set the paths                                                     #
#####################################################################
path = os.getcwd()
script_path = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(script_path, os.pardir))
conf_file = path + '/conf/vap.conf'

#####################################################################
# Parse any command line argurments                                 #
#####################################################################
opts, args = getopt.getopt(sys.argv[1:],"hvc:d:o:",["conf_file=","processing_dir=","output_dir="])
for opt, arg in opts:
    if opt == '-h':
        print ("\n\tYou did not provide all the correct command line arguments\n")
        print ("\tUsage:\n")
        print ("\tvap.py -c <config_file> -d <processing_dir> -o <output_dir> -h -v\n")
        print ("\t-c The Confiuration file path (default: <install_dir>/conf/vap.conf)")
        print ("\t-d The data processing directory for holding the data (local)")
        print ("\t-o The output directory (local)")
        print ("\t-v Verbose mode / Print dialog")
        print ("\t-h Help message")
        print ("\n")
        sys.exit()
    elif opt == '-v':
        verbose = "Yes"
    elif opt in ("-c", "--conf_file"):
        conf_file = arg
    elif opt in ("-d", "--processing_dir"):
        processing_dir = arg
    elif opt in ("-o", "--output_dir"):
        output_dir = arg


#####################################################################
# Get configuration data from config file                           #
#####################################################################
def getVarFromFile(filename):
    import configparser
    global output_dir
    global processing_dir
    global touch_bin
    global mkdir_bin
    global rm_bin
    global mv_bin
    config = configparser.ConfigParser()
    config.read(filename)
    if output_dir == '':
        output_dir = config.get('Configuration', 'output_dir')
    if processing_dir == '':
        processing_dir = config.get('Configuration', 'processing_dir')
    touch_bin = config.get('Configuration', 'touch_bin')
    mkdir_bin = config.get('Configuration', 'mkdir_bin')
    rm_bin = config.get('Configuration', 'rm_bin')
    mv_bin = config.get('Configuration', 'mv_bin')
    if verbose == "Yes":
        print ("\n   Configuration Loaded: " + path + "/conf/vap.conf\n")


getVarFromFile(conf_file)


#####################################################################
# Start the run                                                     #
#####################################################################
if verbose == "Yes":
    print ("\n   Start of Run!\n", file=sys.stdout, flush=True)
    print ("\n   Install Directory:", path, file=sys.stdout, flush=True)

if verbose == "Yes":
    print ("\n\tCurrent Paths\n")
    print ('\t\tFull Sync Directory Path: ', processing_dir, file=sys.stdout, flush=True)
    print ('\t\tFull Output Directory Path: ', output_dir, file=sys.stdout, flush=True)

#####################################################################
# Main Pipeline                                                     #
#####################################################################
if processing_dir:
    if os.path.exists(processing_dir):
        for input_file in os.listdir(processing_dir):
            if input_file.endswith('.avi') or input_file.endswith('.jpg') or input_file.endswith('.png'):
                total_processed = total_processed + 1
                # Do something with "input_file" variable
                if verbose == "Yes":
                    print ('\t\t  File: ', input_file)
                print ("Doing something with Input File", input_file)
            else:
                if verbose == "Yes":
                    print ('\t\tExiting - Not a avi, jpg or png file: ', input_file)
    else:
        print ('\t\tExiting - processing directory does not exists: ', processing_dir)
else:
    print ('\t\tExiting - processing directory was not provided: ')



if verbose == "Yes":
    print ("\n   End of Run!\n", file=sys.stdout, flush=True)

sys.exit()

#####################################################################
#                                                                   #
#                           End of Program                          #
#                                                                   #
#####################################################################