# VAP


VAP - Vision-based AI Pipeline 
============================

 Copyright 2025

 Authors:
 Bing Wang
 Christopher M. Sullivan

 Center for Quantitative Life Science (CEOAS)
 College of Earth, Ocean, and Atmospheric Sciences (CEOAS)
 University Information Technology (UIT)
 Oregon State University
 Corvallis OR, 97331

 Email:
 wangb5@oregonstate.edu
 chris.sullivan@oregonstate.edu

 This program is free for educational and research purposes.  You cannot redistribute 
 OSU_ScreenSign_System or any of the tools. You can modify it in any way for
 educational and research purposes.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY.



Copyright and Porting
=====================

 See the Copyright file for copyright conditions.



Using VAP Pipeline
========================================
 Thank you for choosing to use the VAP Pipeline. This tool 
 was designed to work on a standard Linux / Unix based machine with
 a simple web service. The tool is designed to sync data shared from

 The tool is built with a standard directory structure. You will find the
 main processing script (vap.py) located under the "bin" folder
 of the main install directory. This tool does all the work but gets most of
 the core information from the configuration file located under the "conf"
 folder. In the "conf" folder you will find a default configuration file 
 called "signs.conf". Please refer to the INSTALL file for information about
 the configuration file values. 

 The main processing tool is designed to run by default with no output. The
 tool does have command line options that can be shown using the -h option. 
 Users that are looking to have output as they run the tools should use the -v
 option or verbose mode. Most of the time users will configure the processing
 directory and microsoft path folder in the configuration file but we have 
 command line options for those settings as well for testing purposes. 

        Usage:

        vap.py -c <config_file> -d <processing_dir> -o <output_dir> -h -v

        -c The Confiuration file path (default: <install_dir>/conf/signs.conf)
        -d The data processing directory for holding the data (local)
        -o The MicrosoftShare data for sync (local)
        -v Verbose mode / Print dialog
        -h Help message 

 The processing tool will sync data down to the output_dir entry in the 
 configuration file. Once the data is synced the system will make a working
 copy into the processing_dir entry in the configuration file. The processing
 tool will then convert all the data from each sub folder in the share into
 a single MP4 file. Once the file is created it will then copy that data to
 the document_root entry in the configuration file. 

 Once you have the settings you are looking for you will need to run the
 processing tool in some regular interval. This is generally managed by
 the crontab on Linux / Unix based machines. Please refer to the INSTALL
 file for information about setting up the crontab entry. Users are welcome
 to use other methods to run the tool on the interval of their choice. 



Minimum Requirements for running
================================
 Minimum Requirements are in the INSTALL file.



Installation on Unix/Linux based machine
========================================
 Build instructions are in the INSTALL file.

 The code for VAP was written with portability in mind; however,
 VAP was developed with Python3 and requires minimal libraries. 
 VAP has been tested on the following systems (incomplete), we
 cannot guarantee any compatibility.

 o RedHat/Rocky/CENT-OS 7.x/8.x/9.x Linux 64-bit 
 o Ubuntu 18.04/20.04 Linux 64-bit 
 o SuSE 15.x Linux 64-bit 



Help and Bug Reports
====================

 Please send question or Bugs to:
 chris.sullivan@oregonstate.edu



End
===