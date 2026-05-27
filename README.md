**VAPP: Vision-based AI plankton pipeline**

Bing Wang<sup>1</sup>, Marco Corrales-Ugalde<sup>2</sup>, Elena
Conser<sup>3</sup>, Nicholas F. Howard<sup>2</sup>, Cameron S.
Royer<sup>3</sup>, Savana Stallsmith<sup>2,3</sup>, Jennifer
Fisher<sup>5</sup>, Christopher Sullivan<sup>4</sup>,Su
Sponaugle<sup>3</sup>, Robert K. Cowen<sup>3,4</sup>

<sup>1</sup>Center for Quantitative Life Science, Oregon State
University, Corvallis, OR, 97331, United States

<sup>2</sup><span class="mark">Hatfield Marine Science Center,</span>
Oregon State University, Newport, OR, 97365, United States

<sup>3</sup><span class="mark">Department of Integrative Biology, Oregon
State University, Corvallis, OR, 97331, United States</span>

<sup>4</sup>College of Earth, Ocean, and Atmospheric Sciences, Oregon
State University, Corvallis, OR, 97331, United States

<sup>5</sup><span class="mark">The Cooperative Institute for Marine
Ecosystem and Resources Studies (CIMERS), Newport, OR 97365</span>

VAPP is a vision-based AI plankton pipeline developed to segment and
classify marine and freshwater plankton imaged by shadowgraph systems
(Greer et al., 2025), and to integrate classification results with
environmental data from images and videos captured specifically by the
In Situ Ichthyoplankton Imaging System (ISIIS) or similar imaging
devices. It was implemented and tested on plankton imaging datasets from
ISIIS. Although VAPP was designed with plankton imaging systems in mind,
the generalizable workflow is intended to be used for other image
classification based analyses.VAPP is written in Python and includes a
precompiled segmentation tool, Threshold MSER In Situ Plankton
Segmentation. The pipeline requires minimal third-party libraries and
can run in a Unix/Linux environment on Windows, Linux, and macOS
operating systems.

Pipeline is maintained at
[<u>https://github.com/wangb799/VAPP</u>](https://github.com/wangb799/VAPP)

<span class="mark">\#########################################################\
\# Copyright © 2026 Oregon State University\
\#</span> Wang, B.; Corrales-Ugalde M.; Conser, E.; Howard, N.F.; Royer,
C.S.; Stallsmith, S.; Fisher, J.; Sullivan, C.; Sponaugle, S.; Cowen,
R.K.\
<span class="mark">\# Center for Quantitative and Life Sciences &</span>

<span class="mark">\# College of Earth, Ocean, and Atmospheric
Sciences</span>

<span class="mark">\# Hatfield Marine Science Center &\
\# Oregon State University\
\# Newport, OR 97365\
\# Corvallis, OR 97331\
\# Cite as:</span> Wang, B.; Corrales-Ugalde M.; Conser, E.; Howard,
N.F.; Royer, C.S.; Stallsmith, S.; Fisher, J.; Sullivan, C.; Schmid, M.;
Sponaugle, S.; Cowen, R.K. (2026) VAPP: Vision-based AI plankton
pipeline\
<span class="mark">\# This program is distributed WITHOUT ANY WARRANTY;
without even the implied warranty of\
\# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\
\#\
\# This program is distributed under the GNU GPL v 2.0 or later
license.\
\#\
\# Any User wishing to make commercial use of the Software must contact
the authors or</span>

<span class="mark">\# Oregon State University directly to arrange an
appropriate license.\
\# Commercial use includes (1) use of the software for commercial
purposes, including\
\# integrating or incorporating all or part of the source code into a
product\
\# for sale or license by, or on behalf of, User to third parties, or
(2) distribution\
\# of the binary or source code to third parties for use with a
commercial\
\# product sold or licensed by, or on behalf of, User.\
\#\
\#########################################################</span>

**Getting Started**

The VAPP pipeline consists of the following steps: T-MSER segmentation
(Panaïotis et al., 2022), classification, occurrence merging,
environmental data merging, and final integration of environmental and
classification results. Segmentation and classification can also be run
independently.

**Dependencies**

conda env create -f environment.yml

\# The enviroment.yml contains the required dependencies. This command
creates a conda environment vapp with installation of dependencies.

conda activate vapp\
\# Activate vapp conda environment

**Installation\**
git clone https://gitlab.cqls.oregonstate.edu/sullichr/vapp.git

\# Clone the VAPP repository

Usage and Options

usage: vapp.py \[-h\] \[-i INPUT\] \[-sb SEGMENT_BIN\] \[-ai AI_MODEL\]
\[-mw WEIGHTS\] \[-mo MODELOPT\] \[-en ENVIRONMENTAL\] \[-o OUTPUT\]
\[-g GPU\] \[-c CONFIG\] \[-vv\] {segment, classify}

options:\
-h show this help message and exit\
-i INPUT Input folder containing AVIs or images\
-sb SEGMENT_BIN Path to segmentation binary\
-ai AI_MODEL AI model (yolo, inceptionv3, megadetector, U-Net)\
-mw WEIGHTS Model weights file (.weights,.pt)\
-mo MODELOPT Model advanced option (extra option for the model)\
-en ENVIRONMENTAL Path to environmental data directory (publisher
subdirs inside)\
-o OUTPUT Output directory (will contain segmentation/, classification/,
merge/)\
-g GPU GPU ID (default: 0)\
-c CONFIG Configuration file to use over command line options\
-vv Enable verbose output

positional arguments: {segment, classify}

vapp.py segment \[-h\] \[-i INPUT\] \[-sb SEGMENT_BIN\] \[-o OUTPUT\]
\[-g GPU\] \[-c CONFIG\] \[-vv\]\
<span class="mark">\# segment only</span>

vapp.py classify \[-h\] \[-i INPUT\] \[-ai AIMODEL\] \[-mw WEIGHTS\]
\[-mo MODELOPT\] \[-o OUTPUT\] \[-g GPU\] \[-c CONFIG\] \[-vv\]\
<span class="mark">\# classify only</span>

**Configuration file\**
Options can be provided either through the command line or a
configuration file.

If the same option is specified in both places, the command-line option
takes precedence.

e.g.\
\[Configuration\]

input = test_video

segment-bin =
/home/cqls/wangb5/cowenNFS4/freshwater/pipeline/Threshold-MSER_current/build/segment

ai-model = yolo

weights = best.pt

modelopt =

environmental =
/nfs4/FW_HMSC/Cowen_Lab/MBON/2024/fall/environmental_data

output = test_result

gpu = 0

verbose = False

\[Segmentation\]

delta=4

min-area=50

max-area=400000

threshold=160

signal-to-noise=60

outlier-percent=0.15

variation=100

epsilon=1

top-crop=0

bottom-crop=0

left-crop=66

right-crop=23

\[classification\]

input = test_result/segmentation

ai-model = yolo

weights = best.pt

modelopt =

output = test_result

gpu = 0

verbose = False

**Output\**
The pipeline outputs four subdirectories: classification, measurements,
merge, and segmentation. These directories contain the results for
classification, measurement, merged files for occurrences and
integration of classification and environmental data, and segmentation.

e.g.

├── classification

│ └── CamTop-09-20-2024-00-49-08.947_classification.csv

├── measurements

│ └── CamTop-09-20-2024-00-49-08.947_measure.csv

├── merge

│ ├── test_result_merged_occurrence.csv

│ └── test_video_UNKNOWN_merge_final.csv

└── segmentation

└── CamTop-09-20-2024-00-49-08.947

├── CamTop-09-20-2024-00-49-08.947_classification.csv

├── measurements

│ ├── CamTop-09-20-2024-00-49-08.947_bboxData.csv

│ ├── CamTop-09-20-2024-00-49-08.947.csv

│ └── CamTop-09-20-2024-00-49-08.947_yoloFormat.txt

└── segmentation

└── corrected_crop

├── CamTop-09-20-2024-00-49-08.947_0001_crop_0000.png

├── CamTop-09-20-2024-00-49-08.947_0001_crop_0001.png

├── CamTop-09-20-2024-00-49-08.947_0001_crop_0002.png

**Running the pipeline\**
python bin/vapp.py -c conf/vapp.conf\
\# Run the pipeline using only a config file

python bin/vapp.py --input test_video --segment-bin
/home/cqls/wangb5/cowenNFS4/freshwater/pipeline/Threshold-MSER_current/build/segment
--weights [<u>best.pt</u>](http://best.1000.pt/) --environmental
/nfs4/FW_HMSC/Cowen_Lab/MBON/2024/fall/environmental_data --output
test_result\
\# Run the pipeline using command line options

python bin/vapp.py -c conf/vapp.conf -o test_result

\# Run the pipeline with both of config file and command line options

python bin/vapp.py segment -c conf/vapp.conf -o test_result\
\# Run the segmentation step independently using both a config file and
command line options

python bin/vapp.py segment -c conf/vapp.conf

\# Run the segmentation step independently using only a config file

python bin/vapp.py classify -c conf/vapp.conf -o test_result

\# Run the classification step independently using both a config file
and command-line options

python bin/vapp.py classify -c conf/vapp.conf

\# Run the classification step independently using only a config file

**License**

See the Copyright file for copyright conditions.

**Acknowledgments**

The project has received funding from the US National Science Foundation
under grant numbers OCE-2125407, OCE-1737399, EF-2222214, \[\[NOAA
funding source TBD\]\], The Belmont Forum (through NSF grant number
1927710), Extreme Science and Engineering Discovery Environment (XSEDE)
under grant number OCE-170012, NSF ACCESS (formerly XSEDE) OCE-170012,
and the NOAA-Fisheries Optics Strategic Initiative.

**Help and Bug Reports**

Please send questions or bugs to: chris.sullivan@oregonstate.edu

**References**

Greer, A.T., Duffy, P.I., Walles, T.J.W., Cousin, C., Treible, L.M.,
Aaron, K.D., Nejstgaard, J.C., 2025. Modular shadowgraph imaging for
zooplankton ecological studies in diverse field and mesocosm settings.
Limnology & Ocean Methods 23, 67–86. https://doi.org/10.1002/lom3.10657

Panaïotis, T., Caray–Counil, L., Woodward, B., Schmid, M.S., Daprano,
D., Tsai, S.T., Sullivan, C.M., Cowen, R.K., Irisson, J.-O., 2022.
Content-Aware Segmentation of Objects Spanning a Large Size Range:
Application to Plankton Images. Front. Mar. Sci. 9, 870005.
https://doi.org/10.3389/fmars.2022.870005
