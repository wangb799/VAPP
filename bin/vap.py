#!/usr/bin/env python3
#####################################################################
#
#  OSU/NOAA vap.py Pipeline
#                                                                   
#  Copyright 2025
#
#  OSU CIMERS / OSU CEOAS / NOAA
#  Oregon State University
#  Corvallis, OR 97331
#
#  Created By: Bing Wang and Christopher Sullivan
#  Email Contact: chris.sullivan@oregonstate.edu
#
# ----------------------------------------
# Full automatic video analytics pipeline:
#   0) Environmental data merge             -> merged_environmental.csv
#   1) Segmentation                         -> segmentation/corrected_crop + measurements/<avi>_measure.csv
#   2) Classification                       -> classification/<avi>_classification.csv
#   3) create occurrence merged file        -> merge/all_merged_occurrence.csv
#   4) Final merge with environmental       -> merge/<drive>_<cruise>_merge_final.csv
# ----------------------------------------
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

#####################################################################
# Standard System libraries                                         #
#####################################################################
import os
import re
import csv
import sys
import glob
import argparse
import subprocess
import shutil
import textwrap
from datetime import datetime, timedelta
from collections import defaultdict

#####################################################################
# Third-party libraries                                             #
#####################################################################
import torch
from ultralytics import YOLO

#####################################################################
# ANSI color codes                                                  #
#####################################################################
RED='\033[31;1;202m'
GREEN='\033[32;1;202m'
YELGRN='\033[33;1;202m'
ORANGE='\033[33m'
BLUE='\033[34;1;202m'
PURP='\033[35;1;202m'
TURQ='\033[36;1;202m'
WHITE='\033[37;1;202m'
C_END='\033[0m'


#####################################################################
# Function Definitions                                              #
#####################################################################

def directory(arg):
    if os.path.isdir(arg):
        return os.path.normpath(arg)
    raise argparse.ArgumentTypeError("Not a valid directory path")

def file_empty(path):
    return (not os.path.exists(path)) or (os.path.exists(path) and os.stat(path).st_size == 0)

def parse_iso_like(s):
    """
    Parse a timestamp with microseconds."""
    s = s.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unparsable time: {s}")

def supports_half(gpu: str = "0") -> bool:
    """
    Return True if the given CUDA device supports half-precision inference.
    """
    if not torch.cuda.is_available():
        return False
    try:
        device_index = int(gpu)
        major, minor = torch.cuda.get_device_capability(device_index)
        # FP16 fully supported on compute capability >= 7.0
        return major >= 7
    except Exception:
        return False

#####################################################################
# Environmental data merge                                          #
#####################################################################

def contains_header(ptr):
    """Detect CSV header by checking first token starts with 'Time'."""
    pos = ptr.tell()
    line = ptr.readline()
    ptr.seek(pos)
    return line.startswith("Time")

def get_files(dir_path):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    if len(files) <= 0:
        print(f"\t\t{RED}  Error:{C_END} No files found in directory {dir_path}", file=sys.stderr)
        sys.exit(1)
    return files

class DoubleGen:
    """Yield (last_row_dict, next_row_dict) pairs across a CSV file (DictReader)."""
    def __init__(self, f, fieldnames=None, delimiter=',', header_lines=0):
        import itertools as _it
        self.f = f
        self.fptr = open(f, 'r', newline='')
        self.header_lines = header_lines
        if fieldnames:
            self.reader = csv.DictReader(_it.islice(self.fptr, self.header_lines, None), fieldnames, delimiter=delimiter)
        else:
            self.reader = csv.DictReader(_it.islice(self.fptr, self.header_lines, None), delimiter=delimiter)
        self.fieldnames = self.reader.fieldnames
        self._last = None
        self._next = None
    def __iter__(self): return self
    def __next__(self):
        if self._last is None and self._next is None:
            self._last = next(self.reader)
            self._next = next(self.reader)
        else:
            self._last = self._next
            self._next = next(self.reader)
        return self._last, self._next
    def restart(self):
        self.fptr.seek(0)
        # rebuild reader and skip header lines
        if self.fieldnames:
            self.reader = csv.DictReader(self.fptr, fieldnames=self.fieldnames)
        else:
            self.reader = csv.DictReader(self.fptr)
        next(self.reader)  # skip header
        self._last = next(self.reader)
        self._next = next(self.reader)

def dir_double_gen(directory, fieldnames=None, delimiter=',', header_lines=1):
    if not os.path.exists(directory):
        print(f"\t\t{RED}  Error:{C_END} directory not valid for the generator: {directory}", file=sys.stderr)
        sys.exit(1)
    files = get_files(directory)
    prev_next = None
    for i, f in enumerate(files):
        gen = DoubleGen(f, fieldnames, delimiter, header_lines)
        try:
            new_last, new_next = next(gen)
        except StopIteration:
            continue
        if prev_next is not None:
            yield prev_next, new_last
        yield new_last, new_next
        try:
            for _last, _next in gen:
                yield _last, _next
        except StopIteration:
            pass
        prev_next = _next

def get_time_env_like(line):
    """Parse environmental 'Time' column (ISO-ish or epoch 1904 seconds)."""
    if "Time" not in line:
        raise KeyError("Expected 'Time' column in environmental data")
    val = (line["Time"] or "").strip()
    # ISO-ish
    if re.search(r'\d{4}-\d{2}-\d{2}[T ]?\d{2}:\d{2}:\d{2}\.\d{6}', val):
        try:
            dt = parse_iso_like(val)
            # If ends with Z (UTC), apply the  -7h shift
            if val.endswith("Z"):
                dt = dt - timedelta(hours=7)
            return dt
        except Exception:
            pass
    # Numeric seconds 1904-01-01
    else:
        try:
            seconds = float(val)
            start = datetime(1904, 1, 1)
            # -7h shift
            delta = timedelta(hours=-7, seconds=seconds) 
            return start + delta
        except Exception:
            # fallback
            return parse_iso_like(val)

def closest_time(occ_time, last_pub_time, next_pub_time, max_time_gap=10):
    next_gap = (next_pub_time - occ_time)
    last_gap = (occ_time - last_pub_time)
    if (last_gap > timedelta(seconds=max_time_gap) and next_gap > timedelta(seconds=max_time_gap)):
        return "skip"
    return "last" if last_gap <= next_gap else "next"

def check_env_header(directory, expected_header, delimiter=","):
    env_files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f))]
    env_files.sort()
    if not env_files:
        print(f"\t\t{RED}  Error:{C_END} No files in {directory}", file=sys.stderr)
        sys.exit(1)

    header_file = env_files[0]
    with open(header_file, "r", newline='') as ptr:
        found_header = next(ptr).rstrip("\n").split(delimiter)

        # Normalize headers (strip whitespace, carriage returns, etc.)
        found_header = [h.strip().rstrip('\r') for h in found_header]
        expected_header = [h.strip().rstrip('\r') for h in expected_header]

        if found_header and "Time" in found_header[0]:
            found_header[0] = "Time"

        if found_header != expected_header:
            print(f"\t\t{RED}  Error:{C_END}Headers do not match for {directory}\n"
                  f"\t\t\tExpected: {expected_header}\n"
                  f"\t\t\tFound:    {found_header}",
                  file=sys.stderr)
            sys.exit(1)

def publisher_iterator(d, header):
    files = get_files(d)
    for f in files:
        with open(f, 'r', newline='') as ptr:
            if contains_header(ptr):
                reader = csv.DictReader(ptr, header)
                # skip header line
                next(reader, None)
            else:
                reader = csv.DictReader(ptr, header)
            for line in reader:
                yield line

def merge_environmental(environmental_dir, output_dir=None, header_lines=0, verbose=True):
    """
    Merge environmental publishers by aligning other publishers to the Inclinometer timeline.
    Writes merged_environmental.csv in environmental_dir.
    """
    environmental_dir = os.path.abspath(environmental_dir)
    out_file = os.path.join(environmental_dir, "merged_environmental.csv")
    if output_dir:
        out_file = os.path.join(output_dir, "merged_environmental.csv")

    # Publisher expected headers
    ctd_header = ["Time","Temperature","Conductivity","Pressure","Depth","Salinity","Sound Velocity"]
    altimeter_header = ["Time", "Altitude"]
    ad1216_header = ["Time","Raw Oxygen [V]","Raw PAR [V]","Raw pH [V]","Received Checksum","Computed Checksum"]
    flowmeter_header = ["Time","Horizontal Speed","Vertical Speed"]
    inclinometer_header = ["Time","AccelrationX [milliG]","AccelerationY [milliG]","AccelerationZ [milliG]","Roll [deg]","Pitch [deg]","Temperature [deg C]","Received Checksum","Computed Checksum"]
    if "w22" in environmental_dir:
        gps_header = ["Time","Longitude","Latitude","nmea_quality","nsv","hdop","antenna_height","speed_made_good","course_made_good","mac_epoch_time"]
    else:
        gps_header = ["Time", "Message Type", "UTC DateTime", "Latitude", "Latitude Hemisphere", "Longitude", "Longitude Hemisphere", "Quality", "Number of Satellites", "Horizontal Dilution Precision", "Antenna Altitude", "Antenna Altitude Unit", "Geoidal Seperation", "Geoidal Seperation Unit", "Age", "Station ID", "Checksum"]
    fluorometer_header = ["Time","Chlorophyll Wavelength","Chlorophyll"]

    base_publisher = ("Inclinometer Publisher", inclinometer_header)
    other_publishers = [
        ("CTD Publisher", ctd_header),
        ("Altimeter Publisher", altimeter_header),
        ("AD1216 Publisher", ad1216_header),
        ("Flowmeter Publisher", flowmeter_header),
        ("GPS Publisher", gps_header),
        ("Fluorometer Publisher", fluorometer_header),
    ]

    # Validate headers
    check_env_header(os.path.join(environmental_dir, base_publisher[0]), base_publisher[1])
    for directory, header in other_publishers:
        # accept both with/without trailing slash names for Fluorometer
        d = os.path.join(environmental_dir, directory)
        if not os.path.isdir(d):
            # try variant
            d2 = os.path.join(environmental_dir, directory.rstrip('/'))
            if os.path.isdir(d2):
                d = d2
        check_env_header(d, header)
    if verbose:
        print(f"\t\t{WHITE}   Info:{C_END} Environmental headers verified. Merging publishers")

    # Build output fieldnames: Time + all non-Time non-Checksum from others + Latitude/Longitude last
    merged_fields = ["Time"] + [
        col for publisher, header in other_publishers for col in header
        if (publisher != "GPS Publisher" and "Checksum" not in col and col != "Time")
    ] + ["Latitude", "Longitude"]

    with open(out_file, 'w', newline='') as out_ptr:
        env_writer = csv.DictWriter(out_ptr, merged_fields, extrasaction='ignore')
        env_writer.writeheader()

        # Inclinometer provides the timeline
        inc_iter = publisher_iterator(os.path.join(environmental_dir, base_publisher[0]), base_publisher[1])

        # Initialize other publisher iterators as double_gens
        pub_readers = []
        for pub_dir, pub_header in other_publishers:
            d = os.path.join(environmental_dir, pub_dir)
            if not os.path.isdir(d):
                d2 = os.path.join(environmental_dir, pub_dir.rstrip('/'))
                d = d2 if os.path.isdir(d2) else d
            pub_readers.append(dir_double_gen(d, pub_header))

        last_rows = []
        next_rows = []
        for j, dg in enumerate(pub_readers):
            try:
                l, n = next(dg)
            except StopIteration:
                print(f"\t\t{RED}  Error:{C_END} {other_publishers[j][0]} empty.", file=sys.stderr)
                return out_file
            last_rows.append(l); next_rows.append(n)

        for line_num, inc_row in enumerate(inc_iter, start=1):
            try:
                inc_time = get_time_env_like(inc_row)
            except Exception:
                continue

            merged_row = dict(inc_row)  # start with inclinometer row

            for i, dg in enumerate(pub_readers):
                try:
                    last_time = get_time_env_like(last_rows[i])
                    next_time = get_time_env_like(next_rows[i])
                except Exception:
                    continue

                while inc_time > next_time:
                    try:
                        last_rows[i], next_rows[i] = next(dg)
                        next_time = get_time_env_like(next_rows[i])
                    except StopIteration:
                        # no more rows for this publisher
                        break

                # pick closer
                try:
                    last_time = get_time_env_like(last_rows[i])
                    which = closest_time(inc_time, last_time, next_time, max_time_gap=10)
                    if which == "last":
                        pub_row = last_rows[i]
                    elif which == "next":
                        pub_row = next_rows[i]
                    else:  # skip
                        continue
                    merged_row.update(pub_row)
                    merged_row["Time"] = inc_time
                except Exception:
                    continue

            # Normalize Lat/Lon if present
            try:
                if "Longitude" in merged_row and "Latitude" in merged_row:
                    lon = str(merged_row["Longitude"]).strip()
                    lat = str(merged_row["Latitude"]).strip()
                    # If numeric string > 180 in abs, treat as deg+min (DDDMM.m)
                    def dm_to_deg(num, deg_len):
                        s = str(num)
                        deg = float(s[:deg_len])
                        minutes = float(s[deg_len:])
                        return deg + minutes/60.0
                    try:
                        lonf = float(lon)
                        latf = float(lat)
                        if abs(lonf) >= 180:
                            lonf = dm_to_deg(lon, 3)
                        if abs(latf) >= 90:
                            latf = dm_to_deg(lat, 2)
                        merged_row["Longitude"] = lonf
                        merged_row["Latitude"] = latf
                    except Exception:
                        pass
                    # Apply hemispheres if present
                    if merged_row.get("Latitude Hemisphere", "N") == "S":
                        try: merged_row["Latitude"] = -abs(float(merged_row["Latitude"]))
                        except Exception: pass
                    if merged_row.get("Longitude Hemisphere", "E") == "W":
                        try: merged_row["Longitude"] = -abs(float(merged_row["Longitude"]))
                        except Exception: pass
            except KeyError:
                pass

            env_writer.writerow(merged_row)

    if verbose:
        print(f"\t\t{WHITE}   Info:{C_END} Created Environmental Merge File: {out_file}")
    return out_file

#####################################################################
# Segmentation                                                      #
#####################################################################

def run_segmentation(segment_bin, input_dir, output_dir, seg_kv_args=None, verbose=False):
    seg_root = os.path.join(output_dir, "segmentation")
    measure_root = os.path.join(output_dir, "measurements")
    os.makedirs(seg_root, exist_ok=True)
    os.makedirs(measure_root, exist_ok=True)

    avis = glob.glob(os.path.join(input_dir, "*.avi"))

    if not avis:
        raise FileNotFoundError(f"\t\t\t{RED}  Error:{C_END} No .avi files found in {input_dir}")

    if verbose:
        print(f"\t\t\t{WHITE}   Info:{C_END} Segmenting {len(avis)} avi videos files")

    for avi in avis:
        avi_base = os.path.splitext(os.path.basename(avi))[0]
        avi_out = seg_root

        # Build segmentation command dynamically
        cmd = [segment_bin, "-i", avi, "-o", avi_out]
        if seg_kv_args:
            cmd.extend(seg_kv_args)

        if verbose:
            #print(f"\t\t\t{PURP}Running:{C_END}", " ".join(map(str, cmd)))
            print(f"\t\t\t{PURP}Running:{C_END} ", avi)

        subprocess.run(cmd, check=True)
        # also copy  measurement csv per avi into measurements root
        try:
            out_measure = os.path.join(avi_out, avi_base, "measurements", f"{avi_base}.csv")
            shutil.copy2(out_measure, os.path.join(measure_root, f"{avi_base}_measure.csv"))
        except Exception as e:
            print(f"\t\t\t{RED}  Error:{C_END} Could Not Copy {out_measure}: {e}")

    return seg_root, len(avis)

#####################################################################
# Classification                                                    #
#####################################################################

def classify_one_avi(model, corrected_crop_dir, out_csv, gpu, verbose=False):
    patterns = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    for ext in patterns:
        if glob.glob(os.path.join(corrected_crop_dir, f"*.{ext}")):
            break
    else:
        raise FileNotFoundError(
            print(f"\t\t\t{RED}  Error:{C_END} No image files found in directory: {corrected_crop_dir}")
        )

    if verbose:
        print(f"\t\t\t{PURP}Running:{C_END} ", corrected_crop_dir)

    # prediction over a folder
    use_half = supports_half(gpu)
    results = model(
        corrected_crop_dir,
        verbose=False,
        stream=True,
        batch=32,
        device=f"cuda:{gpu}",
        imgsz=640,
        half=use_half,
    )
    class_names = list(model.names.values())
    wrote_header = False
    n = 0
    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        for res in results:
            n += 1
            fname = os.path.basename(res.path)
            probs = res.probs
            prob_list = [float(x) for x in probs.data.tolist()]
            if not wrote_header:
                writer.writerow(["image"] + class_names)
                wrote_header = True
            writer.writerow([fname] + prob_list)
            if verbose and n % 10000 == 0:
                print(f"\t\t\t{WHITE}Progress:{C_END} Classified {n} images\r")
    return n

def run_classification(model_path, seg_root, output_dir, gpu, verbose=False):
    class_root = os.path.join(output_dir, "classification")
    os.makedirs(class_root, exist_ok=True)
    model = YOLO(model_path)

    avi_dirs = [d for d in glob.glob(os.path.join(seg_root, "*")) if os.path.isdir(d)]
    if verbose:
        print(f"\t\t\t{WHITE}   Info:{C_END} Classifying {len(avi_dirs)} avi crop folders")

    total_imgs = 0
    for avi_dir in avi_dirs:
        avi_base = os.path.basename(avi_dir)
        corrected_crop = os.path.join(avi_dir, "segmentation", "corrected_crop")
        if not os.path.isdir(corrected_crop):
            if verbose:
                print(f"\t\t\t{YELGRN}Warning:{C_END} Skipping {avi_base}: no corrected_crop found")
            continue
        out_csv = os.path.join(avi_dir, f"{avi_base}_classification.csv")
        n = classify_one_avi(model, corrected_crop, out_csv, gpu, verbose=verbose)
        total_imgs += n
        if n > 0:
            #classi_csv.append(out_csv)
            # also copy index CSVs into classification root
            try:
                shutil.copy2(out_csv, os.path.join(class_root, os.path.basename(out_csv)))
            except Exception as e:
                print(f"\t\t\t{RED}  Error:{C_END} Could Not Copy {out_csv}: {e}")
    return class_root, total_imgs

#####################################
# create occurrence file            #
#####################################

def dec_to_micro(decimal_second):
    s = str(decimal_second)
    if len(s) == 3: 
        return 1000 * int(s)
    elif len(s) == 2: 
        return 10000 * int(s)
    elif len(s) == 1: 
        return 100000 * int(s)
    else:
        raise ValueError("Unhandled decimal second")

def new_date_code_parse(avi):
    m = re.search(r"(\d{2})-(\d{2})-(\d{2,4})-(\d{2})-(\d{2})-(\d{2})\.(\d{1,3})", avi)
    if not m: 
        raise ValueError(f"Bad avi: {avi}")

    month, day, year, h, m_, s, dec_s = m.groups()

    if len(year) == 2: 
        year = f"20{year}"
    microseconds = dec_to_micro(dec_s)
    return datetime(int(year), int(month), int(day), int(h), int(m_), int(s), int(microseconds))

def old_date_code_parse(avi):
    m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.(\d{1,3})", avi)
    if not m: 
        raise ValueError(f"Bad avi: {avi}")

    y, day, month, h, m_, s, dec_s = m.groups()
    microseconds = dec_to_micro(dec_s)
    return datetime(int(y), int(month), int(day), int(h), int(m_), int(s), int(microseconds))

def date_code_parse(avi):
    if "Cam" in avi or "Camera" in avi:
        return new_date_code_parse(avi)
    else:
        return old_date_code_parse(avi)

def create_measure_dict(avi, measure_dir):
    """convert measure.csv to a dictionary    """
    csv_path = os.path.join(measure_dir, avi + "_measure.csv")
    md = defaultdict(lambda: "error")
    fps = 19.88 if ("Cam" in avi or "Camera" in avi) else 18.0
    sec_p_frame = 1.0 / fps
    if not os.path.exists(csv_path): 
        sys.exit(f"ERROR: Missing measurement CSV: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline='') as mf:
        mlines = csv.reader(mf)
        try:
            #image,area,major,minor,perimeter,x,y,mean,height
            header = next(mlines)
        except StopIteration:
            return md, header
        video_start_date = date_code_parse(avi)
        for row in mlines:
            if not row or row[0] == "End of Measurement": 
                break
            image = row[0]
            #CamBottom-09-20-2024-05-34-07.274_0001_crop_0000.png
            m = re.search(r"_(\d{4})_crop", image) or re.search(r"_(\d{4})\.", image)
            if not m: 
                continue
            stack_number = int(m.group(1))
            sec_since_start = (stack_number - 1) * sec_p_frame
            frame_date = video_start_date + timedelta(seconds=sec_since_start)
            md[os.path.basename(image)] = [frame_date.isoformat()] + row
    return md, header

def build_occurrence(output_dir, verbose=False):
    measure_root = os.path.join(output_dir, "measurements")
    classi_root = os.path.join(output_dir, "classification")
    merge_root = os.path.join(output_dir, "merge")
    os.makedirs(merge_root, exist_ok=True)

    if os.path.exists(classi_root) and  os.path.exists(measure_root):
        classi_count = len(glob.glob(f"{classi_root}/*_classification.csv"))
        measure_count = len(glob.glob(f"{measure_root}/*_measure.csv"))
        avis_ex = sorted(
            [os.path.basename(f) for f in glob.glob(f"{classi_root}/*_classification.csv")],
            key=lambda f: date_code_parse(f)
        )
        avis_order = [ f.replace("_classification.csv", "") for f in avis_ex ]

    base_outdir = os.path.basename(output_dir)
    combined = os.path.join(output_dir, "merge", f"{base_outdir}_merged_occurrence.csv")
    header_written = False
    #header = "time,area,major,minor,perimeter,x,y,mean,height,predicted_taxon,probability"
    with open(combined, "w", newline='') as out:
        writer = csv.writer(out)
        for avi in avis_order:
            class_csv = os.path.join(classi_root, f"{avi}_classification.csv")
            measure_file = os.path.join(measure_root, f"{avi}_measure.csv")
            if not os.path.exists(class_csv) or not os.path.exists(measure_file):
                if verbose: 
                    print(f"\t\t\t{YELGRN}Warning:{C_END} Measurement File Missing For {avi}: {measure_file}")
                return None

            mdict, mheader = create_measure_dict(avi, measure_root)
            with open(class_csv, "r", newline='') as cf:
                cf_lines = csv.reader(cf)
                cf_header = next(cf_lines)  

                if not header_written:
                    #time,image,area,major,minor,perimeter,x,y,mean,height,predicted_taxon,probability
                    writer.writerow(["time"] + mheader + ["predicted_taxon", "probability"])
                    header_written = True

                for row in cf_lines:
                    if not row: 
                        continue
                    image_name = os.path.basename(row[0])
                    class_names = cf_header[1:]
                    probs = row[1:]
                    try:
                        prob_floats = [float(p) if p != "" else 0.0 for p in probs]
                    except Exception:
                        prob_floats = [0.0 for _ in probs]
                    if not prob_floats: 
                        continue
                    max_idx = max(range(len(prob_floats)), key=lambda i: prob_floats[i])
                    predicted_taxon = class_names[max_idx]
                    probability = prob_floats[max_idx]
                    measure_data = mdict[image_name]
                    measure_data[1] = os.path.basename(measure_data[1])
                    if measure_data == "error":
                        if verbose: 
                            print(f"\t\t\t{RED}  Error:{C_END} Missing Measurement For {image_name}")
                        continue

                    writer.writerow(measure_data + [predicted_taxon, f"{probability:.6f}"])
    
    return combined

#############################################
# Final merge (occurrence + environmental)  #
#############################################

def create_final(input_dir, output_dir, occurrence_file, environmental_file, max_time_gap=2, verbose=True):
    input_name = os.path.basename(os.path.normpath(input_dir))
    parts = input_dir.strip("/").split("/")
    cruises_set = {"MEZCAL", "MBON", "CONCORDE", "OSTRICH", "SPECTRA"}
    cruise_name = "UNKNOWN"
    for cruise in cruises_set:
        if cruise in parts:
            cruise_name = cruise
            break

    if len(parts) > 10:
        subset = [parts[i] for i in [4, 7, 8, 10]]
        prefix = '_'.join(subset)
    elif len(parts) > 8 :
        subset = [parts[i] for i in [4, 6, 7, 8]]
        prefix = '_'.join(subset)
    else:
        prefix = input_name
    final_output = os.path.join(output_dir, "merge", f"{prefix}_{cruise_name}_merge_final.csv")

    # lightweight two-row window over env file
    class _DG(DoubleGen): pass
    env_ptr = _DG(environmental_file, header_lines=0)
    with open(occurrence_file, 'r', newline='') as occ_ptr, open(final_output, 'w', newline='') as final_ptr:
        occ_csv_ptr = csv.DictReader(occ_ptr)
        env_header = env_ptr.fieldnames
        occ_header = occ_csv_ptr.fieldnames
        env_header_no_time = [h for h in env_header if h != 'Time']
        final_header = ["Cruise_name", "HD_ID", "Path"] + occ_header + env_header_no_time
        final_writer = csv.DictWriter(final_ptr, fieldnames=final_header)
        final_writer.writeheader()

        last_env_row, next_env_row = next(env_ptr)
        next_env_time = get_time_env_like(next_env_row)
        last_occ_row = None

        for line_num, occ_row in enumerate(occ_csv_ptr, start=1):
            occ_time = parse_iso_like(occ_row["time"])

            if last_occ_row is not None:
                last_occ_time = parse_iso_like(last_occ_row["time"])
                if last_occ_time > occ_time:
                    if verbose: 
                        print(f"\t\t\t{YELGRN}Warning:{C_END} Occurrence time decreased at line {line_num}")
                    env_ptr.restart()
                    last_env_row, next_env_row = next(env_ptr)
                    next_env_time = get_time_env_like(next_env_row)

            while occ_time > next_env_time:
                try:
                    last_env_row, next_env_row = next(env_ptr)
                except StopIteration:
                    if verbose: 
                        print("\t\t\t{YELGRN}Warning:{C_END} Environmental data exhausted")
                    break
                next_env_time = get_time_env_like(next_env_row)

            last_env_time = get_time_env_like(last_env_row)
            which = closest_time(occ_time, last_env_time, next_env_time, max_time_gap)
            if which == "skip":
                continue
            env_row = last_env_row if which == "last" else next_env_row

            final_row = {k: v for k, v in env_row.items() if k != 'Time'}
            final_row.update(occ_row)
            final_row["Cruise_name"] = cruise_name
            final_row["HD_ID"] = input_name
            final_row["Path"] = input_dir
            if "time" in final_row:
                if isinstance(final_row["time"], datetime):
                    final_row["time"] = final_row["time"].strftime("%Y-%m-%d %H:%M:%S.%f")
                else:
                    final_row["time"] = final_row["time"].replace("T", " ")
            final_writer.writerow(final_row)
            last_occ_row = occ_row

    if verbose:
        print(f"\t\t\t{WHITE}   Info:{C_END} Final merged file: {final_output}")
    return final_output

#####################################################################
# Menu                                                              #
#####################################################################
class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # Customize the error message format
        print(f"ERROR: {self.prog}: {message}", file=sys.stderr)
        # Optionally print the help message
        self.print_help(sys.stderr)
        # Exit with a custom status code (e.g., 2)
        sys.exit(2) 

def menu():
    parser = CustomArgumentParser(description="Full Video Analytics Pipeline: env merge -> segmentation -> classification -> occurrence file ->     final merge")
    parser.add_argument("-i","--input", type=directory, required=False, help="Input folder containing AVI files")
    parser.add_argument("-sb","--segment-bin", required=False, help="Path to segmentation binary")
    parser.add_argument("-ai","--ai-model", required=False, help="AI model (yolo,inceptionv3,megadetector,U-Net)")
    parser.add_argument("-mw","--weights", required=False, help="Model weights file (.weights,.pt)")
    parser.add_argument("-mo","--modelopt", help="Model Advanced Option (Extra option for the model)")
    parser.add_argument("-en","--environmental", type=directory, required=False, help="Path to environmental data directory (publisher subdirs inside)")
    parser.add_argument("-o","--output", required=False, help="Output directory (will contain segmentation/, classification/, merge/)")
    parser.add_argument("-g","--gpu", type=str, default="0", help="GPU ID (default: 0)")
    parser.add_argument("-c","--config", type=str, default="vap.conf", help="Config File To Use over Command Line Options (default: vap.conf)")

    #####################################################################
    # use subcommands                                           #
    #####################################################################
    subparsers = parser.add_subparsers(dest="command")
    segment_parser = subparsers.add_parser("segment")
    segment_parser.add_argument("-i","--input", type=directory, required=False, help="Input folder containing AVI files")
    segment_parser.add_argument("-sb","--segment-bin", required=False, help="Path to segmentation binary")
    segment_parser.add_argument("-o","--output", required=False, help="Output directory (will contain segmentation/, classification/, merge/)")
    segment_parser.add_argument("-g","--gpu", type=str, default="0", help="GPU ID (default: 0)")
    segment_parser.add_argument("-c","--config", type=str, default="vap.conf", help="Config File To Use over Command Line Options (default: vap.conf)")

    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument("-i","--input", type=directory, required=False, help="Input folder containing AVI files")
    classify_parser.add_argument("-mw","--weights", required=False, help="Model weights file (.weights,.pt)")
    classify_parser.add_argument("-mo","--modelopt", help="Model Advanced Option (Extra option for the model)")
    classify_parser.add_argument("-o","--output", required=False, help="Output directory (will contain segmentation/, classification/, merge/)")
    classify_parser.add_argument("-g","--gpu", type=str, default="0", help="GPU ID (default: 0)")
    classify_parser.add_argument("-c","--config", type=str, default="vap.conf", help="Config File To Use over Command Line Options (default: vap.conf)")

    #####################################################################
    # Segmentation parameters                                           #
    #####################################################################
    parser.add_argument("-d", "--delta", type=str, default="4", help="Segmentation parameter -d / --delta (Default: 4)")
    parser.add_argument("-m", "--min-area", type=str, default="50", help="Segmentation parameter -m / --minArea (Default: 50)")
    parser.add_argument("-M","--max-area", type=str, default="400000", help="Segmentation parameter -M / --maxArea (Default: 400000)")
    parser.add_argument("-T","--threshold", type=str, default="160", help="Segmentation parameter -T / --threshold (Default: 160)")
    parser.add_argument("-s","--signal-to-noise", type=str, default="60", help="Segmentation parameter -s / --signal-to-noise (Default: 60)")
    parser.add_argument("-p","--outlier-percent", type=str, default="0.15", help="Segmentation parameter -p / --outlier-percent (Default: 0.15)")
    parser.add_argument("-v","--variation", type=str, default="100", help="Segmentation parameter -v / --variation (Default: 100)")
    parser.add_argument("-e","--epsilon", type=str, default="1", help="Segmentation parameter -e / --epsilon (Default: 1)")
    parser.add_argument("-t","--top-crop", type=str, default="0", help="Segmentation parameter -t / --top-crop (Default: 0)")
    parser.add_argument("-b","--bottom-crop", type=str, default="0", help="Segmentation parameter -b / --bottom-crop (Default: 0)")
    parser.add_argument("-l","--left-crop", type=str, default="66", help="Segmentation parameter -l / --left-crop (Default: 66)")
    parser.add_argument("-r","--right-crop", type=str, default="23", help="Segmentation parameter -r / --right-crop (Default: 23)")

    parser.add_argument("-vv","--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument('-help', action="help", help="Help Message")

    global args
    args = parser.parse_args()

#####################################################################
# Main                                                              #
#####################################################################
def main():

    #####################################################################
    #                                                                   #
    #                         Start of Program                          #
    #                                                                   #
    #####################################################################
    menu()
    t0 = datetime.now()

    seg_flag_map = {
        "delta": "-d",
        "min_area": "-m",
        "max_area": "-M",
        "threshold": "-T",
        "signal_to_noise": "-s",
        "outlier_percent": "-p",
        "variation": "-v",
        "epsilon": "-e",
        "top_crop": "-t",
        "bottom_crop": "-b",
        "left_crop": "-l",
        "right_crop": "-r",
    }

    seg_kv_args = []
    for attr, short_flag in seg_flag_map.items():
        val = getattr(args, attr)
        seg_kv_args.extend([short_flag, str(val)])

    #####################################################################
    # Start the run                                                     #
    #####################################################################
    print("", file=sys.stdout, flush=True)
    print(f"\t{GREEN}VAP - Vision-based AI Pipeline", file=sys.stdout, flush=True)

    if args.verbose:
        print(f"\n\t{WHITE}List of Steps:{C_END}", file=sys.stdout, flush=True)
        print(f"\t\t{WHITE}0){C_END} Enviromental Data Merge", file=sys.stdout, flush=True)
        print(f"\t\t{WHITE}1){C_END} Data Segmentation", file=sys.stdout, flush=True)
        print(f"\t\t{WHITE}2){C_END} Data Classification", file=sys.stdout, flush=True)
        print(f"\t\t{WHITE}3){C_END} Occurrence Creation", file=sys.stdout, flush=True)
        print(f"\t\t{WHITE}4){C_END} Final Merge of Data", file=sys.stdout, flush=True)
        print("", file=sys.stdout, flush=True)

    print(f"\t{GREEN}Starting pipeline at {t0:%Y-%m-%d %H:%M:%S}{C_END}", file=sys.stdout, flush=True)


    if args.command is None:
        #####################################################################
        # 0) Environmental merge                                            #
        #####################################################################
        if args.verbose:
            print(f"\t\t{WHITE}[0/4]{C_END} Environmental data merge")
        env_merged_path = os.path.join(args.environmental, "merged_environmental.csv")
        if file_empty(env_merged_path):
            if args.verbose:
                print(f"\t\t\t   {WHITE}Info:{C_END} No merged environmental file found; building merged_environmental.csv ...")
            env_out_file = merge_environmental(args.environmental, output_dir=args.environmental, header_lines=0, verbose=args.verbose)
            #env_out_file = ""
        else:
            if args.verbose:
                print(f"\t\t\t   {WHITE}Info:{C_END} Found existing merged_environmental.csv — skipping environmental merge.")
            environmental_dir = os.path.abspath(args.environmental)
            env_out_file = os.path.join(environmental_dir, "merged_environmental.csv")



        #####################################################################
        # 1) Segmentation                                                   #
        #####################################################################
        if args.verbose:
            print(f"\t\t{WHITE}[1/4]{C_END} Segmentation")
        seg_root, n_avi = run_segmentation(args.segment_bin, args.input, args.output, seg_kv_args, args.verbose)




        #####################################################################
        # 2) Classification                                                 #
        #####################################################################
        if args.verbose:
            print(f"\t\t{WHITE}[2/4]{C_END} Classification")
        class_root, n_imgs = run_classification(args.weights, seg_root, args.output, args.gpu, args.verbose)



        #####################################################################
        # 3) occurrence creation                                            #
        #####################################################################
        if args.verbose:
            print(f"\t\t{WHITE}[3/4]{C_END} Occurrence Creation")
            print(f"\t\t\t   {WHITE}Info:{C_END} Merging classification with measurement")

        output_dir = args.output
        measure_root = os.path.join(output_dir, "measurements")
        classi_root = os.path.join(output_dir, "classification")

        if os.path.exists(classi_root) and os.path.exists(measure_root):
            print(f"\t\t\t   {WHITE}Info:{C_END} The measure_root or classi_root does exist")
            combined_occ = build_occurrence(output_dir, verbose=args.verbose)
        else:
            print(f"\t\t\t{RED}  Error:{C_END} The measure_root or classi_root does not exist")
            combined_occ = None
            #combined_occ = ""



        #####################################################################
        # 4) Final merge with environmental                                 #
        #####################################################################
        if args.verbose:
            print(f"\t\t{WHITE}[4/4]{C_END} Final Merge of Data")

        if os.path.exists(combined_occ) and os.path.exists(env_out_file):
            print(f"\t\t\t   {WHITE}Info:{C_END} There are occurrence files or environment files ")
            final_csv = create_final(args.input, args.output, combined_occ, env_out_file, max_time_gap=2, verbose=args.verbose)
        else:
            print(f"\t\t\t{RED}  Error:{C_END} There are missing occurrence files or environment files ")
            final_csv = None

    elif args.command == "segment":
        if args.verbose:
            print(f"\t\t{WHITE}[1/4]{C_END} Segmentation")
        seg_root, n_avi = run_segmentation(args.segment_bin, args.input, args.output, seg_kv_args, args.verbose)

    elif args.command == "classify":
        if args.verbose:
            print(f"\t\t{WHITE}[2/4]{C_END} Classification")
        seg_root = args.input
        class_root, n_imgs = run_classification(args.weights, seg_root, args.output, args.gpu, args.verbose)


    #####################################################################
    # End of the run                                                    #
    #####################################################################
    # Summary
    print("", file=sys.stdout, flush=True)
    print(f"\t{GREEN}Pipeline complete!{C_END}")
    
    if args.command is None:
        print(f"\t\t{WHITE}Environmental Merge:{C_END} {env_out_file}")
        print(f"\t\t{WHITE}Segmented files:{C_END} {n_avi}")
        print(f"\t\t{WHITE}Classified Images:{C_END} {n_imgs}")
        print(f"\t\t{WHITE}Combined occurrence:{C_END} {combined_occ}")
        print(f"\t\t{WHITE}Final merged file:{C_END} {final_csv}")

    elif args.command == "segment":
        print(f"\t\t{WHITE}Segmented files:{C_END} {n_avi}")

    elif args.command == "classify":
        print(f"\t\t{WHITE}Classified Images:{C_END} {n_imgs}")

    print(f"\t    {WHITE}   Output directory:{C_END} {args.output}")
    print(f"\t    {WHITE}      Total runtime:{C_END} {datetime.now() - t0}")
    print("", file=sys.stdout, flush=True)

if __name__ == "__main__":
    main()