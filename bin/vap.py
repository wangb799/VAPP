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
import itertools as _it
import textwrap
from datetime import datetime, timedelta
from collections import defaultdict

#####################################################################
# Third-party libraries                                             #
#####################################################################
from ultralytics import YOLO
import torch

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
# Environmental data merge                                          #
#####################################################################

iso_pattern = re.compile(r'\d{4}-\d{2}-\d{2}[T ]?\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?')

def isoformat_parse(date_time: str) -> datetime:
    """
    Parse a timestamp string in several ISO-ish formats, with or without microseconds.
    """
    if date_time is None:
        raise ValueError("Unparsable time: None")

    date_time = date_time.strip()
    if date_time.endswith("Z"):
        date_time = date_time[:-1]

    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_time, fmt)
        except ValueError:
            continue  

    try:
        return datetime.fromisoformat(date_time)
    except Exception:
        raise ValueError(f"Unparsable time: {date_time!r}")


def contains_header(ptr):
    """Detect CSV header by checking first token starts with 'Time'."""
    pos = ptr.tell()
    line = ptr.readline()
    ptr.seek(pos)
    return line.strip().lower().startswith("time")

def combine_pub_csv(dir_path):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    if len(files) <= 0:
        raise FileNotFoundError(f"No files found in directory {dir_path}")
    return files

def _convert_nmea_lat(lat_raw):
    if lat_raw is None:
        return None
    s = str(lat_raw).strip()
    if not s:
        return None

    try:
        val = float(s)
    except ValueError:
        return None

    # If already a decimal degree in [-90, 90], just return it
    if -90.0 <= val <= 90.0:
        return val

    # Assume NMEA ddmm.mmmm (e.g., 4533.12 → 45° + 33.12')
    parts = s.split(".")
    int_part = parts[0]
    if len(int_part) < 4:
        return None

    deg = float(int_part[:-2])
    minutes = float(int_part[-2:] + ("." + parts[1] if len(parts) > 1 else ""))
    return deg + minutes / 60.0


def _convert_nmea_lon(lon_raw):
    if lon_raw is None:
        return None
    s = str(lon_raw).strip()
    if not s:
        return None

    try:
        val = float(s)
    except ValueError:
        return None

    # If already a decimal degree in [-180, 180], just return it
    if -180.0 <= val <= 180.0:
        return val

    # Assume NMEA dddmm.mmmm (e.g., 12345.67 → 123° + 45.67')
    parts = s.split(".")
    int_part = parts[0]
    if len(int_part) < 5:
        return None

    deg = float(int_part[:-2])
    minutes = float(int_part[-2:] + ("." + parts[1] if len(parts) > 1 else ""))
    return deg + minutes / 60.0


def pairwise_csv_rows(directory, fieldnames=None, delimiter=',', header_lines=1):
    """
    Iterate through all CSVs in the directory and yield (last_row, next_row) pairs,
    preserving boundaries between files.
    """
    files = combine_pub_csv(directory)
    prev_last = None

    for f in files:
        with open(f, 'r', newline='') as fp:
            # Skip header lines
            for _ in range(header_lines):
                next(fp, None)

            reader = csv.DictReader(fp, fieldnames=fieldnames, delimiter=delimiter)

            try:
                last_row = next(reader)
            except StopIteration:
                continue

            if prev_last is not None:
                # Bridge last row of previous file with first row of this file
                yield prev_last, last_row

            for next_row in reader:
                yield last_row, next_row
                last_row = next_row

            prev_last = last_row


def format_timestamp(line):
    """Parse environmental 'Time' column (ISO-ish or epoch 1904 seconds)."""
    if "Time" not in line:
        raise KeyError("Expected 'Time' column in environmental data")

    val = (line["Time"] or "").strip()
    if not val:
        raise KeyError("Expected 'Time' column in environmental data")

    val = " ".join(val.split())

    # 1. ISO-like timestamps    
    if iso_pattern.search(val):
        try:
            dt = isoformat_parse(val)
            if val.endswith("Z"):
                dt = dt - timedelta(hours=7)
            return dt
        except Exception:
            pass  # allow numeric parsing below

    # 2. Numeric seconds (epoch 1904)
    try:
        seconds = float(val)
        start = datetime(1904, 1, 1)
        delta = timedelta(hours=-7, seconds=seconds)
        return start + delta
    except Exception:
        pass  # allow final fallback

    # 3. Final fallback ISO parse
    return isoformat_parse(val)

def closest_time(occ_time, last_pub_time, next_pub_time, max_time_gap=10):
    """
    Choose the closest publisher timestamp to occ_time.
    """
    max_delta = timedelta(seconds=max_time_gap)

    # Compute distances to publisher times (None means that side doesn't exist)
    if last_pub_time is not None:
        last_dist = abs(occ_time - last_pub_time)
    else:
        last_dist = None

    if next_pub_time is not None:
        next_dist = abs(next_pub_time - occ_time)
    else:
        next_dist = None

    if last_dist is not None and next_dist is not None:
        if last_dist > max_delta and next_dist > max_delta:
            return "skip"

    if last_dist is None:
        return "next"

    if next_dist is None:
        return "last"

    if last_dist <= next_dist:
        return "last"
    else:
        return "next"

def check_env_header(directory, expected_header, delimiter=","):
    env_files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f))]
    env_files.sort()
    if not env_files:
        raise FileNotFoundError(f"No files found in directory: {directory}")

    header_file = env_files[0]
    with open(header_file, "r", newline='') as ptr:
        found_header = next(ptr).rstrip("\n").split(delimiter)

        # Normalize headers (strip whitespace, carriage returns, etc.)
        found_header = [h.strip().rstrip('\r') for h in found_header]
        expected_header = [h.strip().rstrip('\r') for h in expected_header]

        if found_header and "Time" in found_header[0]:
            found_header[0] = "Time"

        if found_header != expected_header:
            raise ValueError(
                f"Header mismatch in directory: {directory}\n"
                f"Expected: {expected_header}\n"
                f"Found:    {found_header}"
            )


def publisher_iterator(d, header):
    files = combine_pub_csv(d)
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

def ensure_parsed_time(row):
    """Store parsed datetime in row['_parsed_time'] once."""
    if "_parsed_time" not in row:
        raw = row.get("Time")
        try:
            row["_parsed_time"] = format_timestamp(row)
        except Exception as e:
            print(f"\t\t\t{RED}  DEBUG:{C_END}Failed to parse Time value:", repr(raw))
            print("Error:", e)
            row["_parsed_time"] = None
    return row

def run_merge_environmental(environmental_dir, output_dir=None, verbose=True):
    """
    Merge environmental publishers by aligning other publishers to the Inclinometer timeline.
    Writes merged_environmental.csv in environmental_dir.
    """
    environmental_dir = os.path.abspath(environmental_dir)
    env_out_file = os.path.join(environmental_dir, "merged_environmental.csv")
    if output_dir:
        env_out_file = os.path.join(output_dir, "merged_environmental.csv")

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
        print("[0/4] Environmental headers verified. Merging publishers")

    # Build output fieldnames: Time + all non-Time non-Checksum from others + Latitude/Longitude last
    merged_fields = ["Time"] + [
        col for publisher, header in other_publishers for col in header
        if (publisher != "GPS Publisher" and "Checksum" not in col and col != "Time")
    ] + ["Latitude", "Longitude"]

    with open(env_out_file, 'w', newline='') as out_ptr:
        env_writer = csv.DictWriter(out_ptr, merged_fields, extrasaction='ignore')
        env_writer.writeheader()

        # Inclinometer provides the timeline
        inc_iter = publisher_iterator(os.path.join(environmental_dir, base_publisher[0]), base_publisher[1])
        try:
            first_inc_row = next(inc_iter)
        except StopIteration:
            print("Error: No inclinometer data found.", file=sys.stderr)
            return env_out_file

        ensure_parsed_time(first_inc_row)
        first_inc_time = first_inc_row["_parsed_time"]
        
        # Initialize other publisher iterators
        pub_readers = []
        for pub_dir, pub_header in other_publishers:
            d = os.path.join(environmental_dir, pub_dir)
            if not os.path.isdir(d):
                d2 = os.path.join(environmental_dir, pub_dir.rstrip('/'))
                d = d2 if os.path.isdir(d2) else d
            pub_readers.append(pairwise_csv_rows(d, pub_header))

        last_rows = []
        next_rows = []

        for j, dg in enumerate(pub_readers):
            try:
                l, n = next(dg)
                ensure_parsed_time(l)
                ensure_parsed_time(n)
            except StopIteration:
                print(f"StopIteration Error: {other_publishers[j][0]} empty.", file=sys.stderr)
                return env_out_file

            # Align publisher to the inclinometer timeline
            while True:
                n_time = n.get("_parsed_time")
                if n_time is None:
                    try:
                        l, n = next(dg)
                        ensure_parsed_time(l)
                        ensure_parsed_time(n)
                        continue
                    except StopIteration:
                        break

                if n_time >= first_inc_time:
                    break

                try:
                    l, n = next(dg)
                    ensure_parsed_time(l)
                    ensure_parsed_time(n)
                except StopIteration:
                    break

            last_rows.append(l)
            next_rows.append(n)


        for line_num, inc_row in enumerate(_it.chain([first_inc_row], inc_iter), start=1):
            ensure_parsed_time(inc_row)
            inc_time = inc_row["_parsed_time"]

            merged_row = dict(inc_row) 

            for i, dg in enumerate(pub_readers):
                ensure_parsed_time(last_rows[i])
                ensure_parsed_time(next_rows[i])

                while True:
                    next_time = next_rows[i]["_parsed_time"]

                    if next_time is None or inc_time <= next_time:
                        break

                    try:
                        last_rows[i], next_rows[i] = next(dg)
                        ensure_parsed_time(last_rows[i])
                        ensure_parsed_time(next_rows[i])
                    except StopIteration:
                        break
                pub_row = last_rows[i]

                #which = closest_time(inc_time, last_time, next_time, max_time_gap=10)
                #if which == "last":
                #    pub_row = last_rows[i]
                #elif which == "next":
                #    pub_row = next_rows[i]
                #else: 
                #    continue

                merged_row.update(pub_row)
            merged_row["Time"] = inc_time

            # Normalize Lat/Lon if present
            if "Longitude" in merged_row and "Latitude" in merged_row:
                lat_raw = merged_row.get("Latitude")
                lon_raw = merged_row.get("Longitude")

                # Hemisphere fields exist in the GGA-style GPS header
                lat_hemi = merged_row.get("Latitude Hemisphere")
                lon_hemi = merged_row.get("Longitude Hemisphere")

                # 1) Convert numeric / NMEA ddmm.mmmm to decimal degrees
                lat = _convert_nmea_lat(lat_raw)
                lon = _convert_nmea_lon(lon_raw)

                # 2) Apply hemisphere sign if hemisphere columns exist
                if lat is not None:
                    if lat_hemi == "S":
                        lat = -abs(lat)
                    merged_row["Latitude"] = lat

                if lon is not None:
                    if lon_hemi == "W":
                        lon = -abs(lon)
                    merged_row["Longitude"] = lon

            env_writer.writerow(merged_row)

    if verbose:
        print(f"\t\t{WHITE}   Info:{C_END} Created environmental merge: {env_out_file}")
    return env_out_file

#####################################################################
# Segmentation                                                      #
#####################################################################

def run_segmentation(segment_bin, input_dir, output_dir, seg_kv_args=None, verbose=False):
    seg_root = os.path.join(output_dir, "segmentation")
    measure_root = os.path.join(output_dir, "measurements")
    os.makedirs(seg_root, exist_ok=True)
    os.makedirs(measure_root, exist_ok=True)

    video_exts = ("avi", "AVI", "mp4", "MP4", "mpg", "MPG", "mpeg", "MPEG")
    avis = []
    for ext in video_exts:
        avis.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))


    if not avis:
        raise FileNotFoundError(f"\t\t\t{RED}  Error:{C_END} No .avi files found in {input_dir}")

    if verbose:
        print(f"\t\t{WHITE}   Info:{C_END} Segmenting {len(avis)} avi videos files")

    for avi in avis:
        avi_base = os.path.splitext(os.path.basename(avi))[0]
        avi_out = seg_root

        # Build segmentation command dynamically
        cmd = [segment_bin, "-i", avi, "-o", avi_out]
        if seg_kv_args:
            cmd.extend(seg_kv_args)

        if verbose:
            print(f"\t\t\t{PURP}Running:{C_END} ", avi)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Segmentation failed for {avi}: {e}")
            continue

        # also copy  measurement csv per avi into measurements root
        try:
            out_measure = os.path.join(avi_out, avi_base, "measurements", f"{avi_base}.csv")
            shutil.copy2(out_measure, os.path.join(measure_root, f"{avi_base}_measure.csv"))
        except Exception as e:
            print(f"\t\t{RED}  Error:{C_END} could not copy {out_measure}: {e}")

    return seg_root, len(avis)

#####################################################################
# Classification                                                    #
#####################################################################

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

def classify_one_avi(model, corrected_crop_dir, out_csv, gpu, verbose=False):
    patterns = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    for ext in patterns:
        if glob.glob(os.path.join(corrected_crop_dir, f"*.{ext}")):
            break
    else:
        msg = f"No image files found in directory: {corrected_crop_dir}"
        print(f"\t\t\t{RED}  Error:{C_END} {msg}")
        raise FileNotFoundError(msg)

    if verbose:	
        print(f"\t\t\t{PURP}Running:{C_END} ", corrected_crop_dir)

    # prediction over a folder
    use_half = supports_half(gpu)
    class_names = list(model.names.values())
    wrote_header = False
    n = 0

    with torch.inference_mode():
        results = model.predict(
            corrected_crop_dir,
            imgsz=640,
            batch=32,
            stream=True,
            half=use_half,
            device=gpu,
            verbose=False
        )
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
                print(f"\t\t{RED}  Error:{C_END} could not copy {out_csv}: {e}")
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

NEW_CAMERA_PATTERN = re.compile(r"(\d{2})-(\d{2})-(\d{2,4})-(\d{2})-(\d{2})-(\d{2})\.(\d{1,3})")

OLD_CAMERA_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.(\d{1,3})")

def avi_date_parse(avi):
    """
    Parse both new and old camera timestamp formats from .avi names.
    New: MM-DD-YYYY-HH-MM-SS.dec
    Old: YYYYDDMMHHMMSS.dec
    """

    # New camera format
    #m = re.search(r"(\d{2})-(\d{2})-(\d{2,4})-(\d{2})-(\d{2})-(\d{2})\.(\d{1,3})", avi)
    m = NEW_CAMERA_PATTERN.search(avi)
    if m:
        month, day, year, h, m_, s, dec_s = m.groups()
        if len(year) == 2:
            year = "20" + year
        return datetime(
            int(year), int(month), int(day),
            int(h), int(m_), int(s),
            int(dec_to_micro(dec_s))
        )

    # Old camera format
    #m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.(\d{1,3})", avi)
    m = OLD_CAMERA_PATTERN.search(avi)
    if m:
        y, day, month, h, m_, s, dec_s = m.groups()
        return datetime(
            int(y), int(month), int(day),
            int(h), int(m_), int(s),
            int(dec_to_micro(dec_s))
        )

    raise ValueError(f"Unrecognized timestamp format: {avi}")

def create_measure_dict(avi, measure_dir):
    """convert measure.csv to a dictionary    """
    csv_path = os.path.join(measure_dir, avi + "_measure.csv")
    md = {}
    fps = 19.88 if ("Cam" in avi or "Camera" in avi) else 18.0
    sec_p_frame = 1.0 / fps
    if not os.path.exists(csv_path): 
        raise FileNotFoundError(f"Missing measurement CSV: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline='') as mf:
        mlines = csv.reader(mf)
        try:
            #image,area,major,minor,perimeter,x,y,mean,height
            header = next(mlines)
        except StopIteration:
            return md,  []

        video_start_date = avi_date_parse(avi)
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

def run_build_occurrence(output_dir, verbose=False):
    measure_root = os.path.join(output_dir, "measurements")
    classi_root = os.path.join(output_dir, "classification")
    merge_root = os.path.join(output_dir, "merge")
    os.makedirs(merge_root, exist_ok=True)

    if os.path.exists(classi_root) and  os.path.exists(measure_root):
        avis_ex = sorted(
            [os.path.basename(f) for f in glob.glob(f"{classi_root}/*_classification.csv")],
            key=lambda f: avi_date_parse(f)
        )
        avis_order = [ f.replace("_classification.csv", "") for f in avis_ex ]

    base_outdir = os.path.basename(output_dir)
    all_occurrences = os.path.join(output_dir, "merge", f"{base_outdir}_merged_occurrence.csv")
    header_written = False
    #header = "time,area,major,minor,perimeter,x,y,mean,height,predicted_taxon,probability"
    with open(all_occurrences, "w", newline='') as out:
        writer = csv.writer(out)
        for avi in avis_order:
            class_csv = os.path.join(classi_root, f"{avi}_classification.csv")
            measure_file = os.path.join(measure_root, f"{avi}_measure.csv")
            if not os.path.exists(class_csv) or not os.path.exists(measure_file):
                if verbose: 
                    print(f"\t\t\t{YELGRN}Warning:{C_END} Measurement File Missing For {avi}: {measure_file}")
                continue

            mdict, mheader = create_measure_dict(avi, measure_root)
            with open(class_csv, "r", newline='') as cf:
                cf_lines = csv.reader(cf)
                cf_header = next(cf_lines)  
                class_names = cf_header[1:]

                if not header_written:
                    #time,image,area,major,minor,perimeter,x,y,mean,height,predicted_taxon,probability
                    writer.writerow(["time"] + mheader + ["predicted_taxon", "probability"])
                    header_written = True

                for row in cf_lines:
                    if not row: 
                        continue
                    image_name = os.path.basename(row[0])
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
                    measure_data = mdict.get(image_name)
                    if measure_data is None:
                        if verbose: 
                            print(f"\t\t\t{RED}  Error:{C_END} Missing Measurement For {image_name}")
                        continue
                    m2 = measure_data.copy()
                    m2[1] = os.path.basename(m2[1])
                    writer.writerow(m2 + [predicted_taxon, f"{probability:.6f}"])
    
    return all_occurrences

#############################################
# Final merge (occurrence + environmental)  #
#############################################

def pairwise_csv_file(filepath, fieldnames=None, delimiter=',', header_lines=0):
    """
    Yield (last_row, next_row) pairs from a single CSV file.
    """
    with open(filepath, 'r', newline='') as fp:
        # skip header lines
        for _ in range(header_lines):
            next(fp, None)

        reader = csv.DictReader(fp, fieldnames=fieldnames, delimiter=delimiter)

        try:
            last_row = next(reader)
        except StopIteration:
            return  # empty file

        for next_row in reader:
            yield last_row, next_row
            last_row = next_row

def run_create_final(input_dir, output_dir, occurrence_file, environmental_file, max_time_gap=2, verbose=True):
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

    # read environmental header
    with open(environmental_file, 'r') as tmp_fp:
        for row in csv.reader(tmp_fp):
            if row:    
                env_header = row
                break

    # slide window over env file
    env_ptr = pairwise_csv_file(environmental_file)

    with open(occurrence_file, 'r', newline='') as occ_ptr, open(final_output, 'w', newline='') as final_ptr:
        occ_csv_ptr = csv.DictReader(occ_ptr)
        occ_header = occ_csv_ptr.fieldnames
        env_header_no_time = [h for h in env_header if h != 'Time']
        final_header = ["Cruise_name", "HD_ID", "Path"] + occ_header + env_header_no_time
        final_writer = csv.DictWriter(final_ptr, fieldnames=final_header)
        final_writer.writeheader()

        last_env_row, next_env_row = next(env_ptr)
        ensure_parsed_time(last_env_row)
        ensure_parsed_time(next_env_row)

        next_env_time = next_env_row["_parsed_time"]
        last_env_time = last_env_row["_parsed_time"] 
        last_occ_row = None

        for line_num, occ_row in enumerate(occ_csv_ptr, start=1):
            occ_time = isoformat_parse(occ_row["time"])

            if last_occ_row is not None:
                last_occ_time = isoformat_parse(last_occ_row["time"])

                if last_occ_time > occ_time:
                    raise RuntimeError(
                        f"Occurrence time decreased at line {line_num}. "
                        "This should not happen if Occurrence timestamp is in an order."
                    )
                
                """    
                if last_occ_time > occ_time:
                    if verbose: 
                        print(f"\t\t\t{YELGRN}Warning:{C_END} Occurrence time decreased at line {line_num}")
                    env_ptr = pairwise_csv_file(environmental_file)
                    last_env_row, next_env_row = next(env_ptr)
                    ensure_parsed_time(last_env_row)
                    ensure_parsed_time(next_env_row)
                    next_env_time = next_env_row["_parsed_time"]
                """

            while True:
                try:
                    ensure_parsed_time(next_env_row)
                    next_env_time = next_env_row["_parsed_time"]
                except Exception:
                    # next_env_row has no valid time → stop advancing
                    ensure_parsed_time(last_env_row)
                    last_env_time = last_env_row["_parsed_time"]
                    next_env_time = last_env_time
                    break

                # If environmental timestamp has caught up
                if next_env_time is None:
                    raise ValueError(
                        f"Invalid environmental timestamp.\n"
                        f"Row: {next_env_row}\n"
                        f"Raw Time: {repr(next_env_row.get('Time'))}"
                    )

                if last_env_time is None:
                    raise ValueError(
                        f"Invalid environmental timestamp (last row).\n"
                        f"Row: {last_env_row}\n"
                        f"Raw Time: {repr(last_env_row.get('Time'))}"
                    )

                if occ_time <= next_env_time:
                    break

                # Otherwise keep advancing the environmental pointer
                try:
                    last_env_row, next_env_row = next(env_ptr)
                    ensure_parsed_time(last_env_row)
                    ensure_parsed_time(next_env_row)
                except StopIteration:
                    if verbose:
                        print("Warning: environmental data exhausted")
                    break

            ensure_parsed_time(last_env_row)
            last_env_time = last_env_row["_parsed_time"]

            which = closest_time(occ_time, last_env_time, next_env_time, max_time_gap)
            if which == "skip":
                continue
            env_row = last_env_row if which == "last" else next_env_row

            final_row = {k: v for k, v in env_row.items() if k != 'Time'}
            final_row.update(occ_row)
            final_row["Cruise_name"] = cruise_name
            final_row["HD_ID"] = input_name
            final_row["Path"] = input_dir
            if "time" in final_row and isinstance(final_row["time"], datetime):
                # produce ISO-8601
                final_row["time"] = final_row["time"].isoformat()
            
            final_row.pop("_parsed_time", None)
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
    parser.add_argument("-i","--input", required=False, help="Input folder containing AVI or segmentation ")
    parser.add_argument("-sb","--segment-bin", required=False, help="Path to segmentation binary")
    parser.add_argument("-ai","--ai-model", required=False, help="AI model (yolo,inceptionv3,megadetector,U-Net)")
    parser.add_argument("-mw","--weights", required=False, help="Model weights file (.weights,.pt)")
    parser.add_argument("-mo","--modelopt", help="Model Advanced Option (Extra option for the model)")
    parser.add_argument("-en","--environmental", required=False, help="Path to environmental data directory (publisher subdirs inside)")
    parser.add_argument("-o","--output", required=False, help="Output directory (will contain segmentation/, classification/, merge/)")
    parser.add_argument("-g","--gpu", type=str, default="0", help="GPU ID (default: 0)")
    parser.add_argument("-c","--config", type=str, default="vap.conf", help="Config File To Use over Command Line Options (default: vap.conf)")

    #####################################################################
    # use subcommands                                           #
    #####################################################################
    subparsers = parser.add_subparsers(dest="command")
    segment_parser = subparsers.add_parser("segment")
    segment_parser.add_argument("-i","--input", required=False, help="Input folder containing AVI files")
    segment_parser.add_argument("-sb","--segment-bin", required=False, help="Path to segmentation binary")
    segment_parser.add_argument("-o","--output", required=False, help="Output directory (will contain segmentation/, classification/, merge/)")
    segment_parser.add_argument("-g","--gpu", type=str, default="0", help="GPU ID (default: 0)")
    segment_parser.add_argument("-c","--config", type=str, default="vap.conf", help="Config File To Use over Command Line Options (default: vap.conf)")

    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument("-i","--input", required=False, help="Input folder containing segmentatin folder of avi")
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
        required = [args.input, args.output, args.environmental, args.segment_bin, args.weights]
        if any(x is None for x in required):
            parser.error(f"\t\t{RED} Error:{C_END} Full pipeline requires --inpu --output --enviromental --segment_bin --weights")
        #####################################################################
        # 0) Environmental merge                                            #
        #####################################################################
        if args.verbose:
            print(f"\t\t{WHITE}[0/4]{C_END} Environmental data merge")
        env_merged_path = os.path.join(args.environmental, "merged_environmental.csv")
        if (not os.path.exists(env_merged_path)) or os.path.getsize(env_merged_path) == 0:
            if args.verbose:
                print(f"\t\t\t   {WHITE}Info:{C_END} No merged environmental file found; building merged_environmental.csv ...")
            env_out_file = run_merge_environmental(args.environmental, output_dir=args.environmental, verbose=args.verbose)
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
            combined_occ = run_build_occurrence(output_dir, verbose=args.verbose)
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
            final_csv = run_create_final(args.input, args.output, combined_occ, env_out_file, max_time_gap=2, verbose=args.verbose)
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
        print(f"\t    {WHITE}Environmental Merge:{C_END} {env_out_file}")
        print(f"\t        {WHITE}Segmented files:{C_END} {n_avi}")
        print(f"\t      {WHITE}Classified Images:{C_END} {n_imgs}")
        print(f"\t    {WHITE}Combined occurrence:{C_END} {combined_occ}")
        print(f"\t      {WHITE}Final merged file:{C_END} {final_csv}")

    elif args.command == "segment":
        print(f"\t\t{WHITE}Segmented files:{C_END} {n_avi}")

    elif args.command == "classify":
        print(f"\t\t{WHITE}Classified Images:{C_END} {n_imgs}")

    print(f"\t    {WHITE}   Output directory:{C_END} {args.output}")
    print(f"\t    {WHITE}      Total runtime:{C_END} {datetime.now() - t0}")
    print("", file=sys.stdout, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{RED}Fatal Error:{C_END} {e}")
        sys.exit(1)
