#!/usr/bin/env python3

""" Sets up a training area for OD training based on video or images assuming
    the image or video is a thumbnail of the image to classify. 

    Output may be useful for appearance-based comparators for tracking.
"""

import argparse
import tator
import os
import pandas as pd
import progressbar
import shutil
import subprocess
import tempfile

from collections import defaultdict
from pprint import pprint

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                     "Sets up a training area from tator project")
    parser = tator.get_parser(parser)
    parser.add_argument("--section",
                        nargs='+',
                        help='sections to download (pk)')
    parser.add_argument("--padding",
                        type=int,
                        help="Amount of padding to add to bounding box (pixels)",
                        default=1)
    parser.add_argument("--keyname",
                        default="Species",
                        help="Key name to use for the species value in the box")
    parser.add_argument("--discard",
                        nargs="*",
                        help="Discard classes of this name")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    api = tator.get_api(args.host, args.token)
    section = api.get_section(args.section[0])
    project = section.project

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating Species information")
    species_output=os.path.join(args.output_dir, "species.csv")
    # First get the species for the project by getting the set of the
    # species names
    species = defaultdict(lambda : 0)
    global_media_list = []
    for section in args.section:
        media_list = api.get_media_list(project,section=section)
        global_media_list += media_list
        for media in media_list:
            species_name = media.attributes[args.keyname]
            species[species_name] += 1
            
    print("Species Information:")
    pprint(species)
    print("Discarding classes:")
    pprint(args.discard)
            
    names=list(species.keys())
    names.sort()
    name_idx_pairs=[]
    for idx,name in enumerate(names):
        name_idx_pairs.append((name,idx))
        species_df = pd.DataFrame(data=name_idx_pairs,
                                  columns=['Species','Num'])
        species_df.to_csv(species_output, index=False, header=False)


    # generate annotations.csv
    annotations_output=os.path.join(args.output_dir, "totalPopulation.csv")
    # output is img, x1,y1,x2,y2,species-id-0
    images_dir=os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    cols=['img', 'x1', 'y1', 'x2', 'y2', 'species_name']
    local_df=pd.DataFrame(columns=cols)
    local_df.to_csv(annotations_output, header=False,index=False)

    if args.section:
        sections_to_process=args.section
    else:
        sections_to_process=[x.id for x in api.get_section_list(project)]

    print(f"Processing {sections_to_process}")

    bar = progressbar.ProgressBar(redirect_stdout=True)
    for media in bar(global_media_list):
        with tempfile.TemporaryDirectory() as td:
            temp_path = os.path.join(td, media.name)
            frames_path = os.path.join(td,"frames")
            media_path = os.path.join(frames_path,f"{media.id}")
            os.makedirs(media_path)
            species_name = media.attributes.get(args.keyname,None)
            if species_name is None or species_name in args.discard:
                print(f"Skipping {species_name} in media.name")
                continue
            for _ in tator.download_media(api, media, temp_path):
                pass
            ffmpeg_cmd = ["ffmpeg",
                          "-i", temp_path,
                          f"{media_path}/%05d.png"]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            for root, media_dirs, frame_images in os.walk(frames_path):
                for media_dir in media_dirs:
                    shutil.copytree(os.path.join(root,media_dir),
                                    os.path.join(images_dir, media_dir))
                for frame_image in frame_images:
                    rel_image_path = os.path.join("images",
                                                  os.path.basename(root),
                                                  frame_image)
                    datum={'img': rel_image_path,
                           'x1': args.padding,
                           'y1': args.padding,
                           'x2': media.width-args.padding,
                           'y2': media.height-args.padding,
                           'species_name': species_name}
                    datum_df = pd.DataFrame(data=[datum],
                                            columns=cols)
                    datum_df.to_csv(annotations_output,
                                    mode='a',
                                    header=False,
                                    index=False)
