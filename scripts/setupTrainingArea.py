#!/usr/bin/env python3

""" Sets up a training area for OD training """

import argparse
import tator
import os
import pandas as pd
import progressbar
import shutil

from pprint import pprint

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                     "Sets up a training area from tator project")
    parser = tator.get_parser(parser)
    parser.add_argument("--section", nargs='*', help='sections to download (pk)')
    parser.add_argument("--box-type-id", type=int, help='pk of localization type to extract')
    parser.add_argument("--keyname", default="Species", help="Key name to use for the species value in the box")
    parser.add_argument("--squash-species", action="store_true",
                        help="Squash species annotations into a single class")
    parser.add_argument("--squash-species-name", default="object", help="name to use for the single class")

    parser.add_argument("output_dir")
    args = parser.parse_args()

    api = tator.get_api(args.host, args.token)
    box_type = api.get_localization_type(args.box_type_id)
    project = box_type.project

    localizations = api.get_localization_list(project,type=args.box_type_id)
    os.makedirs(args.output_dir, exist_ok=True)
    species_output=os.path.join(args.output_dir, "species.csv")
    if args.squash_species:
        single_species=[(args.squash_species_name, 0)]
        species_df = pd.DataFrame(data=single_species,
                                  columns=['Species','Num'])
        species_df.to_csv(species_output, index=False, header=False)
    else:
        # First get the species for the project by getting the set of the
        # species names
        species = {}
        for localization in localizations:
            species_name = localization.attributes[args.keyname]
            if species_name in species:
                species[species_name] += 1
            else:
                species[species_name] = 1

        print("Species Information:")
        pprint(species)

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


    media_types = api.get_media_type_list(project)
    type_lookup = {t.id : t.dtype for t in media_types}
    for section in sections_to_process:
        medias = api.get_media_list(project,section=section)
        section_obj = api.get_section(section)
        section_dir=os.path.join(images_dir, section_obj.name)
        os.makedirs(section_dir, exist_ok=True)
        bar = progressbar.ProgressBar(redirect_stdout=True)
        for media in bar(medias):
            media_element = media.to_dict() # for compatibility reason
            localizations= api.get_localization_list(project,
                                                     type=args.box_type_id,
                                                     media_id=[media.id])

            is_video = type_lookup[media.meta] == 'video'

            if localizations is None:
                print(f"{media_element['name']}({media_element['id']}) has no localizations")
                continue
            for localization in localizations:
                localization = localization.to_dict() # for compatibility reasons
                frame = localization['frame']
                image_path = os.path.join(section_dir, f"{media_element['name']}_{frame}.png")
                rel_image_path = os.path.relpath(image_path, images_dir)
                if not os.path.exists(image_path):
                    if is_video:
                        temp_path = api.get_frame(media.id,
                                                  frames=[frame])
                        shutil.move(temp_path,image_path)
                    else:
                        for _ in tator.download_media(api, media, image_path):
                            pass
                shape = media_element['media_files']['streaming'][0]['resolution']
                img_width = shape[1]
                img_height = shape[0]
                x1 = localization['x'] * img_width
                y1 = localization['y'] * img_height
                width = localization['width'] * img_width
                height = localization['height'] * img_height
                x2 = x1 + width
                y2 = y1 + height
                if args.squash_species:
                    species_name = args.squash_species_name
                else:
                    species_name = localization['attributes'][args.keyname]

                x1 = max(x1,0)
                x2 = min(x2, img_width)
                y1 = max(y1,0)
                y2 = min(y2, img_height)
                datum={'img': rel_image_path,
                       'x1': round(x1),
                       'y1': round(y1),
                       'x2': round(x2),
                       'y2': round(y2),
                       'species_name': species_name}
                datum_df = pd.DataFrame(data=[datum],
                                 columns=cols)
                datum_df.to_csv(annotations_output,
                                mode='a',
                                header=False,
                                index=False)
