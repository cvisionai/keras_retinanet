#!/usr/bin/env python3

import argparse
import pytator
import os
import pandas as pd

from pprint import pprint

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                     "Sets up a training area from tator project")
    parser = pytator.tator.cli_parser(parser)
    parser.add_argument("--section", nargs='+', help='sections to download')
    parser.add_argument("--box-type-id", type=int, required=True)
    parser.add_argument("--keyname", default="Species", help="Key name to use for the species value in the box")
    parser.add_argument("--squash-species", action="store_true",
                        help="Squash species annotations into a single class")
    parser.add_argument("--squash-species-name", default="object", help="name to use for the single class")

    parser.add_argument("output_dir")
    args = parser.parse_args()

    tator = pytator.Tator(args.url, args.token, args.project)

    os.makedirs(args.output_dir, exist_ok=True)
    species_output=os.path.join(args.output_dir, "species.csv")
    if args.squash_species:
        single_species=[(args.squash_species_name, 0)]
        species_df = pd.DataFrame(data=single_species)
        species_df.to_csv(species_output, index=False, header=False)
    else:
        # First get the species for the project by getting the set of the
        # species names
        localizations = tator.Localization.filter({"type": args.box_type_id})
        species = {}
        for localization in localizations:
            species_name = localization['attributes'][args.keyname]
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
        species_df = pd.DataFrame(data=name_idx_pairs)
        species_df.to_csv(species_output, index=False, header=False)
