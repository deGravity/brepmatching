from zipfile import ZipFile
import pandas as pd
from tqdm import tqdm
from apikey.onshape import Onshape
from pathlib import Path
from argparse import ArgumentParser
import pickle

def main():

    parser = ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    api = Onshape(stack='https://cad.onshape.com', creds=str(Path.home().joinpath('.config','onshapecreds.json')), logging=False)

    with ZipFile(args.input,'r') as zf:
        with zf.open('data/baseline/allVariationsWithBaseline.csv','r') as f:
            variations = pd.read_csv(f)
        variations[:100]
        variations = variations[variations.fail == 0]

        failures = []
        written = set()
        with ZipFile(args.output,'r') as rzf:
            names = rzf.namelist()
            for i in tqdm(range(len(variations))):
                d = variations.iloc[i]

                orig_path = 'data/BrepsWithReference/' + d.ps_orig
                var_path = 'data/BrepsWithReference/' + d.ps_var
                
                if orig_path not in names or var_path not in names:
                    failures.append(d)

    print(f'Finished checking. {len(failures)} failures found.')
    if len(failures) > 0:
        fail_path = args.output + '-failures.pickle'
        with open(fail_path, 'wb') as f:
            pickle.dump(failures, f)
        print(f'Failure cases written to {fail_path}')

if __name__ == '__main__':
    main()
