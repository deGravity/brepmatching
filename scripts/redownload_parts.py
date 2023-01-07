from zipfile import ZipFile
import pandas as pd
from tqdm import tqdm
from apikey.onshape import Onshape
from pathlib import Path
from argparse import ArgumentParser

def main():

    parser = ArgumentParser()
    parser.add_argument('--input', 'str')
    parser.add_argument('--output', 'str')
    args = parser.parse_args()

    api = Onshape(stack='https://cad.onshape.com', creds=str(Path.home().joinpath('.config','onshapecreds.json')), logging=False)

    with ZipFile(args.input,'r') as zf:
        with zf.open('data/baseline/allVariationsWithBaseline.csv','r') as f:
            variations = pd.read_csv(f)
        variations[:100]
        variations = variations[variations.fail == 0]

        failures = []
        written = set()
        with ZipFile(args.output,'w') as wzf:
            for i in tqdm(range(len(variations))):
                d = variations.iloc[i]

                response_orig = api.request(method='get', path=f'/api/partstudios/d/{d.did}/m/{d.mv_orig}/e/{d.eid}/parasolid', query={'includeExportIds':True})
                response_var = api.request(method='get', path=f'/api/partstudios/d/{d.did}/m/{d.mv_var}/e/{d.eid}/parasolid', query={'includeExportIds':True})

                if response_orig.status_code != 200 or response_var.status_code != 200:
                    failures.append(d)
                    continue
                
                orig_path = 'data/BrepsWithReference/' + d.ps_orig
                var_path = 'data/BrepsWithReference/' + d.ps_var

                if orig_path not in written:
                    with wzf.open(orig_path, 'w') as f:
                        f.write(response_orig.text.encode('utf-8'))
                    written.add(orig_path)
                if var_path not in written:
                    with wzf.open(var_path, 'w') as f:
                        f.write(response_var.text.encode('utf-8'))
                    written.add(var_path)