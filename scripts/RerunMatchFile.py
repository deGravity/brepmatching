import matchGenHelpFunctions as mg
import pandas as pd
from zipfile import ZipFile
from tqdm import tqdm
from argparse import ArgumentParser
import os
from client import Client

def main():
    parser = ArgumentParser()
    parser.add_argument('input',type=str)
    parser.add_argument('outdir',type=str)
    args = parser.parse_args()

    # stacks to choose from
    stacks = {
        'cad': 'https://cad.onshape.com'
    }

    # create instance of the onshape client; change key to test on another stack
    c = Client(stack=stacks['cad'], logging=False)
    
    with ZipFile(args.input, 'r') as zf:
        with zf.open('data/baseline/allVariationsWithBaseline.csv') as f:
            variationData = pd.read_csv(f)
    variationData = variationData[(variationData.fail == 0) & (variationData.translationFail == 0)]

    for i in tqdm(range(len(variationData))):
        d = variationData.iloc[i]

        match_outpath = os.path.join(args.outdir, d.baselineMatch)

        did = d.baselinedid
        eid = d.baselineeid
        mv_orig = d.baselineOrig_mid
        mv_var = d.baselineNew_mid

        orginReferences = mg.getReferencesMV(c,did, mv_orig, eid)

        varReferences = mg.getReferencesMV(c, did, mv_var, eid)

        os.makedirs(os.path.dirname(match_outpath), exist_ok=True)
        mg.exportMatches(orginReferences, varReferences, match_outpath)

if __name__ == '__main__':
    main()