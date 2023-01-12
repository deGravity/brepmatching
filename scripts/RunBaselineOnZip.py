from client import Client
import matchGenHelpFunctions as mg
import json 
import pandas as pd
from onshape_client import OnshapeElement as newOnshapeEl
from onshape_client import Client as NewClient
from pathlib import Path
import os
from tqdm import tqdm
from argparse import ArgumentParser
from zipfile import ZipFile
import shutil

def main():

    parser = ArgumentParser()
    parser.add_argument('--inputZip', type=str)
    parser.add_argument('--outputZip', type=str)
    parser.add_argument('--limit', type=int, default=-1)

    args = parser.parse_args()

    print(f'Copying {args.inputZip} to {args.outputZip}')
    shutil.copy(args.inputZip, args.outputZip)

    with ZipFile(args.inputZip, 'r') as zf_in, ZipFile(args.outputZip, 'a') as zf_out:
        with zf_in.open('data/VariationData/all_variations.csv') as f:
            variationData = pd.read_csv(f)
    
        variationData['baselineOrig']= " "
        variationData['baselineNew']= " "
        variationData['baselinedid']= " "
        variationData['baselinewid']= " "
        variationData['baselineeid']= " "
        variationData['baselineOrig_mid']= " "
        variationData['baselineNew_mid']= " "
        variationData['baselineMatch']= " "
        variationData['translationFail'] = 1

        variationData = variationData.copy()

        # stacks to choose from
        stacks = {
            'cad': 'https://cad.onshape.com'
        }

        # create instance of the onshape client; change key to test on another stack
        c = Client(stack=stacks['cad'], logging=False)

        newClient = NewClient()

        limit = args.limit
        if limit < 0:
            limit = len(variationData)

        for idx in tqdm(range(limit)):
            if(variationData.fail.iloc[idx] != 1):
                variationData.translationFail.iloc[idx] = 0
                variationData.baselineOrig.iloc[idx] = "baselineOrig_" + variationData.ps_var.iloc[idx]
                variationData.baselineNew.iloc[idx] = "baselineNew_" + variationData.ps_var.iloc[idx]
                variationData.baselineMatch.iloc[idx] = "baselineMatch_" + variationData.matchFile.iloc[idx]
                
                inPath1 = 'data/BrepsWithReference/' + variationData.ps_orig.iloc[idx] 
                inPath2 = 'data/BrepsWithReference/' + variationData.ps_var.iloc[idx] 
                outPath1 = 'data/baseline/' + variationData.baselineOrig.iloc[idx]
                outPath2 = 'data/baseline/' + variationData.baselineNew.iloc[idx] 
                outPathMatch = 'data/baseline/' + variationData.baselineMatch.iloc[idx] 


                new_doc = newOnshapeEl.create("baselinefixedwithIlyaHelp" + str(variationData.name.iloc[idx]))
                did = new_doc.did
                wid = new_doc.wvmid
                variationData.baselinedid.iloc[idx] = did
                variationData.baselinewid.iloc[idx] = wid
                with zf_in.open(inPath1,'r') as f:
                    importResult = newClient.blob_elements_api.upload_file_create_element(did, wid,file=f,encoded_filename=os.path.basename(inPath1), allow_faulty_parts=True,  translate=True)
                importEid = importResult.id
                try:
                    translationResult = newOnshapeEl.poll_translation_result(importResult.translation_id)
                except Exception as e:
                    variationData.translationFail.iloc[idx] = 1

                if (variationData.translationFail.iloc[idx] == 0): 
                    editEid = translationResult.result_element_ids[0]
                    variationData.baselineeid.iloc[idx] = editEid     
                    transIds = mg.getTransientIds(c, did, wid, editEid)
                    mg.setAttributesWithQueries(c, did, wid, editEid, transIds)
                    variationData.baselineOrig_mid.iloc[idx] = c.historyfromWorkspace(did, wid).json()[0]['microversionId']
                    orginReferences = mg.getReferences (c,did, wid, editEid)
                    ps_data1 = mg.dlParasolid(c, did, wid, editEid, outPath1)
                    if outPath1 not in zf_out.namelist(): # Don't add multiple copies of the input variation
                        with zf_out.open(outPath1,'w') as f:
                            f.write(ps_data1.encode('utf-8'))
                    with zf_in.open(inPath2,'r') as f:
                        result = mg.updateImportFromFile(newClient, did, wid, importEid, f)
                    try:
                        newOnshapeEl.poll_translation_result( json.loads(result.data)['translationId'])  
                    except Exception as e:
                        variationData.translationFail.iloc[idx]  = 1
                    if (variationData.translationFail.iloc[idx] == 0):  
                        variationData.baselineNew_mid.iloc[idx] = c.historyfromWorkspace(did, wid).json()[0]['microversionId']    
                        varReferences = mg.getReferences (c, did, wid, editEid)
                        ps_data2 = mg.dlParasolid(c, did, wid, editEid, outPath2)
                        with zf_out.open(outPath2,'w') as f:
                            f.write(ps_data2.encode('utf-8'))
                        with zf_out.open(outPathMatch,'w') as f:
                            mg.exportMatchesToFile(orginReferences, varReferences, f)
                    #mg.deleteElement(newClient, did, wid, editEid)

                #mg.deleteElement(newClient, did, wid, importEid)
            #new_doc.delete()
        with zf_out.open('data/baseline/allVariationsWithBaseline.csv','w') as f:
            variationData.to_csv(f, index=False, header=True)

if __name__ == '__main__':
    main()