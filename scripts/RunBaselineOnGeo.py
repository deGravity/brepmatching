from client import Client
import matchGenHelpFunctions as mg
import json 
import pandas as pd
from onshape_client import OnshapeElement as newOnshapeEl
from onshape_client import Client as newClient
from pathlib import Path
import os
from tqdm import tqdm
from argparse import ArgumentParser

def main():

    parser = ArgumentParser()
    parser.add_argument('--folder', default='GeoV2FullRun', type=str)
    parser.add_argument('--baselineFolder', default='baselineOnshapeIlyaFixFullRun',type=str)
    parser.add_argument('--rerun', type=bool, action='store_false')
    parser.add_argument('--useRange', type=bool, action='store_true')
    parser.add_argument('--minVal', type=int, default=0)
    parser.add_argument('--maxVal', type=int, default=100)

    args = parser.parse_args()

    folder = args.folder
    baselideFolder = args.baselineFolder

    firstTry = not args.rerun

    minVal = args.minVal
    maxVal = args.maxVal
    toEnd= not args.useRange


    # stacks to choose from
    stacks = {
        'cad': 'https://cad.onshape.com'
    }

    # create instance of the onshape client; change key to test on another stack
    c = Client(stack=stacks['cad'], logging=False)

    newClient = newClient()


    paths = {
    'OriginalBrepsPath' : Path().cwd().parents[0] / 'data' / "OriginalBreps/with_tol",
    'OnshapeBaselinesPath' : Path().cwd().parents[0] / 'data' / folder / "data" / baselideFolder,
    'VariationDataPath' : Path().cwd().parents[0] / 'data' / folder /  "data" / "VariationData",
    'BrepsWithReferencePath' : Path().cwd().parents[0] / 'data' / folder / "data" / "BrepsWithReference"
    }

    if (firstTry):
        os.mkdir(paths['OnshapeBaselinesPath'])
        variationData = pd.read_csv(paths['VariationDataPath'] / 'all_variations.csv')
        variationData['baselineOrig']= " "
        variationData['baselineNew']= " "
        variationData['baselinedid']= " "
        variationData['baselinewid']= " "
        variationData['baselineeid']= " "
        variationData['baselineOrig_mid']= " "
        variationData['baselineNew_mid']= " "
        variationData['baselineMatch']= " "
        variationData['translationFail'] = 1
    else:
        variationData = pd.read_csv(paths['OnshapeBaselinesPath'] / 'allVariationsWithBaseline.csv')


    #for idx in range(0,len(variationData)):

    if (toEnd):
        maxVal = len(variationData)

    print ("minVal = " + str(minVal))
    print ("maxVal = " + str(maxVal))

    for idx in range(minVal, maxVal):
        print (idx)
        if(variationData['fail'][idx] != 1):
            variationData['translationFail'][idx] = 0
            variationData['baselineOrig'][idx] = "baselineOrig_" + variationData['ps_var'][idx]
            variationData['baselineNew'][idx] = "baselineNew_" + variationData['ps_var'][idx]
            variationData['baselineMatch'][idx] = "baselineMatch_" + variationData['matchFile'][idx]

            inPath1 = paths['BrepsWithReferencePath'] / variationData['ps_orig'][idx] 
            inPath2 = paths['BrepsWithReferencePath'] / variationData['ps_var'][idx] 
            outPath1 = paths['OnshapeBaselinesPath'] / variationData['baselineOrig'][idx]
            outPath2 = paths['OnshapeBaselinesPath'] / variationData['baselineNew'][idx] 
            outPathMatch = paths['OnshapeBaselinesPath']/ variationData['baselineMatch'][idx] 



            new_doc = newOnshapeEl.create("baselinefixedwithIlyaHelp" + variationData['name'][idx])
            did = new_doc.did
            wid = new_doc.wvmid
            variationData['baselinedid'][idx]= did
            variationData['baselinewid'][idx]= wid
            importResult = newClient.blob_elements_api.upload_file_create_element(did, wid,file=open(inPath1, "rb"),encoded_filename=inPath1.name, allow_faulty_parts=True,  translate=True)
            importEid = importResult.id
            try:
                translationResult = newOnshapeEl.poll_translation_result(importResult.translation_id)
            except Exception as e:
                variationData['translationFail'][idx] = 1

            if (variationData['translationFail'][idx] == 0): 
                editEid = translationResult.result_element_ids[0]
                variationData['baselineeid'][idx]= editEid     
                transIds = mg.getTransientIds(c, did, wid, editEid)
                mg.setAttributesWithQueries(c, did, wid, editEid, transIds)
                variationData['baselineOrig_mid'][idx]= c.historyfromWorkspace(did, wid).json()[0]['microversionId']
                orginReferences = mg.getReferences (c,did, wid, editEid)
                mg.exportParasolid(c, did, wid, editEid, outPath1)
                result = mg.updateImport(newClient, did, wid, importEid, inPath2)
                try:
                    newOnshapeEl.poll_translation_result( json.loads(result.data)['translationId'])  
                except Exception as e:
                    variationData['translationFail'][idx] = 1
                if (variationData['translationFail'][idx] == 0):  
                    variationData['baselineNew_mid'][idx]= c.historyfromWorkspace(did, wid).json()[0]['microversionId']    
                    varReferences = mg.getReferences (c, did, wid, editEid)
                    mg.exportParasolid(c, did, wid, editEid, outPath2)
                    mg.exportMatches(orginReferences, varReferences, outPathMatch)
                #mg.deleteElement(newClient, did, wid, editEid)

            #mg.deleteElement(newClient, did, wid, importEid)
        #new_doc.delete()
        variationData.to_csv(paths['OnshapeBaselinesPath'] / 'allVariationsWithBaseline.csv', mode='w', index=False, header=True)

if __name__ == '__main__':
    main()