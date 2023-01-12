import json
from apikey.client import Client
from onshape_client import OnshapeElement as newOnshapeEl
from onshape_client import Client as newClient
from onshape_client.oas.models.btm_feature134 import BTMFeature134
from onshape_client.oas.models.btm_parameter_quantity147 import BTMParameterQuantity147
from onshape_client.oas.models.btm_parameter_boolean144 import BTMParameterBoolean144
from onshape_client.oas.models.bt_feature_definition_call1406 import (
    BTFeatureDefinitionCall1406,
)
import random


# Create a Variation
# input: orinalModelInfo orginReferences
# output: variationInfo
def createVariation(c, paths, codeVersion, orinalModelInfo, orginReferences, GeoTransf,TopoTransf ):
    varNumber =  orinalModelInfo['nVariations'] + 1
    varInfo = {
                'name':orinalModelInfo['name'],
                'did':orinalModelInfo['did'],
                'wid':orinalModelInfo['wid'],
                'eid':orinalModelInfo['eid'],
                'link_orig': orinalModelInfo['link'], 
                'mv_orig':orinalModelInfo['mvid'],
                'ps_orig':orinalModelInfo['ps'],
                'mv_var':'TBD',
                'ps_var': orinalModelInfo['name'] + 'V' + str(varNumber) + '.x_t',
                'matchFile':orinalModelInfo['name'] + 'V0' + str(varNumber) + '.json',
                'codeVersion':codeVersion,
                'fail' : 0
    }
    varInfo['seed'] = random.randint(10, 100000) 
    varResult = createVariations(c, varInfo['did'], varInfo['wid'], varInfo['eid'], varInfo['seed'], GeoTransf,TopoTransf )
    stop = 10
    while ((varResult['state'] != "OK") & stop>0):
        if (varResult['state'] != "ABORT"):
            deleteFeature(c, varInfo['did'], varInfo['wid'], varInfo['eid'], varResult['fid'])
        varInfo['seed'] = random.randint(10, 100000) 
        varResult = createVariations(c, varInfo['did'], varInfo['wid'], varInfo['eid'],varInfo['seed'], GeoTransf,TopoTransf)
        stop-=1 
    if (varResult['state'] != "OK"):
        varInfo['fail'] = 1
        print("failed more than 10 times")
    else:
        varReferences = getReferences (c, varInfo['did'], varInfo['wid'], varInfo['eid'])
        exportParasolid(c, varInfo['did'], varInfo['wid'], varInfo['eid'], paths['BrepsWithReferencePath'] / varInfo['ps_var'])
        exportMatches(orginReferences, varReferences, paths['MatchesPath'] / varInfo['matchFile'])
        varInfo['mv_var'] = c.historyfromWorkspace(varInfo['did'], varInfo['wid']).json()[0]['microversionId']
    deleteFeature(c, varInfo['did'], varInfo['wid'], varInfo['eid'], varResult['fid'])
    # variationInfo = {'mvid': mvid1, 'parasolidName' : (modelName+ 'V1'), 'matchFile' : ( modelName+ 'V01.json'), 'codeVersion' : codeVersion}
    # varationData['variations'].append(variationInfo)
    varInfo['link_var'] =  'https://cad.onshape.com/documents/' + varInfo['did'] + '/w/' + varInfo['wid']   + '/m/' + varInfo['mv_var']   +'/e/' + varInfo['eid']
    return varInfo


# Create a new Doc with a orignal verions
# input: modelName (also global paths and client)
# output: orinalModelInfo: (name, did, wid, eid, mvid, ps, nVariations)
#         orginReferences 
def initNewModel(c, paths, modelName):
    new_doc = newOnshapeEl.create(modelName)
    imported_part_studio = new_doc.import_file(paths['OriginalBrepsPath'] / (modelName + ".x_t"), allow_faulty_parts=True)
    orinalModelInfo = {
        'name': modelName,
        'did' : imported_part_studio.did, 
        'wid' : imported_part_studio.wvmid, 
        'eid' : imported_part_studio.eid, 
        'ps' : ( modelName+ 'V0.x_t'),
        'nVariations' : 0 
        }
    setAttributes(c, orinalModelInfo['did'], orinalModelInfo['wid'], orinalModelInfo['eid'])
    orginReferences = getReferences (c, orinalModelInfo['did'], orinalModelInfo['wid'], orinalModelInfo['eid'])
    orinalModelInfo['mvid'] = c.historyfromWorkspace(orinalModelInfo['did'], orinalModelInfo['wid']).json()[0]['microversionId']
    exportParasolid(c, orinalModelInfo['did'], orinalModelInfo['wid'], orinalModelInfo['eid'], paths['BrepsWithReferencePath'] / orinalModelInfo['ps'])
    orinalModelInfo['link'] =  'https://cad.onshape.com/documents/' + orinalModelInfo['did'] + '/w/' + orinalModelInfo['wid']   + '/m/' + orinalModelInfo['mvid']   +'/e/' + orinalModelInfo['eid']
    return[ orinalModelInfo, orginReferences]


def getOnshapeBaselineMatch(c, newClient, inPath1, inPath2, outPath1, outPath2, outPathMatch):
    new_doc = newOnshapeEl.create("testUpdates")
    imported_part_studio = new_doc.import_file(inPath1, allow_faulty_parts=True)
    setAttributes(c, imported_part_studio.did, imported_part_studio.wvmid, imported_part_studio.eid)
    orginReferences = getReferences (c, imported_part_studio.did, imported_part_studio.wvmid, imported_part_studio.eid)
    exportParasolid(c, imported_part_studio.did, imported_part_studio.wvmid, imported_part_studio.eid, outPath1)
    updateImport(newClient, imported_part_studio.did, imported_part_studio.wvmid, imported_part_studio.eid, inPath2)
    varReferences = getReferences (c, imported_part_studio.did, imported_part_studio.wvmid, imported_part_studio.eid)
    exportParasolid(c, imported_part_studio.did, imported_part_studio.wvmid, imported_part_studio.eid, outPath2)
    exportMatches(orginReferences, varReferences, outPathMatch)


def dlParasolid(c, did, wid, eid, filename):
    return c.get_partWithLabels(did, wid, eid).text

def exportParasolid (c, did, wid, eid, filename):
    asm = c.get_partWithLabels(did, wid, eid)
    with open(filename, "w") as outfile:
        outfile.write(str(asm.text))

def deleteFeature(c, did, wid, eid, fid):
    c.deleteFeature(did, wid, eid, fid)

def testAutoreload():
    print ("test24!")

def exportMatches(references1, references2, filename):
    matches = {}

    for key, value1 in references1.items(): 
        if key in references2.keys():
            value2 = references2[key] 
            matches[key] = {"val1" : value1, "val2": value2}
    with open(filename, "w") as outfile:
        json.dump(matches, outfile)


def exportMatchesToFile(references1, references2, outfile):
    matches = {}

    for key, value1 in references1.items(): 
        if key in references2.keys():
            value2 = references2[key] 
            matches[key] = {"val1" : value1, "val2": value2}
    outfile.write(json.dumps(matches).encode('utf-8'))

def createNewDocWithParasolidFile(c, docname, parasolidFile):
    asm = c.new_document(docname)
    did = asm.json()['id']
    wid = asm.json()['defaultWorkspace']['id']
    # upload the parasolid and get the resulting blob element
    tid = '637d4fcd368184174c46ea4e'
    eid =  c.getTranslationInfo(tid).json()['resultElementIds'][0]
    return {'did': did, 'wid': wid, 'eid': eid}


# couldn't get this to work, not sure why, creating and deleting documents for now.
def deleteElement(newClient, did, wid, eid):


    base_url = 'https://cad.onshape.com'
    fixed_url = '/api/elements/d/did/w/wid/e/eid'

    method = 'DELETE'

    headers = {'Accept': 'application/vnd.onshape.v1+json;charset=UTF-8;qs=0.1',
            'X-XSRF-TOKEN': 'application/json;charset=UTF-8; qs=0.09'}


    fixed_url = fixed_url.replace('did', did)
    fixed_url = fixed_url.replace('wid', wid)
    fixed_url = fixed_url.replace('eid', eid)
    
    try:
        response = newClient.api_client.request(method, url=base_url + fixed_url, 
        query_params={}, 
        headers=headers,
        post_params={},
        body={})
        print(response)
    except Exception as e:
        print(e)


def deleteElement(newClient, did, wid, eid):


    base_url = 'https://cad.onshape.com'
    fixed_url = '/api/elements/d/did/w/wid/e/eid'

    method = 'DELETE'

    headers = {'Accept': 'application/vnd.onshape.v1+json;charset=UTF-8;qs=0.1',
            'X-XSRF-TOKEN': 'application/json;charset=UTF-8; qs=0.09'}


    fixed_url = fixed_url.replace('did', did)
    fixed_url = fixed_url.replace('wid', wid)
    fixed_url = fixed_url.replace('eid', eid)
    
    try:
        response = newClient.api_client.request(method, url=base_url + fixed_url, 
        query_params={}, 
        headers=headers,
        post_params={},
        body={})
        print(response)
    except Exception as e:
        print(e)


def updateImportFromFile(newClient, did, wid, eid, aFile):

    base_url = 'https://cad.onshape.com'
    fixed_url = '/api/blobelements/d/did/w/wid/e/eid'

    method = 'POST'

    headers = {'Accept': 'application/vnd.onshape.v1+json; charset=UTF-8;qs=0.1',
            'Content-Type': 'multipart/form-data'}

    fixed_url = fixed_url.replace('did', did)
    fixed_url = fixed_url.replace('wid', wid)
    fixed_url = fixed_url.replace('eid', eid)
    post_params = []
    fileParam = newClient.api_client.files_parameters({'file': [aFile]})
    post_params.extend(fileParam)

    try:
        response = newClient.api_client.request(method, url=base_url + fixed_url, 
        query_params={}, 
        headers=headers,
        post_params=post_params,
        body={})
    except Exception as e:
        print(e)

    return response

def updateImport(newClient, did, wid, eid, filepath):
    aFile = open(filepath, "rb")

    base_url = 'https://cad.onshape.com'
    fixed_url = '/api/blobelements/d/did/w/wid/e/eid'

    method = 'POST'

    headers = {'Accept': 'application/vnd.onshape.v1+json; charset=UTF-8;qs=0.1',
            'Content-Type': 'multipart/form-data'}

    fixed_url = fixed_url.replace('did', did)
    fixed_url = fixed_url.replace('wid', wid)
    fixed_url = fixed_url.replace('eid', eid)
    post_params = []
    fileParam = newClient.api_client.files_parameters({'file': [aFile]})
    post_params.extend(fileParam)

    try:
        response = newClient.api_client.request(method, url=base_url + fixed_url, 
        query_params={}, 
        headers=headers,
        post_params=post_params,
        body={})
    except Exception as e:
        print(e)
    finally:
        aFile.close()

    return response

    


def getReferences (c, did, wid, eid): 
    references = {}

    featureScript = {
    "script" : """function (context is Context, queries is map) {
                var result = {};
                for (var entity in evaluateQuery(context, qAllModifiableSolidBodies()->qOwnedByBody()))
                {
                    
                    var att = getAttribute(context, {
                            "entity" : entity,
                            "name" : "id"
                    });
                    if (att != undefined)
                        result[att] = entity.transientId; 
                }
                return result;
                }""",
    "queries" : [ ]
    }

    asm = c.evaluate_featureScript(did, wid, eid, featureScript);

    notices = asm.json()['notices']; 

    # if (len(notices) > 0) :
    #     print("Error in Get References Not returning----------------------------")
    #     print("Onshape Error: ")
    #     print(asm.json()['notices'][0]['message']['message'])
    #     print("Onshape Warnings: ")
    #     print(asm.json()['notices'][1]['message']['message'])
    #     print("----------------------------")

    # else :
    entities =  asm.json()['result']['message']['value']
    for entity in entities:
        idx = entity["message"]['key']['message']['value']
        val1 = entity["message"]['value']['message']['value']
        references[idx] =  val1

    return (references)


def getReferencesMV(c, did, mv, eid): 
    references = {}

    featureScript = {
    "script" : """function (context is Context, queries is map) {
                var result = {};
                for (var entity in evaluateQuery(context, qAllModifiableSolidBodies()->qOwnedByBody()))
                {
                    
                    var att = getAttribute(context, {
                            "entity" : entity,
                            "name" : "id"
                    });
                    if (att != undefined)
                        result[att] = entity.transientId; 
                }
                return result;
                }""",
    "queries" : [ ]
    }

    asm = c.evaluate_featureScriptMV(did, mv, eid, featureScript);

    notices = asm.json()['notices']; 

    # if (len(notices) > 0) :
    #     print("Error in Get References Not returning----------------------------")
    #     print("Onshape Error: ")
    #     print(asm.json()['notices'][0]['message']['message'])
    #     print("Onshape Warnings: ")
    #     print(asm.json()['notices'][1]['message']['message'])
    #     print("----------------------------")

    # else :
    entities =  asm.json()['result']['message']['value']
    for entity in entities:
        idx = entity["message"]['key']['message']['value']
        val1 = entity["message"]['value']['message']['value']
        references[idx] =  val1

    return (references)



def setAttributes (c, did, wid, eid): 

    feature = {
    "feature" : {
        "type": 134,
        "typeName": "BTMFeature",
        "message": {
            "featureType": "setAttributes",
            "namespace" : "d7534d34ba0e0ebceb76c5adf::v13b3a13d98c77e0c0c891302::e409a257e3570984ca3a884ae::m525a118a6b9bfda49f8bcd93",
            "name": "SetAttributes",
            "parameters": []
            }
        }
    }
    asm = c.add_feature(did, wid, eid, feature)


def getTransientIds (c, did, wid, eid):

    featureScript = {
    "script" : """function (context is Context, queries is map) {
                            var result = {};
                for (var idx, entity in evaluateQuery(context, qAllModifiableSolidBodies()->qOwnedByBody()))
                {
                    
                    result[idx]  =entity.transientId; 
                }
                return result;
                }""",
    "queries" : [ ]
    }

    asm = c.evaluate_featureScript(did, wid, eid, featureScript)
    entities =  asm.json()['result']['message']['value']
    transIds = [None] * len(entities)
    idx =0
    for entity in entities:
        transIds[idx] = entity["message"]['value']['message']['value']
        idx = idx+1
        
    return transIds

def setAttributesWithQueries (c, did, wid, eid, transIds): 

    queries = [None] * len(transIds)
    for i in range(0, len(transIds)):
        queries [i] = {'type': 138, 'typeName': 'BTMIndividualQuery', 'message': { 'geometryIds': [transIds[i]], } }

    feature = {
    "feature" : {
        'type': 134, 
        'typeName': 'BTMFeature', 
        'message': {
            'featureType': 'setAttributesWithQueries', 
            'name': 'SetAttributesWith Queries 1', 
            'parameters': [
                    {'type': 148, 
                    'typeName': 'BTMParameterQueryList', 
                    'message': {'queries': queries, 
                            'parameterId': 'foo'                          
                        }
                    }
            ], 
            'namespace': 'd7534d34ba0e0ebceb76c5adf::v2e6c312a195b6da84147fc96::e409a257e3570984ca3a884ae::mfb7bae41d0d75f63d387c220' 
            }
        }
    }
    asm = c.add_feature(did, wid, eid, feature)



# def createVariationsNewClient (newClient, did, wid, eid, val, GeoTransf,TopoTransf): 
        
#     seedParam= BTMParameterQuantity147( expression=str(val), parameter_id="mySeed" )
#     topoParam = BTMParameterBoolean144(value = True, parameter_id= "TopoTransf" )
#     geoParam = BTMParameterBoolean144(value = False, parameter_id= "GeoTransf")
#     varfeature = BTMFeature134(
#             bt_type="BTMFeature-134",
#             name="CreateVariations 1",
#             feature_type="CreateVariations",
#             namespace = "d7534d34ba0e0ebceb76c5adf::v95167b5eca0cd3643a804bba::ef2f9b0185467d986bff5df1a::m5b7bf0078fc01f58251f532b",
#             parameters=[seedParam, topoParam, geoParam],
#         )

#     print("started to do:  "+ str(val))
#     try:
#         asm = newClient.part_studios_api.add_part_studio_feature(
#             did=did, 
#             wvm="w", 
#             wvmid=wid, 
#             eid=eid,  
#             bt_feature_definition_call_1406= BTFeatureDefinitionCall1406(feature=varfeature),
#             _preload_content=False,
#         )
#         return (asm)
#     except Exception as e:
#         return ("error")
#     finally:
#         return ("finnaly")

    
#     return asm
#     # print("succeeded to execture the feature:  "+ str(val))
#     # print(asm.json())
#     #     return {"state" : "ABORT", "fid" : "no need to delete"}
#     # # print(asm.json()['featureState'])
#     # return {"state" : asm.json()['featureState']['message']['featureStatus'], "fid": asm.json()['feature']['message']['featureId']}


# def createVariationsByHand (newClient, did, wid, eid, val, GeoTransf,TopoTransf): 
        
#     feature = {
#     "feature" : {
#         "type": 134,
#         "typeName": "BTMFeature",
#         "message": {
#             "featureType": "CreateVariations",
#             "namespace" : "d7534d34ba0e0ebceb76c5adf::v95167b5eca0cd3643a804bba::ef2f9b0185467d986bff5df1a::m5b7bf0078fc01f58251f532b",
#             "name": "CreateVariations 1",
#             "parameters": [
#              {
#                 "type": 147,
#                 "typeName": "BTMParameterQuantity",
#                 "message": {
#                 "expression": val,
#                 "parameterId": "mySeed"
#                 }
#             },
#             {
#                 "type": 144,
#                 "typeName": "BTParameterSpecBoolean",
#                 "message": {
#                 "value": TopoTransf,
#                 "parameterId": "TopoTransf"
#                 }
#             },
#             {
#                 "type": 144,
#                 "typeName": "BTParameterSpecBoolean",
#                 "message": {
#                 "value": GeoTransf,
#                 "parameterId": "GeoTransf"
#                 }
#             }   
#             ]
#         }
#         }
#     }

#     base_url = 'https://cad.onshape.com'
#     fixed_url = '/api/partstudios/d/did/w/wid/e/eid/features'

#     method = 'POST'

#     headers = {'Accept': 'application/json;charset=UTF-8; qs=0.09',
#             'Content-Type': ' application/json;charset=UTF-8; qs=0.09',
#             'X-XSRF-TOKEN' : 'LSbTNiSFFN9XwgWZwrEdBg=='}

#     fixed_url = fixed_url.replace('did', did)
#     fixed_url = fixed_url.replace('wid', wid)
#     fixed_url = fixed_url.replace('eid', eid)

#     try:
#         response = newClient.api_client.request(method, url=base_url + fixed_url, 
#         query_params={}, 
#         headers=headers,
#         post_params=[],
#         body=feature)
#         return response
#     except Exception as e:
#         print(e)
#         return("errror!!")
#     finally:
#          return "finnally"

    
    
#     # print("succeeded to execture the feature:  "+ str(val))
#     # print(asm.json())
#     #     return {"state" : "ABORT", "fid" : "no need to delete"}
#     # # print(asm.json()['featureState'])
#     # return {"state" : asm.json()['featureState']['message']['featureStatus'], "fid": asm.json()['feature']['message']['featureId']}



def createVariations (c, did, wid, eid, val, GeoTransf,TopoTransf): 

    feature = {
    "feature" : {
        "type": 134,
        "typeName": "BTMFeature",
        "message": {
            "featureType": "CreateVariations",
            "namespace" : "d7534d34ba0e0ebceb76c5adf::vc59d86f61d68660eb23c9c55::ef2f9b0185467d986bff5df1a::mbbbc0742bca4524a6b090ae6",
            "name": "CreateVariations 1",
            "parameters": [
             {
                "type": 147,
                "typeName": "BTMParameterQuantity",
                "message": {
                "expression": val,
                "parameterId": "mySeed"
                }
            },
            {
                "type": 144,
                "typeName": "BTParameterSpecBoolean",
                "message": {
                "value": TopoTransf,
                "parameterId": "TopoTransf"
                }
            },
            {
                "type": 144,
                "typeName": "BTParameterSpecBoolean",
                "message": {
                "value": GeoTransf,
                "parameterId": "GeoTransf"
                }
            }   
            ]
        }
        }
    }

    try:
        asm = c.add_feature(did, wid, eid, feature)
        if ('status' in asm.json()):
            return {"state" : "ABORT", "fid" : "no need to delete"}
        # print(asm.json()['featureState'])
        return {"state" : asm.json()['featureState']['message']['featureStatus'], "fid": asm.json()['feature']['message']['featureId']}

    except Exception as e:
        print(e)
        return {"state" : "EXEPTION", "fid" : "no need to delete"}



    print("started to do:  "+ str(val))
    asm = c.add_feature(did, wid, eid, feature)
    print("succeeded to execture the feature:  "+ str(val))
    print(asm.json())
    if ('status' in asm.json()):
        return {"state" : "ABORT", "fid" : "no need to delete"}
    # print(asm.json()['featureState'])
    return {"state" : asm.json()['featureState']['message']['featureStatus'], "fid": asm.json()['feature']['message']['featureId']}


def createOldVariations (c, did, wid, eid): 

    val = random.randint(2, 100)
    feature = {
    "feature" : {
        "type": 134,
        "typeName": "BTMFeature",
        "message": {
            "featureType": "CreateVariations",
            "namespace" : "d7534d34ba0e0ebceb76c5adf::v0fc0290e413e20af557eb514::ef2f9b0185467d986bff5df1a::mf4eecbd61a3f151708cedf32",
            "name": "CreateVariations 4",
            "parameters": [
            {
                "type": 147,
                "typeName": "BTMParameterQuantity",
                "message": {
                "expression": "5",
                "parameterId": "n1"
                }
            },
            {
                "type": 147,
                "typeName": "BTMParameterQuantity",
                "message": {
                "expression": "5",
                "parameterId": "n2"
                }
            },
            {
                "type": 147,
                "typeName": "BTMParameterQuantity",
                "message": {
                "expression": "1",
                "parameterId": "n3"
                }
            },
            {
                "type": 147,
                "typeName": "BTMParameterQuantity",
                "message": {
                "expression": val,
                "parameterId": "seed"
                }
            }
            ]
        }
        }
    }
    asm = c.add_feature(did, wid, eid, feature)
    return asm.json()['feature']['message']['featureId']