'''
client
======

Convenience functions for working with the Onshape API
'''

from apikey.onshape import Onshape

import mimetypes
import random
import string
import os
from pathlib import Path


class Client():
    '''
    Defines methods for testing the Onshape API. Comes with several methods:

    - Create a document
    - Delete a document
    - Get a list of documents

    Attributes:
        - stack (str, default='https://cad.onshape.com'): Base URL
        - logging (bool, default=True): Turn logging on or off
    '''

    def __init__(self, stack='https://cad.onshape.com', logging=True):
        '''
        Instantiates a new Onshape client.

        Args:
            - stack (str, default='https://cad.onshape.com'): Base URL
            - logging (bool, default=True): Turn logging on or off
        '''

        self._stack = stack
        self._api = Onshape(stack=stack, logging=logging, creds=str(Path.home().joinpath('.config','onshapecreds.json')))

    
    def get_partWithLabels(self, did, wid, eid, pid = None, config = None, out_path=None):
        endpoint = f'/api/partstudios/d/{did}/w/{wid}/e/{eid}/parasolid'
        query = {'includeExportIds':True}
        if pid is not None:
            query['partIds'] = pid
        if config is not None:
            query['configuration'] = config

        response = self._api.request('get', endpoint, query)

        if response.status_code < 200 or response.status_code >= 300:
            raise Exception(f'Request for parasolid {endpoint} was not successful.')
        
        if out_path is not None:
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        return response
    
    def getTranslationInfo(self, tid):

        return self._api.request('get', '/api/translations/' + tid)



    def new_document(self, name='Test Document', owner_type=0, public=False):
        '''
        Create a new document.

        Args:
            - name (str, default='Test Document'): The doc name
            - owner_type (int, default=0): 0 for user, 1 for company, 2 for team
            - public (bool, default=False): Whether or not to make doc public

        Returns:
            - requests.Response: Onshape response data
        '''

        payload = {
            'name': name,
            'ownerType': owner_type,
            'isPublic': public
        }

        return self._api.request('post', '/api/documents', body=payload)



    def rename_document(self, did, name):
        '''
        Renames the specified document.

        Args:
            - did (str): Document ID
            - name (str): New document name

        Returns:
            - requests.Response: Onshape response data
        '''

        payload = {
            'name': name
        }

        return self._api.request('post', '/api/documents/' + did, body=payload)

    def del_document(self, did):
        '''
        Delete the specified document.

        Args:
            - did (str): Document ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('delete', '/api/documents/' + did)

    def get_document(self, did):
        '''
        Get details for a specified document.

        Args:
            - did (str): Document ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/documents/' + did)

    def list_documents(self):
        '''
        Get list of documents for current user.

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/documents')

    def create_assembly(self, did, wid, name='My Assembly'):
        '''
        Creates a new assembly element in the specified document / workspace.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - name (str, default='My Assembly')

        Returns:
            - requests.Response: Onshape response data
        '''

        payload = {
            'name': name
        }

        return self._api.request('post', '/api/assemblies/d/' + did + '/w/' + wid, body=payload)

    def get_features(self, did, wid, eid):
        '''
        Gets the feature list for specified document / workspace / part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features')

    def get_partstudio_tessellatededges(self, did, wid, eid):
        '''
        Gets the tessellation of the edges of all parts in a part studio.

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/tessellatededges')

    def upload_blob(self, did, wid, filepath='./blob.json'):
        '''
        Uploads a file to a new blob element in the specified doc.
        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - filepath (str, default='./blob.json'): Blob element location
        Returns:
            - requests.Response: Onshape response data
        '''

        chars = string.ascii_letters + string.digits
        boundary_key = ''.join(random.choice(chars) for i in range(8))

        mimetype = mimetypes.guess_type(filepath)[0]
        encoded_filename = os.path.basename(filepath)
        file_content_length = str(os.path.getsize(filepath))
        blob = open(filepath)

        req_headers = {
            'Content-Type': 'multipart/form-data; boundary="%s"' % boundary_key
        }

        # build request body
        payload = '--' + boundary_key + '\r\nContent-Disposition: form-data; name="encodedFilename"\r\n\r\n' + encoded_filename + '\r\n'
        payload += '--' + boundary_key + '\r\nContent-Disposition: form-data; name="fileContentLength"\r\n\r\n' + file_content_length + '\r\n'
        payload += '--' + boundary_key + '\r\nContent-Disposition: form-data; name="file"; filename="' + encoded_filename + '"\r\n'
        payload += 'Content-Type: ' + mimetype + '\r\n\r\n'
        payload += blob.read()
        payload += '\r\n--' + boundary_key + '--'

        return self._api.request('post', '/api/blobelements/d/' + did + '/w/' + wid, headers=req_headers, body=payload)
    



    def upload_parasolid(self, did, wid):
        '''
        Uploads a file to a new blob element in the specified doc.
        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - filepath (str, default='./blob.json'): Blob element location
        Returns:
            - requests.Response: Onshape response data
        '''

        req_headers = {'Accept': 'application/vnd.onshape.v1+json; charset=UTF-8;qs=0.1',
           'Content-Type': 'multipart/form-data'}


        fileinfo = """
            "importantNumber": 0,
            "name": "Bob",
            "2 is less than 1": false
        """
        # build request body
        payload = {
            'file': fileinfo
        }

        return self._api.request('post', '/api/blobelements/d/' + did + '/w/' + wid, headers=req_headers, body=payload)
    


    def part_studio_stl_fromMV(self, did, mid, eid):
        '''
        Exports STL export from a part studio

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        req_headers = {
            'Accept': 'application/vnd.onshape.v1+octet-stream'
        }
        return self._api.request('get', '/api/partstudios/d/' + did + '/m/' + mid + '/e/' + eid + '/stl', headers=req_headers)

    def part_studio_stl(self, did, wid, eid):
        '''
        Exports STL export from a part studio

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        req_headers = {
            'Accept': 'application/vnd.onshape.v1+octet-stream'
        }
        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/stl', headers=req_headers)


    def part_studio_parasolid_fromMV(self, did, mid, eid):
        '''
        Exports STL export from a part studio

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        req_headers = {
            'Accept': 'application/vnd.onshape.v1+octet-stream'
        }
        return self._api.request('get', '/api/partstudios/d/' + did + '/m/' + mid + '/e/' + eid + '/parasolid', headers=req_headers)


    def part_studio_parasolid(self, did, wid, eid):
            '''
            Exports STL export from a part studio

            Args:
                - did (str): Document ID
                - wid (str): Workspace ID
                - eid (str): Element ID

            Returns:
                - requests.Response: Onshape response data
            '''

            req_headers = {
                'Accept': 'application/vnd.onshape.v1+octet-stream'
            }
            return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/parasolid', headers=req_headers)




    def historyfromWorkspace(self, did, wid):

        req_headers = {
            'Accept': 'application/vnd.onshape.v1+octet-stream'
        }
        return self._api.request('get', '/api/documents/d/' + did + '/w/' + wid + '/documenthistory')


    def historyfromMicroversion(self, did, mvid):

        return self._api.request('get', '/api/documents/d/' + did + '/m/' + mvid + '/documenthistory')


    def elements(self, did, mvid):

        return self._api.request('get', '/api/documents/d/' + did + '/m/' + mvid + '/elements')


    def bodyDetails(self, did, mvid, eid):

        return self._api.request('get', '/api/partstudios/d/' + did + '/m/' + mvid + '/e/' + eid + '/bodydetails')

    def getpng(self, did, mvid, eid):
        return self._api.request('get', '/api/partstudios/d/' + did + '/m/' + mvid + '/e/' + eid + '/shadedviews?viewMatrix=0.612,0.612,0,0,-0.354,0.354,0.707,0,0.707,-0.707,0.707,0')

    def getfeaturesFromMV(self, did, mvid, eid):
        return self._api.request('get', '/api/partstudios/d/' + did + '/m/' + mvid + '/e/' + eid + '/features')

    def deleteFeature(self, did, wid, eid, fid):
        return self._api.request('delete', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features/featureid/' + fid)

    def get_features(self, did, wid, eid):
        '''
        Gets the feature list for specified document / workspace / part studio.
        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
        Returns:
            - requests.Response: Onshape response data
        '''

        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features')


    def add_feature(self, did, wid, eid, payload) :
    
        return self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features', body = payload)

    def update_feature(self, did, wid, eid, fid, payload) :
    
        return self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features/featureid/' + fid, body = payload)

    def evaluate_featureScript(self, did, wid, eid, payload) :

        return self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/featurescript', body = payload)

    def evaluate_featureScriptMV(self, did, mv, eid, payload) :

        return self._api.request('post', '/api/partstudios/d/' + did + '/m/' + mv + '/e/' + eid + '/featurescript', body = payload)

    def getFeatureSpecsFromV (self, did, wid, eid) :


        return self._api.request('get', '/api/featurestudios/d/' + did + '/v/' + wid + '/e/' + eid + '/featurespecs')

    def getFeatureSpecsFromW (self, did, wid, eid) :


        return self._api.request('get', '/api/featurestudios/d/' + did + '/W/' + wid + '/e/' + eid + '/featurespecs')

    def update_rollback(self, did, wid, eid, body):
        '''
        Move the rollback bar in the feature list for a part studio (partner stack only)
        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID
            - body (dict): POST body
        Returns:
            - requests.Response: Onshape response data
        '''


        return self._api.request('post', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/features/rollback', body=body)

