from pathlib import Path
import json, sys, os, ast, hashlib, platform, tempfile, shutil

def main():
    repo=Path(sys.argv[1]); url=sys.argv[2]; output=Path(sys.argv[3])
    sys.path.insert(0,str(repo/'python'))
    deps=Path(__file__).parent/'windows-deps'
    if platform.system()=='Windows':
        sys.path.insert(0,str(deps)); os.environ['PYTHONPATH']=str(deps)+os.pathsep+str(repo/'python')
    cache_root=Path(tempfile.mkdtemp(prefix='ouroboros-173-probe-'))
    for key in ('CLOUD_VOLUME_DIR','CLOUD_VOLUME_CACHE_DIR','CLOUD_FILES_DIR'):
        os.environ[key]=str(cache_root/key.lower())
    import numpy as np
    from ouroboros.helpers.volume_cache import CloudVolumeInterface, download_volume, download_volume_geometry
    from ouroboros.helpers.bounding_boxes import BoundingBox
    from ouroboros.helpers.mem import SharedNPManager
    from ouroboros.common.memory_managed import resident_bytes
    cv=CloudVolumeInterface(url)
    box=BoundingBox(BoundingBox.bounds_to_rect(-64,192,-64,192,-64,192))
    bbox, clipped, shape=download_volume_geometry(cv,box)
    result={'platform':platform.platform(),'source_dtype':str(cv.dtype),'requested_bounds':[bbox.minpt.tolist(),bbox.maxpt.tolist()],
        'clipped_bounds':[clipped.minpt.tolist(),clipped.maxpt.tolist()],'buffers':[]}
    source=(repo/'python/ouroboros/helpers/volume_cache.py').read_text()
    tree=ast.parse(source)
    result['full_buffer_zero_assignments']=[node.lineno for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and isinstance(node.value,ast.Constant) and node.value.value==0
        and any(isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='volume_data' for target in node.targets)]
    result['volume_cache_sha256']=hashlib.sha256(source.encode()).hexdigest()
    with SharedNPManager(authkey=b'issue173-probe') as (manager,):
        for shared in (False,True,True):
            volume,*_=download_volume(cv,box,0,use_shared=shared,shm_address=manager.address,shm_authkey=b'issue173-probe')
            record={'shared':shared,'dtype':str(volume.dtype),'nbytes':volume.size() if shared else volume.nbytes}
            if shared:
                record['name']=volume.name
                record['resident_bytes_before_inspection']=(resident_bytes({'name':volume.name,'size':volume.size()}) if platform.system()!='Windows' else None)
                array=volume.array()
            else: array=volume
            record['shape']=list(array.shape)
            record['unwritten_background_zero']=bool(np.all(array[:64,:,:]==0) and np.all(array[:,:64,:]==0) and np.all(array[:,:,:64]==0))
            record['written_nonzero_count']=int(np.count_nonzero(array[64:,64:,64:]))
            result['buffers'].append(record)
            del array
            if shared:
                volume.__exit__(); manager.remove_termed(volume); volume.close()
    result['fresh_segment_names_distinct']=result['buffers'][1]['name']!=result['buffers'][2]['name']
    output.write_text(json.dumps(result,indent=2)+'\n')
    shutil.rmtree(cache_root)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
