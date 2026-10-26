"""Orchestrate existing benchmark.run_once; no new timer or RAM collector."""
from pathlib import Path
import argparse, importlib.util, importlib.metadata, json, os, hashlib, platform, statistics, sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'

def load_harness(repo, name):
    path = repo / 'python/benchmarks/benchmark_run_statistics.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def digest_tree(repo):
    result = {}
    for path in sorted((repo / 'python/ouroboros').rglob('*.py')):
        result[path.relative_to(repo / 'python').as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result

def output_evidence(path, harness):
    import tifffile
    result = harness.tiff_evidence(path)
    with tifffile.TiffFile(path) as tif:
        metadata=[]
        for page in tif.pages:
            metadata.append({tag.name: tag.value for tag in page.tags.values()
                             if tag.name not in ('StripOffsets','StripByteCounts','TileOffsets','TileByteCounts')})
        result.update({'series_shape': list(tif.series[0].shape), 'axes':tif.series[0].axes,
                'metadata_sha256':hashlib.sha256(
                    json.dumps(metadata,sort_keys=True,default=str).encode()).hexdigest(),
                'first_page_metadata':metadata[0]})
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--modified',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--platform-label',required=True)
    args=parser.parse_args()
    folder=(args.output_root or args.root) / args.platform_label
    folder.mkdir(exist_ok=True)
    harnesses={label:load_harness(repo,label) for label,repo in [('baseline',args.baseline),('modified',args.modified)]}
    result={'platform':platform.platform(),'python':sys.version,'versions':{name:importlib.metadata.version(name)
        for name in ('numpy','scipy','tifffile','cloud-volume','psutil')},
        'code':{name:digest_tree(repo) for name,repo in [('baseline',args.baseline),('modified',args.modified)]},
        'harness_sha256':hashlib.sha256((args.baseline/'python/benchmarks/benchmark_run_statistics.py').read_bytes()).hexdigest(),
        'protocol':'Three measured verbose runs of each revision per dataset, alternating before/after order; one unmeasured verbose warmup of each revision. Four workers, one BLAS thread. Existing benchmark.run_once pipeline/data-step timing and canonical RAM collector. TIFF hashing outside timers. Warm local HTTP source and OS caches. Bounded real-data subset, not full historical campaign.',
        'datasets':{}}
    for dataset in ('synthetic','representative'):
        source=args.root/dataset
        fixture=json.loads((source/'fixture.json').read_text())
        annotations=json.loads((source/'annotations.json').read_text())
        annotations['layers']=[layer for layer in annotations['layers'] if layer.get('type')=='annotation']+[
            {'type':'image','name':'local','source':args.source_root+'/'+dataset+'/source'}]
        annotation_path=folder/(dataset+'-annotations.json')
        annotation_path.write_text(json.dumps(annotations))
        options=dict(fixture['slice_options'],neuroglancer_json=str(annotation_path))
        for key,dirname in [('CLOUD_VOLUME_DIR','cloudvolume'),('CLOUD_VOLUME_CACHE_DIR','cloudvolume-cache'),('CLOUD_FILES_DIR','cloudfiles')]:
            os.environ[key]=str(folder/dirname)
        data={'fixture':fixture,'runs':[],'warmups':[]}
        result['datasets'][dataset]=data
        for pair in (-1,0,1,2):
            labels=('baseline','modified') if pair % 2 == 0 or pair == -1 else ('modified','baseline')
            for label in labels:
                runlabel=f'{dataset}-{pair}-{label}'
                print('START',args.platform_label,runlabel,flush=True)
                measured,output=harnesses[label].run_once(folder,'slice',options,True,4,runlabel,300,'pipeline')
                measured['revision']=label
                measured['output']=output_evidence(output,harnesses[label])
                measured['statistics']=[json.loads((folder/path).read_text()) for path in measured['statistics_files']]
                data['warmups' if pair == -1 else 'runs'].append(measured)
                (folder/'results.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
                print('DONE',runlabel,measured['data_operation_elapsed_seconds'],measured['memory']['sampled_peak_bytes'],flush=True)
        for label in ('baseline','modified'):
            selected=[run for run in data['runs'] if run['revision']==label]
            data[label]={field:statistics.median(run[field] for run in selected)
                         for field in ('pipeline_elapsed_seconds','data_operation_elapsed_seconds')}
            data[label]['sampled_peak_bytes']=statistics.median(run['memory']['sampled_peak_bytes'] for run in selected)
        data['all_output_pixels_identical']=len({run['output']['pixel_sha256'] for run in data['runs']})==1
        data['all_output_metadata_identical']=len({run['output']['metadata_sha256'] for run in data['runs']})==1
        (folder/'results.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    for label,repo in [('baseline',args.baseline),('modified',args.modified)]:
        if result['code'][label] != digest_tree(repo):
            raise RuntimeError('Production code changed during measurements')
    print('COMPLETE',folder/'results.json',flush=True)

if __name__=='__main__': main()
