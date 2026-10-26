from pathlib import Path
import argparse, json, os, hashlib, sys
import numpy as np
from cloudvolume import CloudVolume, Bbox

parser=argparse.ArgumentParser(description='Prepare local synthetic and bounded real Octopus fixtures.')
parser.add_argument('--root',type=Path,default=Path(__file__).parent)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--prior-campaign',type=Path,required=True)
args=parser.parse_args()
root = args.root.resolve()
os.environ['CLOUD_VOLUME_DIR'] = str(root / 'cloudvolume')
os.environ['CLOUD_FILES_DIR'] = str(root / 'cloudfiles')
sys.path.insert(0, str(args.baseline/'python/benchmarks'))
from benchmark_run_statistics import make_fixture, write_json

synthetic = root / 'synthetic'
synthetic.mkdir(exist_ok=True)
fixture = make_fixture(synthetic, 256, 4)
write_json(synthetic / 'fixture.json', fixture)

real = root / 'representative'
real.mkdir(exist_ok=True)
campaign = args.prior_campaign
annotations = json.loads((campaign / 'inputs/salivary-annotations.json').read_text())
layer = next(layer for layer in annotations['layers'] if layer['name'] == 'Posterior Salivary Duct 2')
layer['annotations'] = layer['annotations'][:6]
points = np.array([item['point'] for item in layer['annotations']])
low = (np.floor((points.min(axis=0) - 128) / 96) * 96).astype(int)
high = (np.ceil((points.max(axis=0) + 128) / 96) * 96).astype(int)
url = annotations['layers'][0]['source']['url'].removeprefix('precomputed://')
source = CloudVolume(url, cache=str(campaign / 'cache-validated'), progress=False, parallel=1)
print('Downloading representative crop:', low.tolist(), high.tolist(), flush=True)
pixels = source.download(Bbox(low, high))
info = CloudVolume.create_new_info(num_channels=1, layer_type='image', data_type=str(pixels.dtype),
    encoding='raw', resolution=[700,700,700], voxel_offset=low.tolist(), chunk_size=[64,64,64],
    volume_size=(high-low).tolist())
cv = CloudVolume('file://' + str(real / 'source'), info=info, cache=False, compress=False, progress=False)
cv.commit_info()
cv[low[0]:high[0],low[1]:high[1],low[2]:high[2]] = pixels
write_json(real / 'annotations.json', {'layers':[layer, {'type':'image','name':'local','source':'PLACEHOLDER'}]})
options = json.loads((campaign / 'inputs/salivary-slice-options.json').read_text())
options.update(neuroglancer_image_layer='local', neuroglancer_json=str(real / 'annotations.json'))
write_json(real / 'fixture.json', {'slice_options': options,
    'source_url':url, 'annotation_path':'Posterior Salivary Duct 2', 'original_point_indices':list(range(6)),
    'source_shape':list(pixels.shape), 'source_dtype':str(pixels.dtype), 'source_voxel_bytes':pixels.nbytes,
    'source_pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
    'crop_min':low.tolist(),'crop_max':high.tolist(),
    'annotation_original_sha256':hashlib.sha256((campaign / 'inputs/salivary-annotations.json').read_bytes()).hexdigest()})
print('Fixture preparation complete', flush=True)
