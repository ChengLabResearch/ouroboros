import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import psutil
from ouroboros.common import memory_statistics, memory_ipc

source = Path(memory_statistics.__file__).resolve().parents[1]
digest = hashlib.sha256()
files = {}
for path in sorted(source.rglob('*.py')):
    relative = path.relative_to(source.parent).as_posix()
    content = path.read_bytes()
    digest.update(relative.encode())
    digest.update(content)
    files[relative] = hashlib.sha256(content).hexdigest()
packages = sorted((distribution.metadata['Name'], distribution.version)
                  for distribution in importlib.metadata.distributions())
cgroups = {}
for name in ('cpu.max', 'cpuset.cpus.effective', 'memory.max', 'memory.high', 'memory.swap.max'):
    path = Path('/sys/fs/cgroup', name)
    cgroups[name] = path.read_text().strip() if path.exists() else None
print(json.dumps({
    'python': sys.version, 'executable': sys.executable,
    'execution_os': platform.platform(), 'logical_cpus': os.cpu_count(),
    'cpu_model': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')),
    'cpu_affinity': sorted(os.sched_getaffinity(0)),
    'physical_memory_bytes': psutil.virtual_memory().total,
    'shared_memory_filesystem': psutil.disk_usage('/dev/shm')._asdict(),
    'cgroup_limits': cgroups,
    'thread_environment': {key: os.environ.get(key) for key in
                           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
    'build_environment': {key: os.environ.get(key) for key in
                          ('OUR_ENV', 'OUROBOROS_BUILD_REVISION', 'OUROBOROS_BUILD_ID')},
    'production_module': str(memory_statistics.__file__),
    'production_python_sha256': digest.hexdigest(), 'production_files_sha256': files,
    'dependencies': packages,
    'dependency_resolution_sha256': hashlib.sha256(json.dumps(packages).encode()).hexdigest(),
    'identity_contract': memory_ipc.export_process_identity(psutil.Process()),
    'benchmark_python_sha256': hashlib.sha256(Path('/app/benchmarks/benchmark_run_statistics.py').read_bytes()).hexdigest(),
    'dev_shm_entries': sorted(path.name for path in Path('/dev/shm').iterdir()),
}, indent=2))
