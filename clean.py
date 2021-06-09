import os
import shutil
from glob import glob

dirs = [
    './.cache',
    './.pytest_cache',
    './build',
    './dist',
    './docs/build',
    './qmeq.egg-info',
    './qmeq/build',
    ]

for directory, _, _ in os.walk('./'):
    if '__pycache__' in directory:
        dirs.append(directory)

for dr in dirs:
    if os.path.exists(dr):
        shutil.rmtree(dr)
    else:
        pass

dirs = [
    './qmeq/',
    './qmeq/approach/',
    './qmeq/approach/base/',
    './qmeq/approach/elph/',
    './qmeq/builder/',
    './qmeq/specfunc/',
    './qmeq/tests/',
    './qmeq/wrappers/',
    ]

exts = ['*.o', '*.so', '*.pyd', '*.dll', '*.pyc', '*.c', '*.html']

files = []
for dr in dirs:
    for ext in exts:
        files += glob(dr+ext)

for f in files:
    os.remove(f)
