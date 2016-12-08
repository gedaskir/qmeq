import os
#import sys
import subprocess
import shutil
from glob import glob

dirs = ['./.cache',
        './build',
        './dist',
        './docs/build',
        './qmeq.egg-info',
        './qmeq/__pycache__',
        './qmeq/build',
        './qmeq/approach/__pycache__',
        './qmeq/tests/__pycache__',
        './qmeq/tests_/__pycache__']

for dr in dirs:
    try: shutil.rmtree(dr)
    except: pass

files = ( glob('./qmeq/*.pyd') + glob('./qmeq/approach/*.pyd')
         +glob('./qmeq/*.pyc') + glob('./qmeq/approach/*.pyc')
         +glob('./qmeq/*.c') + glob('./qmeq/approach/*.c')
         +glob('./qmeq/*.html') + glob('./qmeq/approach/*.html') )

for f in files:
    os.remove(f)
