import os
#import sys
import subprocess
import shutil
from glob import glob

dirs = ['./build',
        './dist',
        './qmeq.egg-info',
        './qmeq/build',
        './qmeq/docs/build']

for dr in dirs:
    try: shutil.rmtree(dr)
    except: pass

files = ( glob('./qmeq/*.pyd')
         +glob('./qmeq/*.pyc')
         +glob('./qmeq/*.c')
         +glob('./qmeq/fortran/*.o')
         +glob('./qmeq/quadpack/*.o')
         +glob('./qmeq/*.html') )

for f in files:
    os.remove(f)


