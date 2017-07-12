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

files = (glob('./qmeq/*.o')    + glob('./qmeq/approach/*.o')
        +glob('./qmeq/*.so')   + glob('./qmeq/approach/*.so')
        +glob('./qmeq/*.pyd')  + glob('./qmeq/approach/*.pyd')
        +glob('./qmeq/*.dll')  + glob('./qmeq/approach/*.dll')
        +glob('./qmeq/*.pyc')  + glob('./qmeq/approach/*.pyc')
        +glob('./qmeq/*.c')    + glob('./qmeq/approach/*.c')
        +glob('./qmeq/*.html') + glob('./qmeq/approach/*.html')
        +glob('./qmeq/tests/*.pyc') )

for f in files:
    os.remove(f)
