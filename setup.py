from setuptools import setup

import os
import sys
import subprocess

cwd = os.path.abspath(os.path.dirname(__file__))

print "Entering", os.path.join(cwd, 'neumann')
os.chdir('neumann')
if os.name == 'nt': 
    print "Running makefile.bat"
    subprocess.call(['makefile.bat'])
else: 
    print "Running make"
    subprocess.call(['make'])
print "Going back to", cwd
os.chdir('..')

setup(name='neumann',
      version='0.0',
      description='Package for solving master equations',
      #url='http://github.com/xxxx/neumann',
      author='Gediminas Kirsanskas',
      author_email='gediminas.kirsanskas@teorfys.lu.se',
      license='MIT',
      packages=['neumann'],      
      package_data={'neumann': ['*.pyx', '*.c', '*.pyd', '*.o', '*.so',
                                os.path.join('fortran', '*.*'), 
                                os.path.join('quadpack', '*.*')]},
      zip_safe=False)
