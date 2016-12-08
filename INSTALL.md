Installation of QmeQ
====================

For now QmeQ can be installed by building it from source. To be able to use and
build QmeQ you need to have:

* [Python][Python] 2.7 or 3.3 with [setuptools][setuptools] installed,
* [Cython][Cython] 0.22 and a C compiler compatible with it,
* [NumPy][NumPy] package,
* [SciPy][Scipy] package.

The indicated versions are the minimal required versions. Optionally, such
packages are used:

* [Matplotlib][Matplotlib] for plotting,
* [Jupyter][Jupyter] for the tutorials.

An easy way to obtain the above packages is by using Python package manager
[pip][pip]. After [setting up pip][setpip] the above packages can be obtained
by using the following command

```bash
$ pip install cython, numpy, scipy, matplotlib, jupyter
```

QmeQ is installed by going into the [downloaded source][qmeqsrc] directory and
running

```bash
$ python setup.py install
```

We note that the binaries **pip** and **python** have to be in the system path.

C compiler
----------

For **Linux** and **Mac** we recommend to use the C compiler in the conventional
[gcc][gcc] suite, which will be recognized by Cython. For **Windows** the
**Visual Studio** or **Windows SDK C/C++** compiler can be used and more
instructions how to setup these compilers to work with Cython are available
[here][cext].

The 32-bit gcc compiler can be used on Windows with 32-bit Python 2.7. This can
be achieved by using the gcc compiler in [MinGW][mingw] suite by adding

```
[build]
compiler = mingw32
```

to *'path to python'\Lib\distutils\distutils.cfg*. If the file does not exist
then simply create it. We note that instead of using the actual [MinGW][mingw]
it is possible to use Python package [Mingwpy], which can be obtained using

```bash
$ pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy
```

NumPy and ATLAS/OpenBLAS/MKL
----------------------------

For a good performance of the calculations NumPy needs to be linked to
so-called ATLAS, OpenBLAS, or MKL libraries. To check if NumPy is linked go to
Python interpreter and write

```python
import numpy
numpy.show_config()
```

If all of the entries like **atlas\_info**, **openblas\_info**, or **mkl\_info**
says **NOT AVAILABLE** then it is likely that your NumPy does not perform well.

For Windows the NumPy and SciPy libraries linked to MKL can be obtained from
[Unofficial Windows Binaries for Python Extension Packages][cgohlke] by
Christoph Gohlke.

Tests
-----

To run the [tests][qmeqtest] included with QmeQ we use

* [py.test][pytest] testing framework.

To install it run

```bash
$ pip install pytest
```

Then the tests can be performed by calling

```bash
$ cd 'path to qmeq source'/qmeq
$ pytest tests
```

Documentation
-------------

For now QmeQ contains just the [documentation][qmeqdocs] generated from
docstrings in the source code. The QmeQ documentation can be generated in
**html**, **latex**, and other formats using

* [Sphinx][Sphinx] 1.3 package,
* [sphinx-rtd-theme][srtdt] Read the Docs Sphinx theme.

To install the above packages run

```bash
$ pip install sphinx, sphinx-rtd-theme
```

For example, to generate the documentation in **html** format run

```bash
$ cd 'path to qmeq source'/docs
$ make html
```

The generated documentation should be in
*'path to qmeq source'/docs/build/index.html*

[Python]: http://www.python.org
[Cython]: http://cython.org
[NumPy]: http://www.numpy.org
[SciPy]: http://www.scipy.org
[Matplotlib]: http://matplotlib.org
[Jupyter]: http://jupyter.org
[Sphinx]: http://www.sphinx-doc.org
[pytest]: http://doc.pytest.org

[setuptools]: http://setuptools.readthedocs.io
[pip]: http://pip.pypa.io
[setpip]: http://pip.pypa.io/en/stable/installing
[gcc]: http://gcc.gnu.org
[cext]: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
[mingw]: http://www.mingw.org
[mingwpy]: https://mingwpy.github.io
[cgohlke]: http://www.lfd.uci.edu/~gohlke/pythonlibs
[srtdt]: https://github.com/snide/sphinx_rtd_theme

[qmeqdocs]: http://github.com/gedaskir/qmeq/tree/master/docs
[qmeqsrc]: http://github.com/gedaskir/qmeq/archive/master.zip
[qmeqtest]: http://github.com/gedaskir/qmeq/tree/master/qmeq/tests
