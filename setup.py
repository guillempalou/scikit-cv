#! /usr/bin/env python

descr = """Scikit Computer Vision

Computer Vision algorithms based on Scikit-Image and Scikit-learn
Includes: image and video segmentation, optical flow, n-view geometry

"""

DISTNAME = 'skicit-cv'
DESCRIPTION = 'Computer Vision library for Python'
LONG_DESCRIPTION = descr
MAINTAINER = 'Guillem Palou'
MAINTAINER_EMAIL = 'guillem.palou@gmail.com'
URL = 'http://github.com/guillempalou/scikit-cv'
LICENSE = 'MIT'
DOWNLOAD_URL = 'http://github.com/guillempalou/scikit-cv'
VERSION = '0.1dev'
PYTHON_VERSION = (3, 3)
DEPENDENCIES = {
    'numpy': (1, 6),
    'Cython': (0, 17),
    'six': (1, 3),
    'skimage': (0, 9),
    'sklearn': (0, 14),
    'networkx': (1, 8)
    #'numpydoc': (0, 4)
}

import os
import sys
import re
import glob
import setuptools  # setuptools need to be imported before distutils
from distutils.core import setup, Extension
from Cython.Distutils import build_ext

# get the numpoy include directories
from numpy.distutils.misc_util import get_numpy_include_dirs


def configure_extensions():

    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    # search for all cython files and build them as modules
    # in the corresponding subpackage
    packages = setuptools.find_packages('.')

    exts = []

    for package in packages:

        working_path = os.path.join(*package.split('.'))
        pyx_paths = glob.glob(os.path.join(working_path, '*.pyx'))
        pyx_files = [path.split('/')[-1] for path in pyx_paths]

        for pyx_file in pyx_files:
            name = pyx_file[:-4]
            full_path = os.path.join(working_path, pyx_file)

            e = Extension(package + "." + name, [full_path],
                          include_dirs=get_numpy_include_dirs())

            exts.append(e)

    return exts


def write_version_py(filename='skcv/version.py'):
    template = ("# THIS FILE IS GENERATED FROM THE SCIKIT-CV SETUP.PY\n"
                "version='%s'\n"
                )

    vfile = open(os.path.join(os.path.dirname(__file__),
                              filename), 'w')

    try:
        vfile.write(template % VERSION)
    finally:
        vfile.close()


def get_package_version(package):
    version = []
    for version_attr in ('version', 'VERSION', '__version__'):
        if hasattr(package, version_attr) \
                and isinstance(getattr(package, version_attr), str):
            version_info = getattr(package, version_attr, '')
            for part in re.split('\D+', version_info):
                try:
                    version.append(int(part))
                except ValueError:
                    pass
    return tuple(version)


def check_requirements():
    if sys.version_info < PYTHON_VERSION:
        raise SystemExit('You need Python version %d.%d or later.' \
                         % PYTHON_VERSION)

    for package_name, min_version in DEPENDENCIES.items():
        dep_error = False
        try:
            package = __import__(package_name)
        except ImportError:
            dep_error = True
        else:
            package_version = get_package_version(package)
            if min_version > package_version:
                dep_error = True

        if dep_error:
            raise ImportError('You need `%s` version %d.%d or later.' \
                              % ((package_name, ) + min_version))


if __name__ == "__main__":
    check_requirements()

    write_version_py()

    extensions = configure_extensions()

    data_dirs = {'skcv': ['data/*']}

    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        url=URL,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version=VERSION,

        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: C++',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],


        packages=setuptools.find_packages(exclude=['doc']),
        package_data=data_dirs,

        ext_modules=extensions,
        include_package_data=True,
        cmdclass={'build_ext': build_ext},

        zip_safe=False,  # the package can run out of an .egg file
    )


