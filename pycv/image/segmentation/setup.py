
#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    """ Configuration for the directory
    Parameters
    ----------
    parent_package: string
        name of the parent package
    top_path: string
        root of the library
    """
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('segmentation', parent_package, top_path)

    #cython(['_felzenszwalb_cy.pyx'], working_path=base_path)
    #config.add_extension('_felzenszwalb_cy', sources=['_felzenszwalb_cy.c'],
    #                     include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='pycv Developers',
          maintainer_email='scikit-image@googlegroups.com',
          description='Image Segmentation Algorithms',
          url='https://github.com/guillempalou/pycv',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
