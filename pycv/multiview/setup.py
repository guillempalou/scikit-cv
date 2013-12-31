import os

def configuration(parent_package='', top_path=None):
    """ Configuration for the directory
    Parameters
    ----------
    parent_package: string
        name of the parent package
    top_path: string
        root of the library
    """
    from numpy.distutils.misc_util import Configuration

    config = Configuration('multiview', parent_package, top_path)

    config.add_subpackage('autocalibration')
    config.add_subpackage('n_views')
    config.add_subpackage('two_views')

    def add_test_directories(arg, dirname, fnames):
        """ Adds tests directories

        Parameters
        ----------
        dirname: string
            path of the tests

        """
        if dirname.split(os.path.sep)[-1] == 'tests':
            config.add_data_dir(dirname)

    # Add test directories as data directories
    from os.path import isdir, dirname, join
    rel_isdir = lambda d: isdir(join(curpath, d))

    curpath = join(dirname(__file__), './')
    subdirs = [join(d, 'tests') for d in os.listdir(curpath) if rel_isdir(d)]
    subdirs = [d for d in subdirs if rel_isdir(d)]
    for test_dir in subdirs:
        config.add_data_dir(test_dir)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)


