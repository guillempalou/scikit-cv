.PHONY: all clean test

all:
	python setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

test:
	python -c "import skcv, sys, io; sys.exit(skcv.test_verbose())"

doctest:
	python -c "import skcv, sys, io; sys.exit(skcv.doctest_verbose())"

coverage:
	nosetests skcv --with-coverage --cover-package=skcv
