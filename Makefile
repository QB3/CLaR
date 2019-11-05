# simple makefile to simplify repetetive build env management tasks under posix

PYTHON ?= python
PYTESTS ?= pytest

CTAGS ?= ctags

all: clean inplace test

clean-pyc:
	find . -name "*.pyc" | xargs rm -f
	find . -name "__pycache__" | xargs rm -rf

clean: clean-pyc

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code:
	$(PYTESTS) clar

test-doc:
	$(PYTESTS) $(shell find doc -name '*.rst' | sort)

test-coverage:
	rm -rf coverage .coverage
	$(PYTESTS) clar --cov=clar --cov-report html:coverage

test: test-code test-doc test-manifest

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'


.PHONY : doc-plot
doc-plot:
	make -C doc html

.PHONY : doc
doc:
	make -C doc html-noplot

test-manifest:
	check-manifest --ignore doc,celer/tests;
