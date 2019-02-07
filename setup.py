import os

from distutils.core import setup


descr = 'Concomitant Lasso with repetitions'

version = None
with open(os.path.join('clar', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'clar'
DESCRIPTION = descr
MAINTAINER = 'Quentin Bertrand'
MAINTAINER_EMAIL = 'quentin.bertrand@inria.fr'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/QB3/clar.git'
VERSION = version
# URL = 'https://qb3.github.io/clar'

setup(name='clar',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.rst').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      # url=URL,
      download_url=DOWNLOAD_URL,
      packages=['clar'],
      )
