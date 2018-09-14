from setuptools import setup

import aikit


DISTNAME = 'aikit'
DESCRIPTION = 'An automated machine learning framework'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = ''
MAINTAINER_EMAIL = ''
URL = 'https://github.com/societe-generale/aikit'
DOWNLOAD_URL = 'https://pypi.org/project/aikit/#files'
LICENSE = 'MIT'

VERSION = aikit.__version__


def parse_requirements(req_file):
    with open(req_file) as fp:
        _requires = fp.read()
    return _requires


# Get dependencies from requirement files
SETUP_REQUIRES = []
INSTALL_REQUIRES = parse_requirements('requirements.txt')


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.4',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 'Programming Language :: Python :: 3.7',
                                 ('Programming Language :: Python :: '
                                  'Implementation :: CPython'),
                                 ('Programming Language :: Python :: '
                                  'Implementation :: PyPy')
                                 ],
                    install_requires=INSTALL_REQUIRES,
                    setup_requires=SETUP_REQUIRES)

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
