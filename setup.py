from setuptools import setup, find_packages

from aikit import __meta__ as META


DISTNAME = 'aikit'
DESCRIPTION = 'An automated machine learning framework'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = META.__author__
MAINTAINER_EMAIL = ''
URL = META.__website__
DOWNLOAD_URL = 'https://pypi.org/project/aikit/#files'
LICENSE = META.__license__

VERSION = META.__version__


def parse_requirements(req_file):
    with open(req_file) as fp:
        _requires = fp.read()
    return _requires


# Get dependencies from requirement files
SETUP_REQUIRES = ['setuptools', 'setuptools-git', 'wheel']
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
                    classifiers=META.__classifiers__,
                    install_requires=INSTALL_REQUIRES,
                    setup_requires=SETUP_REQUIRES,
                    packages=find_packages(exclude=["tests", "tests.*"]))

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
