from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

setup(
    name = 'neural-boost',
    version = '0.0.5',
    keywords='inference, machine learning, x86, x86_64, avx512, neural network,',
    description = 'Neural Boost targeting to boost inference performance.',
    long_description = 'Neural Boost targeting to boost inference performance.',
    license = 'Apache 2.0',
    url = 'https://github.com/neural-boost/neural-boost',
    author = 'Neural Boost',
    author_email = 'neural_boost@163.com',
    python_requires = '>=3.6',
    packages = find_packages(include=['neural_boost']),
    package_data = {
        'neural_boost': ['*.so']
    },
    # distclass = BinaryDistribution,
    platforms = 'x86_64',
)

