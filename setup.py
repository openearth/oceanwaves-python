from setuptools import setup, find_packages

setup(
    name='oceanwaves',
    version='0.0',
    author='Bas Hoonhout',
    author_email='bas.hoonhout@deltares.nl',
    packages=find_packages(),
    description='A toolbox for ocean wave datasets',
    long_description=open('README.txt').read(),
    install_requires=[
        'numpy',
        'scipy',
        'xarray',
        'pyproj',
        'docopt',
    ],
)
