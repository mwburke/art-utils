from distutils.core import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='art_utils',
      version='0.0.3',
      description='Python utilities for creating generative art',
      license='GPLv3',
      long_description=long_description,
      author='Matthew Burke',
      author_email='matthew.wesley.burke@gmail.com',
      url='https://github.com/mwburke/art-utils',
      packages=['art_utils'],
)
