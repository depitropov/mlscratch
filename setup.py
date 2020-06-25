from setuptools import setup

setup(name='mlscratch',
      version='0.0.1',
      description='A simple library implementing ml algrotihms from scratch.',
      url='https://github.com/depitropov/mlscratch',
      author='Dimitar Epitropov',
      author_email='depitropov@gmail.com',
      license='MIT',
      packages=['mlscratch'],
      zip_safe=False,
      install_requires=[
          'numpy'
      ])
