import setuptools
import os

def get_requirements(path:str=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')):
    with open(path) as f:
        requirements = f.read().splitlines()
    return requirements

setuptools.setup(
    name='tripy',
    version='0.1',    
    description='N/A',
    url='https://github.com/pipparichter/tripy',
    author='Philippa Richter',
    author_email='prichter@caltech.edu',
    packages=['src', 'src.files', 'src.tools'], 
    entry_points={'console_scripts':['build=src.cli:build']},
    install_requires=get_requirements())


# TODO: What exactly is an entry point?
# https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html 