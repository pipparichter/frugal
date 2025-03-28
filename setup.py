import setuptools
import os

def get_requirements(path:str=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')):
    with open(path) as f:
        requirements = f.read().splitlines()
    return requirements

commands = list()
commands += ['prune=src.cli:prune']
commands += ['stats=src.cli:stats']
commands += ['library=src.cli:library']
commands += ['ref=src.cli:ref']
commands += ['embed=src.cli:embed']
commands += ['train=src.cli:train']
commands += ['predict=src.cli:predict']
commands += ['cluster=src.cli:cluster']

setuptools.setup(
    name='frugal',
    version='0.1',    
    description='N/A',
    url='https://github.com/pipparichter/tripy',
    author='Philippa Richter',
    author_email='prichter@caltech.edu',
    packages=['src', 'src.files', 'src.tools', 'src.embed', 'src.embed.embedders'], 
    entry_points={'console_scripts':commands},
    install_requires=get_requirements())


# TODO: What exactly is an entry point?
# https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html 