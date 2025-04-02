import setuptools
import os

def get_requirements(path:str=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')):
    with open(path) as f:
        requirements = f.read().splitlines()
    return requirements

commands = list()
commands += ['prune=src.cli:prune']
commands += ['library=src.cli:library']
commands += ['embed=src.cli:embed']
commands += ['model=src.cli:model']
commands += ['cluster=src.cli:cluster']
commands += ['train_test_split=src.cli:train_test_split']

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