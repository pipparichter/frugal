import setuptools
import os

def get_requirements(path:str=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')):
    with open(path) as f:
        requirements = f.read().splitlines()
    return requirements

commands = ['prune=src.cli:prune', 'stats=src.cli:stats', 'library=src.cli:library', 'label=src.cli:label', 'ref=src.cli:ref', 'embed=src.cli:embed', 'train=src.cli:train', 'predict=src.cli:predict']

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