from setuptools import setup, Extension, Command, find_packages
import os,sys,glob

def read(fname):
    """Quickly read in the README.md file."""
    return open(os.path.join(os.path.dirname(__file__),fname)).read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

dist = setup(name='aemHMF',
             install_requires=['numpy','scipy','cosmocalc','matplotlib','george'],
             version='1.0',
             packages=find_packages(),
             include_package_data=True,
             description='Emulator for the halo mass function using the Aemulus simulation suite.',
             long_description=read('README.md'),
             author='Tom McClintock',
             author_email='tmcclintock@email.arizona.edu',
             url='https://github.com/tmcclintock/Aemulus_HMF',
             cmdclass={'clean': CleanCommand},
             setup_requires=['pytest_runner'],
             tests_require=['pytest']
)
