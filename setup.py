from setuptools import setup, Extension, Command
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

setup(name='tinker_mass_function',
      install_requires=['numpy','scipy','cosmocalc'],
      version='1.0',
      py_modules=['tinker_mass_function','TMF_extras'],
      description='An implementation of the Tinker2010 mass function.',
      long_description=read('README.md'),
      author='Tom McClintock',
      author_email='tmcclintock@email.arizona.edu',
      url='https://github.com/tmcclintock/Mass-Function',
      cmdclass={'clean': CleanCommand})
