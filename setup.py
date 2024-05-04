import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="PyDDC",
  version="1.0",
  author="Sayan Sen, Scott K. Hansen",
  author_email='sayan@bgu.ac.il',
  description="Simulates density-driven convection of single phase CO2--brine mixture",
  license_files=('LICENSE.txt',),
  long_description=long_description,
  long_description_content_type="text/markdown",
  packages=setuptools.find_packages(),
  url='https://gitlab.com/skhansen_researchgroup/PyDDC.git',
  install_requires=['numpy','matplotlib','scipy', 'gstools', 'tqdm', 'h5py'],
)