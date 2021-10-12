
from setuptools import setup
import os
# read the contents of your README file
with open('README.md') as f:
    long_description = f.read()
print(long_description)
print("""
*******************************************************************
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
(c) 2020 Haotian Teng
*******************************************************************
""")

install_requires=[
'numpy==1.18.5',
'sklearn',
'pandas==1.0.5',
'matplotlib==3.2.2',
'xlrd==1.2.0',
'seaborn==0.10.1'
]
extras_require={
  "pytorch": ["pytorch>1.1"]
}
exec(open('fict/_version.py').read())
setup(
  name = 'fict',
  packages = ['fict'], 
  version = __version__,
  include_package_data=True,
  description = 'A .',
  author = 'Haotian Teng, Ye Yuan, Ziv Bar-Joseph',
  author_email = 'havens.teng@gmail.com',
  url = 'https://github.com/haotianteng/fict', 
  download_url = 'https://github.com/haotianteng/fict/archive/1.0.tar.gz', 
  keywords = ['Spatial Transcriptomics','Probabilistic Graphical Model','Clustering'], 
  license="MPL 2.0",
  classifiers = ['License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)'],
  install_requires=install_requires,
  extras_require=extras_require,
  long_description=long_description,
  entry_points={'console_scripts':['fict=fict.run:main'],},
  long_description_content_type='text/markdown',
)
