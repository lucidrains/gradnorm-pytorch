from setuptools import setup, find_packages

setup(
  name = 'gradnorm-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.26',
  license='MIT',
  description = 'GradNorm - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/gradnorm-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'loss balancing',
    'gradient normalization'
  ],
  install_requires=[
    'accelerate',
    'beartype',
    'einops>=0.7.0',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
