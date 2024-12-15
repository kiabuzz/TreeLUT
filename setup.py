from setuptools import setup, find_packages

setup(
    name='treelut',
    version='1.0.0',
    author='Alireza Khataei, Kia Bazargan',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description='An Efficient Alternative to Deep Neural Networks for Inference Acceleration Using Gradient Boosted Decision Trees',
    url='https://github.com/kiabuzz/treelut',
    license='MIT',
    install_requires=['numpy', 'xgboost>=2.1.1']
)