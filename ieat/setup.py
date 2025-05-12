from distutils.core import setup

setup(
    name='ieat',
    version='1.0',
    packages=['ieat',],

    install_requires=[
    	'transformers',
        'torch',
        'numpy',
        'matplotlib',
        'opencv-python',
        'tensorflow',
        'torchvision'
    ]
)