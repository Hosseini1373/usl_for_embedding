from setuptools import setup, find_packages

setup(
    name='usl_embedding',
    version='0.1.0',
    author='Ahmad Hosseini, ZHAW',
    author_email='s.ahmad.hosseini94@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy', 'pandas', 'scikit-learn', 'torch',
    ],
    python_requires='>=3.6',
    description='Implements the USL Technique for embeddings of any data, be it text, image, etc.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Hosseini1373/usl_for_embedding',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'usl_for_embedding=src.main:main'
        ],
    },
)
