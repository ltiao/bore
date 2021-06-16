#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "numpy==1.20.3",
    "scipy==1.6.3",
    "Cython==0.29.23",
    "ConfigSpace==0.4.18",
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', "scikit-learn"]

setup(
    author="Louis C. Tiao",
    author_email='louistiao@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Bayesian Optimization by Density-Ratio Estimation",
    install_requires=requirements,
    extras_require={"hpbandster": ["hpbandster==0.7.4"],
                    "tf": ["tensorflow==2.5.0"],
                    "tf-gpu": ["tensorflow-gpu==2.5.0"]},
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='bore',
    name='bore',
    packages=find_packages(include=['bore', 'bore.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ltiao/bore',
    version='1.4.0',
    zip_safe=False,
)
