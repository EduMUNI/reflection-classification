#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="reflection_classification",
    version='0.1',
    description="Classification of reflection in reflective diaries.",
    long_description="Utilities for preprocessing and classification of sentences "
                     "in candidate teachers' reflective diaries",
    classifiers=[],
    author="To Be Added",
    author_email="tobeadded@tobeadded.com",
    url="gitlab.com",
    license="MIT",
    packages=find_packages(include=["reflection_classification"]),
    use_scm_version={"write_to": ".version", "write_to_template": "{version}\n"},
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        "torch>=1.7",
        "transformers==4.2.1",
        "sentencepiece==0.1.95",
        "protobuf==3.14.0",
        "gensim==3.8.3",
        "scikit-learn==0.23.2",
        "pandas>=1.1.5"
    ],
    # package_data={"reflexive_diaries": ["annotations/*", "models/configs/*"]},
)
