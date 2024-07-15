import os
import json
from setuptools import setup, find_packages


def get_requirements_to_install():
    __curr_location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    requirements_txt_file_as_str = f"{__curr_location__}/requirements.txt"
    with open(requirements_txt_file_as_str, 'r') as reqfile:
        libs = reqfile.readlines()
        for i in range(len(libs)):
            libs[i] = libs[i].replace('\n', '')
    return libs


version_json_path = 'version.json'
version_json = json.load(open(version_json_path))
version = ".".join([str(version_json[i]) for i in ['MAJOR', 'MINOR', 'MICRO']])

setup(
    name='llama3-playground',
    version=version,
    description='',
    long_description='',
    install_requires=get_requirements_to_install(),
    author='Amith Koujalgi',
    author_email='koujalgi.amith@gmail.com',
    packages=find_packages(
        include=[
            'llama3_playground',
            'llama3_playground.*',
        ]),
    py_modules=[
        'llama3_playground.core',
        'llama3_playground.server'
    ],
    entry_points={
        'console_scripts': [
            'playground = llama3_playground.cli:main'
        ],
    },
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ]
)
