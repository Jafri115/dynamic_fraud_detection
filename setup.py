from setuptools import setup, find_packages

def get_requirements(filename):
    with open(filename, encoding='utf-8-sig') as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith('--') and not line.startswith('#') and line != '-e .'
        ]

setup(
    name='wiki-fraud-detection',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=get_requirements('requirements.txt'),
    author='wasif',
    author_email='swasifmurtaza@gmail.com',
)
