from setuptools import setup, find_namespace_packages


def _get_version():
    with open('bbo/__init__.py') as f:
        for line in f:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)
                return g['__version__']


with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name='bbo',
    version=_get_version(),
    description='Black-box optimization library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='songl',
    author_email='songl@lamda.nju.edu.cn',
    url='https://github.com/songlei00/bbo',
    packages=find_namespace_packages(include=['bbo*']),
    include_package_data=True,
    install_requires=required,
    python_requires='>=3.10',
    license='Apache License 2.0',
)