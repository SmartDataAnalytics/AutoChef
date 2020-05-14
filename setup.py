
from setuptools import setup, find_packages
from os import path
from io import open

setup(

    name='autochef',  # Required

    version='1.0.0',  # Required

    description='AutoChef Recipe Generator',

    author='Jonas Weinz',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='jo.we93@gmail.com',  # Optional

    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={'': 'src/'},  # Optional

    packages = ['autochef','autochef.data','autochef.db','autochef.EvolutionaryAlgorithm','autochef.RecipeAnalysis','autochef.Tagging'],

    include_package_data=True,



    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. If you
    # do not support Python 2, you can simplify this to '>=3.5' or similar, see
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.6',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'jupyterlab',
        'dill',
        'graphviz',
        'nltk',
        'networkx',
        'pandas',
        'PyMySQL',
        'pygraphviz',
        'scikit-learn',
        'scipy',
        'numpy',
        'matplotlib',
        'word2vec',
        'conllu',
        'python-crfsuite',
        'plotly',
        'tqdm',
        'ipywidgets'
    ]  # Optional
)