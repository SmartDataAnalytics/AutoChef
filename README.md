# AutoChef

An Evolutionary Algortihm to create new cooking recipes

## requirements

* an envrironment with python3.7.
* For python3.8 or newer the adjacency matrices have to be created manually from the 1M Recipe Dataset as described in the package details below

## installation

* clone this directory: `https://github.com/SmartDataAnalytics/AutoChef.git`
* inside the cloned project, extract the pregenerated compressed adjacency matrices with

```
cd src/autochef/RecipeAnalysis && tar -xf adjacency_matrices.tar.xz
```

* install the cloned package with pip:

```bash
pip install wheel
pip install ./AutoChef
```

## Example Usage:

just perform the following in a jupyter notebook (same as [Example.ipynb](Example.ipynb))

**Import autochef's Evolutionary Algorithm**

```python
import autochef
from autochef.EvolutionaryAlgorithm import EvolutionaryAlgorithm as EA
```

**Setup parameters**

```python
# define the set of input ingredients and it's subset of main ingredients

all_ingredients = ["bread", "butter"]
main_ingredients = ["bread"]

# define range of additional ingredients
min_add = 4
max_add = 13

# population size:
n_pop = 50

# mutations rate (given in number of node mutations for each tree)
mutations = 2

# if pairwise competition is false, just the best n_pop/2 individuals survive
pairwise_competition = True
```

**build initial population**

```python
p = EA.Population(
    all_ingredients,
    main_ingredients,
    min_additional=min_add,
    max_additional=max_add,
    n_population = n_pop,
    mutations=mutations,
    pairwise_competition=pairwise_competition
)
```

**run over a given number of generations**

```python
# run over 10 cycles
fitness_logs = p.run(10)
```

**plot best 3 individuals**

```python
p.plot_population(n_best=3)
```

## Package Details

if you want to modify package data, retrain models or recreate adjacency matrices: here are some notes about the structure of this package (in `/src/autochef/`):

### `./data`

the `./data` folder contains data files used in this thesis.

**NOTE**: due to the [1M_recipes](http://pic2recipe.csail.mit.edu/) licensing, i am not allowed to redistribute it's data. Because of this the data has to be downloaded manually and put indside the folder [`./data/1M_recipes`](./src/autochef/data/1M_recipes/).

### `./db`

Tools to create a mariadb recipe database from the 1M_recipes dataset and working with it. Necessary if you want to recreate adjacency matrices. To create a docker image with a database containing the 1M_recipe dataset execute the following Notebooks:

* [`./db/create_database_docker.ipynb`](src/autochef/db/create_database_docker.ipynb)
  
  * creating a docker image with mariadb

* [`./db/create_database.ipynb`](src/autochef/db/create_database.ipynb)
  
  * filling the created database with the 1M_recipe dataset

### `./Tagging`

(needs the 1M_recipe dataset placed inside the data folder, see above)

Tools to retrain the crf classifier. The package ships already a trained one.

To generate a train dataset for the crf classifier run the python script:

* [`recipe_conllu_generator.py`](src/autochef/Tagging/recipe_conllu_generator.py)

it will generate enumerated `*.conllu` files which can then used to train the classifier with 

* [`CRF_training.ipynb`](src/autochef/Tagging/CRF_training.ipynb)



### `./RecipeAnalysis`

(needs the maria database set up and filled with data)

tools to generate adjacency Matrices. To generate them run the Notebooks

* [`MatrixGeneration.ipynb`](src/autochef/RecipeAnalysis/MatrixGeneration.ipynb)

and 

* [`AdjacencyMatrixRefinement.ipynb`](src/autochef/RecipeAnalysis/AdjacencyMatrixRefinement.ipynb)



the matrices are then serialized and saved with [dill]([dill · PyPI](https://pypi.org/project/dill/)) and can be loaded with dill again (which is done in the Evolutionary algorithm)



### `./EvolutionaryAlgorithm`

needs adjaceny matrices set up in the recipe analysis folder. The package already ships with pregenerated ones, so that you can try the evolutionary algorithm without setting up the 1M recipe database if you only want to run this part. Core Part is the `EvolutionaryAlgorithm.py` module.
