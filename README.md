# Execution instructions

In this file you can find the instructions to run the 
experiments in order to verify the results, and the correct functioning,
of the implemented algorithms.

## Installation

### Get the source code

First, clone the [Project repository](https://github.com/Luca97Cuomo/sna-project) from GitHub into your system.

```bash
git clone https://github.com/Luca97Cuomo/sna-project
```

### Install Dependencies

A `requirements.txt` file is provided to install the dependencies.

It is recommended, not necessary, to install the 
dependencies into a 
[python virtual environment](https://docs.python.org/3/library/venv.html).

> **Python version:** The code has been tested only with `Python 3.9.5` and there is a bug
> in the `logger` if you execute the code with a Python version lower than 3.9.

First, create the virtual environment.

```bash

python3 -m venv venv

```

From the same folder you executed the previous command, activate the Python environment.

```bash

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate  # sh, bash, or zsh
    
```

> To deactivate the virtual environment, simply launch `deactivate` into the venv shell.

Now you should navigate into the project directory and 
install the code dependencies.

```bash

pip install -r requirements.txt

```

The requirements' installation makes our library code globally available to the main scripts that you
should use to run all the experiments. So it is essential for a correct execution.

## Execution

Different main files have been provided in order to test
the exercises functionalities. 
It is assumed that all the commands indicated 
in this file are executed within the active virtual environment and
preferably with the 3.9.5 version of Python and that the requirements are correctly installed.


### Mid term exercise 1 - Clustering algorithms

Execute the following command to test all the clustering algorithms.

```bash

python clustering_main.py

```

If you want to run only a specific clustering algorithm you can delete 
as many experiments as you want from the following lines into the 
`clustering_main.py` files.

```python

clustering = clustering_utils.Clustering([
        (clustering_algorithms.hierarchical_optimized, {"seed": 42, "desired_clusters": 4}),
        (clustering_algorithms.k_means, {"centrality_measure": None, "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10000,
                                         "centers": None}),
        (clustering_algorithms.k_means, {"centrality_measure": "degree_centrality", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10000,
                                         "centers": None}),
        (clustering_algorithms.k_means, {"centrality_measure": "closeness_centrality", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10,
                                         "centers": None}),
        (clustering_algorithms.k_means, {"centrality_measure": "nodes_betweenness_centrality", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10,
                                         "centers": None}),
        (clustering_algorithms.k_means, {"centrality_measure": "pagerank", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 50,
                                         "centers": None}),
        (clustering_algorithms.girvan_newman, {"centrality_measure": "edges_betweenness_centrality", "seed": 42,
                                               "k": 4, "optimized": True}),
        (clustering_algorithms.spectral, {"k": 4})],
        graph, true_clusters, draw_graph=False)

    clustering.evaluate_all()

```

For example if you want to test only the Hierarchical clustering algorithm you
can modify the previous code lines as follows:

```python

    clustering = clustering_utils.Clustering([
        (clustering_algorithms.hierarchical_optimized, {"seed": 42, "desired_clusters": 4})],
        graph, true_clusters, draw_graph=False)

    clustering.evaluate_all()

```

### Mid term exercise 2 - Centrality measure algorithms

Execute the following command to test all the centrality measure algorithms.

```bash

python centrality_main.py

```

If you want to test only some specific centrality measure you can edit
the `centrality_main.py` file in the same way of the Exercise 1.

For example, if you want to execute only the Page rank algorithm and the HITS algorithm
you can leave only the following lines into the `main` function:

```python
def main():
        
    graph, _ = utils.load_graph_and_clusters(PATH_TO_NODES, PATH_TO_EDGES)
    
        centrality = centrality_utils.Centrality([
            (basic_page_rank, {'max_iterations': 100, 'delta_rel': 0.6}),
            (naive_hits, {'max_iterations': 100}),
        ],
            graph)
    
        centrality.evaluate_all()

```

### Mid term exercise 3/4 - Mechanism design

Execute the following command to test all the classifier described into the 2.3 paragraph of the final report.

```bash

python mechanism_design_main.py

```

### Final term exercise 1 - Shapley value and FJ Dynamics

All the implemented functions are tested in:

`src/final_term/exercise_1/test`

In order to reproduce all the tests you can run:

```bash

python -m unittest discover src/final_term/exercise_1/test

```

### Final term exercise 2 - Network analysis

Execute the following command to generate and analyze different nets
with different models and parameters and compare them with the target one.

```bash

python network_analysis_main.py

```

As the middle term Exercise 1 and Exercise 2, you can analyze a specific
model, for example `randomG`, by leaving only the following lines into the `main` function.

```python

def main():
    target_network = load_network(NETWORK_PATH)

    network_analyzer = NetworkAnalyzer(degree_mean_threshold=DEGREE_MEAN_THRESHOLD,
                                       degree_std_threshold=DEGREE_STD_THRESHOLD,
                                       network_diameter_threshold=DIAMETER_THRESHOLD,
                                       network_clustering_coefficient_threshold=AVG_CLUSTERING_COEF_THRESHOLD,
                                       target_network=target_network, network_generation_algorithms_with_kwargs=[
            (randomG, {"n": NETWORK_NUM_NODES, "p": 0.05}),
        ])

    network_analyzer.analyze_all()

```

### Final term exercise 3 - Election manipulation

The file `election_manipulation_main.py` has been used for testing puposes and should not be executed.

In order to use the manipulation function requested by the exercise you should import it into
your main file as follows:

```python

from election_manipulation_main import manipulation

def main():
    
    ...
    
    manipulation(G, p, c, B, b)

```

Note that you should use your main file from the same directory in which is located the file `election_manipulation_main.py`.
You can also copy it into your folder.

The experiment was configured in order to run in a defined amount of time. By default it will
run for 4 hours.
If you want to change it you should edit the constant `MAX_RUNNING_TIME_S` at 
the line `45` of the file `election_manipulation_main.py`.

> N.B.: The constant `MAX_RUNNING_TIME_S` specify the running time in seconds
