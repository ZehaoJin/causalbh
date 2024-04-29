# Causalbh

### [Paper: A Data-driven Causal Discovery of Galaxy--Black Hole Coevolution](https://ui.adsabs.harvard.edu/)
### [1. Installation](#1-installation)
### [2. Causal discovery with BGe exact posterior calculation](#2-causal-discovery-with-bge-exact-posterior-calculation)
### [3. Extensions: PC, FCI and DAG-GFN](#3-extensions-pc-fci-and-dag-gfn)
### [4. Data: Black hole mass - galaxy property catalog](#4-black-hole-mass-galaxy-property-catalog)
### [5. Reproduce paper plots](#5-reproduce-paper-plots)
### [6. Cite this work](#6-cite-this-work)

## 1. Installation
### 1.1. clone this repository to your machine (make sure your machine has a GPU)

    git clone git@github.com:ZehaoJin/causalbh.git
    
or

    git clone https://github.com/ZehaoJin/causalbh.git

### 1.2. Install dependencies
- We highly recommand install dependencies in a virtual python environment via conda:

      conda create --name causalbh
      conda activate causalbh

- To use any gpu features, install [jax](https://jax.readthedocs.io/en/latest/installation.html)
- 
