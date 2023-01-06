## wetree
Make a resampling tree visual after running a WE simulation.

---

This tutorial contains the code necessary to generate a resampling tree with a Wepy simulation hdf5 file. A small, example hdf5 file is provided. This file has a 25 cycle, 12 trajectory dataset and an observable used for coloring of nodes. This tree can be visualized in Gephi.
___

Adapted from: https://gitlab.com/nmroussey/resampling_tree_tutorial 
Converted from compatibility with wepy h5 to westpa h5 files (in progress).

---

## Installation and Requirements
If you have generated a Wepy hdf5 file, your environment should already contain all necessary packages.

Wepy: **pip install wepy[all]**

NumPy, NetworkX, and sys are also required.

---
## Contents of the Repository

1. run_restree_gen.py

The script to run to generate the tree. Arguments are: input h5 file and output file name.

2. restree/

The codes required to make the tree.

3. example_input.wepy.h5

A sample wepy output file.


## Generating The Tree

In your enviornment that contains Wepy do:

python run_restree_gen.py example_input.wepy.h5 test_outfile.gexf

This will generate a .gexf file for use in '[Gephi](https://gephi.org)'.
Data included is nodes and edges, weights/probabilities (for node size), and an extra piece of information named COM, for coloring the tree.

NOTE: No merging and cloning is visible in this tree due to the short length of the simulation. This is not a mistake in the tree generation code.

---

## References

##### Software Packages

'[Wepy](https://github.com/ADicksonLab/wepy)'

'[NumPy](https://numpy.org)'

'[Gephi](https://gephi.org)'
##### Papers

'[Wepy: A Flexible Software Framework for Simulating Rare Events with Weighted Ensemble Resampling](https://pubs.acs.org/doi/10.1021/acsomega.0c03892)' Lotz, S., & Dickson, A., ACS Omega, 2020


---
ENV:
conda install -c condo-forge mdtraj
pip install wepy[all]
Downgrade numpy to 1.23.* (pip install numpy=1.23)
# wetree
