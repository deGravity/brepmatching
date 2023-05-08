# brepmatching
Learning to match brep topology.

### Getting Started

Create a new environment with conda (or mamba) from environment.yml, then
editably install brepmatching with pip (tested with mamba on Linux and
Windows):

```
conda env create -f environment.yml
conda activate brepmatching
python setup.py install
pip install -e .
```

Using this code requires an active Parasolid license and installation, create an environmental variable called `$PARASOLID_BASE` pointing to your
Parasolid install prior to any of the above instructions in order to link against the Parasolid library.
