# laropt

`laropt` (LArSoft Optimisation package), while not being officially part of the LArSoft family, aims at offering a tool to optimize the proprietary `FHiCL` configuration files, through an easy to use and HPC-scalable set of modules, all controlled by a standardised `yaml` format configuration file. 

## Installation 

Installing this package is as simple as running 

```bash
python -m pip install . -e
```

from within a terminal, where LArSoft was already setup. 

It is suggested that the installation happens inside a python virtual environment (a brief starter for this is [provided below](#setting-up-pythons-venv)). This will be useful for the grid submission part, which will require both the (optional) tarball of the local LArSoft/ICARUS- or SBNDCode installation, as well as the installation of the python environment. See the ["Running over the grid" section](#running-over-the-grid) for details on this part. 

### Requirements



### Setting up Python's `venv`

Preliminarly to setting up the python environment it is required to setup FNAL nodes correctly for LArSoft/ICARUS- or SBNDCode use. You can set it up (using ICARUS here as an example) as

```bash
source /cvmfs/icarus.opensciencegrid.org/products/icarus/setup_icarus.sh
setup icaruscode v10_06_00_06p03 -q e26:prof # Or any other latest version of the code
```

At this point you should have python set up and ready to go 

```bash
python3 -V
Out: Python 3.9.15
```
At this point the virtual environment can be created as 

```bash 
python3 -m venv .env
```

and activated using 

```bash
source .env/bin/activate
```

## Usage

### Basic usage



### Running over the grid



## Changelog

