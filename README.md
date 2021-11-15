# efficient_profiled_attacks_extended
This repository contains the artifacts of the paper _Efficient Profiled Side-Channel Analysis of Masked Implementations, Extended_ (TIFS 2021) co-authored by Bronchain, Masure, Durvaux and Standaert. 

## Install
The artifact uses Python3. The required packages can be installed with:
```
pip install -r requirements.txt
```
Please note that PyTorch is a dependency and might require dedicated additional libraries. You can find additional information [here](https://pytorch.org/get-started/locally/).

## Usage 
The provided artifact can be used according to
```
user: python3 main.py --help

usage: main.py [-h] [--shares SHARES] [--bits BITS] [--std STD] [--flaw FLAW] [--repeat REPEAT]

Comparing distinguishers convergence.

optional arguments:
  -h, --help            show this help message and exit
  --shares SHARES, -d SHARES
                        Number of shares.
  --bits BITS, -b BITS  Number of bits in target bits.
  --std STD             Noise standard deviation.
  --flaw FLAW, -f FLAW  Flaw magnitude.
  --repeat REPEAT, -r REPEAT
                        Number of repeated experiments
```

As an example, running
```
python3 main.py -d 3 --std 0.5 -f 0.0 -r 10 -b 4
```
will generate the converge curves for 3 4-bit shares with noise standard deviation of 0.05. The plot will be saved in the corresponding directory. 

## Contact
In order to report issues, please contact Olivier Bronchain (olivier.bronchain at uclouvain.be) or open an issue on github. 
