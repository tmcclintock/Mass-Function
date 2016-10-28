Mass-Function
=============
An implementation of the Tinker et al. 
(http://adsabs.harvard.edu/abs/2008ApJ...688..709T) 
2008 mass function for predictiing the number of dark matter halos 
within some range of masses.

Dependencies
------------
- numpy
- scipy
- Matt Becker's cosmocalc (https://github.com/beckermr/cosmocalc)

Installation
------------
To install simply write
```
python setup.py install
```
if you care about keeping the root directory clean then do
```
python setup.py clean
```

Usage
-----
An example of how to use this code can be found
in the examples/example.py file. The module can
either calculate dn/dM as a function of mass
or the number density within some mass bins,
all as a function of redshift.

The result of running the example should be the following.
![alt text](https://github.com/tmcclintock/Mass-Function/blob/master/figures/mf_example.png)