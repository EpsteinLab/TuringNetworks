# TuringNetworks
Contains the code to simulate and analyze Turing patterns on growing domains from a wide variety of reaction networks.

This project is based on using the new conditions from Van Gorder et al. (2021, J. Math. Bio., 82 (4)) to study the emergence of transient Turing-type instabilities for reaction-diffusion systems on growing domains. In this project, we analyze the two-node Turing networks on growing domains using these conditions - using the same Hill-type reaction-diffusion equations as Scholes et al. (2019, Cell Systems, 9 (3)).

Currently, the NetSims.m file contains the code for the NetSim class, which can be used to run the simulations. As of now (10:30 am, 4/29/21) it runs the simulations, but is very slow due to the large amount of symbolic math used. I will be updating the code later to hopefully speed it up significantly. For an explanation how to use the code, see the Documentation.mlx file. The VG_1_a_iii.mat file is an example of the output of the code - a replica of Van Gorder et al's Figure 1a, section iii that was reproduced with this code. It is there and used in the documentation to see a potential NetSim, without having to actually run it for 15 minutes.

This repository is developed and maintained by Chris Konow, a PhD candidate in the Epstein Lab.
