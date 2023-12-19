This repository contains three notebooks that are related to the paper "Simulating photon counting from dynamic quantum emitters by exploiting zero-photon measurements
" available at https://arxiv.org/abs/2307.16591.

- exact_values.nb is a Mathematica notebook that analytically solves the exact photon scattering probabilities of a two-level system driven by a square pulse.

- square_pulse_statistics.ipynb is a Jupyter notebook that numerically computes the same scattering probabilities using a Fourier transform. This notebook also extends the method to compute the photon-number distribution and threshold detection outcomes for Boson-sampling type experiments using light from the same driven two-level emitter.

- threshold_detection.jl is a Pluto notebook for Julia that demonstrates the potential to optimise the zero-photon generator method to be orders of magnitude faster than in Python.
