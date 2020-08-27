# quantum_baker_scrambling_neqr
An implementation and demonstration of Quantum image scrambling using baker map using NEQR encoding, based on recent journal articles.
The demonstration can be found in demo_simple.ipynb Jupyter notebook, while the developed functions/implementation are present in the *.py files.

Introduction

This notebook demonstrates a simple implementation of the Quantum Baker Map based quantum image scrambling algorithm introduced in [1]. The scrambling algorithm requires the classical image to be encoded according to the NEQR scheme [2]. In this notebook, a stepwise demonstration is provided which utilizes functions developed in the NEQR and Baker map related modules. The PIL library is used for basic image processing.

The NEQR encoder has been implemented as a functional implementation. The NEQR decoder reconstructs the classical image from the quantum state via measurements, which give a random outcome (using Python's pseudorandom number generator) to imitate the quantum measurement behaviour. The Quantum Baker map scrambling circuit is implemented via a gate based circuit, which consists of swap gates and controlled swap gates.

Author: Dhruv Bhatnagar

Background

The Baker Map is a two-dimensional chaotic transform, which has been generalized to the quatum case in [1]. Chaotic maps are well suited for scrambling because they are sensitive to initial values, system parameters and because of their pseudo-randomness.[1]

References

[1] Hou, C., Liu, X., & Feng, S. (2020). Quantum image scrambling algorithm based on discrete Baker map. Modern Physics Letters A, 35(17), 2050145. doi:10.1142/s021773232050145x

[2] Zhang, Y., Lu, K., Gao, Y., & Wang, M. (2013). NEQR: A novel enhanced quantum representation of digital images. Quantum Information Processing, 12(8), 2833-2860. doi:10.1007/s11128-013-0567-z

[3] Google Images and the MNIST Handwritten digits dataset are credited for the input images.

Conclusion and scope for future work

This is a work in progress, and a basic working implementation of NEQR encoding and Quantum image scrambling based on the Baker map has been achieved. An important improvement could be to make this program run efficiently for larger images, of the size which is relevant for real world applications. To that end, techniques such as sparse matrices and boolean optimization are being explored. After an efficient implementation is achieved, applications such as image watermarking can be demonstrated.
