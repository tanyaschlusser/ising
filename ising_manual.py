# -*- coding: utf-8 -*-
"""
ising_manual.py
~~~~~~~~~~~~~~~

A manual implementation of the Ising model using the
approach described by Metropolis and collaborators in 1953.

Usage:

    python ising_pymc3.py [Temperature [width [height]]]
    
    - The default (dimensionless) temperature is T=1.35, and the
      critical temperature Tc, at which magnetism is lost, is 2.269.
    - The default width and height are 42 pixels each.
    - If just the width is given, the image will be a square.

Output:

    Writes an animated GIF with filename 'img/manual/ising_{T}_{w}x{h}.gif',
    with the temperature T and the width and height are  substituted
    in the file name.
"""
from __future__ import print_function

import os
import numpy as np
from array2gif import write_gif


width, height = 42, 42
T = 1.35   # normalized temperature. T_c ~ 2.269


def to_two_color(lattice):
    blue = np.ones(lattice.shape, dtype=np.int) * 255
    red = np.zeros(lattice.shape, dtype=np.int)
    red[lattice < 0] = 255
    green = red
    return np.array([red,green,blue])
    

def get_dH(lattice, trial_location):
    """Get the change in internal energy of the lattice if a charge flips.

     H = - Jij * Sum_ij(s_i s_j) - h * Sum_i (s_i)
       Jij = coupling parameter  # set to 1
       h = external field strength  # set to zero
       s_i, s_j = spin of particle

    so
    H = - Sum_ij(s_i s_j)
    """
    i, j = trial_location
    width, height = lattice.shape
    H = 0
    Hflip = 0
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ii = (i + di) % height
        jj = (j + dj) % width
        H -= lattice[ii, jj] * lattice[i, j]
        Hflip -= lattice[ii, jj] * (-lattice[i, j])
    return Hflip - H


def simulate(N=50):
    # Randomly initialize the spins to either +1 or -1
    lattice = 2 * np.random.randint(2, size=(height, width)) - 1
    snapshots = []
    for snapshot in range(N):
        snapshots.append(to_two_color(lattice))
        print('Net magnetization: {:2.0f}%'
              .format(100.0 * abs(lattice.sum())/lattice.size))
        for step in range(5):
            # Walk through the array flipping atoms.
            for i in range(height):
                for j in range(width):
                    # Compute energy change from the flip.
                    # Use periodic boundary conditions for edges.
                    dE = get_dH(lattice, (i, j))
                    if dE < 0:
                        # lower energy: flip for sure
                        lattice[i, j] = -lattice[i, j]
                    else:
                        # Higher energy: flip with probability
                        probability = np.exp(-dE / T)
                        switch = np.random.rand() < probability
                        if switch:
                            lattice[i, j] = -lattice[i, j]
    return snapshots


def main(t=1.35, w=42, h=42):
    global T, width, height
    T, width, height = t, w, h
    dataset = simulate(80)
    dataset.append(dataset[-1] * 0)
    dataset.append(dataset[-1] * 0)
    filename = 'ising_{}_{}x{}.gif'.format(T, width, height)
    full_path = os.path.join('img', 'manual', filename)
    write_gif(dataset, full_path, fps=8)


def process_args(args):
    if 'h' in args or '--help' in args:
        print(__doc__)
        sys.exit(0)
    T = 1.35
    width = 42
    height = 42
    sys.argv.pop(0)
    if len(sys.argv) > 0:
        T = float(sys.argv[0])
    if len(sys.argv) > 1:
        width = int(sys.argv[1])
        height = width
    if len(sys.argv) > 2:
        height = int(sys.argv[2])
    return T, width, height


if __name__ == '__main__':
    import sys
    T, width, height = process_args(sys.argv)
    main(t=T, w=width, h=height)
