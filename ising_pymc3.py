# -*- coding: utf-8 -*-
"""
ising_pymc3.py
~~~~~~~~~~~~~~

A PyMC3 implementation of the Ising model using the full
energy, and PyMC3's BinaryGibbsMetropolis method.

Usage:

    python ising_pymc3.py [Temperature [width [height]]]
    
    - The default (dimensionless) temperature is T=1.35, and the
      critical temperature Tc, at which magnetism is lost, is 2.269.
    - The default width and height are 42 pixels each.
    - If just the width is given, the image will be a square.

Output:

    Writes an animated GIF with filename 'img/pymc3/ising_{T}_{w}x{h}.gif',
    with the temperature T and the width and height are  substituted
    in the file name.
"""
from __future__ import print_function

import os
import numpy as np
import scipy as sc
import theano.tensor as tt
import pymc3 as pm
from array2gif import write_gif


width, height = 42, 42
T = 1.35   # normalized temperature. T_c ~ 2.269


def to_two_color(lattice):
    blue = np.ones(lattice.shape, dtype=np.int) * 255
    red = np.zeros(lattice.shape, dtype=np.int)
    red[lattice < 1] = 255
    green = red
    return np.array([red, green, blue])

    
class Magnetism(pm.Discrete):
    def __init__(self, Tc, *args, **kwargs):
        super(Magnetism, self).__init__(*args, **kwargs)
        self.Tc = Tc
        self.mode = tt.cast(1, 'int64')  # or -1

    def random(self, point=None, size=None, repeat=None):
        samples = pm.distributions.distribution.generate_samples(
            sc.stats.randint.rvs,
            low=0,
            high=2,
            dist_shape=self.shape,
            size=size
        )
        return samples

    def get_internal_energy(self, lattice):
        """Get the lattice's internal energy.
    
         H = - Jij * Sum_ij(s_i s_j) - h * Sum_i (s_i)
           Jij = coupling parameter  # set to 1
           h = external field strength  # set to zero
           s_i, s_j = spin of particle
    
        so
        H = - Sum_ij(s_i s_j)
        """
        hshift = tt.roll(lattice, 1, axis=1)
        vshift = tt.roll(lattice, 1, axis=0)
        H = -(
            ((2 * lattice - 1) * (2 * hshift - 1)).sum() +
            ((2 * lattice - 1) * (2 * vshift - 1)).sum()
        )
        return H

    def logp(self, value):
        H = self.get_internal_energy(value)
        log_prob = - H / self.Tc
        return log_prob


def simulate(N=50):
    shape = (height, width)
    with pm.Model() as basic_model:
        initial_lattice = sc.stats.randint.rvs(low=0, high=2, size=shape)
        m = Magnetism('m', Tc=T, shape=shape, testval=initial_lattice)
        step = pm.BinaryGibbsMetropolis([m])
        trace = pm.sample(N * 5, step=step)
    dataset = [to_two_color(lookup['m']) for lookup in trace[::5]]
    return dataset
    

def main(t=1.35, w=42, h=42):
    global T, width, height
    T, width, height = t, w, h
    dataset = simulate(80)
    dataset.append(dataset[-1] * 0)
    dataset.append(dataset[-1] * 0)
    filename = 'ising_{}_{}x{}.gif'.format(T, width, height)
    full_path = os.path.join('img', 'pymc3', filename)
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
