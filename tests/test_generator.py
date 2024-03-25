#!/usr/bin/env python3

import numpy as np

from ppafm.ml.AuxMap import AtomicDisks
from ppafm.ml.Generator import GeneratorAFMtrainer
from ppafm.ocl.AFMulator import AFMulator


def test_GeneratorAFMtrainer():
    n_sample = 12
    n_atoms = 10
    batch_size = 5

    class TestTrainer(GeneratorAFMtrainer):
        def on_batch_start(self):
            self.randomize_tip(max_tilt=0.5)

        def on_afm_start(self):
            self.randomize_distance(delta=0.5)

    def generator():
        for _ in range(n_sample):
            sample_dict = {"xyzs": 10 * np.random.rand(n_atoms, 3), "Zs": np.random.randint(1, 16, n_atoms)}
            yield sample_dict

    afmulator = AFMulator(scan_dim=(100, 100, 20), scan_window=((0, 0, 5), (10, 10, 7)))
    aux_maps = [AtomicDisks(scan_dim=(100, 100, 20), scan_window=((0, 0, 5), (10, 10, 7)))]
    trainer = TestTrainer(afmulator=afmulator, aux_maps=aux_maps, sample_generator=generator(), sim_type="LJ", batch_size=batch_size, iZPPs=[8, 54])

    for i_batch, (Xs, Ys, mols, sws) in enumerate(trainer):
        nb = 5 if i_batch < 2 else 2
        assert Xs.shape == (nb, 2, 100, 100, 11)
        assert Ys.shape == (nb, 1, 100, 100)
        assert len(mols) == nb
        for m in mols:
            assert m.shape == (n_atoms, 5)
        assert sws.shape == (nb, 2, 2, 3)

    assert i_batch == 2
