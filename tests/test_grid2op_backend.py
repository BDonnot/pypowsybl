#
# Copyright (c) 2020, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

import os
import warnings
import tempfile
import unittest
import numpy as np

from grid2op.Action import ActionSpace
from grid2op.Rules import AlwaysLegal
from pypowsybl.network import create_ieee14
from pypowsybl.grid2op_backend import PyPowSyBlBackend


class PyPowSyBlBackendTestCases_0(unittest.TestCase):
    """
    This class is used during the development of the grid2op_backend to make sure everything works properly.

    The more extensive test suit of grid2op.Backend, included with grid2op will be performed once the development is
    over.
    """
    def setUp(self) -> None:
        with tempfile.NamedTemporaryFile() as f:
            net = create_ieee14()
            net.dump(f.name)
            self.backend = PyPowSyBlBackend()
            self.backend.load_grid(f.name)
            type(self.backend).env_name = "PyPowSyBlBackendTestCases_0"
        self.backend.assert_grid_correct()
        self.tol = 1e-2  # ultra high for a solver tolerance...
        self._act_space_cls = ActionSpace.init_grid(type(self.backend))
        self.act_space = self._act_space_cls(self.backend, AlwaysLegal())
        self._init_load_p = 1.0 * self.backend._grid.create_loads_data_frame()["p0"].values
        self._init_load_q = 1.0 * self.backend._grid.create_loads_data_frame()["q0"].values
        self._init_gen_p = 1.0 * self.backend._grid.create_generators_data_frame()["target_p"].values
        self._init_gen_v = 1.0 * self.backend._grid.create_generators_data_frame()["target_v"].values
        self.bk_act = type(self.backend).my_bk_act_class()

    def test_is_ok(self):
        """basic test on the resulting grid"""
        assert self.backend.n_load == 11
        assert self.backend.n_gen == 5
        assert self.backend.n_sub == 14

    def test_runpf_active_value(self):
        """only test for now the active part, because, well, the shunts are not coded"""
        conv, reason = self.backend.runpf()
        assert conv
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            p_subs, q_subs, p_bus, q_bus, diff_v_bus = self.backend.check_kirchoff()
        assert np.max(np.abs(p_subs)) <= self.tol
        assert np.max(np.abs(p_bus)) <= self.tol

    def test_change_injections(self):
        """check i can modify properly the injections (to that end i check that if i perform a dc powerflow, it is
        in fact "linear" with the modification of the injection"""
        conv, reason = self.backend.runpf(is_dc=True)
        assert conv
        gen_p = 1.0 * self.backend._prod_p

        mult_fact = 1.5
        act = self.act_space({"injection": {"load_p": mult_fact * self._init_load_p, "prod_p": mult_fact * gen_p}})
        self.bk_act += act
        self.backend.apply_action(self.bk_act)

        conv, reason = self.backend.runpf(is_dc=True)
        assert conv
        assert np.max(np.abs(self.backend._load_p - mult_fact * self._init_load_p)) <= self.tol
        assert np.max(np.abs(self.backend._prod_p - mult_fact * gen_p)) <= self.tol

        # now test i can run an ac_pf and it converges
        conv, reason = self.backend.runpf()
        assert conv

    def test_disco_reco_lines(self):
        """In this test i check that i can connect and disconnect powerlines or transformers"""
        act = self.act_space()
        line_id = 0
        trafo_id = 18

        # disco the powerline
        act.set_line_status = [(line_id, -1)]
        self.bk_act += act
        self.backend.apply_action(self.bk_act)
        conv, reason = self.backend.runpf(is_dc=True)
        assert conv
        assert self.backend._p_or[line_id] == 0.
        assert self.backend._p_ex[line_id] == 0.
        assert self.backend._a_or[line_id] == 0.

        # reco the powerline
        act.set_line_status = [(line_id, +1)]
        self.bk_act = type(self.backend).my_bk_act_class()
        self.bk_act += act
        self.backend.apply_action(self.bk_act)
        conv, reason = self.backend.runpf(is_dc=True)
        assert conv
        assert self.backend._p_or[line_id] != 0.
        assert self.backend._p_ex[line_id] != 0.
        assert self.backend._a_or[line_id] != 0.

        # disco the trafo
        act.set_line_status = [(trafo_id, -1)]
        self.bk_act += act
        self.backend.apply_action(self.bk_act)
        conv, reason = self.backend.runpf(is_dc=True)
        assert conv
        assert self.backend._p_or[trafo_id] == 0.
        assert self.backend._p_ex[trafo_id] == 0.
        assert self.backend._a_or[trafo_id] == 0.

        # reco the trafo
        act.set_line_status = [(trafo_id, +1)]
        self.bk_act = type(self.backend).my_bk_act_class()
        self.bk_act += act
        self.backend.apply_action(self.bk_act)
        conv, reason = self.backend.runpf(is_dc=True)
        assert conv
        assert self.backend._p_or[trafo_id] != 0.
        assert self.backend._p_ex[trafo_id] != 0.
        assert self.backend._a_or[trafo_id] != 0.


if __name__ == '__main__':
    unittest.main()
