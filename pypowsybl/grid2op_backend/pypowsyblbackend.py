#
# Copyright (c) 2020, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

import os
import warnings
import copy
import time
import numpy as np
import pandas as pd

from grid2op.Exceptions import DivergingPowerFlow
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Backend import Backend

import pypowsybl as pypo


class PyPowSyBlBackend(Backend):
    def __init__(self, detailed_infos_for_cascading_failures=False, provider=None):
        Backend.__init__(self, detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        warnings.warn("This backend does not totally implements the grid2op requirement yet. This is an attempt to "
                      "use it!")
        self._nb_real_line_pandapower = None

        self._provider = provider  # can be anything to pass to `pp.loadflow.run_ac(n, provider="Hades2")` call

        self.p_or = None
        self.q_or = None
        self.v_or = None
        self.a_or = None
        self.p_ex = None
        self.q_ex = None
        self.v_ex = None
        self.a_ex = None
        self.load_p = None
        self.load_q = None
        self.load_v = None
        self.storage_p = None
        self.storage_q = None
        self.storage_v = None
        self.prod_p = None
        self.prod_q = None
        self.prod_v = None
        self.line_status = None

    def load_grid(self, path=None, filename=None):
        if path is None and filename is None:
            raise RuntimeError("You must provide at least one of path or file to load a powergrid.")
        if path is None:
            full_path = filename
        elif filename is None:
            full_path = path
        else:
            full_path = os.path.join(path, filename)
        if not os.path.exists(full_path):
            raise RuntimeError("There is no powergrid at \"{}\"".format(full_path))

        # TODO not working completely yet
        self._grid = pypo.network.load(full_path)

        # number of each types of elements
        self.n_sub = self._grid.create_substations_data_frame().shape[0]
        self.n_line = self._grid.create_lines_data_frame().shape[0]
        self.n_line += self._grid.create_2_windings_transformers_data_frame().shape[0]
        self.n_gen = len(self._grid.generators)
        self.n_load = len(self._grid.loads)

        # TODO n_shunt and n_storage
        self.shunts_data_available = False
        self.set_no_storage()

        # which objects belong to which sub
        buses = self._grid.create_buses_data_frame()
        load_buses = self._grid.create_loads_data_frame()["bus_id"].values
        gen_buses = self._grid.create_generators_data_frame()["bus_id"].values
        self.load_to_subid = np.array([np.where(el == buses.index)[0][0] for el in load_buses]).astype(dt_int)
        self.gen_to_subid = np.array([np.where(el == buses.index)[0][0] for el in gen_buses]).astype(dt_int)
        lines_buses = pd.concat([self._grid.create_lines_data_frame()[["bus1_id", "bus2_id"]],
                                 self._grid.create_2_windings_transformers_data_frame()[["bus1_id", "bus2_id"]]])
        self.line_or_to_subid = np.array([np.where(el == buses.index)[0][0] for el in lines_buses["bus1_id"].values]).astype(dt_int)
        self.line_ex_to_subid = np.array([np.where(el == buses.index)[0][0] for el in lines_buses["bus2_id"].values]).astype(dt_int)
        # TODO storage_to_subid
        # TODO shunt_to_subid

        self._compute_pos_big_topo()

        # retrieve the names
        self.name_load = [el for el in self._grid.create_loads_data_frame().index]
        self.name_gen = [el for el in self._grid.create_generators_data_frame().index]
        self.name_line = [el for el in self._grid.create_lines_data_frame().index]
        self.name_line += [el for el in self._grid.create_2_windings_transformers_data_frame().index]
        self.name_sub = [el for el in self._grid.create_buses_data_frame().index]
        # TODO name_storage
        # TODO name_shunt

        # intermediate attribute read from the grid java side
        self.p_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.q_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.v_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.a_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.p_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.q_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.v_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self.a_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        # self.line_status = np.full(self.n_line, dtype=dt_bool, fill_value=np.NaN)  # TODO
        self.load_p = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.load_q = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.load_v = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self.prod_p = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.prod_v = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.prod_q = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self.storage_p = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self.storage_q = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self.storage_v = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)


    def apply_action(self, backendAction=None):
        """
        Here the implementation of the "modify the grid" function.

        From the documentation, it's pretty straightforward, even though the implementation takes ~70 lines of code.
        Most of them being direct copy paste from the examples in the documentation.
        """
        if backendAction is None:
            return

        active_bus, (prod_p, prod_v, load_p, load_q, storage), _, shunts__ = backendAction()

        # TODO code that !
        for gen_id, new_p in prod_p:
            self._grid.gen["p_mw"].iloc[gen_id] = new_p
        for gen_id, new_v in prod_v:
            self._grid.gen["vm_pu"].iloc[gen_id] = new_v  # but new_v is not pu !
            self._grid.gen["vm_pu"].iloc[gen_id] /= self._grid.bus["vn_kv"][self.gen_to_subid[gen_id]]  # now it is :-)

        for load_id, new_p in load_p:
            self._grid.load["p_mw"].iloc[load_id] = new_p
        for load_id, new_q in load_q:
            self._grid.load["q_mvar"].iloc[load_id] = new_q
        # TODO storage
        # TODO shunts

        # TODO topology !!!

    def runpf(self, is_dc=False):
        try:
            beg_ = time.time()
            if is_dc:
                results = pypo.loadflow.run_dc(self._grid)
            else:
                results = pypo.loadflow.run_ac(self._grid, provider=self._provider)
            end_ = time.time()
            self.comp_time += end_ - beg_

            conv = True
            exc_ = None
            for el in results:
                conv = conv and el.status == pypo.loadflow.ComponentStatus.CONVERGED
            if not conv:
                exc_ = DivergingPowerFlow("Powerflow diverge")

            if conv:
                # retrieve the data
                self._generators_info()
                self._loads_info()
                self._lines_or_info()
                self._lines_ex_info()
                # TODO shunt and storage

            return conv, exc_
        except Exception as exc_:
            # of the powerflow has not converged, results are Nan
            return False, exc_

    def get_topo_vect(self):
        # TODO it's not coded...
        return np.ones(self.dim_topo, dtype=dt_int)

    def copy(self):
        return copy.deepcopy(self)

    def reset(self, path=None, grid_filename=None):
        # TODO not coded at the moment
        self.comp_time = 0.
        pass

    def generators_info(self):
        return self.prod_p, self.prod_q, self.prod_v

    def loads_info(self):
        return self.load_p, self.load_q, self.load_v

    def lines_or_info(self):
        return self.p_or, self.q_or, self.v_or, self.a_or

    def lines_ex_info(self):
        return self.p_ex, self.q_ex, self.v_ex, self.a_ex

    # tmp method, to read data once
    def _generators_info(self):
        # carefull with copy / deep copy
        df = self._grid.create_generators_data_frame()
        self.prod_p[:] = df["p"].values
        self.prod_q[:] = df["q"].values
        self.prod_v[:] = 1.0  # TODO !!!!

    def _loads_info(self):
        """
        We chose to keep the same order in grid2op and in pandapower. So we just return the correct values.
        """
        df = self._grid.create_generators_data_frame()
        self.load_p[:] = df["p"].values
        self.laod_q[:] = df["q"].values
        self.load_v[:] = 1.0  # TODO !!!!

    def _lines_or_info(self):
        pass

    def _lines_ex_info(self):
        pass

    # TODO get_line_status, _disconnect_line