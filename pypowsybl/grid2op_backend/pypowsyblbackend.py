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
    def __init__(self,
                 detailed_infos_for_cascading_failures=False,
                 provider=None,
                 parameters=None):
        Backend.__init__(self, detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        warnings.warn("This backend does not totally implements the grid2op requirement yet. This is an attempt to "
                      "use it!")
        self._nb_real_line_pandapower = None

        if provider is None:
            self._provider = 'OpenLoadFlow'
        else:
            # can be anything to pass to `pp.loadflow.run_ac(n, provider="Hades2")` call
            self._provider = provider
        if parameters is None:
            self._parameters = pypo.loadflow.Parameters()
        else:
            if not isinstance(parameters, pypo.loadflow.Parameters):
                raise BaseException(f"When building a PyPowSyBlBackend, the \"parameters\" argument should either be "
                                    f"None (to use the default pypowsybl parameters) or an instance of "
                                    f"\"pypo.loadflow.Parameters\". You provided an object of type: "
                                    f"{type(parameters)}")
            self._parameters = parameters

        # this backend has the possibility to compute the "theta" (voltage angle)
        self.can_output_theta = True

        self._p_or = None
        self._q_or = None
        self._v_or = None
        self._a_or = None
        self._theta_or = None
        self._p_ex = None
        self._q_ex = None
        self._v_ex = None
        self._a_ex = None
        self._theta_ex = None
        self._load_p = None
        self._load_q = None
        self._load_v = None
        self._load_theta = None
        self._prod_p = None
        self._prod_q = None
        self._prod_v = None
        self._gen_theta = None
        self._line_status = None
        self._storage_theta = None  # TODO
        self._storage_p = None
        self._storage_q = None
        self._storage_v = None

        # convert pair unit to kv
        self._gen_vn_kv = None

        self._bus_df = None
        self._volt_df = None

        # for faster copy when object are returned
        self.cst_1 = dt_float(1.0)

    def load_grid(self, path=None, filename=None):
        """
        We suppose that the file given as argument contains one bus per substation.

        Parameters
        ----------
        path
        filename

        Returns
        -------

        """
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
        self._volt_df = self._grid.create_voltage_levels_data_frame()

        df = self._grid.create_generators_data_frame()
        self._gen_vn_kv = 1.0 * self._volt_df.loc[df["voltage_level_id"]]["nominal_v"].values
        self._gen_vn_kv = self._gen_vn_kv.astype(dt_float)

        # number of each types of elements
        self.n_sub = self._grid.create_buses_data_frame().shape[0]  # self._grid.create_substations_data_frame().shape[0]
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

        # retrieve the names of the elements
        self.name_load = np.array([el for el in self._grid.create_loads_data_frame().index])
        self.name_gen = np.array([el for el in self._grid.create_generators_data_frame().index])
        self.name_line = [el for el in self._grid.create_lines_data_frame().index]
        self.name_line += [el for el in self._grid.create_2_windings_transformers_data_frame().index]
        self.name_line = np.array(self.name_line)
        self.name_sub = np.array([el for el in self._grid.create_buses_data_frame().index])

        self._compute_pos_big_topo()  # mandatory for grid2op.Backend


        # TODO name_storage
        # TODO name_shunt

        # intermediate attribute read from the grid java side
        self._p_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._q_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._v_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._a_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._p_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._q_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._v_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._a_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._line_status = np.full(self.n_line, dtype=dt_bool, fill_value=np.NaN)
        self._load_p = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self._load_q = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self._load_v = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self._prod_p = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self._prod_v = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self._prod_q = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self._storage_p = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self._storage_q = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self._storage_v = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)
        self._theta_or = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._theta_ex = np.full(self.n_line, dtype=dt_float, fill_value=np.NaN)
        self._load_theta = np.full(self.n_load, dtype=dt_float, fill_value=np.NaN)
        self._gen_theta = np.full(self.n_gen, dtype=dt_float, fill_value=np.NaN)
        self._storage_theta = np.full(self.n_storage, dtype=dt_float, fill_value=np.NaN)

    def apply_action(self, backendAction=None):
        """
        Here the implementation of the "modify the grid" function.

        From the documentation, it's pretty straightforward, even though the implementation takes ~70 lines of code.
        Most of them being direct copy paste from the examples in the documentation.
        """
        if backendAction is None:
            return

        active_bus, (prod_p, prod_v, load_p, load_q, storage), _, shunts__ = backendAction()

        # modify loads
        if load_p.changed.any():
            df_load_p = pd.DataFrame({'p0': load_p.values[load_p.changed]}, index=self.name_load[load_p.changed])
            self._grid.update_loads_with_data_frame(df_load_p)
        if load_q.changed.any():
            df_load_q = pd.DataFrame({'q0': load_q.values[load_q.changed]}, index=self.name_load[load_q.changed])
            self._grid.update_loads_with_data_frame(df_load_q)

        # modify generators.
        if prod_p.changed.any():
            df_gen_p = pd.DataFrame({'target_p': prod_p.values[prod_p.changed]}, index=self.name_gen[prod_p.changed])
            self._grid.update_generators_with_data_frame(df_gen_p)

        # TODO check pair unit
        if prod_v.changed.any():
            arr_kv = prod_v.values[prod_v.changed]
            arr_pu = arr_kv / self._gen_vn_kv
            df_gen_v = pd.DataFrame({'target_v': arr_pu}, index=self.name_gen[prod_v.changed])
            self._grid.update_generators_with_data_frame(df_gen_v)

        # TODO storage
        # TODO shunts

        # TODO topology !!!

        # line status
        lor_bus = backendAction.get_lines_or_bus()
        lex_bus = backendAction.get_lines_ex_bus()
        for l_id in range(self.n_line):
            if lor_bus[l_id] == -1 or lex_bus[l_id] == -1:
                if self._line_status[l_id]:
                    # disconnect the powerline as it was connected
                    self._grid.disconnect(self.name_line[l_id])
                    self._line_status[l_id] = False
            elif lor_bus[l_id] > 0 and lex_bus[l_id] > 0:
                if not self._line_status[l_id]:
                    # reconnect the powerline as it has been disconnected
                    self._grid.connect(self.name_line[l_id])
                    self._line_status[l_id] = True

    def runpf(self, is_dc=False):
        try:
            self._bus_df = None  # forget the previous information for the buses
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
                self._bus_df = self._grid.create_buses_data_frame()
                self._generators_info()
                self._loads_info()
                self._lines_info()
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
        self._p_or[:] = np.NaN
        self._q_or[:] = np.NaN
        self._v_or[:] = np.NaN
        self._a_or[:] = np.NaN
        self._p_ex[:] = np.NaN
        self._q_ex[:] = np.NaN
        self._v_ex[:] = np.NaN
        self._a_ex[:] = np.NaN
        # self.line_status = np.full(self.n_line, dtype=dt_bool, fill_value=np.NaN)  # TODO
        self._load_p[:] = np.NaN
        self._load_q[:] = np.NaN
        self._load_v[:] = np.NaN
        self._prod_p[:] = np.NaN
        self._prod_v[:] = np.NaN
        self._prod_q[:] = np.NaN
        self._storage_p[:] = np.NaN
        self._storage_q[:] = np.NaN
        self._storage_v[:] = np.NaN
        self._theta_or[:] = np.NaN
        self._theta_ex[:] = np.NaN
        self._load_theta[:] = np.NaN
        self._gen_theta[:] = np.NaN
        self._storage_theta[:] = np.NaN

    # interface to retrieve the information
    def generators_info(self):
        return self.cst_1 * self._prod_p, self.cst_1 * self._prod_q, self.cst_1 * self._prod_v

    def loads_info(self):
        return self.cst_1 * self._load_p, self.cst_1 * self._load_q, self.cst_1 * self._load_v

    def lines_or_info(self):
        return self.cst_1 * self._p_or, self.cst_1 * self._q_or, self.cst_1 * self._v_or, self.cst_1 * self._a_or

    def lines_ex_info(self):
        return self.cst_1 * self._p_ex, self.cst_1 * self._q_ex, self.cst_1 * self._v_ex, self.cst_1 * self._a_ex
    # TODO storage and shunt !

    def get_theta(self):
        return self.cst_1 * self._theta_or,  self.cst_1 * self._theta_ex, self.cst_1 * self._load_theta, \
               self.cst_1 * self._gen_theta, self.cst_1 * self._storage_theta

    # private methods: data are read from java side once and stored as temporary variables
    def _generators_info(self):
        # TODO check the convention for generators
        df = self._grid.create_generators_data_frame()
        self._prod_p[:] = -df["p"].values
        self._prod_q[:] = -df["q"].values
        df_volt = self._bus_df.loc[df["bus_id"]]
        self._prod_v[:] = df_volt["v_mag"].values  # in pu
        self._prod_v[:] *= self._volt_df.loc[df["voltage_level_id"]]["nominal_v"].values
        self._gen_theta[:] = df_volt["v_angle"].values

    def _loads_info(self):
        """
        We chose to keep the same order in grid2op and in pandapower. So we just return the correct values.
        """
        df = self._grid.create_loads_data_frame()
        self._load_p[:] = df["p"].values
        self._load_q[:] = df["q"].values
        df_volt = self._bus_df.loc[df["bus_id"]]
        self._load_v[:] = df_volt["v_mag"].values  # in pu
        self._load_v[:] *= self._volt_df.loc[df["voltage_level_id"]]["nominal_v"].values
        self._load_theta[:] = df_volt["v_angle"].values

    def _lines_info(self):
        df_l = self._grid.create_lines_data_frame()
        df_t = self._grid.create_2_windings_transformers_data_frame()
        self._p_or[:] = np.concatenate((df_l["p1"].values, df_t["p1"].values))
        self._p_ex[:] = np.concatenate((df_l["p2"].values, df_t["p2"].values))
        self._q_or[:] = np.concatenate((df_l["q1"].values, df_t["q1"].values))
        self._q_ex[:] = np.concatenate((df_l["q2"].values, df_t["q2"].values))

        df_lines_nm = pd.concat((df_l["bus1_id"], df_t["bus1_id"]))
        df_lines_nm = df_lines_nm[df_lines_nm != '']
        df_volt_or = self._bus_df.loc[df_lines_nm]
        self._v_or[self._line_status] = df_volt_or["v_mag"].values  # in pu
        ind_volt = pd.concat((df_l["voltage_level1_id"], df_t["voltage_level1_id"]))
        self._v_or[self._line_status] *= self._volt_df.loc[ind_volt]["nominal_v"].values[self._line_status]
        self._theta_or[self._line_status] = df_volt_or["v_angle"].values

        df_lines_nm2 = pd.concat((df_l["bus2_id"], df_t["bus2_id"]))
        df_lines_nm2 = df_lines_nm2[df_lines_nm2 != '']
        df_volt_ex = self._bus_df.loc[df_lines_nm2]
        self._v_ex[self._line_status] = df_volt_ex["v_mag"].values  # in pu
        ind_volt_ex = pd.concat((df_l["voltage_level2_id"], df_t["voltage_level2_id"]))
        self._v_ex[self._line_status] *= self._volt_df.loc[ind_volt_ex]["nominal_v"].values[self._line_status]
        self._theta_ex[self._line_status] = df_volt_ex["v_angle"].values

        # amps are not computed by pypowsybl...
        self._a_or[self._line_status] = np.sqrt(self._p_or[self._line_status]**2 + self._q_or[self._line_status]**2)
        self._a_or[self._line_status] /= np.sqrt(3.) * self._v_or[self._line_status]
        self._a_ex[self._line_status] = np.sqrt(self._p_ex[self._line_status]**2 + self._q_ex[self._line_status]**2)
        self._a_ex[self._line_status] /= np.sqrt(3.) * self._v_ex[self._line_status]

        # handle non connected lines (they are not updated in the lines above)
        self._v_or[~self._line_status] = 0.
        self._v_ex[~self._line_status] = 0.
        self._a_or[~self._line_status] = 0.
        self._a_ex[~self._line_status] = 0.

        # powsybl assign Nan to non connected lines, in grid2op we expect 0.
        self._p_or[~self._line_status] = 0.
        self._p_ex[~self._line_status] = 0.
        self._q_or[~self._line_status] = 0.
        self._q_ex[~self._line_status] = 0.

    def _disconnect_line(self, id_):
        self._grid.disconnect(self.name_line[id_])

    # TODO get_line_status, get_topo_vect
