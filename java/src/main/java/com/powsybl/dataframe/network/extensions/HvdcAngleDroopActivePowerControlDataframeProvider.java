/**
 * Copyright (c) 2021, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package com.powsybl.dataframe.network.extensions;

import com.google.auto.service.AutoService;
import com.powsybl.dataframe.network.NetworkDataframeMapper;
import com.powsybl.dataframe.network.NetworkDataframeMapperBuilder;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControl;

import java.util.Objects;
import java.util.stream.Stream;

/**
 * @author Christian Biasuzzi <christian.biasuzzi@soft.it>
 */
@AutoService(NetworkExtensionDataframeProvider.class)
public class HvdcAngleDroopActivePowerControlDataframeProvider implements NetworkExtensionDataframeProvider {

    @Override
    public String getExtensionName() {
        return HvdcAngleDroopActivePowerControl.NAME;
    }

    private Stream<HvdcAngleDroopActivePowerControl> itemsStream(Network network) {
        return network.getHvdcLineStream()
                .map(g -> (HvdcAngleDroopActivePowerControl) g.getExtension(HvdcAngleDroopActivePowerControl.class))
                .filter(Objects::nonNull);
    }

    @Override
    public NetworkDataframeMapper createMapper() {
        return NetworkDataframeMapperBuilder.ofStream(this::itemsStream)
                .stringsIndex("id", ext -> ext.getExtendable().getId())
                .doubles("droop", HvdcAngleDroopActivePowerControl::getDroop)
                .doubles("p0", HvdcAngleDroopActivePowerControl::getP0)
                .booleans("enabled", HvdcAngleDroopActivePowerControl::isEnabled)
                .build();
    }
}