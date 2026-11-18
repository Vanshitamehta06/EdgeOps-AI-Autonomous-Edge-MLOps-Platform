"""
Test suite for Industrial PLC OPC UA Simulation and Closed-Loop Bridge.
Verifies IEC 62541 server instantiation, DB1 tag discovery, real-time read/write,
and closed-loop interlock command roundtrip.
"""

import asyncio
import pytest
from asyncua import Client, ua
from industrial.opc_ua_plc_server import SiemensS7PLCServer

TEST_ENDPOINT = "opc.tcp://127.0.0.1:4842/edgeops/test_plc/"


@pytest.mark.asyncio
async def test_opcua_server_and_tags():
    """Verify that the Siemens S7 PLC server publishes DB1 tags and handles values."""
    plc = SiemensS7PLCServer(endpoint=TEST_ENDPOINT)
    
    # Run server task in background
    server_task = asyncio.create_task(plc.start(update_interval_sec=0.2, max_cycles=15))

    client = Client(url=TEST_ENDPOINT)
    connected = False
    for attempt in range(15):
        await asyncio.sleep(0.4)
        try:
            await client.connect()
            connected = True
            break
        except Exception:
            continue

    assert connected, "Failed to connect to OPC UA server within timeout"

    try:
        ns_idx = await client.get_namespace_index("http://siemens.industry.edge/s7-300/predictive_maintenance")
        root = client.nodes.root
        objects = await root.get_child(["0:Objects"])
        db1_path = [f"{ns_idx}:Siemens_S7_300_Rack0_Slot2", f"{ns_idx}:DB1_DriveTelemetry"]
        db1_node = await objects.get_child(db1_path)

        # 1. Read Speed tag
        speed_node = await db1_node.get_child(f"{ns_idx}:DB1_DBD8_Rotational_Speed_RPM")
        speed_val = await speed_node.read_value()
        assert isinstance(speed_val, float)
        assert speed_val > 0

        # 2. Test closed-loop trip command write
        trip_node = await db1_node.get_child(f"{ns_idx}:DB1_DBX22_1_Fault_Trip")
        await trip_node.write_value(ua.DataValue(ua.Variant(True, ua.VariantType.Boolean)))
        trip_val = await trip_node.read_value()
        assert trip_val is True

        # Reset trip
        await trip_node.write_value(ua.DataValue(ua.Variant(False, ua.VariantType.Boolean)))
        trip_val_reset = await trip_node.read_value()
        assert trip_val_reset is False

    finally:
        await client.disconnect()
        plc.stop()
        await server_task
