"""
Siemens Industrial Edge Connector & Gateway Client (IEC 62541 OPC UA Bridge).
Subscribes to PLC DB1 tags, forwards real-time telemetry to the EdgeOps AI inference engine,
and writes back closed-loop Trip / Interlock commands to the PLC if an anomaly is detected.
"""

import asyncio
import logging
import argparse
from pathlib import Path
import httpx
from asyncua import Client, ua

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] [Edge Gateway] %(message)s")
logger = logging.getLogger("Industrial_Edge_Bridge")

DEFAULT_OPCUA_ENDPOINT = "opc.tcp://127.0.0.1:4840/edgeops/simulated_s7_plc/"
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"


class IndustrialEdgeBridge:
    def __init__(self, opcua_url: str = DEFAULT_OPCUA_ENDPOINT, api_url: str = DEFAULT_API_URL):
        self.opcua_url = opcua_url
        self.api_url = api_url
        self.client = Client(url=self.opcua_url)
        self.tag_nodes = {}
        self.is_connected = False

    async def connect(self):
        """Connect to the Siemens OPC UA PLC and locate DB1 tag nodes."""
        logger.info(f"Connecting to Industrial PLC at {self.opcua_url}...")
        await self.client.connect()
        self.is_connected = True
        logger.info("OPC UA Session established with PLC.")

        # Discover nodes in namespace
        root = self.client.nodes.root
        objects = await root.get_child(["0:Objects"])
        
        # Locate namespace index
        ns_idx = await self.client.get_namespace_index("http://siemens.industry.edge/s7-300/predictive_maintenance")

        db1_path = [f"{ns_idx}:Siemens_S7_300_Rack0_Slot2", f"{ns_idx}:DB1_DriveTelemetry"]
        db1_node = await objects.get_child(db1_path)

        # Cache tag variable nodes
        tag_names = [
            "DB1_DBD0_Air_Temperature_K",
            "DB1_DBD4_Process_Temperature_K",
            "DB1_DBD8_Rotational_Speed_RPM",
            "DB1_DBD12_Torque_NM",
            "DB1_DBD16_Tool_Wear_MIN",
            "DB1_DBS20_Machine_Type",
            "DB1_DBX22_0_Motor_Running",
            "DB1_DBX22_1_Fault_Trip"
        ]

        for tag in tag_names:
            node = await db1_node.get_child(f"{ns_idx}:{tag}")
            self.tag_nodes[tag] = node

        logger.info(f"Successfully mapped {len(self.tag_nodes)} PLC tag handles.")

    async def read_telemetry(self) -> dict:
        """Read all physical sensor values from PLC DB1 tags."""
        air_temp = await self.tag_nodes["DB1_DBD0_Air_Temperature_K"].read_value()
        proc_temp = await self.tag_nodes["DB1_DBD4_Process_Temperature_K"].read_value()
        rpm = await self.tag_nodes["DB1_DBD8_Rotational_Speed_RPM"].read_value()
        torque = await self.tag_nodes["DB1_DBD12_Torque_NM"].read_value()
        tool_wear = await self.tag_nodes["DB1_DBD16_Tool_Wear_MIN"].read_value()
        m_type = await self.tag_nodes["DB1_DBS20_Machine_Type"].read_value()

        return {
            "Type": str(m_type),
            "Air temperature [K]": float(air_temp),
            "Process temperature [K]": float(proc_temp),
            "Rotational speed [rpm]": float(rpm),
            "Torque [Nm]": float(torque),
            "Tool wear [min]": float(tool_wear)
        }

    async def write_plc_trip_command(self, trip: bool):
        """Write closed-loop trip command to PLC interlocking relay tag."""
        trip_node = self.tag_nodes.get("DB1_DBX22_1_Fault_Trip")
        if trip_node:
            await trip_node.write_value(ua.DataValue(ua.Variant(trip, ua.VariantType.Boolean)))

    async def run(self, poll_interval_sec: float = 1.0, max_iterations: int = None):
        """Main bridging loop: poll PLC -> run edge inference -> closed-loop feedback."""
        await self.connect()
        iteration = 0

        async with httpx.AsyncClient(timeout=3.0) as http_session:
            while True:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Completed {max_iterations} bridge cycles. Exiting.")
                    break

                try:
                    # 1. Read PLC Tags
                    payload = await self.read_telemetry()
                    iteration += 1

                    # 2. Forward to Edge Inference API
                    t0 = asyncio.get_event_loop().time()
                    resp = await http_session.post(self.api_url, json=payload)
                    t_latency_ms = (asyncio.get_event_loop().time() - t0) * 1000.0

                    if resp.status_code == 200:
                        result = resp.json()
                        pred = result.get("prediction", 0)
                        prob = result.get("probability", 0.0)
                        cause = result.get("root_cause", "None")

                        if pred == 1:
                            logger.warning(
                                f"[PLC Sample #{iteration:03d}] ⚠️ ANOMALY DETECTED! Prob: {prob*100:.1f}% | "
                                f"Root Cause: {cause} | Inference Latency: {t_latency_ms:.2f} ms"
                            )
                            # Closed-loop action: write Trip interlock to PLC
                            await self.write_plc_trip_command(True)
                            logger.warning(">>> CLOSED-LOOP ACTION: Asserted DB1_DBX22_1_Fault_Trip=TRUE on PLC.")
                        else:
                            logger.info(
                                f"[PLC Sample #{iteration:03d}] Normal state (Prob: {prob*100:.2f}%) | "
                                f"Latency: {t_latency_ms:.2f} ms"
                            )
                            await self.write_plc_trip_command(False)
                    else:
                        logger.error(f"Inference API returned HTTP {resp.status_code}: {resp.text}")

                except Exception as ex:
                    logger.error(f"Error in bridge iteration #{iteration}: {ex}")

                await asyncio.sleep(poll_interval_sec)

        await self.client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Edge OPC UA Telemetry Bridge")
    parser.add_argument("--opcua-url", type=str, default=DEFAULT_OPCUA_ENDPOINT, help="OPC UA Server URL")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Edge inference API endpoint")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--cycles", type=int, default=None, help="Stop after N cycles")
    args = parser.parse_args()

    bridge = IndustrialEdgeBridge(opcua_url=args.opcua_url, api_url=args.api_url)
    try:
        asyncio.run(bridge.run(poll_interval_sec=args.interval, max_iterations=args.cycles))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Edge Bridge stopped by operator.")
