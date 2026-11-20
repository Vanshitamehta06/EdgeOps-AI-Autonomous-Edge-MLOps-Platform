"""
Simulated Siemens S7 PLC OPC UA Server (IEC 62541).
Emulates an industrial S7-300 / S7-1500 PLC with CP communication processor exposing
Data Block 1 (DB1) telemetry tags from machine sensors (drives, spindles, temperature probes).
"""

import asyncio
import logging
import random
from pathlib import Path
import pandas as pd
from asyncua import Server, ua
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] [OPC-UA PLC] %(message)s")
logger = logging.getLogger("PLC_OPC_UA_Server")

DEFAULT_ENDPOINT = "opc.tcp://127.0.0.1:4840/edgeops/simulated_s7_plc/"
SERVER_NAME = "Siemens_S7_300_PredictiveMaintenance_Simulator"


class SiemensS7PLCServer:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, dataset_path: str = None):
        self.endpoint = endpoint
        self.dataset_path = Path(dataset_path) if dataset_path else config.DATASET_PATH
        self.server = Server()
        self.is_running = False
        self.tags = {}

    async def initialize(self):
        """Configure OPC UA address space mimicking Siemens S7 DB1 variables."""
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        self.server.set_server_name(SERVER_NAME)

        # Register Industrial Namespace
        uri = "http://siemens.industry.edge/s7-300/predictive_maintenance"
        idx = await self.server.register_namespace(uri)

        # Create Siemens S7 PLC Object & DB1 Data Block Folder
        objects = self.server.nodes.objects
        plc_folder = await objects.add_folder(idx, "Siemens_S7_300_Rack0_Slot2")
        db1_folder = await plc_folder.add_folder(idx, "DB1_DriveTelemetry")

        # Map PLC DB1 variables using clean identifier names (no dots in BrowseNames)
        self.tags["Air_Temp_K"] = await db1_folder.add_variable(
            idx, "DB1_DBD0_Air_Temperature_K", 300.0, ua.VariantType.Float
        )
        self.tags["Process_Temp_K"] = await db1_folder.add_variable(
            idx, "DB1_DBD4_Process_Temperature_K", 310.0, ua.VariantType.Float
        )
        self.tags["Rotational_Speed_RPM"] = await db1_folder.add_variable(
            idx, "DB1_DBD8_Rotational_Speed_RPM", 1500.0, ua.VariantType.Float
        )
        self.tags["Torque_NM"] = await db1_folder.add_variable(
            idx, "DB1_DBD12_Torque_NM", 40.0, ua.VariantType.Float
        )
        self.tags["Tool_Wear_MIN"] = await db1_folder.add_variable(
            idx, "DB1_DBD16_Tool_Wear_MIN", 0.0, ua.VariantType.Float
        )
        self.tags["Machine_Type"] = await db1_folder.add_variable(
            idx, "DB1_DBS20_Machine_Type", "M", ua.VariantType.String
        )
        self.tags["Motor_Running"] = await db1_folder.add_variable(
            idx, "DB1_DBX22_0_Motor_Running", True, ua.VariantType.Boolean
        )
        self.tags["Fault_Trip"] = await db1_folder.add_variable(
            idx, "DB1_DBX22_1_Fault_Trip", False, ua.VariantType.Boolean
        )

        # Make variables writable by clients
        for tag in self.tags.values():
            await tag.set_writable()

        logger.info(f"Configured 8 Siemens DB1 tags in namespace index {idx}.")

    async def start(self, update_interval_sec: float = 1.0, max_cycles: int = None):
        """Start the OPC UA server and publish live sensor telemetry stream."""
        await self.initialize()
        
        # Load telemetry records
        data_file = Path(self.dataset_path)
        if data_file.exists():
            df = pd.read_csv(data_file)
            logger.info(f"Loaded {len(df)} telemetry records from {data_file.name}")
        else:
            df = None
            logger.warning(f"Telemetry file {data_file} not found. Using synthetic generator.")

        self.is_running = True
        cycles = 0

        async with self.server:
            logger.info(f"OPC UA Server listening at: {self.endpoint}")
            logger.info(f"Simulating S7-300 cyclical telemetry publish every {update_interval_sec}s...")

            row_idx = 0
            while self.is_running:
                if max_cycles and cycles >= max_cycles:
                    logger.info(f"Reached max cycles ({max_cycles}). Stopping PLC server.")
                    break

                if df is not None and len(df) > 0:
                    row = df.iloc[row_idx]
                    air_temp = float(row.get("Air temperature [K]", 300.0))
                    proc_temp = float(row.get("Process temperature [K]", 310.0))
                    rpm = float(row.get("Rotational speed [rpm]", 1500.0))
                    torque = float(row.get("Torque [Nm]", 40.0))
                    tool_wear = float(row.get("Tool wear [min]", 10.0))
                    m_type = str(row.get("Type", "M"))
                    row_idx = (row_idx + 1) % len(df)
                else:
                    air_temp = round(300.0 + random.uniform(-1.0, 1.5), 2)
                    proc_temp = round(air_temp + 10.0 + random.uniform(-0.5, 0.5), 2)
                    rpm = round(1538.0 + random.uniform(-50.0, 50.0), 1)
                    torque = round(39.9 + random.uniform(-5.0, 5.0), 2)
                    tool_wear = round(cycles % 250, 1)
                    m_type = "M"

                # Update tags
                await self.tags["Air_Temp_K"].write_value(ua.DataValue(ua.Variant(float(air_temp), ua.VariantType.Float)))
                await self.tags["Process_Temp_K"].write_value(ua.DataValue(ua.Variant(float(proc_temp), ua.VariantType.Float)))
                await self.tags["Rotational_Speed_RPM"].write_value(ua.DataValue(ua.Variant(float(rpm), ua.VariantType.Float)))
                await self.tags["Torque_NM"].write_value(ua.DataValue(ua.Variant(float(torque), ua.VariantType.Float)))
                await self.tags["Tool_Wear_MIN"].write_value(ua.DataValue(ua.Variant(float(tool_wear), ua.VariantType.Float)))
                await self.tags["Machine_Type"].write_value(ua.DataValue(ua.Variant(str(m_type), ua.VariantType.String)))
                await self.tags["Motor_Running"].write_value(ua.DataValue(ua.Variant(True, ua.VariantType.Boolean)))

                cycles += 1
                if cycles % 5 == 0 or cycles == 1:
                    logger.info(
                        f"Cycle #{cycles:04d} published -> Speed: {rpm:6.1f} RPM | Torque: {torque:4.1f} Nm | "
                        f"Temp: {air_temp:5.1f} K | Wear: {tool_wear:5.1f} min"
                    )

                await asyncio.sleep(update_interval_sec)

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulated S7 PLC OPC UA Telemetry Server")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="OPC UA endpoint URI")
    parser.add_argument("--interval", type=float, default=1.0, help="Sensor publish interval in seconds")
    parser.add_argument("--max-cycles", type=int, default=None, help="Stop after N cycles (optional)")
    args = parser.parse_args()

    plc = SiemensS7PLCServer(endpoint=args.endpoint)
    try:
        asyncio.run(plc.start(update_interval_sec=args.interval, max_cycles=args.max_cycles))
    except (KeyboardInterrupt, SystemExit):
        logger.info("PLC Server halted by operator.")
