# 2026-07-01T14:26:29.182702
import vitis

client = vitis.create_client()
client.set_workspace(path="vitis_ws_new")

platform = client.get_component(name="wafer_npu_platform")
status = platform.build()

comp = client.get_component(name="wafer_npu_app")
comp.build()

status = platform.build()

comp.build()

