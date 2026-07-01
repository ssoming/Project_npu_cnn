# 2026-07-01T14:11:59.977954
import vitis

client = vitis.create_client()
client.set_workspace(path="vitis_ws_new")

platform = client.get_component(name="wafer_npu_platform")
status = platform.build()

comp = client.get_component(name="wafer_npu_app")
comp.build()

vitis.dispose()

