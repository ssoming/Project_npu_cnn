# 2026-07-01T13:42:05.754701
import vitis

client = vitis.create_client()
client.set_workspace(path="vitis_ws_new")

comp = client.get_component(name="wafer_npu_app")
comp.build()

vitis.dispose()

