# 2026-07-01T13:09:07.373175
import vitis

client = vitis.create_client()
client.set_workspace(path="vitis_ws_new")

comp = client.create_app_component(name="wafer_npu_app",platform = "$COMPONENT_LOCATION/../wafer_npu_platform",domain = "standalone_domain",template = "empty")

vitis.dispose()

