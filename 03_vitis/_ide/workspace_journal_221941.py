# 2026-07-01T13:12:12.831970
import vitis

client = vitis.create_client()
client.set_workspace(path="vitis_ws_new")

comp = client.create_app_component(name="wafer_npu_app",platform = "$COMPONENT_LOCATION/../wafer_npu_platform/export/wafer_npu_platform/wafer_npu_platform.xpfm",domain = "standalone_domain",template = "empty_application")

vitis.dispose()

