# 2026-07-01T13:06:15.836931
import vitis

client = vitis.create_client()
client.set_workspace(path="vitis_ws_new")

platform = client.create_platform_component(name = "wafer_npu_platform",hw_design = "$COMPONENT_LOCATION/../../C_vivado/wafer_npu.xsa",os = "standalone",cpu = "ps7_cortexa9_0",domain_name = "standalone_domain",generate_dtb = True)

platform = client.get_component(name="wafer_npu_platform")
status = platform.build()

vitis.dispose()

