# 2026-07-01T13:08:32.699390
import vitis

client = vitis.create_client()
client.set_workspace(path="vitis_ws_new")

comp = client.create_app_component(name="wafer_npu_app",platform = "/tmp/claude-1000/-home-kimyujeong-workspace-ondevice-2-09-wafer-npu-c/4437d63d-1bfa-44d4-b029-ecefa49e89d5/scratchpad/wafer_npu_platform",domain = "standalone_domain",template = "empty")

vitis.dispose()

