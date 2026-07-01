# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "")
  file(REMOVE_RECURSE
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/ps7_cortexa9_0/standalone_domain/bsp/include/sleep.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/ps7_cortexa9_0/standalone_domain/bsp/include/xiltimer.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/ps7_cortexa9_0/standalone_domain/bsp/include/xtimer_config.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/ps7_cortexa9_0/standalone_domain/bsp/lib/libxiltimer.a"
  )
endif()
