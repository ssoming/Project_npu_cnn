# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "")
  file(REMOVE_RECURSE
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/include/diskio.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/include/ff.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/include/ffconf.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/include/sleep.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/include/xilffs.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/include/xilffs_config.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/include/xilrsa.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/include/xiltimer.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/include/xtimer_config.h"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/lib/libxilffs.a"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/lib/libxilrsa.a"
  "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/vitis_ws_new/wafer_npu_platform/zynq_fsbl/zynq_fsbl_bsp/lib/libxiltimer.a"
  )
endif()
