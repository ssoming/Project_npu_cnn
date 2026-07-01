# ============================================================================
# package_ip.tcl
#   End-to-end flow for the wafer NPU project:
#     1. Package axi_lite_cnn_wrapper.v (+ cnn_core_fsm/conv_layer/maxpool/
#        global_avg_pool/fc) as Vivado IP "cnn_npu" (v1.0), with AXI4-Lite
#        interface inference.
#     2. Create a block design: ZYNQ7 Processing System (UART1 on MIO,
#        M_AXI_GP0 enabled) + cnn_npu_0, auto-connected via AXI
#        interconnect/SmartConnect + Processor System Reset, base address
#        fixed at 0x43C00000.
#     3. Generate the HDL wrapper, run synthesis + implementation +
#        write_bitstream.
#     4. Export a fixed (bitstream-included) hardware platform .xsa for
#        Vitis.
#
#   Run with: vivado -mode batch -source package_ip.tcl -log package_ip.log
# ============================================================================

set proj_dir   "/home/kimyujeong/workspace_ondevice_2/09.wafer_npu_project/C_vivado"
set proj_name  "C_vivado"
set ip_repo_dir "$proj_dir/ip_repo_cnn_npu"
set bd_name    "wafer_npu_bd"
set xsa_out    "$proj_dir/wafer_npu.xsa"
set num_jobs   8

puts "INFO: ===== STEP 0: open project ====="

# Clean ALL Vivado-managed generated-artifact directories before opening the
# project, so every run starts from a truly fresh state. This is required
# for idempotent re-runs of this script: re-packaging the same IP VLNV
# (user.org:user:cnn_npu:1.0) into the same ip_repo_dir while a STALE IP
# cache from a previous run still exists (under C_vivado.cache/ip,
# C_vivado.ip_user_files, or the BD's generated products in C_vivado.gen)
# causes Vivado to mark the IP "hidden" and fail to create the BD cell with
# "Create IP failed with errors" / "IP is hidden" - seen when iterating on
# this script after a successful first build. None of these directories
# contain anything we authored by hand (that all lives under C_vivado.srcs
# and this .tcl file itself), so deleting them is safe.
foreach d [list "$proj_dir/ip_repo_cnn_npu" "$proj_dir/$proj_name.cache" \
                "$proj_dir/$proj_name.ip_user_files" "$proj_dir/$proj_name.gen" \
                "$proj_dir/$proj_name.runs" "$proj_dir/$proj_name.sim"] {
    if {[file exists $d]} {
        puts "INFO: removing stale generated-artifact dir: $d"
        file delete -force $d
    }
}

open_project "$proj_dir/$proj_name.xpr"

# ----------------------------------------------------------------------
# STEP 1: package cnn_npu IP from the existing sources_1 fileset
#   (axi_lite_cnn_wrapper.v + cnn_core_fsm/conv_layer/maxpool/
#   global_avg_pool/fc.v are already the entire contents of sources_1 -
#   see the file list check performed before writing this script - so no
#   separate throwaway packaging project is needed; ipx::package_project
#   is pointed at a fresh root_dir and imports exactly those files.)
# ----------------------------------------------------------------------
puts "INFO: ===== STEP 1: package cnn_npu IP ====="

file mkdir $ip_repo_dir

set_property top axi_lite_cnn_wrapper [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $ip_repo_dir -vendor user.org -library user -taxonomy /UserIP \
    -import_files -set_current true

set_property name        cnn_npu           [ipx::current_core]
set_property display_name "CNN NPU (WM-811K wafer defect classifier)" [ipx::current_core]
set_property description "AXI4-Lite wrapped CNN NPU (conv/pool/gap/fc pipeline) for wafer defect classification" [ipx::current_core]
set_property version     1.0               [ipx::current_core]
set_property vendor      user.org          [ipx::current_core]
set_property library     user              [ipx::current_core]

# --- AXI4-Lite / clock / reset interface inference ---
# ipx::package_project already auto-infers the whole s_axi_* port group as
# one bus interface named 's_axi' of abstraction 'xilinx.com:interface:
# aximm:1.0' (Vivado 2024.2 has no separate axi4lite_rtl abstraction - full
# AXI4 and AXI4-Lite share the aximm_rtl definition; since our port list
# omits every AXI4-only signal - awlen/awsize/awburst/arlen/arsize/arburst/
# awid/arid/etc - Vivado's detector should set PROTOCOL=AXI4LITE on it
# automatically). Confirmed below rather than assumed, and forced if not.
if {[llength [ipx::get_bus_interfaces s_axi -of_objects [ipx::current_core]]] == 0} {
    puts "ERROR: s_axi bus interface was not auto-inferred by package_project"
    exit 1
}
set s_axi_busif [ipx::get_bus_interfaces s_axi -of_objects [ipx::current_core]]
set proto_param [ipx::get_bus_parameters -quiet PROTOCOL -of_objects $s_axi_busif]
if {[llength $proto_param] > 0} {
    puts "INFO: s_axi PROTOCOL auto-detected as [get_property VALUE $proto_param]"
    if {[get_property VALUE $proto_param] != "AXI4LITE"} {
        set_property value AXI4LITE $proto_param
        puts "INFO: forced s_axi PROTOCOL to AXI4LITE"
    }
} else {
    ipx::add_bus_parameter PROTOCOL $s_axi_busif
    set_property value AXI4LITE [ipx::get_bus_parameters PROTOCOL -of_objects $s_axi_busif]
    puts "INFO: added and set s_axi PROTOCOL=AXI4LITE (was not present)"
}

# these two succeed even though package_project's auto-scan typically finds
# them too (see the INFO messages already logged above) - kept explicit
# per the requirement to include interface inference in this script.
ipx::infer_bus_interface s_axi_aclk    xilinx.com:signal:clock_rtl:1.0 [ipx::current_core]
ipx::infer_bus_interface s_axi_aresetn xilinx.com:signal:reset_rtl:1.0 [ipx::current_core]

# FREQ_HZ parameter on the clock interface (avoids a packaging warning and
# lets downstream BD clock-frequency propagation see this port explicitly)
if {[llength [ipx::get_bus_parameters -quiet FREQ_HZ -of_objects [ipx::get_bus_interfaces s_axi_aclk -of_objects [ipx::current_core]]]] == 0} {
    ipx::add_bus_parameter FREQ_HZ [ipx::get_bus_interfaces s_axi_aclk -of_objects [ipx::current_core]]
}

ipx::create_xgui_files    [ipx::current_core]
ipx::update_checksums     [ipx::current_core]
ipx::save_core            [ipx::current_core]

close_project

puts "INFO: ===== STEP 1 done: IP packaged at $ip_repo_dir ====="

# ----------------------------------------------------------------------
# reopen the main project (packaging closed it) and register the IP repo
# ----------------------------------------------------------------------
open_project "$proj_dir/$proj_name.xpr"

set_property ip_repo_paths [list $ip_repo_dir] [current_project]
update_ip_catalog -rebuild

set cnn_vlnv [get_ipdefs -filter {NAME == "cnn_npu"}]
if {[llength $cnn_vlnv] == 0} {
    puts "ERROR: cnn_npu IP not found in catalog after update_ip_catalog"
    exit 1
}
set cnn_vlnv [lindex $cnn_vlnv 0]
puts "INFO: cnn_npu VLNV = $cnn_vlnv"

# ----------------------------------------------------------------------
# STEP 2: block design
# ----------------------------------------------------------------------
puts "INFO: ===== STEP 2: block design ====="

# idempotent: if a previous attempt already created this BD (e.g. a retry
# after an earlier step failed), remove it and its backing files first so
# create_bd_design doesn't fail with "already exists".
if {[llength [get_files -quiet "$bd_name.bd"]] > 0} {
    remove_files -quiet "$bd_name.bd"
}
set bd_file_path "$proj_dir/$proj_name.srcs/sources_1/bd/$bd_name"
if {[file exists $bd_file_path]} {
    file delete -force $bd_file_path
}

create_bd_design $bd_name

# --- ZYNQ7 Processing System ---
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]

# apply the board's default preset (DDR/FIXED_IO/clocking for Zybo Z7-20),
# then explicitly force the two things this project actually needs:
# UART1 on MIO, and M_AXI_GP0 enabled - set explicitly rather than trusting
# only the board preset, since the preset's UART/GP0 defaults can vary
# across board-file versions.
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} \
    [get_bd_cells processing_system7_0]

set_property -dict [list \
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART1_GRP_FULL_ENABLE {0} \
    CONFIG.PCW_UART1_UART1_IO {MIO 48 .. 49} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {40} \
] [get_bd_cells processing_system7_0]
# NOTE: FCLK0 default is 50MHz (period 20ns). A first build at 50MHz failed
# timing by a small margin (WNS=-0.190ns, only 4 endpoints) entirely inside
# global_avg_pool's single-cycle round-half-away-from-zero divide-by-N_PIX
# logic (31 logic levels, mostly a CARRY4 chain - integer division by a
# non-power-of-two constant is inherently a fairly deep combinational
# circuit). Rather than pipeline that divide in already-verified RTL, the
# clock is lowered to 40MHz (25ns period, ~5ns of margin over the ~19.9ns
# critical path) - this NPU's runtime is dominated by cycle COUNT (millions
# of cycles per inference regardless), not clock speed, so trading 50->40MHz
# costs a small constant-factor slowdown in an already non-real-time
# workload, in exchange for zero risk to already-unit-tested logic.

# --- cnn_npu_0 ---
set cnn [create_bd_cell -type ip -vlnv $cnn_vlnv cnn_npu_0]

# --- connect PS7 M_AXI_GP0 -> cnn_npu_0 S_AXI ---
#   this automation inserts the AXI interconnect/SmartConnect AND the
#   Processor System Reset block automatically (both required whenever a
#   fresh AXI master/slave pair is wired up), satisfying both of those
#   requirements in one step.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Master "/processing_system7_0/M_AXI_GP0" \
    intc_ip "New AXI Interconnect" \
    Clk "Auto" \
} [get_bd_intf_pins cnn_npu_0/s_axi]

# --- fix cnn_npu_0's base address at 0x43C00000 ---
# NOTE: calling plain `assign_bd_address` first (auto-assign) and then
# trying to `set_property offset ...` on the resulting segment fails with
# "Cannot change read-only property 'offset'" - the segment object
# returned by get_bd_addr_segs -of_objects is a read-only view once
# auto-assigned. The correct way to pin a specific address is to pass
# -offset/-range directly to assign_bd_address itself (before any other
# auto-assign touches this segment), which is what we do here.
set cnn_seg [get_bd_addr_segs -of_objects [get_bd_cells cnn_npu_0]]
if {[llength $cnn_seg] == 0} {
    puts "ERROR: no address segment found for cnn_npu_0 - check AXI connection"
    exit 1
}
assign_bd_address -target_address_space [get_bd_addr_spaces processing_system7_0/Data] \
    $cnn_seg -offset 0x43C00000 -range 4K -force

# assign any other still-unassigned segments in the design with defaults
assign_bd_address

regenerate_bd_layout
validate_bd_design

save_bd_design

puts "INFO: ===== STEP 2 done: block design validated ====="

# ----------------------------------------------------------------------
# STEP 3: HDL wrapper, synthesis, implementation, bitstream
# ----------------------------------------------------------------------
puts "INFO: ===== STEP 3: generate wrapper + build ====="

make_wrapper -files [get_files "$bd_name.bd"] -top
add_files -norecurse "$proj_dir/$proj_name.gen/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v"
update_compile_order -fileset sources_1
set_property top "${bd_name}_wrapper" [current_fileset]
update_compile_order -fileset sources_1

generate_target all [get_files "$bd_name.bd"]

reset_run synth_1
launch_runs synth_1 -jobs $num_jobs
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: synth_1 did not complete successfully"
    exit 1
}
puts "INFO: synth_1 complete"

launch_runs impl_1 -to_step write_bitstream -jobs $num_jobs
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: impl_1 did not complete successfully"
    exit 1
}
puts "INFO: impl_1 + write_bitstream complete"

# ----------------------------------------------------------------------
# STEP 4: export hardware platform (.xsa) for Vitis
# ----------------------------------------------------------------------
puts "INFO: ===== STEP 4: write hardware platform ====="

open_run impl_1
write_hw_platform -fixed -include_bit -force $xsa_out

if {[file exists $xsa_out]} {
    puts "INFO: ===== SUCCESS: $xsa_out generated ====="
} else {
    puts "ERROR: $xsa_out was not created"
    exit 1
}
