# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "ACC_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "ADDR_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BIAS_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C0" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C1" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C2" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C3" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONV1_BIAS_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONV1_FRAC_BITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONV1_WEIGHT_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONV2_BIAS_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONV2_FRAC_BITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONV2_WEIGHT_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONV3_BIAS_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONV3_FRAC_BITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CONV3_WEIGHT_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_S_AXI_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_S_AXI_DATA_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DATA_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FC_BIAS_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FC_FRAC_BITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FC_WEIGHT_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IMG0_H" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IMG0_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "NUM_CLASSES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SCORE_W" -parent ${Page_0}


}

proc update_PARAM_VALUE.ACC_W { PARAM_VALUE.ACC_W } {
	# Procedure called to update ACC_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ACC_W { PARAM_VALUE.ACC_W } {
	# Procedure called to validate ACC_W
	return true
}

proc update_PARAM_VALUE.ADDR_W { PARAM_VALUE.ADDR_W } {
	# Procedure called to update ADDR_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.ADDR_W { PARAM_VALUE.ADDR_W } {
	# Procedure called to validate ADDR_W
	return true
}

proc update_PARAM_VALUE.BIAS_W { PARAM_VALUE.BIAS_W } {
	# Procedure called to update BIAS_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BIAS_W { PARAM_VALUE.BIAS_W } {
	# Procedure called to validate BIAS_W
	return true
}

proc update_PARAM_VALUE.C0 { PARAM_VALUE.C0 } {
	# Procedure called to update C0 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C0 { PARAM_VALUE.C0 } {
	# Procedure called to validate C0
	return true
}

proc update_PARAM_VALUE.C1 { PARAM_VALUE.C1 } {
	# Procedure called to update C1 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C1 { PARAM_VALUE.C1 } {
	# Procedure called to validate C1
	return true
}

proc update_PARAM_VALUE.C2 { PARAM_VALUE.C2 } {
	# Procedure called to update C2 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C2 { PARAM_VALUE.C2 } {
	# Procedure called to validate C2
	return true
}

proc update_PARAM_VALUE.C3 { PARAM_VALUE.C3 } {
	# Procedure called to update C3 when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C3 { PARAM_VALUE.C3 } {
	# Procedure called to validate C3
	return true
}

proc update_PARAM_VALUE.CONV1_BIAS_FILE { PARAM_VALUE.CONV1_BIAS_FILE } {
	# Procedure called to update CONV1_BIAS_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV1_BIAS_FILE { PARAM_VALUE.CONV1_BIAS_FILE } {
	# Procedure called to validate CONV1_BIAS_FILE
	return true
}

proc update_PARAM_VALUE.CONV1_FRAC_BITS { PARAM_VALUE.CONV1_FRAC_BITS } {
	# Procedure called to update CONV1_FRAC_BITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV1_FRAC_BITS { PARAM_VALUE.CONV1_FRAC_BITS } {
	# Procedure called to validate CONV1_FRAC_BITS
	return true
}

proc update_PARAM_VALUE.CONV1_WEIGHT_FILE { PARAM_VALUE.CONV1_WEIGHT_FILE } {
	# Procedure called to update CONV1_WEIGHT_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV1_WEIGHT_FILE { PARAM_VALUE.CONV1_WEIGHT_FILE } {
	# Procedure called to validate CONV1_WEIGHT_FILE
	return true
}

proc update_PARAM_VALUE.CONV2_BIAS_FILE { PARAM_VALUE.CONV2_BIAS_FILE } {
	# Procedure called to update CONV2_BIAS_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV2_BIAS_FILE { PARAM_VALUE.CONV2_BIAS_FILE } {
	# Procedure called to validate CONV2_BIAS_FILE
	return true
}

proc update_PARAM_VALUE.CONV2_FRAC_BITS { PARAM_VALUE.CONV2_FRAC_BITS } {
	# Procedure called to update CONV2_FRAC_BITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV2_FRAC_BITS { PARAM_VALUE.CONV2_FRAC_BITS } {
	# Procedure called to validate CONV2_FRAC_BITS
	return true
}

proc update_PARAM_VALUE.CONV2_WEIGHT_FILE { PARAM_VALUE.CONV2_WEIGHT_FILE } {
	# Procedure called to update CONV2_WEIGHT_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV2_WEIGHT_FILE { PARAM_VALUE.CONV2_WEIGHT_FILE } {
	# Procedure called to validate CONV2_WEIGHT_FILE
	return true
}

proc update_PARAM_VALUE.CONV3_BIAS_FILE { PARAM_VALUE.CONV3_BIAS_FILE } {
	# Procedure called to update CONV3_BIAS_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV3_BIAS_FILE { PARAM_VALUE.CONV3_BIAS_FILE } {
	# Procedure called to validate CONV3_BIAS_FILE
	return true
}

proc update_PARAM_VALUE.CONV3_FRAC_BITS { PARAM_VALUE.CONV3_FRAC_BITS } {
	# Procedure called to update CONV3_FRAC_BITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV3_FRAC_BITS { PARAM_VALUE.CONV3_FRAC_BITS } {
	# Procedure called to validate CONV3_FRAC_BITS
	return true
}

proc update_PARAM_VALUE.CONV3_WEIGHT_FILE { PARAM_VALUE.CONV3_WEIGHT_FILE } {
	# Procedure called to update CONV3_WEIGHT_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CONV3_WEIGHT_FILE { PARAM_VALUE.CONV3_WEIGHT_FILE } {
	# Procedure called to validate CONV3_WEIGHT_FILE
	return true
}

proc update_PARAM_VALUE.C_S_AXI_ADDR_WIDTH { PARAM_VALUE.C_S_AXI_ADDR_WIDTH } {
	# Procedure called to update C_S_AXI_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_S_AXI_ADDR_WIDTH { PARAM_VALUE.C_S_AXI_ADDR_WIDTH } {
	# Procedure called to validate C_S_AXI_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.C_S_AXI_DATA_WIDTH { PARAM_VALUE.C_S_AXI_DATA_WIDTH } {
	# Procedure called to update C_S_AXI_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_S_AXI_DATA_WIDTH { PARAM_VALUE.C_S_AXI_DATA_WIDTH } {
	# Procedure called to validate C_S_AXI_DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.DATA_W { PARAM_VALUE.DATA_W } {
	# Procedure called to update DATA_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DATA_W { PARAM_VALUE.DATA_W } {
	# Procedure called to validate DATA_W
	return true
}

proc update_PARAM_VALUE.FC_BIAS_FILE { PARAM_VALUE.FC_BIAS_FILE } {
	# Procedure called to update FC_BIAS_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FC_BIAS_FILE { PARAM_VALUE.FC_BIAS_FILE } {
	# Procedure called to validate FC_BIAS_FILE
	return true
}

proc update_PARAM_VALUE.FC_FRAC_BITS { PARAM_VALUE.FC_FRAC_BITS } {
	# Procedure called to update FC_FRAC_BITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FC_FRAC_BITS { PARAM_VALUE.FC_FRAC_BITS } {
	# Procedure called to validate FC_FRAC_BITS
	return true
}

proc update_PARAM_VALUE.FC_WEIGHT_FILE { PARAM_VALUE.FC_WEIGHT_FILE } {
	# Procedure called to update FC_WEIGHT_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FC_WEIGHT_FILE { PARAM_VALUE.FC_WEIGHT_FILE } {
	# Procedure called to validate FC_WEIGHT_FILE
	return true
}

proc update_PARAM_VALUE.IMG0_H { PARAM_VALUE.IMG0_H } {
	# Procedure called to update IMG0_H when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IMG0_H { PARAM_VALUE.IMG0_H } {
	# Procedure called to validate IMG0_H
	return true
}

proc update_PARAM_VALUE.IMG0_W { PARAM_VALUE.IMG0_W } {
	# Procedure called to update IMG0_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IMG0_W { PARAM_VALUE.IMG0_W } {
	# Procedure called to validate IMG0_W
	return true
}

proc update_PARAM_VALUE.NUM_CLASSES { PARAM_VALUE.NUM_CLASSES } {
	# Procedure called to update NUM_CLASSES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NUM_CLASSES { PARAM_VALUE.NUM_CLASSES } {
	# Procedure called to validate NUM_CLASSES
	return true
}

proc update_PARAM_VALUE.SCORE_W { PARAM_VALUE.SCORE_W } {
	# Procedure called to update SCORE_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SCORE_W { PARAM_VALUE.SCORE_W } {
	# Procedure called to validate SCORE_W
	return true
}


proc update_MODELPARAM_VALUE.C_S_AXI_DATA_WIDTH { MODELPARAM_VALUE.C_S_AXI_DATA_WIDTH PARAM_VALUE.C_S_AXI_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_S_AXI_DATA_WIDTH}] ${MODELPARAM_VALUE.C_S_AXI_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.C_S_AXI_ADDR_WIDTH { MODELPARAM_VALUE.C_S_AXI_ADDR_WIDTH PARAM_VALUE.C_S_AXI_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_S_AXI_ADDR_WIDTH}] ${MODELPARAM_VALUE.C_S_AXI_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.IMG0_W { MODELPARAM_VALUE.IMG0_W PARAM_VALUE.IMG0_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IMG0_W}] ${MODELPARAM_VALUE.IMG0_W}
}

proc update_MODELPARAM_VALUE.IMG0_H { MODELPARAM_VALUE.IMG0_H PARAM_VALUE.IMG0_H } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IMG0_H}] ${MODELPARAM_VALUE.IMG0_H}
}

proc update_MODELPARAM_VALUE.C0 { MODELPARAM_VALUE.C0 PARAM_VALUE.C0 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C0}] ${MODELPARAM_VALUE.C0}
}

proc update_MODELPARAM_VALUE.C1 { MODELPARAM_VALUE.C1 PARAM_VALUE.C1 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C1}] ${MODELPARAM_VALUE.C1}
}

proc update_MODELPARAM_VALUE.C2 { MODELPARAM_VALUE.C2 PARAM_VALUE.C2 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C2}] ${MODELPARAM_VALUE.C2}
}

proc update_MODELPARAM_VALUE.C3 { MODELPARAM_VALUE.C3 PARAM_VALUE.C3 } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C3}] ${MODELPARAM_VALUE.C3}
}

proc update_MODELPARAM_VALUE.NUM_CLASSES { MODELPARAM_VALUE.NUM_CLASSES PARAM_VALUE.NUM_CLASSES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NUM_CLASSES}] ${MODELPARAM_VALUE.NUM_CLASSES}
}

proc update_MODELPARAM_VALUE.DATA_W { MODELPARAM_VALUE.DATA_W PARAM_VALUE.DATA_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DATA_W}] ${MODELPARAM_VALUE.DATA_W}
}

proc update_MODELPARAM_VALUE.BIAS_W { MODELPARAM_VALUE.BIAS_W PARAM_VALUE.BIAS_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BIAS_W}] ${MODELPARAM_VALUE.BIAS_W}
}

proc update_MODELPARAM_VALUE.ACC_W { MODELPARAM_VALUE.ACC_W PARAM_VALUE.ACC_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ACC_W}] ${MODELPARAM_VALUE.ACC_W}
}

proc update_MODELPARAM_VALUE.SCORE_W { MODELPARAM_VALUE.SCORE_W PARAM_VALUE.SCORE_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SCORE_W}] ${MODELPARAM_VALUE.SCORE_W}
}

proc update_MODELPARAM_VALUE.ADDR_W { MODELPARAM_VALUE.ADDR_W PARAM_VALUE.ADDR_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.ADDR_W}] ${MODELPARAM_VALUE.ADDR_W}
}

proc update_MODELPARAM_VALUE.CONV1_FRAC_BITS { MODELPARAM_VALUE.CONV1_FRAC_BITS PARAM_VALUE.CONV1_FRAC_BITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV1_FRAC_BITS}] ${MODELPARAM_VALUE.CONV1_FRAC_BITS}
}

proc update_MODELPARAM_VALUE.CONV2_FRAC_BITS { MODELPARAM_VALUE.CONV2_FRAC_BITS PARAM_VALUE.CONV2_FRAC_BITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV2_FRAC_BITS}] ${MODELPARAM_VALUE.CONV2_FRAC_BITS}
}

proc update_MODELPARAM_VALUE.CONV3_FRAC_BITS { MODELPARAM_VALUE.CONV3_FRAC_BITS PARAM_VALUE.CONV3_FRAC_BITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV3_FRAC_BITS}] ${MODELPARAM_VALUE.CONV3_FRAC_BITS}
}

proc update_MODELPARAM_VALUE.FC_FRAC_BITS { MODELPARAM_VALUE.FC_FRAC_BITS PARAM_VALUE.FC_FRAC_BITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FC_FRAC_BITS}] ${MODELPARAM_VALUE.FC_FRAC_BITS}
}

proc update_MODELPARAM_VALUE.CONV1_WEIGHT_FILE { MODELPARAM_VALUE.CONV1_WEIGHT_FILE PARAM_VALUE.CONV1_WEIGHT_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV1_WEIGHT_FILE}] ${MODELPARAM_VALUE.CONV1_WEIGHT_FILE}
}

proc update_MODELPARAM_VALUE.CONV1_BIAS_FILE { MODELPARAM_VALUE.CONV1_BIAS_FILE PARAM_VALUE.CONV1_BIAS_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV1_BIAS_FILE}] ${MODELPARAM_VALUE.CONV1_BIAS_FILE}
}

proc update_MODELPARAM_VALUE.CONV2_WEIGHT_FILE { MODELPARAM_VALUE.CONV2_WEIGHT_FILE PARAM_VALUE.CONV2_WEIGHT_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV2_WEIGHT_FILE}] ${MODELPARAM_VALUE.CONV2_WEIGHT_FILE}
}

proc update_MODELPARAM_VALUE.CONV2_BIAS_FILE { MODELPARAM_VALUE.CONV2_BIAS_FILE PARAM_VALUE.CONV2_BIAS_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV2_BIAS_FILE}] ${MODELPARAM_VALUE.CONV2_BIAS_FILE}
}

proc update_MODELPARAM_VALUE.CONV3_WEIGHT_FILE { MODELPARAM_VALUE.CONV3_WEIGHT_FILE PARAM_VALUE.CONV3_WEIGHT_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV3_WEIGHT_FILE}] ${MODELPARAM_VALUE.CONV3_WEIGHT_FILE}
}

proc update_MODELPARAM_VALUE.CONV3_BIAS_FILE { MODELPARAM_VALUE.CONV3_BIAS_FILE PARAM_VALUE.CONV3_BIAS_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CONV3_BIAS_FILE}] ${MODELPARAM_VALUE.CONV3_BIAS_FILE}
}

proc update_MODELPARAM_VALUE.FC_WEIGHT_FILE { MODELPARAM_VALUE.FC_WEIGHT_FILE PARAM_VALUE.FC_WEIGHT_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FC_WEIGHT_FILE}] ${MODELPARAM_VALUE.FC_WEIGHT_FILE}
}

proc update_MODELPARAM_VALUE.FC_BIAS_FILE { MODELPARAM_VALUE.FC_BIAS_FILE PARAM_VALUE.FC_BIAS_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FC_BIAS_FILE}] ${MODELPARAM_VALUE.FC_BIAS_FILE}
}

