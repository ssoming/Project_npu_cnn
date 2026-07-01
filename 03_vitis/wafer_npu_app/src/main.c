/*
 * wafer_npu_app - PC <-> cnn_npu (AXI-Lite, base 0x43C00000) UART bridge.
 *
 * Protocol (UART1, 115200 baud, matches the Zybo Z7-20 board's on-board
 * USB-UART bridge - PCW_UART1_UART1_IO = MIO 48..49 per the board preset):
 *
 *   [IDLE] --(rx byte == 0xAA)--> [RECV_IMAGE]
 *   [RECV_IMAGE] --(2304 bytes received; each pixel: write IMG_ADDR=i,
 *                    then IMG_DATA=pixel)--> [START_INFER]
 *   [START_INFER] --(write CTRL.start=1)--> [WAIT_DONE]
 *   [WAIT_DONE] --(poll STATUS until done bit set)--> [SEND_RESULT]
 *   [SEND_RESULT] --(read CLASS_RESULT; send 1 result byte + 0x55)--> [IDLE]
 *
 *   Any byte received in IDLE that isn't 0xAA is discarded (resync).
 *
 * IMPORTANT: this board exposes exactly one physical UART (PS UART1 wired
 * to the on-board USB-UART bridge - there is no second channel to route
 * debug output to instead). So xil_printf/print/putchar are never called
 * anywhere in this file - only XUartPs_Send/XUartPs_Recv are used, and only
 * for the protocol bytes above. A previous version of this project mixed
 * debug prints into the same UART used for the binary protocol, corrupting
 * it; this file avoids that entire class of bug by construction rather than
 * by convention (no stdio call exists in this translation unit at all).
 */

#include "xparameters.h"
#include "xil_io.h"
#include "xuartps.h"
#include "xstatus.h"

/* ---- cnn_npu (axi_lite_cnn_wrapper) register map, base 0x43C00000 ---- */
#define CNN_NPU_BASEADDR    XPAR_CNN_NPU_0_BASEADDR
#define REG_CTRL            (CNN_NPU_BASEADDR + 0x00u)
#define REG_STATUS          (CNN_NPU_BASEADDR + 0x04u)
#define REG_SW_RESET        (CNN_NPU_BASEADDR + 0x08u)
#define REG_CLASS_RESULT    (CNN_NPU_BASEADDR + 0x0Cu)
#define REG_IMG_DATA        (CNN_NPU_BASEADDR + 0x10u)
#define REG_IMG_ADDR        (CNN_NPU_BASEADDR + 0x14u)

#define CTRL_START_BIT      0x1u
#define STATUS_DONE_MASK    0x2u
#define CLASS_RESULT_MASK   0xFu

#define IMG_W          48u
#define IMG_H          48u
#define IMG_PIXELS     (IMG_W * IMG_H)   /* 2304 */

#define PROTO_START_BYTE   0xAAu
#define PROTO_END_BYTE     0x55u

#define UART_BAUD_RATE     115200

static XUartPs Uart;

static int uart_init(void)
{
    XUartPs_Config *cfg = XUartPs_LookupConfig(XPAR_XUARTPS_0_BASEADDR);
    if (cfg == NULL) {
        return XST_FAILURE;
    }
    if (XUartPs_CfgInitialize(&Uart, cfg, cfg->BaseAddress) != XST_SUCCESS) {
        return XST_FAILURE;
    }
    XUartPs_SetBaudRate(&Uart, UART_BAUD_RATE);
    XUartPs_SetOperMode(&Uart, XUARTPS_OPER_MODE_NORMAL);
    return XST_SUCCESS;
}

/* polled, blocking single-byte I/O - the only thing ever written to or
 * read from this UART is protocol bytes (see file header) */
static inline u8 uart_recv_byte(void)
{
    u8 b;
    while (XUartPs_Recv(&Uart, &b, 1) == 0) {
    }
    return b;
}

static inline void uart_send_byte(u8 b)
{
    while (XUartPs_Send(&Uart, &b, 1) == 0) {
    }
}

int main(void)
{
    if (uart_init() != XST_SUCCESS) {
        /* UART itself is broken - there is no channel to report this on,
         * so just halt rather than run with a peripheral that didn't init */
        while (1) {
        }
    }

    /* known-clean start for the CNN core before the first inference */
    Xil_Out32(REG_SW_RESET, 1u);

    for (;;) {
        /* ---- IDLE: wait for the start byte, discard anything else ---- */
        u8 b;
        do {
            b = uart_recv_byte();
        } while (b != PROTO_START_BYTE);

        /* ---- RECV_IMAGE: stream 2304 pixels straight into the core ---- */
        for (u32 i = 0; i < IMG_PIXELS; i++) {
            u8 pixel = uart_recv_byte();
            Xil_Out32(REG_IMG_ADDR, i);
            Xil_Out32(REG_IMG_DATA, (u32)pixel);
        }

        /* ---- START_INFER ---- */
        Xil_Out32(REG_CTRL, CTRL_START_BIT);

        /* ---- WAIT_DONE ---- */
        u32 status;
        do {
            status = Xil_In32(REG_STATUS);
        } while ((status & STATUS_DONE_MASK) == 0u);

        /* ---- SEND_RESULT ---- */
        u32 class_result = Xil_In32(REG_CLASS_RESULT) & CLASS_RESULT_MASK;
        uart_send_byte((u8)class_result);
        uart_send_byte(PROTO_END_BYTE);
    }

    return 0;
}
