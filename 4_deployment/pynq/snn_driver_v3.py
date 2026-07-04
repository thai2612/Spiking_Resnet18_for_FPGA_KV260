"""
SNN ResNet-18 PYNQ Driver for KV260 - Version 3
Using direct MMIO access
"""

import time
import numpy as np
from pynq import Overlay, MMIO, allocate


class SNNOverlay:
    # AXI-Lite base addresses (from Vivado address map)
    CTRL_ADDR = 0xa0000000      # s_axi_control
    CTRL_R_ADDR = 0xa0010000    # s_axi_control_r

    # Register offsets for s_axi_control
    AP_CTRL = 0x00
    AP_RETURN = 0x10

    # Register offsets for s_axi_control_r
    INPUT_ADDR_LO = 0x10
    INPUT_ADDR_HI = 0x14
    WEIGHTS_ADDR_LO = 0x1c
    WEIGHTS_ADDR_HI = 0x20
    OUTPUT_ADDR_LO = 0x28
    OUTPUT_ADDR_HI = 0x2c

    def __init__(self, bitstream_path):
        """Initialize SNN overlay."""
        print(f"Loading bitstream: {bitstream_path}")
        self.overlay = Overlay(bitstream_path)

        # Create MMIO objects for direct register access
        self.mmio_ctrl = MMIO(self.CTRL_ADDR, 0x100)
        self.mmio_ctrl_r = MMIO(self.CTRL_R_ADDR, 0x100)

        # Verify IP is idle
        ap_ctrl = self.mmio_ctrl.read(self.AP_CTRL)
        print(f"AP_CTRL = 0x{ap_ctrl:02x} (ap_idle={bool(ap_ctrl & 0x4)})")

        self.weights_buf = None
        print("Overlay initialized successfully")

    def load_weights(self, weights_path):
        """Load network weights from binary file into CMA buffer."""
        weights_data = np.fromfile(weights_path, dtype=np.uint8)
        self.weights_buf = allocate(shape=(len(weights_data),), dtype=np.uint8)
        self.weights_buf[:] = weights_data
        self.weights_buf.flush()
        print(f"Loaded {len(weights_data)} bytes of weights")
        print(f"Weights physical address: 0x{self.weights_buf.physical_address:016x}")

    def predict(self, image_256x256x3):
        """Run inference on a single image."""
        if image_256x256x3.shape != (256, 256, 3):
            raise ValueError(f"Expected shape (256, 256, 3), got {image_256x256x3.shape}")

        if self.weights_buf is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

        # Allocate buffers
        in_buf = allocate(shape=(196608,), dtype=np.int8)
        out_buf = allocate(shape=(2,), dtype=np.int32)

        # Copy image to input buffer
        in_buf[:] = image_256x256x3.flatten().astype(np.int8)
        in_buf.flush()

        print(f"Input addr:   0x{in_buf.physical_address:016x}")
        print(f"Weights addr: 0x{self.weights_buf.physical_address:016x}")
        print(f"Output addr:  0x{out_buf.physical_address:016x}")

        # Write addresses to control_r registers
        self.mmio_ctrl_r.write(self.INPUT_ADDR_LO, int(in_buf.physical_address & 0xFFFFFFFF))
        self.mmio_ctrl_r.write(self.INPUT_ADDR_HI, int(in_buf.physical_address >> 32))
        self.mmio_ctrl_r.write(self.WEIGHTS_ADDR_LO, int(self.weights_buf.physical_address & 0xFFFFFFFF))
        self.mmio_ctrl_r.write(self.WEIGHTS_ADDR_HI, int(self.weights_buf.physical_address >> 32))
        self.mmio_ctrl_r.write(self.OUTPUT_ADDR_LO, int(out_buf.physical_address & 0xFFFFFFFF))
        self.mmio_ctrl_r.write(self.OUTPUT_ADDR_HI, int(out_buf.physical_address >> 32))

        # Start HLS IP
        print("Starting inference...")
        start_time = time.time()
        self.mmio_ctrl.write(self.AP_CTRL, 0x01)

        # Wait for ap_done with timeout
        timeout = 120  # seconds
        poll_count = 0
        while True:
            ap_ctrl = self.mmio_ctrl.read(self.AP_CTRL)
            if ap_ctrl & 0x02:  # ap_done
                break
            if time.time() - start_time > timeout:
                raise TimeoutError(f"HLS IP did not complete within {timeout}s. AP_CTRL=0x{ap_ctrl:02x}")
            poll_count += 1
            if poll_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Waiting... {elapsed:.1f}s (AP_CTRL=0x{ap_ctrl:02x})")
            time.sleep(0.1)

        elapsed = time.time() - start_time
        print(f"Inference completed in {elapsed:.2f}s")

        # Read ap_return (argmax result)
        pred = self.mmio_ctrl.read(self.AP_RETURN)

        # Read output buffer
        out_buf.invalidate()
        result = np.array(out_buf, dtype=np.int32).copy()

        # Cleanup
        in_buf.freebuffer()
        out_buf.freebuffer()

        return result, pred

    def predict_class(self, image_256x256x3):
        """Run inference and return predicted class only."""
        _, pred = self.predict(image_256x256x3)
        return int(pred)

    def debug_status(self):
        """Print debug information."""
        ap_ctrl = self.mmio_ctrl.read(self.AP_CTRL)
        print(f"AP_CTRL: 0x{ap_ctrl:02x}")
        print(f"  ap_start: {bool(ap_ctrl & 0x01)}")
        print(f"  ap_done:  {bool(ap_ctrl & 0x02)}")
        print(f"  ap_idle:  {bool(ap_ctrl & 0x04)}")
        print(f"  ap_ready: {bool(ap_ctrl & 0x08)}")
        print(f"AP_RETURN: {self.mmio_ctrl.read(self.AP_RETURN)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python snn_driver_v3.py <bitstream> <weights> <test_image>")
        sys.exit(1)

    bitstream = sys.argv[1]
    weights = sys.argv[2]
    test_image = sys.argv[3]

    snn = SNNOverlay(bitstream)
    snn.load_weights(weights)

    image = np.fromfile(test_image, dtype=np.int8).reshape(256, 256, 3)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    logits, pred = snn.predict(image)
    print(f"\n=== RESULTS ===")
    print(f"Logits: {logits}")
    print(f"Predicted class: {pred}")
    print(f"Class meaning: {'Crack' if pred == 1 else 'No crack'}")
