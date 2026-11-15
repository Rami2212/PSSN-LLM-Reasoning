"""Optional AMD SMI energy-counter readings for measured GPU inference."""
from __future__ import annotations


class GpuEnergyMeter:
    """Read the GPU's cumulative energy counter; gracefully disable if unavailable."""

    def __init__(self, device_index=0):
        self._smi = None
        self._device = None
        self.reason = None
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            devices = amdsmi.amdsmi_get_processor_handles()
            if device_index >= len(devices):
                raise RuntimeError(f"AMD SMI found {len(devices)} GPUs; requested index {device_index}")
            self._smi = amdsmi
            self._device = devices[device_index]
            self._read()
        except Exception as exc:
            self.reason = f"AMD SMI energy counter unavailable: {type(exc).__name__}: {exc}"
            self.close()

    @property
    def available(self):
        return self._smi is not None and self._device is not None

    def _read(self):
        reading = self._smi.amdsmi_get_energy_count(self._device)
        energy_uj = float(reading["energy_accumulator"]) * float(reading["counter_resolution"])
        return energy_uj / 1_000_000

    def snapshot_joules(self):
        if not self.available:
            return None
        try:
            return self._read()
        except Exception as exc:
            self.reason = f"AMD SMI energy read failed: {type(exc).__name__}: {exc}"
            self.close()
            return None

    def delta_joules(self, before, after):
        if before is None or after is None or after < before:
            return None
        return after - before

    def close(self):
        if self._smi is not None:
            try:
                self._smi.amdsmi_shut_down()
            except Exception:
                pass
        self._smi = None
        self._device = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
