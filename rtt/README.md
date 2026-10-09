# SEGGER RTT (kws_pdm hop PCM)

Upstream: [SEGGERMicro/RTT](https://github.com/SEGGERMicro/RTT)
Release: **V8.58.0** · commit `4d8feab3150f86f37a9d323ddc88d6cdf5673072`
License: [LICENSE.md](LICENSE.md)

## Channel split (Suite `pdm_rtt_stream`)

| Channel | Contents |
|---|---|
| **1** | Hop PCM: 16 kHz mono **int16**, 320 samples/hop. Not a label. |
| SWO / ITM | `pred=`, `Frame N peak=`, `live pred=` |

Channel 1 is hop PCM, not a label. Do not parse channel 0 as PCM.

Compile with `-DKWS_PDM_RTT_PCM=ON` (default **OFF**). Control block in
`.bss` (TCM). Channel-1 ring is `NSX_MEM_SRAM_BSS`. After each hop the
firmware `SCB_CleanDCache()` so J-Link SWD sees `WrOff`.

Host: [`../host/rtt_pcm_dump.py`](../host/rtt_pcm_dump.py). Stop `nsx view`
first (J-Link OB is exclusive).
