# `.modraw` tag reference: how to read every `$TAG`

**Status:** reference doc, reverse-engineered from the Matlab reader `mod_som_read_epsi_files_v4.m` in the `MOD_fish_lib` repo, the version used by the current pipeline. It describes how the bytes of each tag are laid out. `modfish.modraw` implements the subset listed in its module docstring.

> **Which version this documents.** Earlier acquisition firmware (`mod_som_read_epsi_files_v1.m`/`v2.m`, in `read_epsi_files_old_versions/`) used different, inconsistent per-tag byte offsets that changed 2-3 times between 2021 and 2023. This page documents only the **current, unified frame format** (in use since ~May 2021, read by `v3.m`/`v4.m`), the one found in recent deployments.

## The big picture: two frame families, plus two one-off metadata blocks

Every `.modraw` file is one long stream of blocks. Almost all of them start with `$` and end with `*XX\r\n` (`XX` = a 2-hex-digit checksum), but the *interior* structure splits into two families:

| Family | Tags | Header style |
|---|---|---|
| **SOM tag frame** | `EFE4`, `SB49`/`SB41`, `ALTI`, `ISAP`, `ACTU`, `SEGM`, `SPEC`, `AVGS`, `RATE`, `APF0`/`APF1`/`APF2`, `ECOP` (fluorometer), `TTV1`/`TTV2`/`TTV3`, `VNAV` (VectorNav) | Fixed 5-field SOM header: sync, 4-char tag, hex timestamp, hex length, header checksum |
| **NMEA-style sentence** | `GPGGA`/`INGGA`, `GPZDA`/`INZDA` (GPS), and the `VNMAR`/`VNYPR` sentences inside `VNAV` frame payloads | `$...*XX\r\n` wrapper around comma-separated ASCII. GPS sentences sit bare in the stream with the ordinary `T<10-digit>` laptop-clock prefix. VectorNav sentences sit inside the payload of a `$VNAV` SOM frame, each preceded by a 16-hex-char timestamp |

Two more blocks are metadata, read once per file (not repeating data records):

| Tag | What it is | Format |
|---|---|---|
| `SOM3` | Mission/hardware setup: mission & vehicle name, firmware rev, git id, serial numbers, per-sensor config sub-modules (`CALENDAR`, `EFE`, `SBE49`/`SBE41`, `SDIO`, `VOLT`, `ALTI`) | Fixed-layout binary struct |
| `DCAL` | SBE49 calibration coefficients, dumped straight from the CTD's own `.cal` file | Plain-text `name=value`, one per line |

> **The declared header length is not a safe read boundary.** The file's
> first line declares `header_file_size_inbytes`, and it is tempting to
> read exactly that many bytes as "the header" and start scanning blocks
> after it. On a MOTIVE 2025 fixture that boundary lands *inside* the block
> stream: the declared size is 7069 bytes, but the first `$DCAL` block
> starts at byte 3050, along with the first `$SOM3` and a run of early
> `$SB49`, `$EFE4`, and `$ECOP` frames. A reader that skips to the declared
> length before scanning silently drops all of them. Frame the whole file
> and use the frame boundaries the scanner finds.

---

## Family 1: the SOM tag frame

```
 $  EFE4  0000018f449d43f1  00001900  *3A   <...DATA...>   *8F \r\n
 │   │           │              │      │         │           │
 │   │           │              │      │         │           └─ CRLF (part of trailing checksum field)
 │   │           │              │      │         └─ DATA payload, length given by the hex-length field
 │   │           │              │      └─ header checksum: "*" + 2 hex digits
 │   │           │              └─ hex length: 8 hex chars = byte length of DATA
 │   │           └─ hex timestamp: 16 hex chars (ms since power-on, or ms since 1970-01-01)
 │   └─ 4-char tag code
 └─ sync byte
```

| Field | Offset (chars after `$`) | Length (chars) | Contents |
|---|---|---|---|
| sync | 0 | 1 | literal `$` |
| tag | 1 | 4 | tag code, e.g. `EFE4`, `SB49`, `ALTI` |
| hex timestamp | 5 | 16 | hex milliseconds, since power-on for small values, since the 1970 epoch when the decoded value exceeds `1e9` |
| hex length | 21 | 8 | hex byte-count of DATA (exceptions: `ALTI`/`ISAP`, see below) |
| header checksum | 29 | 3 | `*` + 2 hex digits, XOR of the 29 bytes from `$` through the hex length (holds for all 2927 frames in the two `tests/data` fixtures; `modfish` checks only that the digits are hex) |
| DATA | 32 | per hex length | tag-specific payload, see table below |
| trailing checksum | end - 5 | 5 | `*` + 2 hex digits + `\r\n`, XOR of the DATA payload bytes |

`mod_som_read_epsi_files_v4.m` builds this shared layout once into a `tag` struct and reuses it for every tag in this family, so one parsing helper covers all of them.

> **Why the tag codes are all 4 characters.** `EFE` and `ALT` look like 3-letter tags in the Matlab regex patterns (`\$EFE`, `\$ALT`), but the byte offsets above only work if the tag field is exactly 4 characters. On disk the tags are `EFE4` and `ALTI`. The regexes still match because they only require the first 3 letters. The 4th character becomes part of the tag field.

### Per-tag DATA payload

| Tag | Sensor / data | Sample rate | Records per block | DATA payload structure |
|---|---|---|---|---|
| `EFE4` | Shear probes, FP07 thermistors, accelerometers (raw ADC) | 320 Hz (7-ch epsi) or 160 Hz (3-ch FCTD); MOTIVE 2025 FCTD files run 7 channels at ~320 Hz instead (`t1 t2 f1 c1 a1 a2 a3`, all unipolar), so the 160 Hz/3-channel case does not hold for them | 80 | Repeating elements: 8-byte little-endian timestamp + 3 bytes/channel x (7 or 3 channels), 24-bit big-endian ADC counts per channel. 80 records of 7 channels make a 2320-byte payload |
| `SB49` | SBE49 CTD | 16 Hz | `Meta_Data.CTD.sample_per_record` | Repeating records: 16 hex-char timestamp + 24 ASCII-hex chars = `T_raw`(6) `C_raw`(6) `P_raw`(6) `PT_raw`(4), raw engineering counts (needs `$DCAL` or a `.cal` file to convert to T/C/P) |
| `SB41` | SBE41 CTD (e.g. APEX float) | 1 Hz | `Meta_Data.CTD.sample_per_record` | Repeating records: 16 hex-char timestamp + 28 ASCII chars = comma-separated `P,T,S` decimal text (already engineering units, no cal needed; `C` is not transmitted, comes back `NaN`) |
| `ALTI` | MOD altimeter board | on demand, 1 value/block | 1 | **Exception, unverified against real bytes (see note below):** the "hex length" field isn't a length. It holds the raw distance reading as ASCII text (units of 10 µs), converted via `x * 1e-5 s * 1500 m/s` sound speed with no factor 1/2, as the Matlab reader does; whether the reading is one-way or round-trip time is unverified. DATA payload after the header is empty; the frame ends right at the header checksum's `*hh\r\n`, with no separate trailing checksum field. |
| `ISAP` | ISA500 altimeter board | on demand, 1 value/block | 1 | Similar exception to `ALTI`: distance comes back directly in meters as ASCII text, read from a slightly nonstandard offset past the normal header (treat it as "everything after the header, as text") |
| `ACTU` | Actuator | - | - | Tag is matched and counted, but `v4` does not currently decode a payload (`act` always returns `[]`) |
| `SEGM` | Onboard raw time-segment (firmware-computed) | 320 Hz, NFFT=2048 samples/segment | 1 segment/block | 8-byte timestamp + 3 channels (`t1_volt`, `s1_volt`, `a3_g`) x 2048 samples, each sample a 4-byte little-endian IEEE-754 float |
| `SPEC` | Onboard power spectrum (firmware-computed) | derived from 320 Hz, NFFT/2=1024 freq bins | 1 spectrum/block | 8-byte timestamp + 3 channels x 1024 bins, each a 4-byte float |
| `AVGS` | Onboard averaged spectrum | derived from 160 Hz, NFFT/2=1024 bins | 1 spectrum/block | Same layout as `SPEC`; channels labeled `t1_k`/`s1_k`/`a3_g` |
| `RATE` | Onboard dissipation-rate summary | 1 value/block | 1 | 8-byte timestamp + 10 channels x 4-byte float: `pressure, temperature, salinity, dpdt, chi, chi_fom, epsilon, epsi_fom, nu, kappa` |
| `APF0`/`APF1` | APEX float telemetry, profile summary | metadata-driven | `sample_cnt` (in metadata) | 38-byte packed metadata header, then `sample_cnt` variable-format sample records, see [APF metadata layout](#apf-metadata-header-apf0apf1) below |
| `APF2` | APEX float telemetry, newer fixed format | 1 record/block | 1 | Fixed record: timestamp(2B) + pressure/temperature/salinity/dpdt/epsilon/chi/kcutoff_shear/fcutoff_temp/epsi_fom/chi_fom (4-byte floats each), then two `nfft/2`-length averaged spectra (shear, thermal gradient, accel), all 4-byte floats. No metadata header (hardcoded `nfft=2048`). |
| `ECOP` | Tridente fluorometer / backscatter sensor | 16 Hz per the Matlab reader (hardcoded); MOTIVE 2025 files actually carry ~32 blocks/s, with duplicated timestamps between blocks and occasional binary-garbage payloads (decode to NaN instead of a record) | 1 | 16 hex-char timestamp + 12 ASCII-hex chars = three 4-hex-char (16-bit) raw counts: `bb` (backscatter), `chla`, `fDOM`, each normalized `(raw/65535 - 0.5)/scale` |
| `TTV1`/`TTV2`/`TTV3` | Travel-time flow meter (up to 3 transducer pairs) | 16 Hz | 10 | Repeating records: 16 hex-char timestamp + 19-byte binary payload, see [TTV record layout](#ttv-record-layout) below |
| `VNAV` | VectorNav IMU | 40 Hz on MOTIVE 2024 and 2025 files | 10 sentences on the 2024 test fixture | Concatenated NMEA-style sentences, each `<16 hex-char ms timestamp>$VNMAR,...*hh\r\n`, see [Family 2](#family-2-nmea-style-sentences-vnav-gps) below |

> **ALTI layout is an unverified hypothesis.** A search of ~700 `.modraw`
> files across both MOTIVE cruises (skq202521s: all 98 EPSI25 files plus
> the first ~300 FCTD25 files; skq202417s: the first 300 EPSI24 files)
> found no `$ALTI` (or other `ALT`-prefixed) frame. The header-only layout
> above (no payload, frame ending at `*hh\r\n` right after the reading) is
> read out of the Matlab v4 code and has never been checked against a real
> `$ALTI` byte stream. `ALTI` is documented as "on demand", so its absence
> may just mean no sampled deployment triggered it.

### APF metadata header (`APF0`/`APF1`)

```c
uint32_t daq_timestamp;                 // 4 bytes
uint16_t profile_id;                    // 2 bytes
uint16_t modsom_sn;                     // 2 bytes
uint16_t efe_sn;                        // 2 bytes
uint32_t firmware_rev;                  // 4 bytes
uint16_t nfft;                          // 2 bytes
uint16_t nfftdiag;                      // 2 bytes
mod_som_apf_probe_t probe1;             // 5 bytes (type:1, sn:2, cal:2)
mod_som_apf_probe_t probe2;             // 5 bytes
uint8_t  comm_telemetry_packet_format;  // 1 byte  (1 or 2, controls sample record layout below)
uint8_t  sd_format;                     // 1 byte
uint16_t sample_cnt;                    // 2 bytes (number of sample records that follow)
uint32_t voltage;                       // 4 bytes
uint16_t end_metadata;                  // 2 bytes, always 0xFFFF
```
Total: 38 bytes, immediately followed by `sample_cnt` sample records. Record layout depends on `packet_format`:

| `packet_format` | Per-sample fields |
|---|---|
| `1` | timestamp(2B), pressure(4B float), packed dissrate (3B: epsilon+chi bit-packed 12-bit each), fom (1B: epsi_fom+chi_fom bit-packed 4-bit each) |
| `2` | adds temperature(4B), salinity(4B), dpdt(4B), kcutoff_shear(4B), fcutoff_temp(4B), then `nfftdiag` averaged shear/thermal-gradient/accel spectral values (2 bytes each, "foco"-encoded) |

### TTV record layout

```
 16 hex chars     4B float      4B float      4B float     1B      2B         2B
┌──────────────┬─────────────┬─────────────┬─────────────┬──────┬──────────┬──────────┐
│  timestamp   │  tof_up     │  tof_down   │   dtof      │ err  │ up ADC   │ dn ADC   │
│  (hex ASCII) │ (upstream   │ (downstream │ (delta      │ code │ peak     │ peak     │
│              │  time of    │  time of    │  time of    │      │ (uint16) │ (uint16) │
│              │  flight)    │  flight)    │  flight)    │      │          │          │
└──────────────┴─────────────┴─────────────┴─────────────┴──────┴──────────┴──────────┘
      16 chars        4 bytes       4 bytes       4 bytes    1B      2B         2B
```
19 data bytes per record (after the 16-char hex timestamp), 10 records per `$TTV` block, 16 Hz.

> **Legacy ASCII TTV format still referenced in code.** An older, human-readable TTV format also appears in comments of the Matlab reader: `$TTV...*2C...00:28:07 447 ms-000000050 ps,+650 mV,+651 mV,078, 078*12`. The binary layout above supersedes it (`ttv.data.ttv_format = 2` is hardcoded in the current parser). The parsing code for format 1 is still present, commented out, in case older files need it.

---

## Family 2: NMEA-style sentences (VNAV, GPS)

VectorNav sentences travel inside SOM frames. A `$VNAV` frame (Family 1 header and trailer) has a payload made of complete sentences, each preceded by its own hex timestamp:

```
 $VNAV<16 hex ts><8 hex len>*hh   0000018f449d43f1$VNMAR,...*4C\r\n   0000018f449d4400$VNMAR,...*51\r\n   ...   *hh\r\n
 └──── Family 1 frame header ───┘ └─────────────── payload: repeated sentences ───────────────────┘   └ frame trailer
```

One sentence inside the payload:

```
 0000018f449d43f1   $VNMAR   ,0.12,0.03,-0.98,0.01,0.00,9.79,...   *4C \r\n
 │                    │        │                                    │
 │                    │        │                                    └─ checksum + CRLF
 │                    │        └─ comma-separated ASCII fields
 │                    └─ tag (VNMAR/VNYPR for vecnav)
 └─ hex timestamp, BEFORE the "$", not after it like Family 1
```

The sentence checksum is the NMEA XOR of the bytes between `$` and `*`. On the 2024 test fixture every `$VNAV` payload holds 10 complete `$VNMAR` sentences, so no sentence is split across frames. The frame's own trailing checksum covers the whole payload.

GPS does not match this layout: FCTD `$GPGGA`/`$INGGA` and `$GPZDA`/`$INZDA` sentences sit bare in the stream, outside any SOM frame, with no hex timestamp before `$`, only the ordinary `T<10-digit>` laptop-clock prefix. See the note below the table.

| Tag | Sensor | Timestamp offset | Payload |
|---|---|---|---|
| `VNMAR` | VectorNav IMU, compass/accel/gyro packet, inside `$VNAV` frames | 16 hex chars immediately before `$` | 9 comma-separated ASCII floats: compass (x,y,z, gauss), acceleration (x,y,z, m/s²), gyro (x,y,z, rad/s) |
| `VNYPR` | VectorNav IMU, yaw/pitch/roll packet, inside `$VNAV` frames | 16 hex chars immediately before `$` | 3 comma-separated ASCII floats: yaw, pitch, roll (degrees) |
| `GPGGA`/`INGGA` | GPS position | not a hex timestamp: FCTD files carry the ordinary `T<10-digit>` laptop-clock prefix used elsewhere in the file, and a GGA sentence's own `hhmmss.ss` field is time-of-day only, no date | Standard NMEA-0183 GGA sentence, comma-separated: field 3 = latitude (`ddmm.mmmm`), field 4 = `N`/`S`, field 5 = longitude (`dddmm.mmmm`), field 6 = `E`/`W` |
| `GPZDA`/`INZDA` | GPS UTC date and time | same `T<10-digit>` prefix as GGA | Standard NMEA-0183 ZDA sentence: `hhmmss.ss,dd,mm,yyyy,...` |

> **GGA has no timestamp prefix, and no date.** Byte inspection of MOTIVE
> FCTD files (2024 and 2025) found no 10-hex-char timestamp before
> `$GPGGA`/`$INGGA`, only the `T<10-digit>` laptop prefix. GGA's own
> `hhmmss.ss` field gives time-of-day with no date, so an absolute
> timestamp needs pairing each GGA with the date carried by a nearby
> `$GPZDA`/`$INZDA` sentence in the same stream. The talker prefix is not
> fixed either: MOTIVE 2025 files use `$GPGGA`/`$GPZDA`, MOTIVE 2024 files
> (cruise skq202417s) use `$INGGA`/`$INZDA` for the same sentences.

---

## The one-off metadata blocks

### `$SOM3`: mission/hardware setup

Not a repeating data record. It is read once per file (if present) to populate `Meta_Data` before any data tags are parsed. Fixed-layout binary struct, parsed by `mod_som_read_setup_from_raw.m`:

| Field | Length | Contents |
|---|---|---|
| `size` | 4 bytes | total struct size (also selects whether a `gitid` field is present: 24 bytes if `size` is 864 or 896) |
| `header` | 8 bytes | ASCII |
| `mission_name` | 24 bytes | ASCII |
| `vehicle_name` | 24 bytes | ASCII |
| `firmware` | 40 bytes | ASCII version string |
| `gitid` | 0 or 24 bytes | ASCII git commit id, only present for some firmware builds |
| `rev` | 8 bytes | ASCII |
| `sn` | 8 bytes | ASCII |
| `initialize_flag` | 4 bytes | uint32 |

Followed by named, self-length-prefixed sub-modules (`CALENDAR`, `EFE`, `SBE49`/`SBE41`, `SDIO`, `VOLT`, `ALTI`) with per-sensor configuration. Each is located by string search rather than a fixed offset, since not every deployment has every module.

### `$DCAL`: SBE calibration coefficients

Not a repeating data record either. Plain-text lines of `name=value`, one coefficient per line, parsed by `get_CalSBE_v2.m`. It's the CTD manufacturer's own calibration sheet (temperature: `ta0`-`ta3`, `toffset`; conductivity: `g`,`h`,`i`,`j`,`pcor`,`tcor`,`cslope`; pressure coefficients follow), copied verbatim into the file header at the start of a deployment so raw `SB49` engineering counts can be converted to physical units without a separate `.cal` file.

---

## Quick lookup

| Tag | Family | Sensor |
|---|---|---|
| `SOM3` | metadata | mission/hardware setup |
| `DCAL` | metadata | SBE calibration coefficients |
| `GPGGA`/`INGGA`, `GPZDA`/`INZDA` | NMEA | GPS |
| `EFE4` | SOM frame | epsi shear/FP07/accel raw ADC |
| `SB49`/`SB41` | SOM frame | CTD |
| `ALTI` | SOM frame | MOD altimeter |
| `ISAP` | SOM frame | ISA500 altimeter |
| `ACTU` | SOM frame | actuator (tag only, no payload decode) |
| `VNAV` | SOM frame | VectorNav IMU, carries `VNMAR`/`VNYPR` |
| `VNMAR`/`VNYPR` | NMEA, inside `VNAV` | VectorNav IMU |
| `SEGM` | SOM frame | onboard raw time-segment |
| `SPEC` | SOM frame | onboard power spectrum |
| `AVGS` | SOM frame | onboard averaged spectrum |
| `RATE` | SOM frame | onboard dissipation-rate summary |
| `APF0`/`APF1`/`APF2` | SOM frame | APEX float telemetry |
| `ECOP` | SOM frame | Tridente fluorometer/backscatter |
| `TTV1`/`TTV2`/`TTV3` | SOM frame | travel-time flow meter |

## Provenance

Derived by reading `mod_som_read_epsi_files_v4.m` in `MOD_fish_lib` (the header-offset struct and each tag's per-block parsing loop). No firmware spec document was available. Byte offsets and field names are transcribed directly from the parsing code; a couple of details (the `ALTI`/`ISAP` "hex length field is actually the reading" quirk, and the exact `ISAP` data offset) are called out as quirks rather than presented as clean spec because the code itself treats them inconsistently with the rest of the frame family.
