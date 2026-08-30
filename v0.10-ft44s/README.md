# v0.10.0 / ft44s — superseded, kept for the record

Every file here was produced by Sieve **v0.10.0**, model `ft44s_best_fp16.onnx`, calibration
bias **0.43**. Upstream has since shipped v0.11.0 (`ft5s`) and v0.12.0 (`ft58s`), and moved the
bias to **0.30**. Numbers in this directory are not comparable to numbers in the parent
directory and must not be pooled with them.

These files predate `sio.py`, so they are bare JSON lists with no model stamp — which is exactly
why they are quarantined in a directory whose name is the version instead of sitting at the root
looking current. `sio.require_version()` refuses to read them as if they were stamped.

They are kept because my GitHub issues #48 and #49 report numbers measured here, and a reader
who wants to check those numbers needs the data they were computed from, not the data that
replaced it.
