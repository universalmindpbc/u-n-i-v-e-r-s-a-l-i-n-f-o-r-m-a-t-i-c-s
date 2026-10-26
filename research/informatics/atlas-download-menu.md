# Atlas dataset download menu

Candidate fca9e0c adds Download beside Geometry Summary, outside the tablist.
An independent action group leaves room for Connect AI when implemented.
At narrower widths actions sit beneath the wrapping geometry tabs. The menu
links to the existing complete M1–M48 CSV, data bundle and JSON; record-specific
exports remain in their existing locations. No AI connection is claimed.

Live staging checks passed at 1440×900, 1366×900 and 390×844:
- Keyboard Enter opens; Escape closes and returns focus.
- Outside click and selecting a download close the disclosure.
- CSV, ZIP and JSON downloads match the local dataset byte for byte.
- Action remains available through geometry tabs and record changes.
- M summary hides the geometry toolbar; returning restores it.
- No horizontal page overflow or JavaScript errors.
- Media paused on exit; all browser checks used dedicated headless Chromium
  with --mute-audio and a stubbed fullscreen API.

Existing Epsilon and combined geometry checks passed at 1440×900, 1366×768
and 390×844, including above-fold diagram/sequence positioning and native
qualification boundaries. Staging only; production unchanged.
