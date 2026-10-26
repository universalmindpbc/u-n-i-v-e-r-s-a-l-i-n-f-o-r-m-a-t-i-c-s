# Retinal triangle centring — staging

Source commit: `431581f0`. Base staging: `58bbd6a5`.

The pink triangle mesh now connects adjacent cell centres inside a symmetrical circular boundary. The previous construction selected triangle origins then extended each triangle right and down. The replacement retains alignment to the cell lattice and centres the complete mesh at the circle centre.

Only the Vision script and its cache version in the shared index changed on the website. The source interactions file was updated and the script rebuilt. Notion wording, styles, other diagrams and Quantum files are unchanged.

Local syntax and whitespace checks passed. Dedicated Chromium launched with `--mute-audio` confirmed before navigation. Public staging was checked at widths 1470, 390 and 320 pixels. In each view the SVG mesh bounds centre is exactly (220, 200), every edge endpoint matches a cell centre within 0.001 SVG units, and all endpoints lie inside the outer circle. The three layer controls work by keyboard; there are no browser errors. Media was paused and every automated browser context closed after verification.

The existing Panthur staging deployment uploaded two changed files. The index and versioned `vision-science.js?v=2` match source bytes. The unversioned script URL retained a previous cached response; the page loads version 2.

Proof: https://staging.universalmind.tv/informatics/?entry=research&area=vision

Production has not been modified.
