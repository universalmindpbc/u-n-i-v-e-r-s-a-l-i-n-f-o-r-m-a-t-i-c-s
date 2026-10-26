# Reduction to M9 — comparison v1

M18P24 Alpha, primary Gamma and Beta reduce exactly to M9P24 in original phase. Gamma: 18,3,3,18,15,15 → 9,3,3,9,6,6. Beta: 4,8,16,14,10,2 → 4,8,7,5,1,2. Recognised cyclic order 1,2,4,8,7,5 is supplementary; the original start remains 4. Zero is displayed as 9 only in the reduced comparison.

For m=9k, (F_n mod m) mod 9 = F_n mod 9. Gamma selection by divisibility by 3 is preserved and modular reduction commutes with interior sums. Python checks all 111 multiples of 9 through M999: Alpha, Gamma, Beta and Gamma indices agree with repeated M9 cycles. This is a divisibility consequence, not a statistical discovery. Wolfram independently confirms M18 Alpha, Gamma and Beta; the audit of all 111 is Python-only.

M18 has period24, so Gamma positions match on the same clock. Reduced Beta uses nine positions. Longer native periods require explicit folding into24-position cycles; whole-circle geometry is not automatically identical. Native Epsilon is unchanged and remains null for M18.

Research question: do these shared reduced patterns have shared biological relevance, or does native-modulus complexity matter? Evidence, simulations and appropriate experimental validation are still needed. CUDA/CUDA-Q modelling is planned, not performed. No WSS score or higher spatial dimensional interpretation is established.

## Verification record

- All24 native JSON records compared against the preceding Git release after removing only the added comparison field: identical.
- All111 applicable moduli passed exact sequence/phase/index comparison. This does not constitute full geometry proof for M25–M1001.
- M18 Wolfram result: Alpha equality True, Gamma raw 0,3,3,0,6,6, Beta4,8,7,5,1,2, Beta equality True.
- Notion:24 comparison statuses saved/read back; M9/M18 Alpha, Beta, Gamma values read back and matched. M18 native Epsilon remains Null.
- Complete download bundle includes comparison CSV, full JSON,111-modulus audit and Wolfram output.
- Staging browser verification:1440×900 and390×844 M18 comparison displays all3 exact matches; M6 hides the inapplicable comparison; M9 shows reference status. All24 records and97 diagrams passed at both viewports. No horizontal overflow or JavaScript errors; returning to landing pauses the video.
- Live download checks:CSV, full ZIP and JSON match local release bytes. All236 public assets match SHA-256 checksums when requested with the release version. Release query parameters prevent stale cached JS/data/bundles. Dedicated headless Chromium used --mute-audio; native fullscreen was stubbed; all automated browser sessions closed.
