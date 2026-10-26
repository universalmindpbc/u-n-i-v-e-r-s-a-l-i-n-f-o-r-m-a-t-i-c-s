# Vision Atlas science explainer

Source: Ben's updated Notion page, `4047b9c9-9c56-80c0-9dea-d2a96398dd1b`, last edited 25 October 2026 at 02:26:54 UTC. The page snapshot is retained in `notion-source.json`.

## Content handling

The 13 unique prose paragraphs are preserved in full. Repeated copies of the same Notion blocks are displayed once. Bold and italic emphasis is retained. Three named publication links are resolved to their original Nature Portfolio article pages; the original Google redirect references remain available in the source snapshot. The site presents the text as research stimulus. The diagrams are labelled illustrative geometry or an illustrative model.

## Scope and coordination

Only `Informatics/index.html`, `Informatics/vision-science.css`, and `Informatics/vision-science.js` change the site. The index receives one stylesheet/script include. The new component appends to `#area-vision`; all CSS is scoped to `.vision-science`.

Ben authorized coordination with the active Quantum chat. That chat owns Quantum Atlas files and the Quantum iframe cache version in the shared index. Its completed staging release is integrated before this deployment. Quantum assets are verified byte-identical to that release.

## Verification

The verification script uses a separate headless Chromium process with `--mute-audio`, confirms that process argument before navigation, prevents native fullscreen, and pauses all media and closes each context after checks. Local QA routes the permanent staging URL to the candidate assets inside the isolated browser; it starts no local web server and creates no temporary website.

Widths: 1470, 1024, 390, and 320 pixels. Checks cover source paragraph parity, appended placement, existing comparison captions, cell/triangle/hexagon controls, keyboard moiré rotation, section navigation, preserved scope, viewport overflow, reference destinations, and browser errors.

The reading panel was strengthened after visual inspection revealed the background film behind the lower text. The second local verification pass passed at all four widths.

## Staging release

The final Notion fetch confirmed the same page revision immediately before release. The parallel Quantum staging record `e0452dda` was integrated first. Source commits `205899a1` and `637f194b` contain the isolated Vision change.

The existing `deploy-panthur-staging.sh informatics` uploaded exactly three changed assets: the shared index and the two new Vision assets. The final public staging route passed the same four-viewport checks. All eight checked source assets (the three changed files plus five Quantum assets) match their public staging bytes. Production's Informatics index hash is unchanged.

Proof: https://staging.universalmind.tv/informatics/?entry=research&area=vision

The live screenshots and `live/verification.json` record rendered layout and interaction checks. `integration.json` records exact preservation of the parallel Quantum files, its iframe version and the inherited deployment manifest before deployment. `staging-assets.json` and `production-unchanged.json` record HTTP byte comparisons.

Production was not deployed. No API or GPT functionality changed in this release.
