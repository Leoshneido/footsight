# ERRORS.md

Log of approaches that took more than 2 attempts to work. Checked before suggesting approaches to similar tasks.

## Team classification — jersey color feature for k-means clustering (2026-09-10)
**What didn't work:**
- **Raw BGR median of the full-width torso crop** — shadow/pose changed brightness enough to push 2 light-blue Man City players into the "officials" cluster (20/22 on the Barça/City still). Full-width crop also let background and neighboring players bleed in.
- **Median (hue, saturation) in HSV** — fixed the shadow case (22/22 on Barça/City), but failed on the Bayern/Bodø still: red (hue ~5-8) and yellow (hue ~23-27) are only ~15-20 hue units apart, and the saturation spread within a team (~35-100 units) swamped that gap, so both teams merged into one cluster.
- **(Hue, saturation) z-score normalized** — made it worse. The officials' extreme hue inflates the overall hue std, shrinking the team-vs-team gap after normalization. Rescaling can't fix this: the saturation noise is genuinely larger than the hue signal, not just in different units.

**What worked:** Hue only (1D), sampled from the center 50% of the box width. Brightness and saturation dropped entirely. Fixed both stills' team split; a regression test fails before the fix and passes after.

**Note for next time:**
- For jersey color, start with hue only. Add another channel only with real-still evidence that it carries signal, and try weighting it before trying normalization.
- Synthetic flat-color test rectangles can't reproduce real photo noise. Always verify against a real still, not just unit tests.
- Still open: fixed `k=3` breaks when a 4th hue group is present (goalkeeper kit, non-player detections). See MEMORY.md.

## Ball detection on broadcast stills (2026-09-10)
**What didn't work:**
- **Faster R-CNN COCO "sports ball" (label 37)** — found nothing even at 0.02 confidence. The ball is ~12-15px in a 3020x1700 frame, too small for a general detector.
- **Crop + upscale around the players, then the learned model** — with 22 players spread edge to edge, the crop was the whole frame, so nothing changed.
- **Classical HSV white-blob threshold, no morphology / 3x3 opening** — merged the ball with the pitch line it sat on (circularity 0.38), and a naive search of the whole padded box picked up a player's white sock.

**What worked:** A classical HSV threshold (low saturation, high brightness) in a padded window near each player's feet, then 5x5 morphological opening, then contour filter by size and circularity ≥0.65. Found the real ball, confirmed by visual crop.

**Note for next time:** For small objects (ball-sized) in broadcast frames, skip general-purpose learned detectors and go straight to classical segmentation. 5x5 was the smallest opening kernel that separated the ball from a line; 3x3 was too weak.
