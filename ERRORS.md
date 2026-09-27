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

## Team classification round 3 — splitting teams after the detector change (2026-09-26)
**What didn't work:**
- **k=2 hue clustering on the `player` boxes alone** — sound in principle (the football model labels goalkeepers and referees separately, so only two teams should remain), but the model hands the main referee over as a `player` at 0.91 confidence. That single yellow hue (~34) sits ~65 units from the nearest team while the two teams are ~15 apart, so it took one cluster and both real teams merged into the other: 21 icons one color, 1 the other.
- **Trimming hue outliers by Tukey fence (1.5×IQR), first pass** — fixed the Barça/Feyenoord and Bayern stills, but over-fires on the 12.00.14 still: 2 real Barça players were trimmed because their arms-wide pose made the torso sample mostly green pitch, and grass hue (40-60) reads as an outlier next to their real ~125.

**What worked (implemented and verified on all 4 stills, 2026-09-26):** dropping grass-hued pixels from the torso sample before taking the median moves those two players from 43→129 and 42→126, while the real referee stays an outlier at 34.

**Note for next time:**
- When a jersey hue lands in the 40-60 band, suspect the grass in the crop before suspecting the clustering. The torso window is a rectangle; a stretched or diving player fills a fraction of it.
- Broadcast watermarks get detected as players (0.65-0.73 across stills). Filtering by box shape fails — a stretching player's box was 94x103, nearly square. Filtering by confidence fails too: raising the threshold to 0.75 drops a real, partly hidden Bodø player scoring 0.72 on the Bayern still. The watermark also projects inside the pitch lines, so the off-pitch filter can't catch it. What worked: `--remove-detections`, clicking it away by hand.

## Team classification round 4 — striped kits across the red hue seam (2026-09-26)
**What didn't work:**
- **Straight-line median hue (the existing approach)** — a Barça player on 11.59.40, split ~50/50 between claret (hue 170-179 and 0-10, straddling the seam) and blue, read 75: a hue the jersey doesn't contain. That put him with Feyenoord.
- **Circular median for both the trim and the team split** — fixed that player, but Barça's circular medians spread over 130-179, which widened the Tukey fence until the referee (33) and the watermark got through on 2 of the 3 Barça stills.
- **Hue histogram per player as the feature (k-means on 18-bin histograms, fence on distance to centroid)** — trimmed the referee everywhere, but also trimmed or misplaced 5-6 real players across the stills, including 2 Bayern players.

**What worked:** a hybrid. The trim runs on the straight-line median (reliable for the referee on every still), then the team split runs on the circular median, with each hue as a point on a unit circle for k-means. Every case with known ground truth is correct on all 4 stills. (A first write-up claimed it also fixed 2 "players" in front of an ad board. They turned out to be one ball boy behind the touchline, who is dropped as off-pitch anyway.)

**Note for next time:**
- A single hue number can't describe a two-color kit. Before changing the feature, look at the per-player pixel histograms: that is what showed the 50/50 split.
- Before calling a box a player, check where it projects. A person in front of an ad board may be standing off the pitch.
- Weak spot: the trim still uses the straight-line median, so a striped player whose linear median falls far enough into the gap could be trimmed as an official. Not seen yet.

## Ball detection on broadcast stills (2026-09-10)
**What didn't work:**
- **Faster R-CNN COCO "sports ball" (label 37)** — found nothing even at 0.02 confidence. The ball is ~12-15px in a 3020x1700 frame, too small for a general detector.
- **Crop + upscale around the players, then the learned model** — with 22 players spread edge to edge, the crop was the whole frame, so nothing changed.
- **Classical HSV white-blob threshold, no morphology / 3x3 opening** — merged the ball with the pitch line it sat on (circularity 0.38), and a naive search of the whole padded box picked up a player's white sock.

**What worked:** A classical HSV threshold (low saturation, high brightness) in a padded window near each player's feet, then 5x5 morphological opening, then contour filter by size and circularity ≥0.65. Found the real ball, confirmed by visual crop.

**Note for next time:** For small objects (ball-sized) in broadcast frames, skip general-purpose learned detectors and go straight to classical segmentation. 5x5 was the smallest opening kernel that separated the ball from a line; 3x3 was too weak.
