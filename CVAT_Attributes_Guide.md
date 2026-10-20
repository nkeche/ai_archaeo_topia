# CVAT Attribute Annotation Guide for Mound Symbols

*Description of CVAT attributes (where annotations are marked and the model is trained on them)*

> **The surroundings around the symbols are also important for the model, so they must be marked as well. Attributes must also be corrected on objects that are NOT MOUNDS but which the model has marked as mounds.**

> **NB! "Crossed" and "Overlapped" do not mean only that the symbol itself is directly "crossed" or "overlapped" — it also applies when its immediate surroundings are. See examples below.**

Some of the examples below show cases affected by more than one additional symbol/line, etc.

![Three mound symbols outlined in red next to elevation marks](cvat_images/image1.png)

![Triangulation point symbol with elevation value](cvat_images/image2.png)

We count these as mounds — the lower one is a triangulation point!!!

## True / False Attributes

- **`affected_by_colored_pencil`** — presence of additional coloring (pencil, pen, marker, etc. on the map)

  ![Example of colored pencil marking on the map](cvat_images/image3.png)
  ![Example of colored pencil marking on the map](cvat_images/image4.png)

- **`blurred_or_bad_print`** — smudged / faint print

  ![Example of blurred or poor-quality print](cvat_images/image5.png)

- **`crossed_by_contour`** — crossed by a contour line / isoline

  ![Example of a mound symbol crossed by a contour line](cvat_images/image6.png)

- **`crossed_by_forestation_line`** — crossed by forest / forest symbols / vineyard; the background does not necessarily have to be green

  ![Example of a mound symbol crossed by a forestation line](cvat_images/image7.png)

- **`crossed_by_grid`** — crossed by the map grid (grid lines on 1:5,000 scale maps)

  ![Example of a mound symbol crossed by the map grid](cvat_images/image8.png)

- **`crossed_by_powerline`** — crossed by a power line

  There are various types of symbols on the map for power lines/transmission lines. Some may no longer exist on the ground (they may be from old poles), but that doesn't matter for computer vision 🙂!

  ![Example of a mound symbol crossed by a power line](cvat_images/image9.png)
  ![Example of a mound symbol crossed by a power line](cvat_images/image10.png)
  ![Example of a mound symbol crossed by a power line](cvat_images/image11.png)

  More examples of power lines/transmission lines:

  ![Power line symbols with unfilled circles](cvat_images/image12.png)

  All these lines with unfilled circles are power lines.

  ![Power line symbols with arrows](cvat_images/image13.png)

  The ones with arrows are clear.

- **`crossed_by_road`** — crossed by a road — dirt/field/forest road, asphalt, railway line, etc.

  ![Example of a mound symbol crossed by a road](cvat_images/image14.png)
  ![Example of a mound symbol crossed by a road](cvat_images/image15.png)

- **`has_absolute_elevation_mark`** — has an absolute elevation (above sea level) mark

  ![Example of a mound symbol with an absolute elevation mark](cvat_images/image8.png)

- **`has_relative_height_mark`** — has a height value relative to the surrounding terrain; if the height value is shared among several symbols, mark `true` for each of them

  ![Example of a mound symbol with a relative height mark](cvat_images/image16.png)
  ![Example of a mound symbol with a relative height mark](cvat_images/image17.png)

  In this case it is clear which symbol the value belongs to — the one with the absolute elevation mark and the dot inside it (a dot inside the mound symbol means it has an absolute elevation above sea level).

  ![Example of elevation mark attribution](cvat_images/image18.png)

  Here it is not clear which symbol the value belongs to, so we mark `true` for all three.

  ![Example of ambiguous elevation mark attribution across three symbols](cvat_images/image19.png)

  Here we mark the absolute elevation for the eastern mound, and the relative height for both mounds.

- **`has_trig_point`** — has a triangulation point

  ![Example of a mound symbol with a triangulation point](cvat_images/image20.png)
  ![Example of a mound symbol with a triangulation point](cvat_images/image2.png)

- **`overlaps_other_mound`** — overlaps with another mound; there are various examples, also depending on the presence of other symbols

  ![Example of overlapping mound symbols](cvat_images/image21.png)
  ![Example of overlapping mound symbols](cvat_images/image22.png)

  All of these must be marked as `overlaps_other_mound`.

  ![Example showing which symbols should be marked as overlapping](cvat_images/image23.png)

  Only the two at the SE end should be marked `overlaps_other_mound`.

## Multi-Value Attributes

- **`water_line_crossing`**:

  | Value | Meaning |
  | :--- | :--- |
  | `none` | No water line crosses this symbol. |
  | `surface` | A stream or river crosses it — *this is a data error, since the mound is an elevated landform and surface water flows through low-lying areas* — **THIS IS NOT ALWAYS THE CASE, because sometimes there are small streams that are not in a gully/ravine (see below).** |
  | `underground` | A pipe or culvert crosses it, which can run under anything and is normal. |
  | `unreviewed` | No one has reviewed this yet. |

  ![Example of a water line crossing a mound symbol](cvat_images/image24.png)

  `surface`

## Additional Examples

![Combined example of a mound symbol crossed by a forestation line](cvat_images/image6.png)

- `crossed_by_forestation_line` — `true`
- `has_absolute_elevation_mark` — `true`
- `has_relative_height_mark` — `true`

![Combined example of a mound symbol crossed by the grid and a road](cvat_images/image25.png)

- `crossed_by_grid` — `true`
- `crossed_by_road` — `true`
- `has_absolute_elevation_mark` — `true`
- `has_relative_height_mark` — `true`
- `has_trig_point` — `true`
