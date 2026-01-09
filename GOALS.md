A system to create abstract pixel art backgrounds.

Overall system takes in canvas dimensions, a four-color palette, and one of eight lighting directions.


Colors:
bkg
shadow
light
highlight


From there, output four layers and a composite image.  

STATIC VERSION (level one)

algorithm to create the image:

background layer is just the background color flat

for the other layers, divide the canvas into a 3*3 grid.  (rounding is fine)

each lighting direction (TOP_LEFT, TOP_MIDDLE, TOP_RIGHT, ...) corresponds to one of the eight cells excluding the center.

Based on lighting direction, choose that cell as LIGHT_CELL.  then also choose DARK_CELL as the cell farthest from LIGHT_CELL (opposite corner or opposite side).

Then define LIGHT_HALF as the six cells closest to the LIGHT_CELL.  This will either be a 2*3 rectangle if the LIGHT_CELL is on a midpoint side, or a triangle containing the middle diagonal and three more cells if the LIGHT_CELL is a corner.  DARK_HALF will be the other 3 cells.

2 of the six cells in light_half will get chosen as populated: LIGHT_CELL plus two other randomly chosen.  DARK_CELL will get populated too.

Generate one blob for each of the populated cells.  What is a blob?

A blob is a contiguous region of a color with a center point.  Build an organic shape using a noise function or probability distribution to expand around the center point, adding pixels to the boundary.  at each iteration, some pixels should be added but not in every direction.  Do floor(N/2) iterations where N is the diagonal of the rectangle of the region (the cell).  The goal is organic, fluid, wispy shapes.

Set the center point of each populated cell as the center of the cell (rounded).  in DARK_CELL, make a blob with the shadow color on its own layer.

in all populated cells in the light half, create similar blobs using the light color in their own layer

in LIGHT_CELL only, create an additional layer containing one blob of highlight color.

I want to audition various blob creation algorithms to make smooth organic shapes as in the shapes in @layers.png.

blobs can extend beyond their cells but they will remain centered in their cells.


TEST NOTES:  whenever generating output, put all new files in their own folder labeled by date and time.


DYNAMIC VERSION (level two)



