GOAL: create four-layer pixel art infinite animation.

Arguments:
dimensions in x and y pixels of canvas
scale factor of window 
four colors [bkg, shadow, light, highlight]
scroll direction (N, NW, W, SW, S, SE, E, NE)


Paint whole canvas bottom layer with bkg color

Then create Blobs of the other three colors and animate them.

A Blob is a contiguous region of pixels, like a noisy ellipse shape.
A Blob has a center point (x, y) which gets translated in the scroll direction every frame
The whole Blob gets translated by center point but it also morphs each frame.
The goal is for a Blob to be an organic, bubbly, wispy shape.  
Keep track of the boundary of a blob (array of pixels?  something more sophisticated, an edge curve?) and fill in the interior pixels.  It is OK for holes to form in the interior of the Blob
The boundary should morph every frame, sometimes contracting, sometimes expanding, like Ricci flow or clouds or boiling shapes.  It is OK if the Blobs form long galactic arms, they don't need to be uniform.  It should be a noisy organic process that produces leaf-like cloud-like amoeba-like shapes.

Create a Blob offscreen (opposite scroll direction) as an ellipse with a center point and random major/minor axes.  Then let the blob morph so its shape changes as it begins to scroll.  Once it comes on screen it should already have a somewhat nonelliptical shape.

Each frame, check for long flat edges, rectangles of width or height 1 px (destroy those), and singleton pixels(destroy).  If a long flat edge is found, add some noise to the edge before rendering that frame.  I don't want flat edges longer than 5 pixels.

Destroy a blob once all its pixels are off the canvas because of the scroll.

The ultimate goal is shader code that runs at 60fps smoothly.

Create blobs with the following probabiltiy distribution: 0.6 light blob, 0.3 shadow blob, and the remaining 0.1 make a highlight blob as a subblob that remains within the interior of a light blob (spawn it at the same time as a light blob with center point 1-5 pixels in the scroll direction from the center point of the light blob, the highlight blob should be in the interior of its light blob)