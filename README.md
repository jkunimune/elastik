# elastik
 A new and improved suite of map projections based on a polydimensional mesh

 The Elastic projections are a map projections of a new breed that uses
 interpolation on a mesh to minimize and control the distortion in maps of the
 whole Earth like never before.

## Data sources
 I didn't include the geographic data that some of the code is run on because
 some of it is kind of big, and it mite need to be updated periodically in the
 future.
 
 To run the `optimize_cuts.py` script, which locates and saves
 continental divides for use as interruptions, you will need the USGS EROS
 Center's global 30 arcsec elevation data set (GTOPO30, DOI: 10.5066/F7DF6PQS).
 Download it as GeoTIF files and put it in data/elevation/.  The script will load
 and assemble it automatically.
 
 To run the `calculate_weights.py` script, which computes a map of the distance of
 each point on earth from the nearest shoreline, you will need the coastline and
 land polygon datasets from Natural Earth (naturalearthdata.com).   Download the
 zip files and put them in data/.  The script will unzip them automatically.
 You specifically need
 - Natural Earth 10m coastline dataset
 - Natural Earth 10m minor islands coastline dataset
 - Natural Earth 110m land dataset
