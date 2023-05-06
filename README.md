# Elastic Earth projections

 The Elastic projections are a map projections of a new breed that uses
 interpolation on a mesh to minimize and control the distortion in maps of the
 whole Earth like never before.
 
![Elastic Earth I projection with mesh](examples/mesh-1.svg "Elastic Earth I projection with mesh")
![Elastic Earth II projection with mesh](examples/mesh-2.svg "Elastic Earth II projection with mesh")
![Elastic Earth III projection with mesh](examples/mesh-3.svg "Elastic Earth III projection with mesh")

## Using the maps

 The maps are defined, not with equations like most maps are, but with tables of coordinates that must be interpolated.
 The tables are stored in two different formats.
 The first format, which I recommend using, is HDF5. HDF5 (hierarchial data format) is a self-explanatory 

## Using the code

 Most of the code is python scripts that you can just run.
 I didn't make a PyPI requirements file because I'm lazy; maybe I'll do that later.
 here are the important scripts:
 - `elastik.py` generates a bunch of nice maps based on pregenerated map projections
 - `create_map_projection.py` generates a map projection based on pregenerated weights and meshes
 - `calculate_weights.py` generates greyscale images that can be used as weights for new map projections (requires coastline data; see below)
 - `build_mesh.py` generates a mesh specifying the rough configuration of a new map projection based on a manually specified or pregenerated cut file.
 - `find_drainage_divides.py` generates a cut file based on the borders between watersheds (requires elevation data; see below)
 - `run_all_scripts.py` executes `build_mesh.py`, `calculate_weights.py`, and `create_map_projections.py` in that order on all of their possible inputs

 I've tried to include all dependencies so that PyPI installs and data files are the only things you need to add.
 However, it's worth noting that if you want to edit some of the code *or* run `create_map_projection.py` on an OS other than Windows,
 you'll need to pay attention to the C library, `sparse.c`.
 There's a CMakeLists file if that's your jam, but I don't really know how to use it;
 it's just left over from when I briefly used CLion.
 When I recompile the C library, I go to the root directory in VS Developer Command Prompt and use
 ~~~bash
 cl /D_USR_DLL /D_WINDLL src/sparse.c /link /DLL /OUT:lib/libsparse.dll
 ~~~
 Naturally, if you're not on Windows, it'll be something different.
 Hopefully you can figure it out.  Make sure the command prompt you use
 has the same architecture as the Python distribution you're running!

## Data sources
 I didn't include the geographic data that some of the code is run on because
 some of it is kind of big, and it mite need to be updated periodically in the
 future.

 To run the `calculate_weights.py` script, which computes a map of the distance of
 each point on earth from the nearest shoreline, you will need the coastline and
 land polygon datasets from Natural Earth (naturalearthdata.com).   Download the
 zip files and put them in data/.  The script will unzip them automatically.
 You specifically need
 - Natural Earth 10m coastline dataset
 - Natural Earth 10m minor islands coastline dataset
 - Natural Earth 110m land dataset

 To run the `find_drainage_divides.py` script, which locates and saves
 continental divides for use as interruptions, you will need the USGS EROS
 Center's global 30 arcsec elevation data set (GTOPO30, DOI: 10.5066/F7DF6PQS).
 Download it as GeoTIF files and put it in data/elevation/.  The script will load
 and assemble it automatically.
