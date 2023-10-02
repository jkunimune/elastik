# Elastic Earth projections

The Elastic Earth projections are map projections of a new breed that uses
interpolation on a mesh to minimize and control the distortion in maps of the
whole Earth like never before.

This repository contains both [the data files](projection) that define the Elastic Earth projections
and [the source code](src) used to create them.
If you're interested in making maps using the Elastic Earth projections,
see [§ Using the projections](#Using the projections) below.
If you're interested in using the code to create new map projections like Elastic Earth,
see [§ Using the code](#Using the code) below that.

![Elastic Earth I projection with mesh](examples/mesh-1.svg "Elastic Earth I projection with mesh")
![Elastic Earth II projection with mesh](examples/mesh-2.svg "Elastic Earth II projection with mesh")
![Elastic Earth III projection with mesh](examples/mesh-3.svg "Elastic Earth III projection with mesh")

## Using the projections

The map projections are defined, not with equations like most maps are,
but with tables of coordinates that must be interpolated.
This section explains how to do that.
The tables are given in two file formats, which can both be found in [`projection/`](projection).
I've coded up two demonstrations for those who learn best by example:
[a Python implementation](src/elastik.py)
that uses the HDF files and bilinear interpolation, and
[a Java implementation](https://github.com/jkunimune/Map-Projections/blob/master/src/maps/Elastik.java)
that uses the plain text files and Hermite spline interpolation.
The Java implementation also implements the inverse-projection using Levenberg-Marquardt iteration.

### Map projection structure

Each Elastic Earth projection can be divided into two to three *sections*.
Each section covers some portion of the globe and defines the projection for points in that portion.
Specifically, each section has a table that, at certain latitudes and longitudes,
defines the corresponding x and y values.
Figure 1 below shows an example of such an array of points both in latitude and longitude, and in x and y.
Note that the points do not completely cover the globe.
That's because most latitudes and longitudes are outside of the region this section covers.

![A section's point array in two coordinate systems](resources/images/diagram-1.png)

Projecting points within a section is a question of taking latitudes and longitudes not explicitly listed in the table of points,
and determining the appropriate x and y values based on nearby points that *are* explicitly listed.
This problem is generally known as 2D interpolation.
It is a common technique in the computational sciences (as well as in the definition of the Robinson projection),
and as such you will find many interpolation routines written in every major programming language.
There are multiple applicable interpolation methods depending on how fancy you want to make it;
the simplest one is [bilinear interpolation](), but [spline interpolation]() will yield a smoother result.
Figure 2 below shows the same section as before with geographic data interpolated onto it.
The shaded region represents the area where interpolation is possible.

![A section with geographic data interpolated on it in two coordinate systems](resources/images/diagram-2.png)

To map the whole globe, we simply combine all of the sections.
Figure 3 below shows three sections fitting together to form a complete map.

![A complete map made of three sections fit together](resources/images/diagram-3.png)

Note that there is significant overlap between them.
In addition, some geographic features are present on multiple sections in different places,
such as Tierra del Fuego which appears in both the upper left and lower right of the map.
Some maps may intentionally use this redundancy.
For example, it's somewhat common in conventional maps to show the Chukchi peninsula on both
the left side of the map (since it's in the Western Hemisphere) and
the right side (where it connects to the rest of Siberia).

However, in most situations this redundancy is unnecessary and confusing.
For this reason, each section has a boundary that defines precisely
which latitudes and longitudes it applies to.
These boundaries are mutually exclusive,
so every point on the globe is contained by exactly one.
Figure 4 below shows the example section from before with its boundary drawn.

![A section with its boundary shown in two coordinate systems](resources/images/diagram-4.png)

Thus, to remove the repeated regions, simply check each point before projecting it
to determine which section contains it,
and interpolate its x and y using only that section.
Figure 5 below shows the result of clipping the sections in this way.

![A complete map made of three sections clipped by their boundaries](resources/images/diagram-5.png)

And there you have it, a world map on an Elastic Earth projection!

Inverting the map projections is possible but computationally challenging.
This is a fundamental limitation of mesh-based projection.
To help with the process, a table is provided that, at certain x and y coordinates,
gives the corresponding latitudes and longitudes (assuming bilinear interpolation).
For x and y coordinates that do not fall on the mesh,
the given latitude and longitude are the ones whose projection is as close as possible.
You can get a pretty good approximation of the inverse-projection by using 2D interpolation on this inverse table,
and then removing all points that fall outside of the map area.

There are two main caveats.
One is that the inverse table cannot completely account for places where two sections of the mesh overlap.
A single set of x and y coordinates in one of those regions
can be projected from multiple sets of latitude and longitude,
but the table only provides one of those sets.
In practice those overlap regions are small enough that it doesn't really matter.
The other caveat is that interpolation is not exact,
so a raster map made using 2D interpolation on the inverse table
will not quite line up with a vector map made using the projection as defined above.
For an exact inverse-projection, apply an iterative solver
like Newton–Raphson, Levenberg–Marquardt, or Nelder–Mead,
using 2D interpolation on the inverse table as the initial guess.

### File format

The tables are stored in two different formats.

The first format, which I recommend using, is [HDF](https://www.hdfgroup.org/solutions/hdf5/).
HDF (hierarchical data format) is a self-describing file format for structured heterogeneous data.
HDF files can be opened with the program [HDFView](https://www.hdfgroup.org/downloads/hdfview/),
and are supported by libraries in a variety of programming languages
including Python, C, Java, and MATLAB.
Because metadata and hierarchy information is encoded, the contents of a HDF file are fairly intuitive,
and in principle one can figure out how to use the Elastic Earth HDF files
without any auxiliary explanation.

Each Elastic Earth HDF file contains the following information:
- The projected boundary of the map projection.
- The minimum and maximum x and y coordinates of the map projection.
- The list of section names.
- A group for each section including
  - The latitudes at which the projection is defined.
  - The longitudes at which the projection is defined.
  - The table of x and y coordinates corresponding to the given latitudes and longitudes (with undefined values set to *NaN*).
  - The boundary on the globe.
- A group for the inverse table including
  - The x values at which the inverse-projection is defined.
  - The y values at which the inverse-projection is defined.
  - The table of latitudes and longitudes corresponding to the given x and y values (values that don't fall on the mesh).

The twoth format is plain text.
Plain text files can be opened with a variety of programs (Notepad is the default on Windows),
and can be read natively in any programming language.
I provide text files because I know HDF can be intimidating for the less technically savvy,
and because installing HDF can be kind of tricky.
However, implementing the projections using text files will be more work,
as you'll need to write code to separate and parse the various numbers and tables.

Each Elastic Earth plain text file contains the following components, in this order:
- A header for the map projection, stating the number of sections.
- A header for each section followed by
  - A header for the section's boundary on the globe, stating the number of vertices in the boundary, followed by
    - The list of vertices in the boundary.
      Each row is a latitude and the corresponding longitude, in degrees.
  - A header for the table of x and y values, stating the number of rows and columns in the table, followed by
    - The table of x and y values corresponding to certain latitudes and longitudes.
      Each row corresponds to one latitude, and each pair of columns to one longitude.
      Each pair of adjacent values is an x value followed by the corresponding y value, in kilometers.
      Undefined values are set to *NaN*.
      The latitudes and longitudes are not explicitly given;
      the latitudes are evenly spaced between -90° and 90° (inclusive),
      and the longitudes are evenly spaced between -180° and 180° (inclusive).
- A header for the map projection's projected boundary, stating the number of vertices in the boundary, followed by
  - The list of vertices in the projected boundary.
    Each row is an x value followed by the corresponding y value, in kilometers.
- A header for the inverse table, stating the number of rows and columns in the table, followed by
  - The minimum x value of the inverse table, the minimum y value of the inverse table,
    the maximum x value of the inverse table, and the maximum y value of the inverse table, in kilometers.
  - The table of latitudes and longitudes corresponding to certain x and y values.
    Each row corresponds to one x value and each pair of columns to one y value.
    Each pair of adjacent values is a latitude followed by the corresponding longitude, in degrees.
    The x and y values are not explicitly given;
    they are evenly spaced between the given minimums and maximums.

## Using the code

Most of the code is Python scripts that you can just run.
All can be found in `src/`.
Here are the important ones:
- `elastik.py` generates a bunch of [nice maps](examples) based on pregenerated map projections.
- `create_map_projection.py` generates a map projection based on pregenerated weights and meshes.
- `calculate_weights.py` generates grayscale images that can be used as weights for new map projections (requires coastline data; see below).
- `build_mesh.py` generates an unoptimized mesh specifying the rough configuration of a new map projection based on a manually specified or pregenerated cut file.
- `find_drainage_divides.py` generates a cut file based on the boundaries between watersheds (requires elevation data; see below).
- `run_all_scripts.py` executes `build_mesh.py`, `calculate_weights.py`, `create_map_projections.py`, and `elastik.py` in that order on all of their possible inputs.

Some of these have PyPI dependencies, which are enumerated in `requirements.txt`.
You'll also likely need to pay attention to the C library, `sparse.c`.
A compiled DLL file is included in this repository,
but depending on your system you'll likely need to recompile it yourself.
I do this by going to the root directory in VS Developer Command Prompt and using
~~~bash
cl /D_USR_DLL /D_WINDLL src/sparse.c /link /DLL /OUT:lib/libsparse.dll
~~~
Naturally, if you're not on Windows, it'll be something different.
Hopefully you can figure it out.  Make sure the command prompt you use
has the same architecture as the Python distribution you're running!

### Data sources

I didn't include the geographic data that some of the code depends on because
some of it is kind of big, and it might need to be updated in the
future.

To run the `calculate_weights.py` script, you will need some coastline and
land polygon datasets from Natural Earth ([naturalearthdata.com](https://www.naturalearthdata.com)).
Download the zip files and put them in `resources/shapefiles/`. 
The script will unzip them automatically.
You specifically need
- Natural Earth 10m coastline dataset
- Natural Earth 10m minor islands coastline dataset
- Natural Earth 110m land dataset

To run the `find_drainage_divides.py` script,
you will need the USGS EROS Center's global 30 arcsec elevation dataset
(GTOPO30, DOI: [10.5066/F7DF6PQS](https://doi.org/10.5066/F7DF6PQS)).
Download it from [EarthExplorer](https://earthexplorer.usgs.gov/) as GeoTIF files
and put them in `resources/elevation/`.
The script will load and assemble it automatically.
You will also need the Natural Earth 10m rivers with lake centerlines dataset
([naturalearthdata.com](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-rivers-lake-centerlines/)).
Download it as a zip file and put it in `resources/shapefiles/`.

To run the `elastic.py` script,
you will need the World Wildlife Fund's terrestrial ecoregions of the world dataset
([worldwildlife.org](https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world)).
Download the zip file and put it in `resources/shapefiles/`.
You will also need several datasets from Natural Earth (naturalearthdata.com).
Download them as zip files and put them in `resources/shapefiles/`.
- Natural Earth 10m bathymetry A–K datasets
- Natural Earth 50m coastline dataset
- Natural Earth 50m rivers with lake centerlines dataset with scale ranks and tapering
- Natural Earth 50m ocean dataset
- Natural Earth 110m admin 0 countries dataset
- Natural Earth 110m lakes dataset
