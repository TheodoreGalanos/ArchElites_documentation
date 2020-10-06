## Introduction

The structure of ArchElites can be seen by the organization of its source code. The library provides classes for building individual designs and a collection of designs either from a seed population or from evolved individuals. Currently only geometric primitives are supported as genomes, specifically each individual requires a list of points, along with number of points in each polygon, and a list of heights. IO routines are provided to read a collection of individuals from disk and also save an offspring to disk during evolution. Each individual has a drawing method that allows for exporting a heightmap, which is currently used for performance evaluation. Finally, evaluation happens by contacting a pretrained model which is deployed in a simple format through sockets.

Classes are named using `CamelCase` (capital letters at the start of each word). functions, methods and variable names are `lower_case_underscore` (lowercase with an underscore representing a space between words).

## ArchElites Basics

After starting Python, import the archelites module with

```python
import arch_elites as ae 
```

### Population

The following basic individual classes are provided as Python classes:

[Collection](source:classes/Collection):

:  A class to create a collection of individuals based on genetic information that is either extracted from the generative grasshopper model or from local files describing each individual design.

[Individual](source:classes/Individual):

:  A class to create an individual, along with its properties, out of a collection.

[Offspring](source:classes/Offspring):

:  A class to create an offspring, along with its properties, from evolution (crossover and/or mutation) of 2 or more individuals. 

### Operators

The following basic evolutionary operators are provided as Python functions:

[crossover](functions/crossover):

:  A function that executes crossover of genetic material between two Individuals or Offspring according to a crossover probability **cross_prob** and a required amount of genetic material from target individual **cross_gen**.

[fi_crossover](functions/fi_crossover):

:  A function that executes feasible-infeasible style crossover of genetic material between two Individuals or Offspring according to a crossover probability **cross_prob**. The class returns an individual with a feasible-infeasible fitness **fi_fitness** which corresponds to the number of intersections in its genetic material (polygons).

[polynomial_bounded_mutation](functions/polynomial_bounded_mutation):

:   A function that defines polynomial bounded mutation, as defined in {cite}`deb2002`. Mutations are applied directly to the individual, which is then returned as output.

[crossover_mutation](functions/crossover_mutation):

:   A function that defines a random uniform crossover and polynomial bounded mutation on two input individuals. Mutations are applied directly to both individuals, which are then returned as output.

### Geometry

A collection of functions that create, adjust or evaluate geometric primitives. The utilities currently implemented are:

[create_path](Source:Functions/create_path):

:  A function that splits a list of points into an array of (x, y) coordinates.

[create_shapely_polygons](functions/create_shapely_polygons):

:  A function that generates shapely polygons out of points and splits (indices indicating where a list of points is split into polygons) of an individual.

[find_intersections](functions/find_intersections):

:  A function that finds intersections between a seed source polygon and a list of target polygons.

[centroids](functions/centroids):

:  A function that calculates the centroids of a collection of polygons.

[get_features](functions/get_features):

:  A function that calculates features for an individual. Currently supported features are:

* floor space index (FSI): gross floor area / area of aggregation
* ground space index (GSI): footprint / area of aggregation
* open space ratio (OSR): (1-GSI)/FSI
* building height (L): FSI/GSI
* tare (T): (area of aggregation - footprint) / area of aggregation

[draw_polygons](functions/draw_polygons):

:  A function that outputs an image heightmap of an individual design.

### Evaluation

Functions that evaluate performance for each individual. The current performance metric implemented is pedestrian wind comfort, which is evaluated according to the internationally accepted Lawson classification (see image, just for reference). 

![lawson criteria graph](https://clqtg10snjb14i85u49wifbv-wpengine.netdna-ssl.com/wp-content/uploads/2020/01/lawson-graph-1-option-3-lines.png)

[wind_comfort](functions/wind_comfort):

:  A function that calculates pedestrian wind comfort for an input individual, based on the result of the inference that was received from the pretrained model.

```[notes]
Currently, only two categories are calculated by the wind_comfort function: 
sitting long and uncomfortable.
```

### Inference

Function that handles the communication with the pretrained model, sending the necessary input and receiving the output evaluation for each individual.

[run_inference](functions/run_inference):

:  A function to run inference on the (locally) deployed pretrained model.

### Utilities

A collection of useful utilities that are used throughout the library for specific tasks.

[genetic_similarity](functions/genetic_similarity):

:  A function that calculates similarity between individuals created in Grasshopper, using information concerning design input parameters saved in each individual's naming convention. Currently uses cosine similarity.

[diverse_pairs](functions/diverse_pairs):

:  A function that outputs indices of a specified number of individuals that are as different as possible from the input individual, based on their genetic similarity.

[create_map](functions/create_map):

:  A function that creates an RGB color map out of an input list of color values.

[height_to_color](functions/height_to_color):

:  A function that translates a building height value to a color, based on the given color map.

[draw_polygons](functions/draw_polygons):

:  A function that outputs a PIL image of a collection of polygons and colors.

[plot_heatmap](functions/plot_heatmap):

:  A function that saves a plot of the the MAP-Elites grid map.



```{warning}
The current implementation is quite brittle and extremely specific to the domain and case studies involved! Focus for the near future is to make it work for a more general classes of inputs (e.g. JSON or geoJSON schemas).
```

### Bibliography

```{bibliography} references.bib

```

