## Overview

### What is this exactly

A python implementation of MAP-Elites, adjusted for inputs and workflows related to architectural design.  The code was initially inspired by Stefano Fioravanzo’s implementation (https://github.com/StefanoFioravanzo/MAP-Elites) and was adjusted where it was necessary to support the domain of architectural design and when new functionalities were implemented.

### Current supported features

The library currently supports the standard MAP-Elites algorithm, adjusted to architectural design, with the following features:

* individuals are single buildings and can have arbitrary massing designs
* individuals’ genotype is defined by two parameters: polygons and heights
* evaluation uses a surrogate (pretrained) model for real-time performance assessment (currently wind comfort and wind flow)
* evolution happens through polynomial bounded mutation, inspired by {cite}`deb2002`, and geometrical crossover with or without feasibility constraint 
* selection happens through individual’s quality and their curiosity score, inspired by {cite}`stanton2016curiosity`
* the history of elite designs for each cell in the behavioral space is recorded and accessible after the run
* behavioral characterization can happen on the objectives space (performance metrics) and on the design or input space (e.g. density, area, etc.) 

### TODO List:

The goal is to implement various interesting approaches, ideas, and algorithms in the library in order to cover the state-of-the-art of the MAP-Elites implementation space. The main priorities are listed below, a more detailed diagram can be seen in the image:

- [ ] generative model(s) for individual (design) generation
- [ ] generative model(s) for design space embeddings
- [ ] generative model(s) for elite hypervolume embeddings
- [ ] generative model(s) for BCs generation
- [ ] sliding-boundary MAP-Elites
- [ ] CVT MAP-Elites
- [ ] Multi-emmitters
- [ ] Surprise score
- [ ] Parallel / batch inference on CPU and GPU



```{figure} ./introduction.assets/todo.jpg
---
width: 800px
height: 500px
align: center
---
ToDO List
```


```{bibliography} references.bib

```

