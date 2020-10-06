# Running an experiment

## Generate Initial Population

Currently there are two ways to acquire an initial population:

* Use Grasshopper and [DecodingSpaces](https://toolbox.decodingspaces.net/#aboutToolbox) and generate a population with characteristics close to what the case study involves.
* Use real world shape files

In both cases, the extent of the urban area should be 250 m x 250 m in order to comply with the pretrained model doing the inference. Currently, the buildings are scaled to 512x512 images (within GH, but that can also happen outside) in order to allow for more detailed heightmaps for the model.

## Set experiment parameters

The parameters that detail each experiment can be set through the configuration file. An example file can be seen below.

### config.ini

```ini
[collection]
# collection folder
input_dir = F:\PhD_Research\Output\CaseStudies\MAP-Elites\experiments\real_world\NewYork\data
# color map file
color_file = F:\PhD_Research\Output\CaseStudies\MAP-Elites\experiments\real_world\NewYork\color_map.csv
# number of parameters in the generative model, used only for GH-generated individuals
n_genes = 5
# number of pairs to create when selecting by distance, used only for GH-generated individuals
n_pairs = 5
# boundary domain size for individuals in m x m, should be equal to the size of the bounding box the individuals were created in.
size_x = 512
size_y = 512

[mapelites]
# random seed
seed = 59
# number of initial random samples: how many individuals to seed the map with
bootstrap_individuals = 1500
# number of map elites iterations: how many generations to run
iterations = 100
# number of curious individuals to select from: the number of most curious individuals to select during evolution
n_curious = 50
# True: solve a minimization problem. False: solve a maximization problem
minimization = False
# show the plot or not at the end
interactive = False

[plotting]
# Set to true to highlight the best fitness value in the final plot
highlight_best = True

[quality_diversity]
# Define the quality function.
name = 'percentage of dangerous zones'
# Define the behavioral dimensions
dimensions = fsi,gsi,osr,mh,tare,dangerous,sitting
# Define which behavioral dimensions will be used for the experiment
n_bins = 5,6
# Define the discretization for each behavioral dimension
step_fsi = 0.16
step_gsi = 0.014
step_osr = 0.02
step_mh = 0.5
step_tare = 0.025
step_dangerous = 1250
step_sitting = 1250
# Define heatmap bins for feature dimensions
# Name each bin as `bin_{name of behavioral dimension}
# Note: The bins must be defined by numbers, except for the `inf` label which can be defined ether at the beginning or at the end of the bins.
bin_FSI = -inf,0,8,inf
bin_GSI = -inf,0,0.7,inf
bin_OSR = -inf,0,1,inf
bin_MH = -inf,0,25,inf
bin_TARE = -inf,0.25,1.0,inf
bin_dangerous = -inf,0,62500,inf
bin_sitting = -inf,0,62500,inf

[crossover]
#indgen: float(0,1), amount of genetic material to keep from parent 1.
indgen = 0.5
#indprob: float(0,1), probability for each building polygon to be kept.
indpb = 0.5

[mutation]
#eta: float, highest values keep mutations close to parent individual
eta = 20.0
#low: float, lowest possible building height
low = 5.0
#up: float, highest possible building height
up = 100.0
#mut_pb: float(0,1), the probability for each building to be mutated
mut_pb = 0.25

```

## MAP-Elite Step

After initialization, the whole run can be described as follows:

```pseudocode
for i in iterations:
	for j in generated_individuals:
		select n_curious individuals
		select 2 individuals from n_curios at random
		create offspring through crossover
		mutate offspring through mutation
		run inference()
		wind_comfort()
		x, y = map_to_grid()
		if (offspring.performance > grid[x,y]):
			add offspring to grid
			save_to_disk()
		else:
			discard offspring
	plot_heatmap()
save_performances()
save_curiosity_score()
```

Each generation typically runs for 250-500 individuals, depending on the map 30,000-100,000 evaluations are possible. The feasible-infeasible crossover has not been tested yet, however it would involve the addition of another population of individuals (so no discarding) and a step of decreasing their fitness (with fitness=0 meaning they are now feasible) at each iteration.