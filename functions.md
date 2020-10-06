

## Functions

### create_path

```python
def create_path(polygon):
	"""
	A function that splits a series of polygon points into an (x, y) array of
	coordinates.
	
	:param polygon: A polygon (i.e. array of x, y sequences) creates by a design
	software or extracted from a geometry file.
	"""
	
	x_coords = polygon[0::2]
	y_coords = polygon[1::2]

	return np.hstack((x_coords, y_coords))
```

### create_shapely_polygons

```python
def create_shapely_polygons(points, splits):
	"""
	A function that generates shapely polygons out of points and splits (indices
	on where to split the points) of each individual.
	
	:param points: a list of points for each individual.
	:param splits: a list of indices that show where to split the point list in
	order to create individual polygons.
	"""
	
	polygons = np.array(np.vsplit(points.reshape(-1, 1), np.cumsum(splits)))[:-1]

	shapes = []
	for poly in polygons:
		path = create_path(poly)
		shapes.append(Polygon(path))

	return np.array(shapes)
```

### find_intersections

```python
def find_intersections(seed_polygon, target_polygons):
	"""
	A function that finds intersections between a seed polygon and a list of
	candidate polygons.
	
	:param seed_polygon: A shapely polygon.
	:param target_polygons: A collection of shapely polygons.
	"""
	
	intersect_booleans = []
	for i, poly in enumerate(target_polygons):
		intersect_booleans.append(seed_polygon.intersects(poly))

	return intersect_booleans
```

### centroids

```python
def centroids(polygons):
	"""
	A function that calculates the centroids of a collection of shapely polygons.
	
	:param polygons: A collection of shapely polygons.
	"""
	
	centroids = []
	for polygon in polygons:
		xy = polygon.centroid.xy
		coords = np.dstack((xy[0], xy[1])).flatten()
		centroids.append(coords)

	return centroids
```

### get_features

```python
def get_features(footprints, heights, boundary=(512, 512), b_color="white"):
	"""
	Calculates urban features for a set of footprints and heights. Features
	include:
	floor space index (FSI): gross floor area / area of aggregation
	ground space index (GSI): footprint / area of aggregation
	oper space ratio (OSR): (1-GSI)/FSI
	building height (L): FSI/GSI
	tare (T): (area of aggregation - footprint) / area of aggregation
	
	:param footprints: list of areas for each building of an individual
	:param heights: list of heights for each building of an individual
	"""
	
	#calculate aggregation A
	area_of_aggregation = boundary[0] * boundary[1]
	#calculate GFA
	gross_floor_area = np.multiply(footprints, np.ceil(heights/4)).sum()
	total_footprint = np.array(footprints).sum()

	fsi = gross_floor_area / area_of_aggregation
	gsi = total_footprint / area_of_aggregation
	osr = (1-gsi) / fsi
	mean_height = fsi / gsi
	tare = (area_of_aggregation - total_footprint) / area_of_aggregation

	return fsi, gsi, osr, mean_height, tare
```

### wind_comfort

```python
def wind_comfort(experiment_folder):
	"""
	A function that calculates fitness values for the individuals that were just
    infereced by the pretrained model. Note: hardcoded for now in the server code.
    
	:param experiment_folder: The folder where the inference data is saved in.
	"""

	#get inference data from the experiment folder
	lawson_results = glob.glob(experiment_folder + '/lawson*.npy')
	total_area = glob.glob(experiment_folder + '/area*.npy')

	lawson_sitting = []
	lawson_dangerous = []

	for result, area in zip(lawson_results, total_area):

		lawson = np.load(result)
		area = np.load(area)

		unique, counts = np.unique(lawson, return_counts=True)

		try:
            #turn 512 x 512 to 250 x 250
			sitting = np.sum(counts[np.where(unique<=2)[0]])/4.2 
		except:
			sitting = 0

		try:
            #turn 512 x 512 to 250 x 250
			dangerous = np.sum(counts[np.where(unique>=4)[0]])/4.2
		except:
			dangerous = 0

		sitting_percentage = (sitting / area.item()) * 100
		dangerous_percentage = (dangerous / area.item()) * 100

	return sitting_percentage, dangerous_percentage, sitting, dangerous
```

### run_inference

```python
def run_inf(port=5559, host='127.0.0.1', timeout=1000):
	"""
	Function to run inference on the local server where the pretrained model is
    running.
    
	:param port: The port through which the communication with the model happens.
	:param host: The local address of the server.
	:param timeout: A specified amount of time to wait for a response from the 
	server before closing.
	"""
	
	# Connect to server
	#print("Connecting to server...")

	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.connect((host,port))

		s.sendall(b'run model!')

		#send message to server
		#print(f'Sending request...')

		data = s.recv(1024).decode()
		#print (data)
		server_msg_ = data
```

### genetic_similarity

```python
def genetic_similarity(ind_files, n_genes):
	"""
	Calculates similarity between individuals created in Grasshopper, using
	information (input parameters) saved in each individual's name convention.
	
	:param ind_files: A Collection of individuals generated through grasshopper.
	:param n_genes: The total number of parameters in the Grasshopper parametric
	model.
	"""
	
	genomes = []
	for file in ind_files:
		genomes.append(file.split('\\')[-1].split('_')[:-1][1::2])

	genome = np.array(genomes).reshape(-1, n_genes).astype(int)
	similarity = cosine_similarity(genomes)

	return similarity
```

### diverse_pairs

```python
def diverse_pairs(similarity_matrix, n_pairs):
	"""
	A function to find the indices of a specified number of individuals, for each
	parent, which are as different as possible from the parent, based on their
	genetic similarity.
	
	:param similarity_matrix: The similarity matrix computed for all individuals.
	:param n_pairs: The number of most dissimilar individuals to find for each
	parent.
	"""
	
	diverse_pairs = []
	for ind in similarity_matrix:
		diverse_pairs.append(np.argpartition(ind, n_pairs)[:n_pairs])

	return diverse_pairs
```

### create_map

```python
def create_cmap(color_file):
	"""
	Creates an RGB color map out of a list of color values.
	
	:param color_file: A csv file of the color gradient used to generate the
	heightmaps.
	"""
	
	colors = np.loadtxt(color_file, dtype=str)
	cmap = []
	for color in colors:
		cmap.append(color.split(',')[0])

	cmap = np.array(cmap, dtype=int)

	return cmap
```

### height_to_color

```python
def height_to_color(cmap, height):
	"""
	Translates a building height value to a color, based on the given color map.
	
	:param cmap: A color map.
	:param height: A building height
	"""
	
	if(height > len(cmap)-1):
		color_value = 0
	else:
		modulo = height % 1
		if(modulo) == 0:
			color_value = cmap[height]
		else:
			minimum = floor(height)
			maximum = ceil(height)

			min_color = cmap[minimum+1]
			max_color = cmap[maximum+1]

			color_value = min_color + ((min_color-max_color) * modulo)

	return [color_value, color_value, color_value]
```

### draw_polygons

```python
def draw_polygons(polygons, colors, im_size=(512, 512), b_color="white", fpath=None):
	"""
	A function that draws a PIL image of a collection of polygons and colors.
	
	:param polygons: A list of shapely polygons.
	:param colors: A list of R, G, B values for each polygon.
	:param im_size: The size of the input geometry.
	:param b_color: The color of the image background.
	:param fpath: The file path to use if saving the image to disk.
	"""
	
	image = Image.new("RGB", im_size, color=b_color)
	draw = aggdraw.Draw(image)

	for poly, color in zip(polygons, colors):
		# get x, y sequence of coordinates for each polygon
		xy = poly.exterior.xy
		coords = np.dstack((xy[0], xy[1])).flatten()
		# create a brush according to each polygon color
		brush = aggdraw.Brush((color[0], color[1], color[2]), opacity=255)
		# draw the colored polygon on the draw object
		draw.polygon(coords, brush)

	#create a PIL image out of the aggdraw object
	image = Image.frombytes("RGB", im_size, draw.tobytes())

	if(fpath):
		image.save(fpath)

	return draw, image
```

### plot_heatmap

```
too hacky to share, TBD :D
```

### crossover

```python
def crossover(ind1, ind2, indgen, indprob, parent_ids, verbose=False):
	"""
	Executes an initialization of the population within a collection by creating an individual from two
	provided individuals. The buildings are selected from ind1 according to the	*indpb* probability, while
	*indgen* defines how much genetic material will be used from ind1.

	:param ind1: The first individual participating in the crossover.
	:param ind2: The second individual participating in the crossover.
	:param indgen: The minimum amount of genetic material to be taken from
	ind1.
	:param indpb: Independent probability for each building polygon to be kept.
	:returns: A tuple of two individuals.
	"""

	# get geometry information from individuals
	heights1, polygons1, colors1, centroids1, size1 = ind1.heights, ind1.polygons, ind1.colors, ind1.centroids, ind1.size
	heights2, polygons2, colors2, centroids2, size2 = ind2.heights, ind2.polygons, ind2.colors, ind2.centroids, ind2.size
	if(verbose):
		print("Individual 1 has {} buildings".format(len(polygons1)))
		print("Individual 2 has {} buildings".format(len(polygons2)))

	#keep a minimum amount of genetic material from parent 1, according to indgen

	selection = []

	#some hacky stuff to avoid weird or failed crossover when individuals have only a few buildings
	if(len(polygons2) < 4 or len(polygons1) < 4):

		if(len(polygons1) > 4):
			poly_ids = random.sample(list(np.arange(0, len(polygons1))), int(len(polygons1) * indgen))
		else:
			poly_ids = [random.randrange(0, len(polygons1))]
		if(verbose):
			print('total selected buildings from individual 1: {} out of {}'.format(len(poly_ids), len(polygons1)))

		p1 = polygons1[poly_ids]
		hts1_ = heights1[poly_ids]
		centroids1_ = centroids1[poly_ids]
		colors1_ = colors1[poly_ids]
		assert p1.shape[0] == hts1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0]
		assert len(p1.shape) == len(hts1_.shape)
		assert len(centroids1_.shape) == len(colors1_.shape)

		intersection_matrix = np.zeros((p1.shape[0], len(polygons2)))
		for k, p in enumerate(p1):
			intersection_matrix[k, :] = geom.find_intersections(p, polygons2)
		bools = np.sum(intersection_matrix, axis=0).astype(bool)
		mask = ~bools

		p2 = polygons2[mask]
		hts2_ = heights2[mask]
		colors2_ = colors2[mask]
		assert p2.shape[0] == hts2_.shape[0] == colors2_.shape[0]
		assert len(p2.shape) == len(hts2_.shape)

		#join polygons from both parents and assign colors
		polygons_cross = np.hstack((p1, p2))
		colors_cross = np.vstack((colors1_, colors2_))
		heights_cross = np.hstack((hts1_, hts2_))
		assert polygons_cross.shape[0] == colors_cross.shape[0] == heights_cross.shape[0]
		if(verbose):
			print("{} buildings present in offspring".format(polygons_cross.shape[0]))
		offspring = Offspring(polygons_cross, colors_cross, heights_cross, size1, parent_ids)
	else:
		poly_ids = random.sample(list(np.arange(0, len(polygons1))), int(len(polygons1) * indgen))
		if(verbose):
			print('total selected buildings from individual 1: {} out of {}'.format(len(poly_ids), len(polygons1)))

		# keep only selected buildings from the first individual and throw away all intersecting
		# building from the second individual
		p1 = polygons1[poly_ids]
		hts1_ = heights1[poly_ids]
		centroids1_ = centroids1[poly_ids]
		colors1_ = colors1[poly_ids]
		assert p1.shape[0] == hts1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0]
		assert len(p1.shape) == len(hts1_.shape)
		assert len(centroids1_.shape) == len(colors1_.shape)

		#find intersections for each polygon selected in p1
		intersection_matrix = np.zeros((p1.shape[0], len(polygons2)))
		for k, p in enumerate(p1):
			intersection_matrix[k, :] = geom.find_intersections(p, polygons2)
		bools = np.sum(intersection_matrix, axis=0).astype(bool)
		mask = ~bools

		p2 = polygons2[mask]
		hts2_ = heights2[mask]
		colors2_ = colors2[mask]
		assert p2.shape[0] == hts2_.shape[0] == colors2_.shape[0]
		assert len(p2.shape) == len(hts2_.shape)

		#join polygons from both parents and assign colors
		polygons_cross = np.hstack((p1, p2))
		colors_cross = np.vstack((colors1_, colors2_))
		heights_cross = np.hstack((hts1_, hts2_))
		assert polygons_cross.shape[0] == colors_cross.shape[0] == heights_cross.shape[0]
		if(verbose):
			print("{} buildings present in offspring".format(polygons_cross.shape[0]))
		
		offspring = Offspring(polygons_cross, colors_cross, heights_cross, size1, parent_ids)

	return offspring
```

### fi_crossover

```python
def feasible_infeasible_crossover(ind1, ind2, indprob):
	"""
	Executes an FI crossover and calculates fitness with respect to infeasibility

	:param ind1: The first individual participating in the crossover.
	:param ind2: The second individual participating in the crossover.
	:param indgen: The minimum amount of genetic material to be taken from
	ind1.
	:param indpb: Independent probability for each building polygon to be kept.
	:returns: An individual.
	"""

	# get geometry information from individuals
	heights1, polygons1, colors1, centroids1, size1 = ind1.heights, ind1.polygons, ind1.colors, ind1.centroids, ind1.size
	heights2, polygons2, colors2, centroids2, size2 = ind2.heights, ind2.polygons, ind2.colors, ind2.centroids, ind2.size
	#print("Individual 1 has {} buildings".format(len(polygons1)))
	#print("Individual 2 has {} buildings".format(len(polygons2)))

	#select from ind1
	probs = np.array([uniform(0,1) for j in range(0, len(polygons1))])
	selection = [probs < indprob]
	poly_ids = np.arange(0, len(polygons1))[tuple(selection)]
	p1 = polygons1[poly_ids]
	hts1_ = heights1.reshape(-1, 1)[poly_ids]
	centroids1_ = centroids1[poly_ids]
	colors1_ = colors1[poly_ids]
	assert p1.shape[0] == hts1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0]
	assert len(p1.shape) == len(hts1_.shape)
	assert len(centroids1_.shape) == len(colors1_.shape)

	#select from ind2
	probs = np.array([uniform(0,1) for j in range(0, len(polygons2))])
	selection = [probs < indprob]
	poly_ids = np.arange(0, len(polygons2))[tuple(selection)]
	p2 = np.array(polygons2)[poly_ids]
	hts2_ = heights2.reshape(-1, 1)[poly_ids]
	centroids2_ = centroids2[poly_ids]
	colors2_ = colors2[poly_ids]
	assert p2.shape[0] == hts2_.shape[0] == centroids2_.shape[0] == colors2_.shape[0]

	# join material from both individuals
	p_cross = np.hstack((p1, p2))
	hts_cross = np.vstack((hts1_, hts2_))
	colors_cross = np.vstack((colors1_, colors2_))
	#keep a minimum amount of genetic material from parent 1, according to indgen

	# calculate feasibility fitness through self-intersection
	intersection_matrix = np.zeros((len(p_cross), len(p_cross)))
	for k, p in enumerate(p_cross):
		intersection_matrix[k, :] = intersection(p, p_cross)
	#remove self-intersection for each polygon
	intersection_events = np.sum(intersection_matrix, axis=0)-1
	fi_fitness = np.where(intersection_events>0)[0].shape[0]/intersection_events.shape[0]

	offspring = Offspring(p_cross, colors_cross, hts_cross, size1)
	offspring.fi_fitness = fi_fitness

	return offspring
```

### polynomial_bounded_mutation

```python
def polynomial_bounded_mutation(ind, cmap, eta: float, low: float, up: float, mut_pb: float):
		"""Return a polynomial bounded mutation, as defined in the original NSGA-II paper by Deb et al.
		Mutations are applied directly on `individual`, which is then returned.
		Inspired from code from the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py).

		Parameters
		----------
		:param individual
			The individual to mutate.
		:param eta: float
			Crowding degree of the mutation.
			A high ETA will produce mutants close to its parent,
			a small ETA will produce offspring with more differences.
		:param low: float
			Lower bound of the search domain.
		:param up: float
			Upper bound of the search domain.
		:param mut_pb: float
			The probability for each item of `individual` to be mutated.
		"""
		mut_heights = copy(ind.heights)
		for i in range(len(ind.heights)):
			if random.random() < mut_pb:
				x = ind.heights[i].astype(float)
				if(x<low):
					x=low
				if(x>up):
					x=up
				delta_1 = (x - low) / (up - low)
				delta_2 = (up - x) / (up - low)
				rand = random.random()
				mut_pow = 1. / (eta + 1.)

				if rand < 0.5:
					xy = 1. - delta_1
					val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
					delta_q = val**mut_pow - 1.
				else:
					xy = 1. - delta_2
					val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
					delta_q = 1. - val**mut_pow

				x += delta_q * (up - low)
				x = min(max(x, low), up)
				if(math.isnan(x)):
					x = random.randrange(low, up)
				mut_heights[i] = x

		mut_colors = np.array([util.height_to_color(cmap, height) for height in mut_heights])
		offspring = Offspring(ind.polygons, mut_colors, mut_heights, ind.size, ind.parent_ids)

		return offspring
```

### crossover_mutation

```python
def crossover_mutation(ind1, ind2, eta: float, low: float, up: float, mut_pb: float, cross_pb: 'default'):
		"""Return a polynomial bounded mutation, on individuals produced from a uniform random crossover of 2 seed
        individuals.

		Parameters
		----------
		:param ind1, in2
			The individuals to mutate.
		:param eta: float
			Crowding degree of the mutation.
			A high ETA will produce mutants close to its parent,
			a small ETA will produce offspring with more differences.
		:param low: float
			Lower bound of the search domain.
		:param up: float
			Upper bound of the search domain.
		:param mut_pb: float
			The probability for each item of `individual` to be mutated.
		:param cross_pb: float
			The probability for each item of `individual` to be crossed over.
		"""
		mut_heights_1 = copy(ind1.heights)
		mut_heights_2 = copy(ind2.heights)

		# crossover first
		if (cross_pb == 'default'):
			for i, ht in enumerate(ind1.heights):
				if (random.random() < 1/len(ind1.heights)):
					mut_heights_1[i] = ind2.heights[random.randrange(0, len(ind2.heights)-1)]

			for i, ht in enumerate(ind2.heights):
				if (random.random() < 1/len(ind2.heights)):
					mut_heights_2[i] = ind1.heights[random.randrange(0, len(ind1.heights)-1)]
		else:
			for i, ht in enumerate(ind1.heights):
				if (random.random() < cross_pb):
					mut_heights_1[i] = ind2.heights[random.randrange(0, len(ind2.heights)-1)]

			for i, ht in enumerate(ind2.heights):
				if (random.random() < cross_pb):
					mut_heights_2[i] = ind1.heights[random.randrange(0, len(ind1.heights)-1)]

		# mutate after
		for i in range(len(ind1.heights)):
			if (random.random() < mut_pb):
				x = ind1.heights[i].astype(float)
				delta_1 = (x - low) / (up - low)
				delta_2 = (up - x) / (up - low)
				rand = random.random()
				mut_pow = 1. / (eta + 1.)

				if rand < 0.5:
					xy = 1. - delta_1
					val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
					delta_q = val**mut_pow - 1.
				else:
					xy = 1. - delta_2
					val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
					delta_q = 1. - val**mut_pow

				x += delta_q * (up - low)
				x = min(max(x, low), up)
				mut_heights_1[i] = x

		# mutate after
		for i in range(len(ind2.heights)):
			if (random.random() < mut_pb):
				x = ind2.heights[i].astype(float)
				delta_1 = (x - low) / (up - low)
				delta_2 = (up - x) / (up - low)
				rand = random.random()
				mut_pow = 1. / (eta + 1.)

				if rand < 0.5:
					xy = 1. - delta_1
					val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
					delta_q = val**mut_pow - 1.
				else:
					xy = 1. - delta_2
					val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
					delta_q = 1. - val**mut_pow

				x += delta_q * (up - low)
				x = min(max(x, low), up)
				mut_heights_2[i] = x


		mut_colors_1 = np.array([height_to_color(cmap, height) for height in mut_heights_1]).astype(int)
		mut_colors_2 = np.array([height_to_color(cmap, height) for height in mut_heights_2]).astype(int)

		offspring_1 = Offspring(ind1.polygons, mut_colors_1, mut_heights_1, ind1.size, ind1.parent_ids)
		offspring_2 = Offspring(ind2.polygons, mut_colors_2, mut_heights_2, ind2.size, ind2.parent_ids)

		return offspring_1, offspring_2
```

