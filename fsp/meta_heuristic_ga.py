import numpy as np
import random
import time

def generate_initsol(instance, popul_size=3):#returns an np array containing jobs permutations
	#initial array
	nb_jobs = instance.get_jobs_number()
	arr = np.array([])
	res = []
	arr = [int(i) for i in range(nb_jobs)]
	for i in range(popul_size):
		res.append(np.random.permutation(arr))

	res=np.array(res)
	return res,len(res)


#Fitness function: un individu a une forte probabilité de se reproduire si il a une bonne qualité (c-à-d une grande valeur de fitness)
def fitness(instance, chromosome):
	return 1 / instance.makespan(chromosome)

#Selection: sélectionne certaines solutions pour former la population
# intermédiaire afin de lui appliquer les opérateurs de croisement et de mutation. 
# Chaque élément de la population est sélectionné avec une probabilité selon sa valeur d’adaptation.
def selection(popul, size, Pc, fits):
	pool_size=round(Pc*size) #définir le nombre d'individus à selectionner pour la reproduction
	
	matting_pool=[]

	indeces = fits.argsort()[-pool_size:][::-1]
	for index in indeces:
		matting_pool.append(popul[index])

	return np.array(matting_pool)

#Crossover: interchanger les gènes situés entres les points considérés et est appliqué avec une probabilité Pc.
#On utilisera ici le croisement à 2 points car des études faites ont constaté que ce type de croisement est très efficace pour le FSP
def crossover(n, matting_pool, Pc):
	offsprings = []
	nb_cross = round(Pc * len(matting_pool))
	for i in range (nb_cross):
		#generate randomly 2 crossover points
		points = np.random.randint(n-1,size=2)
		first_pt = points[0]
		second_pt = points[1]
		
		while True:
			if (first_pt != second_pt): 
				break
			second_pt=random.randint(0,n-1)

		if (first_pt > second_pt):
				t = first_pt
				first_pt = second_pt
				second_pt = t
		
		#choose randomly 2 parents
		rand_ind = np.random.choice(len(matting_pool), size=2)
		parents = matting_pool[rand_ind]
		# parents = np.random.choice(matting_pool, size=2, replace=False)
		parent1 = parents[0]#[0]
		parent2 = parents[1]#[0]    

		offspring = parent1
		parts_parent1 = []
		parts_parent1[0:first_pt] = parent1[0:first_pt]

		if (first_pt != 0):
			parts_parent1[first_pt+1:first_pt+(n-second_pt)+1] = parent1[second_pt:n]
		else:
			parts_parent1[first_pt:first_pt+(n-second_pt)+1] = parent1[second_pt:n]
		idx = first_pt
		for i in parent2:
			if (i not in parts_parent1):
				offspring[idx] = i
				idx += 1
		offsprings.append(offspring)

	return offsprings

#Mutation: consiste à choisir un ou deux bits aléatoirement, puis les inverser. L'opérateur de mutation s'applique avec une certaine probabilité Pm, appelée
#taux de mutation
# Dans le cas d'une mutation par changement de poste, une tâche située à un poste est supprimée et placée à un autre poste. Ensuite, tous les autres postes 
#sont déplacés en conséquence. Les deux positions sont choisies au hasard.
def mutation(n, offsprings, Pm):
	for offspring in offsprings:
		pos = np.random.choice(n, size=2, replace=False)
			
		if pos[0] > pos[1]:
				t = pos[0]
				pos[0] = pos[1]
				pos[1] = t
		
		remJob = offspring[pos[1]]
		
		for i in range(pos[1], pos[0], -1):
				offspring[i] = offspring[i-1]
				
		offspring[pos[0]] = remJob
	offsprings = np.array(offsprings)
	return offsprings

#new_generation: la nouvelle génération est constituée des meilleurs individus de la population
def new_generation(n, init_sol, popul_size, fits):
	new_gen=np.array([])
	new_gen=selection(init_sol,popul_size,1,fits)
	if (len(new_gen)<popul_size):
		adds=popul_size-len(new_gen)
		i=0
		k=0
		while i<adds:
			for j in init_sol:
				j=np.array(j)
				varr=any((j == x).all() for x in new_gen)
				if (not varr):
					new_gen=np.append(new_gen,j[None,:],axis=0)           
					i+=1
					if (i==adds):
						break		
	return new_gen

def get_results(instance, popul_size = 12, nb_generations = 100, Pc = 0.9, Pm = 0.06):
	n = instance.get_jobs_number()
	start = time.time()
	#Generate an inital population
	init_pop,size = generate_initsol(instance,popul_size)

	for _ in range(nb_generations):
		#Reproduction: pick the best chromosomes (tournament selection)
		#evaluation
		fits = np.array([])
		for i in range(len(init_pop)):
			fits = np.append(fits,fitness(instance, init_pop[i]))#fits déja calculés
		#reproduction
		matting_pool = selection(init_pop,size,Pc,fits)

		#Crossover: 
		offsprings = crossover(n,matting_pool,Pc)

		#Mutation
		offsprings = mutation(n,offsprings,Pm)
		#Selection: 
		init_pop = np.concatenate((init_pop,offsprings),axis=0)
		#evaluation
		fits = np.array([])
		for i in range(len(init_pop)):
			fits = np.append(fits,fitness(instance,init_pop[i]))#fits déja calculés
		init_pop = new_generation(n,init_pop,size,fits)
		
	#End While, select the best solution from the last generation.
	#choose the best sequence according to the chromosomes' fitnesses => last evaluation
	fits = np.array([])
    
	for i in range(len(init_pop)):
		fits = np.append(fits,fitness(instance,init_pop[i]))#fits déja calculés
	
	indeces = fits.argsort()[-1:][::-1]
	best_sol = init_pop[indeces[0]]

	return {
			"C_max" : instance.makespan(best_sol),
			"order" : best_sol.tolist(),
			"time" : time.time() - start
	}