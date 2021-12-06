# -*- coding: utf-8 -*-
import sys
from random import randint
from random import uniform
import numpy as np
from scipy.spatial.distance import cdist
from skimage import data, io, filters

sys.setrecursionlimit(11500)

class Diffusion(object):
    """General class for all types of diffusion."""
    def __init__(self,mif_size=5,pob=20,initial_diff=[(50,50)],
                p0=0.3, max_iter=15):
        self._pob = pob
        self._p0 = p0
        self.max_iter = max_iter
        self.mif_size = mif_size
        self.iteration = 0
        self._infected_pop = []
        self._tmp_adopted = []
        self._clean = False
        self._initial_diff = initial_diff
        self.time_series = []
        self.mif_size = mif_size

    def initialize_mif(self,mif_size):
        """MIF initiation"""
        x = np.linspace(0.5,mif_size - 0.5,mif_size)
        y = np.linspace(0.5,mif_size - 0.5,mif_size)
        xv,yv = np.meshgrid(x,y)
        points = np.array(list(zip(np.ravel(xv),np.ravel(yv))))
        center = np.array([[mif_size/2 + 0.5,mif_size/2 + 0.5]])
        #print(points)
        #print(center)
        dist = cdist(center,points)
        dist = dist/np.sum(dist)
        #ALL: needs to be different to respect the p0 of the user
	# print(type(mif_size), type(mif_size/2), mif_size/2)
        dist.reshape(mif_size, mif_size)[int(mif_size/2 + 0.5), int(mif_size/2 + 0.5)] = self._p0
        dist = dist/np.sum(dist)
        return np.cumsum(dist)


    def _mif2delta(self,index):
        """Returns a tuple with increases to obtain the propagated data."""

        return np.unravel_index(index,(self.mif_size,self.mif_size))

    def _select_from_mif(self):
        """Returns one direction (pob_adress) from MIF."""
        rnd = uniform(0,1)
        index = np.nonzero(self._mif>rnd)[0][0]
        return self._mif2delta(index)

    def _clean_adopters(self):
        """Cleans and initialize de new simulation."""

        self._infected_pop = []
        self._tmp_adopted = []
        self._pop_array = np.zeros((len(np.ravel(self.space)),self._pob),
                                    dtype=np.bool)
        self.time_series = []
        for c in self._initial_diff:
            self.space[c[0],c[1]] = 1
            # Modify de original settlers:
            index = self._space2pop_index(c)
            self._pop_array[index][0] = True
            self._infected_pop.append((index,0))

        self._clean = False


class SimpleDiffusion(Diffusion):
    """Simple diffusion spatial model based on Hägerstrand

    1.- Homogeneous and isotropic space
    2.- Single initial diffuser
    3.- .... other assumptions ...

    : param N: int Number of rows in simulation space
    : param M: int Number of columns in simulation space
    : param mif_size: int Size of the matrix (square) of the MIF (must be non)
    : pop param: int population in each cell
    : param initial_diff: [(int, int)] List of diffuser coordinates
                                     initials
    : param p0: float Probability of self-diffusion
    : param max_iter: int Maximum number of iterations

    : attribute space: np.array (M, N, dtype = np.int8) The available space
    : attribute _pop_array: np.array (M * N, pob, dtype = np.bool) array of settlers
                           in each cell
    : attribute _infected_pop: list (space_idx, int) List of the indexes of the
                                adopting cells. The first entry is the
                                flattened index of the cell in the space array and
                                the second is the number of the villager in
                                pop_array. That is, the list of addresses
                                of each infected villager.
    : attribute results: np.array ((M, N, max_iter)) Saves the results of each
                        iteration.
    : attribute time_series: list int Propagations for each iteration
    : attribute _clean: bool Indicates if we have saved results.

    """

    def __init__(self,N=100,M=100,mif_size=5,pob=20,initial_diff=[(50,50)],
                p0=0.3, max_iter=15):

        super().__init__(mif_size, pob, initial_diff, p0, max_iter)
        # super(SimpleDiffusion,self).__init__(mif_size,pob,initial_diff,
        #             p0, max_iter)
        self.M = M
        self.N = N
        self.space = np.zeros((self.N,self.M),dtype=np.int8)
        self._pop_array = np.zeros((len(np.ravel(self.space)),pob),
                                    dtype=np.bool)
        self.result = np.zeros((M,N,max_iter),dtype=np.int8)
        for c in initial_diff:
            if c[0] > M or c[1] > N:
                raise ValueError("The coordinates of diffusors aren't in the space")
            # Modify original settlers:
            index = self._space2pop_index(c)
            self._pop_array[index][0] = True
            self._infected_pop.append((index,0))
        if self.mif_size%2 == 0:
            raise ValueError("The size of MIF needs to be non")
        else:
            self._mif = self.initialize_mif(self.mif_size)

    def initialize_mif(self,mif_size):
        return super(SimpleDiffusion,self).initialize_mif(self.mif_size)

    def _propagate(self,pob_adress):
	'''
	It propagates to the settlers in pob_adress if it's non-adopter.

         : param pob_adress: (int, int) the address of the inhabitant to propagate.
                             The first entry is the index (flattened) in space
                             and the second is the number of the villager in the cell
        '''

        # I review if it's no-adopter
        if self._pop_array[pob_adress[0]][pob_adress[1]] == False:
            self._pop_array[pob_adress[0]][pob_adress[1]] = True
            self._tmp_adopted.append(pob_adress)
            #print "I enfected to "  + str(pob_adress)

        else:
            pass


    def _space2pop_index(self,index):
        """Transform the space index in the index of pop_array
	:param index (int,int) index to transform
        """
        # print(type(index), index)
        return np.ravel_multi_index(index,dims=(self.M,self.N))

    def _pop2space_index(self,index):
        """Returns the tuple (i,j) correspondent to the flattened index."""
        return np.unravel_index(index, (self.M,self.N))

    def _mif2delta(self,index):
        """Returns one tuple with the increase of propagated data."""
        return super(SimpleDiffusion,self)._mif2delta(index)

    def _random_adress(self):
        """Returns a random address (pob_adress)."""
        return (randint(0,(self.M*self.N) - 1),randint(0,self._pob - 1))

    def _select_from_mif(self):
        """Returns a random address from MIF (pob_adress)."""
        return super(SimpleDiffusion,self)._select_from_mif()

    def _get_propagation_adress(self,adress):
        """Returns a random address (pop_adress) propagated from MIF"""

        #print 'It propagates to ' + str(adress)
        delta = self._select_from_mif()
        delta = (delta[0] - int(self.mif_size/2+0.5),delta[1] - int(self.mif_size/2+0.5))
        space_adress = self._pop2space_index(adress[0])
        prop_space_adress = (space_adress[0] + delta[0],
                              space_adress[1] + delta[1])
        try:
            habitant = randint(0,self._pob - 1)
            return (self._space2pop_index(prop_space_adress),habitant)
        except ValueError:
            return self._get_propagation_adress(adress)

    def _clean_adopters(self):
        """Cleans and initialize before a new simulation."""
        return super(SimpleDiffusion,self)._clean_adopters()

    def spatial_diffusion(self):
        """Hagerstrand propagation."""

        #If we have results, we need to clean and initialize
        if self._clean:
            self._clean_adopters()

        if self.iteration == (self.max_iter or
                              np.sum(self._pop_array) >= self.M*self.N*self._pob):
            print("I finish")
            print("There are %i adopters from a total of %i settlers"\
                    % (np.sum(self._pop_array),self.M*self.N*self._pob))
            print("The total of realized iterations is %i" % self.iteration)
            self.iteration = 0
            self._clean = True
            return None
        else:
            for adress in self._infected_pop:
                propagated_adress = self._get_propagation_adress(adress)
                self._propagate(propagated_adress)

            self._infected_pop.extend(self._tmp_adopted)
            #print "There are %i adopters" % len(self._infected_pop)
            self.result[:,:,self.iteration] = np.sum(self._pop_array,
                                                axis=1).reshape(self.M,self.N)
            self.time_series.append(len(self._tmp_adopted))
            self.iteration += 1
            self._tmp_adopted = []
            return self.spatial_diffusion()

    def random_diffusion(self):
        """Random propagation throught space."""

        # If we have results, we need to clean and initialize
        if self._clean:
            self._clean_adopters()

        if self.iteration == (self.max_iter or
                              np.sum(self._pop_array) >= self.M*self.N*self._pob):
            #self.space = np.sum(s._pop_array,axis=1).reshape(s.M,s.N)
            print("I finish")
            print("There are %i adopters from a total of %i settlers" \
                    % (np.sum(self._pop_array),self.M*self.N*self._pob))
            print("The total fo realized iterations is %i" % self.iteration)
            self.iteration = 0
            self._clean = True
            return None
        else:
            for adress in self._infected_pop:
                rand_adress = self._random_adress()
                if adress == rand_adress:
                    #ALL: we need to change, we can get the same twice
                    rand_adress = self._random_adress()

                self._propagate(rand_adress)

            self._infected_pop.extend(self._tmp_adopted)
            #print "There are %i adopters" % len(self._infected_pop)
            self.result[:,:,self.iteration] = np.sum(self._pop_array,
                                                axis=1).reshape(self.M,self.N)
            self.time_series.append(len(self._tmp_adopted))
            self.iteration += 1
            self._tmp_adopted = []
            return self.random_diffusion()

    def mixed_diffusion(self,proportion=0.5):
        """ Mixe of the two types of diffusions.

            In each iteration, the algorithm randomly select diffused points in the space according 
		to it proportion.
            param proportion: float Proportion of adopters that makes the spatial diffusion.
        """

        if proportion < 0 or proportion > 1:
            raise ValueError("The proportion needs to be between 0 and 1.")

        # If we have results, we need to clean and initialize
        if self._clean:
            self._clean_adopters()

        if self.iteration == (self.max_iter or
                              np.sum(self._pop_array) >= self.M*self.N*self._pob):
            #self.space = np.sum(s._pop_array,axis=1).reshape(s.M,s.N)
            print("I finish")
            print("There are %i adopters in a total of %i settlers" \
                    % (np.sum(self._pop_array),self.M*self.N*self._pob))
            print("the total of realized iterations is %i" % self.iteration)
            self.iteration = 0
            self._clean = True
            return None
        else:
            for adress in self._infected_pop:
                rnd = uniform(0,1)
                if rnd <= proportion:
                    propagated_adress = self._get_propagation_adress(adress)
                    self._propagate(propagated_adress)
                else:
                    rand_adress = self._random_adress()
                    if adress == rand_adress:
                        #ALL: we need to change, we can get the same twice
                        rand_adress = self._random_adress()

                    self._propagate(rand_adress)

            self._infected_pop.extend(self._tmp_adopted)
            #print "There are %i adopters" % len(self._infected_pop)
            self.result[:,:,self.iteration] = np.sum(self._pop_array,
                                                axis=1).reshape(self.M,self.N)
            self.time_series.append(len(self._tmp_adopted))
            self.iteration += 1
            self._tmp_adopted = []
            return self.mixed_diffusion(proportion)

class AdvancedDiffusion(Diffusion):
    '''Spatial diffusion model based on Hägerstrand, with heterogeneous space.

    1.- Isotropic space
    2.- Single initial diffuser
    3.- .... other assumptions ...

    : param N: int Number of rows and columns in simulation space
    : param mif_size: int Size of the matrix (square) of the MIF (must be non)
    : pop param: int maximum population in each cell
    : param density: int Number of initial population nuclei.
    : param amplitude: float Amplitude of the Gaussian filter to blur the population.
    : param initial_diff: [(int, int)] List of diffuser coordinates
                                     initials
    : param p0: float Probability of self-diffusion
    : param max_iter: int Maximum number of iterations

    : attribute space: np.array (N, N, dtype = np.int8) The available space
    : attribute _pop_array: np.array (N * N, pob, dtype = np.bool) array of settlers
                           in each cell
    : attribute _infected_pop: list (space_idx, int) List of the indexes of the
                                adopting cells. The first entry is the
                                flattened index of the cell in the space array and
                                the second is the number of the villager in
                                pop_array. That is, the list of addresses
                                of each infected settler.
    : attribute results: np.array ((N, N, max_iter)) Saves the results of each
                        iteration.
    : attribute time_series: list int Propagations for each iteration
    : attribute _clean: bool Indicates if we have saved results.

    '''

    def __init__(self,N=100,mif_size=5,pob=20,initial_diff=[(50,50)],
                p0=0.3, max_iter=25,densidad=20,amplitud=4.0):
        super(AdvancedDiffusion,self).__init__(mif_size,pob,initial_diff, p0,
                                                max_iter)
        self.N = N
        self.densidad = densidad
        self.amplitud = amplitud
        self.space = np.zeros((self.N,self.N),dtype=np.int8)
        points = self.N * np.random.random((2, self.densidad ** 2))
        self.space[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
        self.space = filters.gaussian_filter(self.space, sigma= self.N / (self.amplitud * self.densidad))
        #reescalamos al valor de la pob máxima y convertimos a entero:
        self.space *= self._pob / self.space.max()
        self.space = self.space.astype(np.int8)
        self._pop_array = np.zeros((len(np.ravel(self.space)),self._pob),
                                    dtype=np.bool)
        self.result = np.zeros((self.N,self.N,max_iter),dtype=np.int8)
        for c in initial_diff:
            if c[0] > self.N or c[1] > self.N:
                raise ValueError("The coordinates of initial diffusers aren't in the space")
            #We modify orignial settlers
            index = self._space2pop_index(c)
            self._pop_array[index][0] = True
            self._infected_pop.append((index,0))

        if self.mif_size%2 == 0:
            raise ValueError("The size of MIF needs to by non (square matrix")
        else:
            self._mif = self.initialize_mif(self.mif_size)

    def _space2pop_index(self,index):
        """Transform the spatial index of the index of pop_array.
        :param index (int,int) index to transform
        """
        return np.ravel_multi_index(index,dims=(self.N,self.N))

    def _pop2space_index(self,index):
        """Return the tuple (i,j) correpondent to the flattened index."""
        return np.unravel_index(index,dims=(self.N,self.N))

    def _mif2delta(self,index):
        """Returns one tuple with the increases for the propagated data."""
        return super(AdvancedDiffusion,self)._mif2delta(index)

    def _select_from_mif(self):
        """Returns one direction (pob_adress) from the MIF."""
        return super(AdvancedDiffusion,self)._select_from_mif()

    def _random_adress(self):
        """Returns one random address (pob_adress)."""
        i = randint(0,self.N - 1)
        j = randint(0,self.N - 1)
        pop_idx = self._space2pop_index((i,j))
        #space_idx = self._pop2space_index(i*j)
        return (pop_idx,randint(0,self.space[i,j] - 1))

    def _get_propagation_adress(self,adress):
        """Returns a propagated address from MIF (pop_adress)."""

        #print "Propagated to: " + str(adress)
        delta = self._select_from_mif()
        delta = (delta[0] - self.mif_size/2,delta[1] - self.mif_size/2)
        space_adress = self._pop2space_index(adress[0])
        prop_space_adress = (space_adress[0] + delta[0],
                              space_adress[1] + delta[1])
        try:
            habitant = randint(0,self.space[prop_space_adress[0],prop_space_adress[1]])
            return (self._space2pop_index(prop_space_adress),habitant)
        except ValueError as e:
            return self._get_propagation_adress(adress)

    def _propagate(self,pob_adress):
        """Propagates to the no adopter settler

        :param pob_adress: (int,int) direction of settler to propagate.
                            The first input is the flattened index in space and the second 
			    is the number of settler in each cell
        """

        # Review if it's no-adopter
        if self._pop_array[pob_adress[0]][pob_adress[1]] == False:
            self._pop_array[pob_adress[0]][pob_adress[1]] = True
            self._tmp_adopted.append(pob_adress)
            #print "I infected to "  + str(pob_adress)

        else:
            pass

    def _clean_adopters(self):
        """Cleans and initialized before simulation."""
        return super(AdvancedDiffusion,self)._clean_adopters()

    def spatial_diffusion(self):
        """Hagerstrand diffusion."""

        # If we have previous results, we clean and initialize
        if self._clean:
            self._clean_adopters()

        if self.iteration == (self.max_iter or
                              np.sum(self._pop_array) >= self.M*self.N*self._pob):
            print("I finish")
            print("There are %i adopters from a total of %i settlers" \
                    % (np.sum(self._pop_array),self.N * self.N * self._pob))
            print("The total of iterations is %i" % self.iteration)
            self.iteration = 0
            self._clean = True
            return None
        else:
            for adress in self._infected_pop:
                propagated_adress = self._get_propagation_adress(adress)
                self._propagate(propagated_adress)

            self._infected_pop.extend(self._tmp_adopted)
            #print "There are %i adopters" % len(self._infected_pop)
            self.result[:,:,self.iteration] = np.sum(self._pop_array,
                                                axis=1).reshape(self.N,self.N)
            self.time_series.append(len(self._tmp_adopted))
            self.iteration += 1
            self._tmp_adopted = []
            return self.spatial_diffusion()

    def random_diffusion(self):
        """Propagates randomly throught space."""

        #If we have previous results, we clean and initialize
        if self._clean:
            self._clean_adopters()

        if self.iteration == (self.max_iter or
                              np.sum(self._pop_array) >= self.N*self.N*self._pob):
            #self.space = np.sum(s._pop_array,axis=1).reshape(s.M,s.N)
            print("I finish")
            print("There are %i adopters from a total of %i settlers" \
                    % (np.sum(self._pop_array),self.N*self.N*self._pob))
            print("The total of iterations is %i" % self.iteration)
            self.iteration = 0
            self._clean = True
            return None
        else:
            for adress in self._infected_pop:
                rand_adress = self._random_adress()
                if adress == rand_adress:
                    #ALL: we need to change because it can be the same
                    rand_adress = self._random_adress()

                self._propagate(rand_adress)

            self._infected_pop.extend(self._tmp_adopted)
            #print "There are %i adopters" % len(self._infected_pop)
            self.result[:,:,self.iteration] = np.sum(self._pop_array,
                                                axis=1).reshape(self.N,self.N)
            self.time_series.append(len(self._tmp_adopted))
            self.iteration += 1
            self._tmp_adopted = []
            return self.random_diffusion()
