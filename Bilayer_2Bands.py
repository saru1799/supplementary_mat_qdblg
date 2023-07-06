import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import numbers
import h5py
from matplotlib import pyplot as plt
import matplotlib as mpl
from os.path import exists
from os import remove
import warnings
from scipy.interpolate import interp1d
from scipy.integrate import dblquad
from scipy.special import ellipk

class Bilayer_2Bands:
    """
    Class containing the necessary to solve the eigenvalue/eigenvector
    problem  M v = wei E v for bilayer graphene (BLG) in a central electrostatic potential.
    The matrices are obtained used the finite element method (FEM) and
    all is built with sparse matrices.
    """
    def __init__(self, alpha, v_r, delta_r, r_max, N, gap_size = None):
        """
        Definition of the phisical parameters (alpha, v_r, delta_r and gap_size) and the parameters for the FEM
        (r_max and N) needen to solve the problem.
        Args:
            alpha : Float. Value for hbar²/(2m_eff)
            v_r : callable or constant. Average electric potential as a function of rho.
            delta_r : callable or constant. Interlayer asymmetry as a function
                    of rho.
            r_max : Float. Maximum rho (radial coordinate) to consider. 
                    we will take rho in [h, r_max-h]
            N : Integer. # of divisions in space
            gap_size: specifies the limitation of energy that define the gap. The program then takes it in the form:
                      max and min limitand values for what we consider the gap [gap_max, gap_min].
                      Default value: None. If it is None, e take the value of v_p and v_m for rho_max. If it is just a number, 
                      we assume that the bands are symmetric and set the limits as [|value|, -|value|]. If it is an array with
                      two values, make sure the biggest value is first. It can be then specified with the function 
                      self.set_gapsize().
        """
       
        # Save the parameters of the problem
        self.alpha = alpha
        self.N = N
        self.rhos = np.linspace(0, r_max, N+2)
        self.h = self.rhos[1]-self.rhos[0]
        
        #  Obtain the potential and the interayer assumetry for the different radial coordinates (rhos)
        if isinstance(v_r, numbers.Number):
            vs = np.full(N+2, v_r)
        elif callable(v_r):
            vs = v_r(self.rhos)
        else:
            raise ValueError('Please insert a valid potential: number or callable.')
        self.sigma = np.mean(vs)   # Mean value of the potential ~ central energy

        if isinstance(delta_r, numbers.Number):
            deltas = np.full(N+2, delta_r)
        elif callable(delta_r):
            deltas = delta_r(self.rhos)
        else:
            raise ValueError('Please insert a valid delta: number or callable.')
        self.vp = vs+np.abs(deltas)/2
        self.vm = vs-np.abs(deltas)/2
        
        #  Obtain the matrices for the FEM
        
        self.M_k1 = self._M_k1(N)
        self.M_k1_lp1 = self._M_k1_lp1(N)
        self.M_k1_lm1 = self._M_k1_lm1(N)

        self.M_k2 = self._M_k2(N)
        self.M_k2_lp1 = self._M_k2_lp1(N)
        self.M_k2_lm1 = self._M_k2_lm1(N)

        self.M_k3 = self._M_k3(N)

        self.M_v = self._M_v(self.vp, self.vm)
        self.M_v_lp1 = self._M_v_lp1(self.vp, self.vm)
        self.M_v_lm1 = self._M_v_lm1(self.vp, self.vm)
        
        self.W = self._wei(N)*self.h**2
        self.W_lp1 = self._wei_lp1(N)*self.h**2
        self.W_lm1 = self._wei_lm1(N)*self.h**2
        self.filename = None  # initialize the variable
        self.gap_size = self._get_gap(gap_size)


    def set_filename(self, filename):
        self.filename = filename


    def set_gapsize(self, gap_size):
        self.gap_size = self._get_gap(gap_size)


    def solve(self, l, k):
        """ 
        Solves the eigenvalue/eigenvector relation for an specified l for the given problem, which is the one saved in self
        (Bilayer_2Bands object).
        Args:
            l : Integer. Quantum number for angular momentum
            k : Integer. How many eigenvalues/eigenvectors we return.
        Returns:
            eigvals: Numpy array of lenght k. Contains the eigenvalues sorving the eigenvalue/eigenvector
                for a given l of the specified problem. The chosen eigenvalues are the first k  around
                the mean value of the electric potential.
            u_p: Numpy array of shape (N, k). Contains top layer eigenvectors solving the eigenvalue/eigenvector
                for a given l of the specified problem.  Each of the k eigenvectors is a column of the output matrix.
            u_m: Numpy array of shape (N, k). Contains bottom layer eigenvectors solving the eigenvalue/eigenvector
                for a given l of the specified problem.  Each of the k eigenvectors is a column of the output matrix.
        """
        # For a given l, we solve (α[Mk1 + 2l*Mk2+(l²-1)Mk3] +h²Mv) u= W E u
        if l==1:
            self.M = self.alpha*(self.M_k1_lp1 +2*l*self.M_k2_lp1) + self.h**2*self.M_v_lp1 
            eigvals, eigvec_fem = eigsh(self.M, 
                                        k=k, 
                                        M=self.W_lp1, 
                                        sigma=self.sigma)
            u_p = eigvec_fem[1::2,:]
            u_m = eigvec_fem[2::2,:]
            u_p = np.concatenate([np.expand_dims(eigvec_fem[0,:], axis = 0),u_p,np.zeros([1,k])], axis =0)
            u_m = np.concatenate([np.zeros([1,k]),u_m,np.zeros([1,k])], axis =0)
            return eigvals, u_p, u_m
        elif l==-1:
            self.M = self.alpha*(self.M_k1_lm1+2*l*self.M_k2_lm1)+self.h**2*self.M_v_lm1 #l was missing here
            eigvals, eigvec_fem = eigsh(self.M, 
                                        k=k, 
                                        M=self.W_lm1, 
                                        sigma=self.sigma)
            u_p = eigvec_fem[1::2]
            u_m = eigvec_fem[2::2]
            u_p = np.concatenate([np.zeros([1,k]),u_p,np.zeros([1,k])], axis =0)
            u_m = np.concatenate([np.expand_dims(eigvec_fem[0,:], axis = 0),u_m,np.zeros([1,k])], axis =0)

            return eigvals, u_p, u_m
        else:
            self.M = self.alpha*(self.M_k1+2*l*self.M_k2+(l**2-1)*self.M_k3)+self.h**2*self.M_v #l was missing here
            eigvals, eigvec_fem = eigsh(self.M, k=k, M=self.W, sigma=self.sigma)
            u_p = eigvec_fem[0::2]
            u_m = eigvec_fem[1::2]
            u_p = np.insert(u_p, [0, self.N], [[0], [0]], axis=0)
            u_m = np.insert(u_m, [0, self.N], [[0], [0]], axis=0)
            return eigvals, u_p, u_m
    
    
    def save(self, filename, ls, k, rewrite = False):
        """
        Solves the Schördinger equation for bilayer graphene for the specified problem in the object and saves 
        eigenenergies/eigenfunctions in a hdf5 file.
        Creates a filename.hdf5 file if this does not exists. If the filaname already exists and is linked to the
        object, interprets as if it has already been saved and does not save it again.
        """
        if self.filename is None and exists(f'{filename}.hdf5'):
            if rewrite:
                remove(f'{filename}.hdf5')
            else:
                raise Exception(f'Please remove the existing file named {filename}.hdf5')
        if self.filename == filename and exists(f'{self.filename}.hdf5'):
            print('Already saved')
        
        else:
            with h5py.File(f'{filename}.hdf5', "a") as f:
                l_len = len(ls)

                #  We save the things we already have
                f.create_dataset('ls', data=ls)  # different considered angular numbers
                f.create_dataset('k', data=[k])
                f.create_dataset('rhos', data=self.rhos)
                f.create_dataset('v_p', data=self.vp)  # Possitive potential v(rho)+Delta/2
                f.create_dataset('v_m', data=self.vm)  # Negative potential v(rho)-Delta/2
                f.create_dataset('alpha', data=[self.alpha])# alpha = hbar²/(2m_eff)


                #  Create the datasets to save all the energies and wfs
                energies = f.create_dataset('energies', (l_len, k), dtype='f') # energies
                u_pdat = f.create_dataset('u_p', (l_len, self.N+2, k), dtype='f')  # top layer eigenvector
                u_mdat = f.create_dataset('u_m', (l_len, self.N+2, k), dtype='f')  # bottom layer eigenvector
                
                # Solve the problem and store it in the dataset
                for i, l in enumerate(ls):
                    eigvals, u_p, u_m = self.solve(l, k)
                    energies[i, :] = eigvals
                    u_pdat[i, :, :] = u_p
                    u_mdat[i, :, :] = u_m
                self.filename = filename


    def plot(self, plot_type, rmax = None, filename = None, create = False, variables = None):
        """
        Function set the necessary specifications to plot the energy levels and wavefunctions as a function of rho of BLG.
        
        The function permits several ways to plot. If you have already saved the hdf5 file using the current object,
        it will plot self.filename and the filename should not be specified. If this calculations were done before,
        we recomend to use the self.set_filename(filename) function to store the correct filename that was saven before.
        Anyway, you could specify the filename and the function can plot the data there.
        Also, you can solve the problem, save the data in a hdf5 file by setting create to True and specifying the variables.
        
        Args:
            plot_type: what to plot. Options: 'all', 'correct', 'correct_absolute', 'inside'.
                - all: plot all the obtained energies and wavefunctions.
                - correct: plot the obtained energies and wavefunctions ommiting the ones with numerical errors.
                - correct_absolute: plot the obtained energies and absolute value of the wavefunctions ommiting
                the ones with numerical errors.
                - inside: plot the correct energies and wavefunctions that lie inside the gap.
            rmax: number. Maximum radial coordinate to take into account.
            filename: hdf5 file containing the data to plot. If None, we plot self.filename. If that does not exist, we can create
                some hdf5 file and plot that. For that, see the create and variables args.
            create: boolean. If true, and the filename does not exist, solves the Schrödinger equation, saves it, and solves it.
                the variables for the solvation must be specified. 
            variables: 2 dimensional array containing [lmax, k]. Maximum quantum number l (lmax) and number of energies (k)
                to take into account when creating a new filename to plot. Create must be set to True to do this.
                If filename is None, the defect filename that is set is 'l_{lmax}_k_{k}.hdf5'.
        Output:
            produces a plot for every angular quantum number l with the energies, the top layer wavefunctions as a function
            of the radial coordinate and the bottom layer wavefunctions as a function of rho. Which energies/wavefunctions
            to plot is especified by the plot_type.
        """
        if plot_type not in ['all', 'correct', 'correct_absolute', 'inside']:
            warnings.warn(f'Caution! The specified type is not in the implemented ones: \
                          \'all\', \'correct\', \'correct_absolute\', \'inside\'\
                          We set the default, \'correct\'')
            plot_type = 'correct'
        if filename is None:
            if exists(f'{self.filename}.hdf5'):
                self._plot(self.filename, plot_type, rmax)
            else:
                if create and variables is not None:
                    lmax, k = variables
                    filename = f'l_{lmax}_k_{k}'  # defect filename
                    self.save(filename, np.arange(0, lmax+1, 1), k)
                    self._plot(self.filename, plot_type, rmax)
                elif create and variables is None:
                    raise Exception('Please specify the arg valiables = [lmax, k] in order to create the file')
                else:
                    raise Exception('Write create = True and specify  the arg valiables = [lmax, k] / choose a existing filename/ save the data with the chosen ls and k')
        else:
            if exists(f'{filename}.hdf5'):
                warnings.warn(f'Caution! The plotted dataset {filename} does not necesarily correspond with the object')
                self._plot(filename, plot_type, rmax)
            else:
                if create and variables is not None:
                    lmax, k = variables
                    self.save(filename, np.arange(0, lmax+1, 1), k)
                    self._plot(self.filename, plot_type, rmax)
                elif create and variables is None:
                    raise Exception('Please specify the arg valiables = [lmax, k] in order to create the file')
                else:
                    raise Exception('Write create = True and specify  the arg valiables = [lmax, k] / choose a \
                                    existing filename/ save the data with the chosen ls and k')


    def get_inside_and_correct(self, energies_l, u_p_l, u_m_l, only_energies = False):
        """
        Function to take the energies and wavefunctions that have no numerical errors and that lie inside the gap.
        Args:
            energies_l: array of shape (k) with the energies for a iven angular quantum number l.
            u_p_l, u_m_l: arrays of shape (N, k). Top, bottom layer eigenvectors for a given angular quantum number l.
        Returns:
            energies and eigenvectors that have no numerical errors and that lie inside the gap.    
        """
        take = np.bitwise_and(np.invert(np.bitwise_or(np.isclose(energies_l, self.vp[0]), np.isclose(energies_l,self.vm[0]))),
                             np.bitwise_and(energies_l<self.gap_size[0], energies_l>self.gap_size[1]))
        if only_energies:
            return energies_l[take]
        else:
            return energies_l[take], u_p_l[:, take], u_m_l[:, take]


    def get_correct(self, energies_l, u_p_l, u_m_l):
        """
        Function to take the energies and wavefunctions that have no numerical errors.
        Args:
            energies_l: array of shape (k) with the energies for a given angular quantum number l.
            u_p_l, u_m_l: arrays of shape (N, k). Top, bottom layer eigenvectors for a given angular quantum number l.
        """
        take = np.invert(np.bitwise_or(np.isclose(energies_l, self.vp[0]), np.isclose(energies_l, self.vm[0])))
        return energies_l[take], u_p_l[:, take], u_m_l[:, take]


    def _plot(self, filename, plot_type, rmax):
        """
        Funtion to plot the energies and wavefunction in the filename file.
        Args:
            plot_type: what to plot. Options: 'all', 'correct', 'correct_absolute', 'inside'.
                - all: plot all the obtained energies and wavefunctions.
                - correct: plot the obtained energies and wavefunctions ommiting the ones with numerical errors.
                - correct_absolute: plot the obtained energies and absolute value of the wavefunctions ommiting
                the ones with numerical errors.
                - inside: plot the correct energies and wavefunctions that lie inside the gap.
            rmax: number. Maximum radial coordinate to take into account.
            filename: hdf5 file containing the data to plot. If None, we plot self.filename. If that does not exist, we can create
                some hdf5 file and plot that. For that, see the create and variables args.
        Output:
            produces a plot for every angular quantum number l with the energies, the top layer wavefunctions as a function
            of the radial coordinate and the bottom layer wavefunctions as a function of rho. Which energies/wavefunctions
            to plot is especified by the plot_type.
        """
        try:
            with h5py.File(f'{filename}.hdf5', 'r') as f:
                ls = f['ls']
                rhos = f['rhos'][:]
                k = f['k'][0]
                cmap1 = mpl.colormaps['spring']
                cmap2 = mpl.colormaps['winter']
                energies = f['energies']
                u_p = f['u_p']
                u_m = f['u_m']
                
                v_thres = f['v_p'][0]
                
                if rmax is not None:
                    irmax = np.argmin(np.abs(rhos-rmax))
                else:
                    irmax = len(rhos)
                
                for ln, l in enumerate(ls):
                    energy_l = energies[ln, :]
                    u_p_l = u_p[ln, :, :]
                    u_m_l = u_m[ln, :, :]
                    if plot_type == 'inside':
                        energy_l, u_p_l, u_m_l = self.get_inside_and_correct(energy_l, u_p_l, u_m_l)
                    elif plot_type == 'correct' or plot_type == 'correct_absolute':
                        energy_l, u_p_l, u_m_l = self.get_correct(energy_l, u_p_l, u_m_l)
                        if plot_type == 'correct_absolute':
                            u_p_l = np.abs(u_p_l)
                            u_m_l = np.abs(u_m_l)
                    # else: its all so we sdo not care.
                    fig, axes = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 5, 5]}, figsize=(15, 5))
                    fig.suptitle(f'l={l}')
                    axes[2].sharey(axes[1])
                    for i, en in enumerate(energy_l):
                        if en > v_thres:
                            cmap = cmap1
                        else:
                            cmap = cmap2
                        axes[0].plot([0, 1], np.full(2, en), c=cmap(i))
                        axes[1].plot(rhos[:irmax], u_p_l[:irmax, i], c=cmap(i))
                        axes[2].plot(rhos[:irmax], u_m_l[:irmax, i], c=cmap(i), label=f'n={i}')
                    axes[0].set_title('Energies')
                    axes[1].set_title(fr'$u_{ {l - 1} }^+$')
                    axes[2].set_title(fr'$u_{ {l + 1}}^-$')
                    if k<21:
                        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
                    plt.show()
        except OSError:
            print('Please, use the self.save(filename, ls, k) statement to create the corresponding file to plot the results')
    
    def _get_gap(self, gap_size):
        """
        Funtion to obtain the gap_size in the standard form of [gap_max, gap_min].
        Args:
            gap_size: It can be:
                - None (defect option when the provided format is not valid): gets the gap from the potentials, taking the v+ and v- in rho_max.
                - Array of dimension 2: with the values [gap_max, gap_min]. It ensures that the values are in descending order.
                - Number: assumes symmetric gap and sets the gap as [|gap_size|, -|gap_size|].
        """
        if gap_size is None:
            gap_size = np.array([self.vp[-1], self.vm[-1]])
        elif hasattr(gap_size, "__len__"):
            if len(gap_size) == 2:
                gap_size = np.sort(gap_size)[::-1]
            else:
                warnings.warn(f'The gap_size format was not correct. Taking default')
                gap_size = np.array([self.vp[-1], self.vm])
        else:
            gap_size = np.array([np.abs(gap_size), -np.abs(gap_size)])  
        return gap_size
    
    
    def print_inside_gap(self, filename=None):
        """
        Function to get and print how many (correct) values we hae inside the gap for eah angular quantum number l.
        Args:
            filename: hdf5 file with the calculations. defect value: None = gets self.filename if it exists
        """
        filename = self.comprobe_filename(filename)
        with h5py.File(f'{filename}.hdf5', 'r') as f:
            ls = f['ls']
            energies = f['energies']
            for ln, l in enumerate(ls):
                energy_l = energies[ln]
                energy_l = self.get_inside_and_correct(energy_l, None, None, only_energies=True)  
                print(f'l={l}, states inside the gap ={len(energy_l)}')
                
    def comprobe_filename(self, filename):
        """
        Function to see if the dataset is created and properly stored.
        """
        if filename is None:
            if exists(f'{self.filename}.hdf5'):
                return(self.filename)
            else:
                print('Please, use the self.save(filename, ls, k) statement to create the corresponding file to plot the results')
        else:
            if exists(f'{filename}.hdf5'):
                warnings.warn(f'Caution! The plotted dataset {filename} does not necesarily correspond with the object')
                return filename

    ##################### STATIC METHODS #############################
    
    ######################### FEM MATRICES ###########################
    @staticmethod
    def _M_k1(N):
        ns = np.arange(1, N+1, 1)
        ns_ph = ns+0.5
        line_m1p1 = np.dstack((-2*ns, ns_ph)).flatten()
        line_m3p3 = np.zeros(2*N)
        line_m3p3[::2] = ns_ph
        M = diags([line_m3p3, line_m1p1, line_m1p1, line_m3p3], [-3, -1, 1, 3],
                 shape=(2*N, 2*N))
        return M
    
    @staticmethod
    def _M_k1_lp1(N):
        'substitutes M_k1 if l = +1'
        ns = np.arange(1, N+1, 1)
        ns_ph = ns+0.5
        line_m1p1 = np.dstack((-2*ns, ns_ph)).flatten()
        #add zero in front
        line_m1p1 = np.concatenate(([0.],line_m1p1))

        line_m3p3 = np.zeros(2*N)
        line_m3p3[::2] = ns_ph
        #add zero in front
        line_m3p3 = np.concatenate(([0.],line_m3p3))
        #added +-2 diagoal 
        line_m2p2 = np.zeros(2*N-1)
        line_m2p2[0] = 0.5 
        # added line +-2, modified shape 
        M = diags([line_m3p3, line_m2p2, line_m1p1, line_m1p1, line_m2p2, line_m3p3], [-3,-2, -1, 1,2, 3],
                 shape=(2*N+1, 2*N+1))
        return M
    
    @staticmethod
    def _M_k1_lm1(N):
        'substitutes M_k1 if l = -1'
        ns = np.arange(1, N+1, 1)
        ns_ph = ns+0.5
        line_m1p1 = np.dstack((-2*ns, ns_ph)).flatten()
        #add 0.5 in front
        line_m1p1 = np.concatenate(([0.5],line_m1p1))

        line_m3p3 = np.zeros(2*N)
        line_m3p3[::2] = ns_ph
        #add zero in front
        line_m3p3 = np.concatenate(([0.],line_m3p3))
   
        # modified shape 
        M = diags([line_m3p3, line_m1p1, line_m1p1, line_m3p3], [-3,-1, 1, 3],
                 shape=(2*N+1, 2*N+1))
        return M
    
    @staticmethod
    def _M_k2(N):
        line_m1p1 = np.zeros(2*N)
        line_m1p1[1::2] = np.full(N, -0.5)
        line_m3p3 = np.zeros(2*N)
        line_m3p3[::2] = np.full(N, 0.5)
        M = diags([line_m3p3, line_m1p1, line_m1p1, line_m3p3], [-3, -1, 1, 3],
                 shape=(2*N, 2*N))
        return M
    
    @staticmethod
    def _M_k2_lp1(N):
        'substitutes M_k2 if l = +1'
        line_m1p1 = np.zeros(2*N)
        line_m1p1[1::2] = np.full(N, -0.5)
        #add zero in front
        line_m1p1 = np.concatenate(([0.],line_m1p1))

        line_m3p3 = np.zeros(2*N)
        line_m3p3[::2] = np.full(N, 0.5)
        #add zero in front
        line_m3p3 = np.concatenate(([0.],line_m3p3))

        #added +-2 diagoal 
        line_m2p2 = np.zeros(2*N-1)
        line_m2p2[0] = 0.5 
        # added line +-2, modified shape 

        M = diags([line_m3p3, line_m2p2, line_m1p1, line_m1p1, line_m2p2, line_m3p3], [-3,-2, -1, 1,2, 3],
                 shape=(2*N+1, 2*N+1))
        return M
    
    @staticmethod
    def _M_k2_lm1(N):
        'substitutes M_k2 if l = -1'
        line_m1p1 = np.zeros(2*N)
        line_m1p1[1::2] = np.full(N, -0.5)
        #add -0.5 in front
        line_m1p1 = np.concatenate(([-0.5],line_m1p1))

        line_m3p3 = np.zeros(2*N)
        line_m3p3[::2] = np.full(N, 0.5)
        #add zero in front
        line_m3p3 = np.concatenate(([0.],line_m3p3))
        # modified shape
        M = diags([line_m3p3, line_m1p1, line_m1p1, line_m3p3], [-3, -1, 1, 3],
                shape=(2*N+1, 2*N+1))
        return M

    @staticmethod
    def _M_k3(N):
        ns = np.arange(1, N+1, 1)
        #fn = -2*ns-(ns-1)**2*log_no_errors(ns)+(ns+1)**2*np.log(1+1/ns) # diocane occhio ai segni
        #gn = 0.5+ns-ns*(ns+1)*np.log(1+1/ns)
        fn = -2*ns-(ns-1)**2*Bilayer_2Bands.log1p_noinf(-1/ns)+(ns+1)**2*np.log1p(1/ns)
        gn = 0.5+ns-ns*(ns+1)*np.log1p(1/ns)

        line_m1p1 = np.dstack((fn, gn)).flatten()
        line_m3p3 = np.zeros(2*N)
        line_m3p3[::2] = gn
        M = diags([line_m3p3, line_m1p1, line_m1p1, line_m3p3], [-3, -1, 1, 3],
                 shape=(2*N, 2*N))
        return M
    
    @staticmethod
    def log1p_noinf(x):
        return np.piecewise(x+0.,[x>-1,x<=-1], [lambda y: np.log1p(y), 0.])
    
    @staticmethod
    def _M_v(vp, vm):
        N = len(vp)-2
        vp_m = np.roll(vp, 1) # v^+_{n-1}
        vp_p = np.roll(vp, -1) # v^+_{n+1}
        vm_m = np.roll(vm, 1) # v^-_{n-1}
        vm_p = np.roll(vm, -1) # v^-_{n+1}

        ns = np.arange(0, N+2, 1)
        dv1 = ((5*ns-2)*vp_m + 30*ns*vp + (5*ns+2)*vp_p)/60
        dv2 = ((5*ns-2)*vm_m + 30*ns*vm + (5*ns+2)*vm_p)/60

        pv1 = ((5*ns+2)*vp + (5*ns+3)*vp_p)/60
        pv2 = ((5*ns+2)*vm + (5*ns+3)*vm_p)/60

        # we now do not take n=0, N+1
        line_0 = np.dstack((dv1, dv2)).flatten()[2:-2]
        line_m2p2 = np.dstack((pv1, pv2)).flatten()[2:-2]
        M = diags([line_m2p2, line_0, line_m2p2], [-2, 0, 2], 
                 shape = (2*N, 2*N))
        return M
    
    @staticmethod
    def _M_v_lp1(vp, vm):
        'substitutes M_v if l = +1'
        N = len(vp)-2
        vp_m = np.roll(vp, 1) # v^+_{n-1}
        vp_p = np.roll(vp, -1) # v^+_{n+1}
        vm_m = np.roll(vm, 1) # v^-_{n-1}
        vm_p = np.roll(vm, -1) # v^-_{n+1}

        ns = np.arange(0, N+2, 1)
        dv1 = ((5*ns-2)*vp_m + 30*ns*vp + (5*ns+2)*vp_p)/60
        dv2 = ((5*ns-2)*vm_m + 30*ns*vm + (5*ns+2)*vm_p)/60

        pv1 = ((5*ns+2)*vp + (5*ns+3)*vp_p)/60
        pv2 = ((5*ns+2)*vm + (5*ns+3)*vm_p)/60

        # we now do not take n=0, N+1
        line_0 = np.dstack((dv1, dv2)).flatten()[2:-2]
        # add first element 
        first_element = vp[0]/20. + vp[1]/30.
        line_0 = np.concatenate(([first_element], line_0))
        # add 0 
        line_m2p2 = np.dstack((pv1, pv2)).flatten()[2:-2]
        line_m2p2 = np.concatenate(([0.], line_m2p2))
        # add first diagonal 
        line_m1p1 = np.zeros(2*N)
        line_m1p1[0] = vp[0]/30. + vp[1]/20.
        # added first diagonal, modified shape
        M = diags([line_m2p2, line_m1p1,line_0, line_m1p1, line_m2p2], [-2,-1, 0, 1,2], 
                 shape = (2*N+1, 2*N+1))
        return M
    
    @staticmethod
    def _M_v_lm1(vp, vm):
        'substitutes M_v if l = -1'
        N = len(vp)-2
        vp_m = np.roll(vp, 1) # v^+_{n-1}
        vp_p = np.roll(vp, -1) # v^+_{n+1}
        vm_m = np.roll(vm, 1) # v^-_{n-1}
        vm_p = np.roll(vm, -1) # v^-_{n+1}

        ns = np.arange(0, N+2, 1)
        dv1 = ((5*ns-2)*vp_m + 30*ns*vp + (5*ns+2)*vp_p)/60
        dv2 = ((5*ns-2)*vm_m + 30*ns*vm + (5*ns+2)*vm_p)/60

        pv1 = ((5*ns+2)*vp + (5*ns+3)*vp_p)/60
        pv2 = ((5*ns+2)*vm + (5*ns+3)*vm_p)/60

        # we now do not take n=0, N+1
        line_0 = np.dstack((dv1, dv2)).flatten()[2:-2]
        # add first value
        line_0 = np.concatenate(([vm[0]/20. + vm[1]/30.], line_0))

        line_m2p2 = np.dstack((pv1, pv2)).flatten()[2:-2]
        # add first value
        line_m2p2 = np.concatenate(([vm[0]/30. + vm[1]/20.], line_m2p2))

        # modified shape
        M = diags([line_m2p2, line_0, line_m2p2], [-2, 0, 2], 
                 shape = (2*N+1, 2*N+1))
        return M

    @staticmethod
    def _wei(N):
        ns = np.arange(1, N+1, 1)
        ns2 = np.dstack((ns, ns)).flatten()
        line_0 = 2*ns2/3

        line_m2p2 = (2*ns2+1)/12
        M = diags([line_m2p2, line_0, line_m2p2], [-2, 0, 2], 
                 shape = (2*N, 2*N))
        return M
    
    @staticmethod
    def _wei_lp1(N):
        'substitutes wei if l=+1'
        ns = np.arange(1, N+1, 1)
        ns2 = np.dstack((ns, ns)).flatten()
        line_0 = 2*ns2/3
        # add first element
        line_0 = np.concatenate(([1/12.] , line_0))

        line_m2p2 = (2*ns2+1)/12
        # add 0
        line_m2p2 = np.concatenate(([0], line_m2p2))
        # add first diagonal 
        line_m1p1 = np.zeros(2*N)
        line_m1p1[0] = 1/12.
        #added diagonal, modified shape
        M = diags([line_m2p2, line_m1p1, line_0, line_m1p1, line_m2p2], [-2,-1, 0,1, 2], 
                 shape = (2*N+1, 2*N+1))
        return M
    
    @staticmethod
    def _wei_lm1(N):
        'substitutes wei if l=-1'
        ns = np.arange(1, N+1, 1)
        ns2 = np.dstack((ns, ns)).flatten()
        line_0 = 2*ns2/3
        #added value
        line_0 = np.concatenate(([1/12.] , line_0))

        line_m2p2 = (2*ns2+1)/12
        # added value 
        line_m2p2 = np.concatenate(([1/12.], line_m2p2))
        # modified shape 
        M = diags([line_m2p2, line_0, line_m2p2], [-2, 0, 2], 
                 shape = (2*N+1, 2*N+1))
        return M

    
######################### HUBBARD PARAMETERS ################################

    def get_coupling_and_tunneling(self, R, l1, l2, n1, n2, filename=None, prec=1e-5):
        """
        Function to get the coupling and the tunneling integrals between a wavefunction [l1, n1] centered at the origin
        and another [l2, n2] centered at R.
        Args:
            R: 2 dimensional array. Polar coordinates [module, angle] of the relative distance
            between the spinors. 
            l1: integer. Quantum number for angular momentum for the wavefunction centered at 0.
            l2: integer. Quantum number for angular momentum for the wavefunction centered at R.
            n1: integer (>0). Quantum number determining the energy level for the wavefunction centered at 0.
            n2: integer (>0). Quantum number determining the energy level for the wavefunction centered at R.
        KArgs:
            filename: hdf5 file with the calculations. defect value: None = gets self.filename if it exists.
            prec: float. Minimum recision for the integrals. Default 1e-5.
        Returns:
            coupling , tunneling: values for the coupling and the tunneling  between a wavefunction [l1, n1] 
            centered at the origin and another [l2, n2] centered at R.
        """
        filename = self.comprobe_filename(filename)
        with h5py.File(f'{filename}.hdf5', 'r') as f:
            ls = f['ls'][:]
            ln1 = np.where(ls==l1)[0][0] 
            ln2 = np.where(ls==l2)[0][0]
            if ln1.size==0 or ln2.size==0:
                raise Exception('Some of the asked ls have not been calculated in the given dataset')
            energies_all = f['energies']
            u_p_all = f['u_p']
            u_m_all = f['u_m']

            v_p_interp = interp1d(self.rhos, self.vp, bounds_error=False, fill_value=self.vp[-1])
            v_m_interp  = interp1d(self.rhos, self.vp, bounds_error=False, fill_value=self.vm[-1])

            _, u_p1, u_m1 = self.get_inside_and_correct(energies_all[ln1], u_p_all[ln1], u_m_all[ln1])
            _, u_p2, u_m2 = self.get_inside_and_correct(energies_all[ln2], u_p_all[ln2], u_m_all[ln2])
            psi_1 = Interpolate_Spinor(self.rhos, u_p1[:, n1], u_m1[:, n1], l1)
            psi_2 = Interpolate_Spinor(self.rhos, u_p2[:, n2], u_m2[:, n2], l2)

            return self.integrate_fg(psi_1, psi_2, R, self.rhos[-1], prec),\
                -self.integrate_fg(psi_1, lambda r, theta:  np.array([[v_p_interp(r)- self.vp[-1], 0], [0, v_m_interp(r)-self.vm[-1]]])
                             @ psi_2(r, theta), R, self.rhos[-1], prec)
        
    def get_coupling_and_tunneling_distance(self, Rs, l1, l2, n1, n2, filename=None, prec=1e-5):
        """
        Function to get the coupling and the tunneling integrals between a wavefunction [l1, n1] centered at the origin
        and another [l2, n2] centered at a distance R. Thus, R is only the relative distance (theta_R=0).
        These R are in the array Rs.
        Args:
            Rs: distance (or array of distances) between the wavefunctions.
            l1: integer. Quantum number for angular momentum for the wavefunction centered at 0.
            l2: integer. Quantum number for angular momentum for the wavefunction centered at R.
            n1: integer (>0). Quantum number determining the energy level for the wavefunction centered at 0.
            n2: integer (>0). Quantum number determining the energy level for the wavefunction centered at R.
        KArgs:
            filename: hdf5 file with the calculations. defect value: None = gets self.filename if it exists.
            prec: float. Minimum recision for the integrals. Default 1e-5.
        Returns:
            coup_and_tunn: array of size (len(Rs), 2) with the values for the coupling (coup_and_tunn[:, 0])
            and the tunneling (coup_and_tunn[:, 1]) for each R in Rs  between a wavefunction [l1, n1] 
            centered at the origin and another [l2, n2] centered at  R.
        """
        filename = self.comprobe_filename(filename)
        with h5py.File(f'{filename}.hdf5', 'r') as f:
            ls = f['ls'][:]
            ln1 = np.where(ls==l1)[0][0]  # ln
            ln2 = np.where(ls==l2)[0][0]
            if ln1.size==0 or ln2.size==0:
                raise Exception('Some of the asked ls have not been calculated in the given dataset')
            energies_all = f['energies']
            u_p_all = f['u_p']
            u_m_all = f['u_m']

            v_p_interp = interp1d(self.rhos, self.vp, bounds_error=False, fill_value=self.vp[-1])
            v_m_interp  = interp1d(self.rhos, self.vm, bounds_error=False, fill_value=self.vm[-1])

            _, u_p1, u_m1 = self.get_inside_and_correct(energies_all[ln1], u_p_all[ln1], u_m_all[ln1])
            _, u_p2, u_m2 = self.get_inside_and_correct(energies_all[ln2], u_p_all[ln2], u_m_all[ln2])
            psi_1 = Interpolate_Spinor(self.rhos, u_p1[:, n1], u_m1[:, n1], l1)
            psi_2 = Interpolate_Spinor(self.rhos, u_p2[:, n2], u_m2[:, n2], l2)
            
            coup_and_tunn = np.zeros((len(Rs), 2))
            for i, R_mod in enumerate(Rs):
                coup_and_tunn[i, 0] = self.integrate_fg(psi_1, psi_2, np.array([R_mod, 0]), self.rhos[-1], prec)
                coup_and_tunn[i, 1] = -self.integrate_fg(psi_1, lambda r, theta:  \
                                                   np.array([[v_p_interp(r)- self.vp[-1], 0],
                                                             [0, v_m_interp(r)-self.vm[-1]]])
                                                   @ psi_2(r, theta), np.array([R_mod, 0]), self.rhos[-1], prec)
            return coup_and_tunn

        
    def get_n_rhos(self, l, n, filename=None):
        """
        Function to get the electron density with the quantum numbers [l, n].
        Args:
            l: integer. Quantum number for angular momentum.
            n: integer (>0). Quantum number determining the energy level.
        KArgs:
            filename: hdf5 file with the calculations. defect value: None = gets self.filename if it exists
        Returns:
            electron density with the quantum numbers [l, n]
        """
        filename = filename = self.comprobe_filename(filename)
        with h5py.File(f'{filename}.hdf5', 'r') as f:
            ls = f['ls'][:]
            ln = np.where(ls == l)[0][0]
            energies = f['energies']
            u_p = f['u_p']
            u_m = f['u_m']
            ln = np.where(ls == l)[0][0]
            energy_l, u_p_l, u_m_l = self.get_inside_and_correct(energies[ln, :], u_p[ln, :, :], u_m[ln, :, :])
            return (u_p_l[:, n]**2+u_m_l[:, n]**2)/2/np.pi


    def get_coulomb_pot(self, l, n,  Rs=None, rmax=None, filename=None):
        # R_var: los puntos donde obtenemos V(R)
        """
        Function to get the Coulomb potential by the electron density with quantum numbers [l, n] 
        in the points Rs.
        Args:
            l: integer. Quantum number for angular momentum.
            n: integer (>0). Quantum number determining the energy level.
        KArgs:
            Rs: radial coordinate values for whose we obtain the coulomb potential. Defect value: None: we take
            the coordinates till rmax or, if this is not provided, we take self.rhos.
            rmax: only taken into account if Rs is None. maximum value of the radial coordinate 
            for whose we obtain the coulomb potential 
            filename: hdf5 file with the calculations. defect value: None = gets self.filename if it exists
        Returns:
            Rs: radial coordinate values for whose we obtain the coulomb potential.
            coulomb_pot: Coulomb potential by the electron density with quantum numbers [l, n] 
            in the points Rs.
            n_r: electron density with quantum numbers [l, n]
        """
        n_r = self.get_n_rhos(l, n, filename=filename)
        if Rs is None:
            if rmax is None:
                Rs = self.rhos
            else:
                Rs = np.linspace(0, rmax, 1000)
        coulomb_pot = np.zeros_like(Rs)
        for i, R in enumerate(Rs): #Tenim V(R)
            coulomb_pot[i] = 1.44*np.trapz(np.nan_to_num(self.rhos*n_r*4/(self.rhos+R)*
                                            ellipk(4*self.rhos*R/(self.rhos+R)**2), posinf=0, neginf=0), self.rhos) 
        return Rs, coulomb_pot, n_r
    
    
    def get_coulomb_interaction(self, l1, l2, n1, n2, Rs_c, filename = None):
        """
        Gets the (off-site) coulomb interaction energy between the charges (with [l1, n1]) at
        r and the potential created by the charge density (with [l2, n2]) at R, where those R are given by Rs_c.
        Args:
            Rs: distance (or array of distances) between the wavefunctions.
            l1: integer. Quantum number for angular momentum for the wavefunction centered at 0.
            l2: integer. Quantum number for angular momentum for the wavefunction centered at R.
            n1: integer (>0). Quantum number determining the energy level for the wavefunction centered at 0.
            n2: integer (>0). Quantum number determining the energy level for the wavefunction centered at R.
            Rs_c: distances considered between both charge densities.
        KArgs:
            filename: hdf5 file with the calculations. defect value: None = gets self.filename if it exists.
        Returns:
            U_R: array with all the coulomb interaction energies at distances Rs_c
        """
        filename = self.comprobe_filename(filename)
        n_r1 = self.get_n_rhos(l1, n1, filename=filename)
        Rs, coulomb_pot, _ = self.get_coulomb_pot(l2, n2, filename=filename)
        coulomb_pot_int = interp1d(Rs, coulomb_pot, bounds_error=False, fill_value=0)
        n_r_int = interp1d(self.rhos, n_r1, bounds_error=False, fill_value=0)
        U_R = np.zeros(Rs_c.shape)
        for i, R in enumerate(Rs_c):
            U_R[i] = 0.5*self.integrate_fg(lambda r, theta: n_r_int(r), lambda r, theta: coulomb_pot_int(r),
                                  np.array([R, 0]), R_max=self.rhos[-1], prec=1e-5) # does not depend in theta_R
        return U_R
    
        

    @staticmethod
    def integrate_fg(f:callable, g: callable, R: np.array, R_max: float, prec: float):
        """
        Performs the integral for f(θ, r)g(θ', r') θ in [- π, π] and r in [0, R_max]
        where r', θ' are the radial coordinates for R' = \vec{r} - R.
        Args:
            f: function to integrate centered at 0.
            g: function to integrate centered at R.
            R: point where funtion g is centered.
            R_max: maximum integration point for the radial coordinate.
            prec: precision to integrate
        Returns:
            Value of the integral
        """
        def fun(r, theta,  f, g, R):
            rx = r*np.cos(theta)
            ry = r*np.sin(theta)
            rp = np.linalg.norm(np.array([rx, ry]) -R)
            thetap =np.arctan2(ry-R[1], rx-R[0])
            return np.real(r*np.vdot(f(r, theta), g(rp, thetap)))
        # dblquad discards the real part, but we have seen that the imaginary part is already 0.
        return dblquad(fun, -np.pi, np.pi, 0, R[0], args=(f, g, R), epsabs=prec)[0]\
            +  dblquad(fun, -np.pi, np.pi, R[0], R_max, args=(f, g, R), epsabs=prec)[0]


class Interpolate_Spinor():
    def __init__(self, rhos, u_p, u_m, l):
        """
        Class containing the functions of the spinors for BLG. Stores the information of these
        spinors for a given n and l, 
        Args:
            rhos: radial points where we know u_p(r) and u_m(r)
            u_p, u_m: numpy arrays of lenght N that contain u_+ and u_-,
            already with specified quantum numbers n (energy level) and 
            l (angular momentum).
            l: integer. Quantum number for angular momentum.
        """
        self.u_p = interp1d(rhos, u_p, bounds_error= False, fill_value=0)
        self.u_m = interp1d(rhos, u_m, bounds_error= False, fill_value=0)
        self.l = l
        
    def __call__(self, r, theta):
        """
        Args:
            r: float. Radial coordinate to evaluate the spinor.
            theta: float. Angular coordinate to evaluate the spinor (in rad).
        Returns:
            psi(r, theta): 2 dimensional complex array that is the spinor of our wavefunction.
        """
        return np.array([self.u_p(r)*np.exp((self.l-1)*1.j*theta), 
                         self.u_m(r)*np.exp((self.l+1)*1.j*theta)], dtype=complex)/np.sqrt(2*np.pi)

