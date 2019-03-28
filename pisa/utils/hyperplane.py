import numpy as np
import collections, copy
from scipy.optimize import curve_fit

import inspect
import numba
from pisa import FTYPE, TARGET, ureg
from pisa.utils import vectorizer
from numba import guvectorize, int32, float64


def get_num_args(func) :
    '''
    Function for grabbing the number of arguments to a function
    Handles (1) stand apythin functions, (2) numpy unfuncs
    '''

    #TODO numba funcs

    if isinstance(func, np.ufunc):
        return func.nargs
    else :
        return len(inspect.getargspec(func).args)


def linear_hyperplane_func(p,m,out) :
    '''
    Linear hyperplane functional form
    '''
    result = m * p
    np.copyto(src=result,dst=out)


def exponential_hyperplane_func(p,a,b,out) :
    '''
    Exponential hyperplane functional form
    '''
    result = a * np.exp(b*p)
    np.copyto(src=result,dst=out)


class Hyperplane(object) :
    '''
    A class defining the hyperplane

    Contains :
      - The common intercept
      - Each systematic parameter, inside which the functional form is defined

    The class has a fit method for fitting the hyperplane to some data (e.g. 
    discrete systematics sets)

    I imagine :
      - Having one instance of this per bin, per type (e.g. nue_cc, nu_nc, etc)
      - Storing this to a file (somehow, either pickle, or json the function stored as a str)
      - Loading the hyperplans from a file in the PISA stage and calling the same `evaluate` method (possibly with some numba-fication)
    '''

    def __init__(self,binning,sys_params,initial_intercept=None) :

        # Store args
        self._binning = binning
        self._initial_intercept = initial_intercept

        # Containers for storing fitting information
        self._fit_complete = False
        self._fit_normed_maps = False
        self._fit_maps_raw = None
        self._fit_maps_used = None

        # Store sys params as dict for ease of lookup
        self._sys_params = collections.OrderedDict()
        for sys_param in sys_params :
            assert sys_param.name not in self._sys_params, "Duplicate sys param name found : %s" % sys_param.name
            self._sys_params[sys_param.name] = sys_param

        # Set a default initial intercept value if none provided
        if self._initial_intercept is None :
            self._initial_intercept = 0.

        # Create the fit parameter arrays
        # Have one fit per bin
        self._intercept = np.full(self._binning.shape,initial_intercept,dtype=FTYPE)
        for sys_param in self._sys_params.values() :
            sys_param.init_fit_param_arrays(self._binning)


    @property
    def intercept(self) :
        return self._intercept


    @property
    def sys_params(self) :
        return self._sys_params


    @property
    def sys_param_names(self) :
        return self._sys_params.keys()


    def evaluate(self,sys_param_values,bin_idx=None) :
        '''
        Evaluate the hyperplane, using the systematic parameter values provided
        Uses the current internal values for all functional form parameters
        '''

        #
        # Check inputs
        #

        # Determine number of sys param values (per sys param)
        # This will be >1 when fitting, and == 1 when evaluating the hyperplane within the stage
        num_sys_param_values = np.asarray(sys_param_values.values()[0]).size

        # Check same number of values for all sys params
        for k,v in sys_param_values.items() :
            n = np.asarray(v).size
            assert n == num_sys_param_values, "All sys params must have the same number of values"

        # Determine whether use single bin or not
        single_bin_mode = bin_idx is not None


        #
        # Prepare output array
        #

        # Determine shape of output array
        # Two possible cases, with limitations on both based on how the sys param functional forms are defined
        if not single_bin_mode:
            # Case 1 : Calculating for all bins simultaneously (e.g. `bin_idx is None`)
            #          Only support a single scalar value for each systematic parameters
            #          Use case is evaluating the hyperplanes during the hyperplane stage
            assert num_sys_param_values == 1, "Can only provide one value per sys param when evaluating all bins simultaneously"
            for v in sys_param_values.values() :
                assert np.isscalar(v), "sys param values must be a scalar when evaluating all bins simultaneously"
            out_shape = self._binning.shape
            bin_idx = Ellipsis

        else :
            # Case 2 : Calculating for multiple sys param values, but only a single bin
            #          Use case is fitting the hyperplanes fucntional form fit params
            out_shape = (num_sys_param_values,)

        # Create the output array
        out = np.full(out_shape,np.NaN,dtype=FTYPE)


        #
        # Evaluate the hyperplane
        #

        # Start with the intercept
        for i in range(num_sys_param_values) :
            if single_bin_mode :
                out[i] = self._intercept[bin_idx]
            else :
                np.copyto( src=self._intercept[bin_idx], dst=out[bin_idx] )
        # Evaluate each individual parameter
        for k,p in self._sys_params.items() :
            p.evaluate(sys_param_values[k],out=out,bin_idx=bin_idx)

        return out


    @property
    def sys_params_nominal_values(self) :
        return collections.OrderedDict([ (name,param.nominal_value) for name,param in self._sys_params.items() ])


    def fit(self,sys_param_values,maps,norm=True) :
        '''
        Fit the function/shape parameters 
        Writes the results directly into this data structure
        '''

        #
        # Check inputs
        #

        # Check the systematic parameter values
        assert isinstance(sys_param_values,collections.Mapping)
        assert set(sys_param_values.keys()) == set(self.sys_param_names), "`sys_param_values` keys do not match the hyperplane's systematic params"
        num_datasets = len(sys_param_values.values()[0])
        assert np.all(np.array([ len(x) for x in sys_param_values.values()]) == num_datasets), "Each systematic parameter must have one value per dataset"

        # Check there is a "nominal" dataset
        #TODO

        # Check the maps
        #TODO number, binning, ...

        # Save the sys param values used for fitting in the param objects (useful for plotting later)
        for sys_param_name,sys_param_dataset_values in sys_param_values.items() :
            self._sys_params[sys_param_name]._fitting_sys_dataset_values = np.array(sys_param_dataset_values)


        #
        # Format things before getting started
        #

        # Format the fit `x` values : [ [sys param 0 values], [sys param 1 values], ... ]
        x = np.asarray([ param_vals for param_name,param_vals in sys_param_values.items() ], dtype=FTYPE)

        # Normalise bin values
        if norm :
            nominal_map = maps[0] #TOOD More general, check has the expected niminal values, etc
            normed_maps = [ m/nominal_map for m in maps ]
            maps_to_use = normed_maps
        else :
            maps_to_use = maps


        #
        # Loop over bins
        #

        for bin_idx in np.ndindex(self._binning.shape) : #TODO grab from input map


            #
            # Format this bin's data for fitting
            #

            # Format the fit `y` values : [ bin value 0, bin_value 1, ... ]
            y = np.asarray([ m.nominal_values[bin_idx] for m in maps_to_use ], dtype=FTYPE)

            # Checks
            assert x.shape[0] == len(self._sys_params)
            assert x.shape[1] == y.size

            # Get flat list of the fit param guesses
            p0 = np.array( [self._intercept[bin_idx]] + [ sys_param._fit_params[bin_idx,i_fp][0] for sys_param in self._sys_params.values() for i_fp in range(sys_param._num_fit_params) ], dtype=FTYPE )

            # Define a callback function for use with `curve_fit`
            #   x : sys params
            #   p : func/shape params
            def callback(x,*p) :

                #TODO Once only? What about bin_idx?

                # Unflatten list of the func/shape params, and write them to the hyperplane structure
                self._intercept[bin_idx] = p[0]
                i = 1
                for sys_param in self._sys_params.values() :
                    for j in range(sys_param._num_fit_params) :
                        bin_fit_idx = tuple( list(bin_idx) + [j] )
                        sys_param._fit_params[bin_fit_idx] = p[i]
                        i += 1

                # Unflatten sys param values
                sys_params_unflattened = collections.OrderedDict()
                for i in range(len(self._sys_params)) :
                    sys_param_name = self._sys_params.keys()[i]
                    sys_params_unflattened[sys_param_name] = x[i]

                return self.evaluate(sys_params_unflattened,bin_idx=bin_idx)


            #
            # Fit
            #

            # Define the EPS (step length) used by the fitter
            # Need to take care with floating type precision, don't want to go smaller than the FTYPE being used by PISA can handle
            eps = np.finfo(FTYPE).eps
 
            # Perform fit
            #TODO limit all params to [0,1] as we do for minimizers?
            popt, pcov = curve_fit(
                callback,
                x,
                y,
                p0=p0,
                maxfev=1000,
                epsfcn=eps,
            )

            # Check the fit was successful
            #TODO

            # Write the fit results back to the hyperplane structure
            self._intercept[bin_idx] = popt[0]
            i = 1
            for sys_param in self._sys_params.values() :
                for j in range(sys_param._num_fit_params) :
                    sys_param._fit_params[bin_idx,j] = popt[i]
                    i += 1

        #
        # Done
        #

        # Record some provenance info about the fits
        self._fit_complete = True
        self._fit_normed_maps = norm
        self._fit_maps_raw = maps
        self._fit_maps_used = maps_to_use


    def get_on_axis_mask(self,sys_param_name) :
        '''
        TODO
        '''

        assert sys_param_name in self.sys_param_names

        num_fitting_datasets = self._sys_params.values()[0]._fitting_sys_dataset_values.size
        on_axis_mask = np.ones((num_fitting_datasets,),dtype=bool)

        # Loop over sys params
        for sys_param in self._sys_params.values() :

            # Ignore the chosen param
            if sys_param.name  != sys_param_name :

                # Define a "nominal" mask
                on_axis_mask = on_axis_mask & np.isclose(sys_param._fitting_sys_dataset_values,sys_param.nominal_value) 

        return on_axis_mask


    def report(self,bin_idx=None) :
        '''
        String version of the hyperplane contents
        '''

        # Fit results
        print(">>>>>> Fit parameters >>>>>>")
        bin_indices = np.ndindex(self._binning.shape) if bin_idx is None else [bin_idx]
        for bin_idx in bin_indices :
            print("  Bin %s :" % (bin_idx,) )
            print("     Intercept : %0.3g" % (self._intercept[bin_idx],) )
            for sys_param in self._sys_params.values() :
                print("     %s : %s" % ( sys_param.name, ", ".join([ "%0.3g"%sys_param._fit_params[(bin_idx,i,)] for i in range(sys_param._num_fit_params) ])) )
        print("<<<<<< Fit parameters <<<<<<")



class SysParam(object) :
    '''
    A class defining the systematic parameter in the hyperplane
    Use constructs this by passing the functional form (as a function)
    '''

    def __init__(self,name,nominal_value,func_name,initial_fit_params=None) :

        # Store basic members
        self._name = name
        self._nominal_value = nominal_value

        # Handle functional form fit parameters
        self._fit_params = None # Fir params container, not yet populated
        self._initial_fit_params = initial_fit_params # The initial values for the fit parameters

        # Record information relating to the fitting
        self._fitted = False # Flag indicating whether fit has been performed
        self._fitting_sys_dataset_values = None # The values of this sys param in each of the fitting datasets


        #
        # Init the functional form
        #

        # Get the function
        self._func = self.get_func(func_name)

        # Get the number of functional form parameters
        # This is the functional form function parameters, excluding the systematic paramater and the output object
        #TODO Does this support the GPU case?
        self._num_fit_params = get_num_args(self._func) - 2

        # Check and init the fit param initial values
        #TODO Add support for per bin values?
        if initial_fit_params is None :
            # No values provided, use 0 for all
            self._initial_fit_params = np.zeros(self._num_fit_params,dtype=FTYPE)
        else :
            # Use the provided initial values
            self._initial_fit_params = np.array(self._initial_fit_params)
            assert self._initial_fit_params.size == self._num_fit_params, "'initial_fit_params' should have %i values, found %i" % (self._num_fit_params,self._initial_fit_params.size)


    def get_func(self,func_name) :
        '''
        Find the function defining the hyperplane functional form.

        User specifies this by it's string name, which must correspond to one 
        of the pre-defined functions.
        '''

        assert isinstance(func_name,basestring), "'func_name' must be a string"

        # Form the expected function name
        hyperplane_func_suffix = "_hyperplane_func"
        full_func_name = func_name + hyperplane_func_suffix

        # Find all functions
        all_hyperplane_functions = { k:v for k,v in globals().items() if k.endswith(hyperplane_func_suffix) }
        assert full_func_name in all_hyperplane_functions, "Cannot find hyperplane function '%s', choose from %s" % (func_name,[f.split(hyperplane_func_suffix)[0] for f in all_hyperplane_functions])
        return all_hyperplane_functions[full_func_name]


    def init_fit_param_arrays(self,binning) :
        '''
        Create the arrays for storing the fit parameters
        Have one fit per bin, for each parameter
        The shape of the `self._fit_params` arrays is: (binning shape ..., num fit params )
        '''

        arrays = []

        self._binning_shape = binning.shape

        for fit_param_initial_value in self._initial_fit_params :

            fit_param_array = np.full(self._binning_shape,fit_param_initial_value,dtype=FTYPE)
            arrays.append(fit_param_array)

        self._fit_params = np.vstack(arrays).T


    @property
    def name(self) :
        return self._name

    @property
    def nominal_value(self) :
        return self._nominal_value

    @property
    def fit_params(self) :
        return self._fit_params


    def evaluate(self,sys_param,out,bin_idx=None) :
        '''
        Evaluate the functional form for the given `sys_param` values.
        Uses the current values of the fit parameters.
        By default evaluates all bins, but optionally can specify a particular bin (used when fitting).
        '''

        #TODO properly use SmartArrays

        # Create an array to file with this contorubtion
        this_out = np.full_like(out,np.NaN,dtype=FTYPE)

        # Form the arguments to pass to the functional form
        # Need to be flexible in terms of the number of fit parameters
        args = [sys_param]
        for i_fit_param in range(self._num_fit_params) :
            fit_param_idx = (bin_idx,i_fit_param,) #TODO Does this work?
            args += [self._fit_params[fit_param_idx]]
        args += [this_out]

        # Call the function
        self._func(*args)

        # Add to overall hyperplane result
        out += this_out



if __name__ == "__main__" : 

    import sys

    #TODO turn this into a PASS/FAIL test, and add more detailed test of specific functions

    #
    # Create hyperplane
    #

    # Define the various systematic parameter sin the hyperplane
    sys_params = [
        SysParam( name="foo", nominal_value=0., func_name="linear", initial_fit_params=[1.], ),
        SysParam( name="bar", nominal_value=10., func_name="exponential", initial_fit_params=[1.,-1.], ),
    ]

    # Define binning
    from pisa.core.binning import OneDimBinning, MultiDimBinning
    binning = MultiDimBinning([OneDimBinning(name="reco_energy",domain=[0.,10.],num_bins=10,units=ureg.GeV,is_lin=True)])

    # Create the hyperplane
    hyperplane = Hyperplane( 
        binning=binning,
        sys_params=sys_params, # Specify the systematic parameters
        initial_intercept=0., # Intercept value (or first guess for fit)
    )


    #
    # Create fake datasets
    #

    from pisa.core.map import Map, MapSet

    # Just doing something quick here for demonstration purposes
    # Here I'm only assigning a single value per dataset, e.g. one bin, for simplicity, but idea extends to realistic binning

    # Define the values for the parameters for each dataset
    # Assuming the first is the nominal dataset
    dataset_param_values = {
        "foo" : [0., 0., 0.,  0., -1.,+1.],
        "bar" : [10.,20.,30.,-10.,10.,10.],
    }

    num_datasets = len(dataset_param_values.values()[0])

    # Only consider one particle type for simplicity
    particle_key = "nue_cc"

    # Create a dummy "true" hyperplane that can be used to generate some fake bin values for the dataset 
    true_hyperplane = copy.deepcopy(hyperplane)
    true_hyperplane._intercept.fill(3.)
    if "foo" in true_hyperplane._sys_params :
        true_hyperplane._sys_params["foo"]._fit_params[...,0].fill(2.)
    if "bar" in true_hyperplane._sys_params :
        true_hyperplane._sys_params["bar"]._fit_params[...,0].fill(1.)
        true_hyperplane._sys_params["bar"]._fit_params[...,1].fill(-0.1)

    print("\nTruth hyperplane report:")
    print true_hyperplane.report()

    # Create each dataset, e.g. set the systematic parameter values, calculate a bin count
    datasets_sys_param_values = []
    datasets_mapsets = []
    for i in range(num_datasets) :

        # Get a dict of the param values
        sys_param_vals = { param._name:dataset_param_values[param._name][i] for param in true_hyperplane._sys_params.values() }

        # Generate some histogrammed data using the fake "truth" hyperplane
        dataset_hist = true_hyperplane.evaluate(sys_param_vals)
        dataset_mapset = MapSet([ Map(name=particle_key,binning=binning,hist=dataset_hist) ])

        datasets_sys_param_values.append(sys_param_vals)
        datasets_mapsets.append(dataset_mapset)


    #
    # Fit hyperplanes
    #

    # Perform fit
    hyperplane.fit(
        sys_param_values={ k:[datasets_sys_param_values[i][k] for i in range(num_datasets)] for k in datasets_sys_param_values[0].keys() },
        maps=[ mapset[particle_key] for mapset in datasets_mapsets ],
        norm=False,
    )

    # Report the results
    print("\nFitted hyperplane report:")
    hyperplane.report()

    # Check the fitted parameter values match the truth
    print("\nChecking fit recovered truth...")
    assert np.allclose( hyperplane.intercept, true_hyperplane.intercept )
    for sys_param_name in hyperplane.sys_param_names :
        assert np.allclose( hyperplane.sys_params[sys_param_name].fit_params, true_hyperplane.sys_params[sys_param_name].fit_params )
    print("... fit was successful!\n")


    #
    # Plot
    #

    import matplotlib.pyplot as plt

    def plot_bin_fits(hyperplane,bin_idx) :

        # Create the figure
        fig,ax = plt.subplots(1,len(hyperplane.sys_params))

        # Get bin values for this bin only
        #TODO errors
        chosen_bin_values = [ m.nominal_values[bin_idx] for m in hyperplane._fit_maps_used ]

        # Loop over systematics
        for i_sys,sys_param in enumerate(hyperplane.sys_params.values()) :

            # Get the plot ax
            plot_ax = ax if len(hyperplane.sys_params) == 1 else ax[i_sys]

            # Define a mask for selecting on-axis points only
            on_axis_mask = hyperplane.get_on_axis_mask(sys_param.name)

            # Plot the points from the datasets used for fitting
            x = np.asarray(sys_param._fitting_sys_dataset_values)[on_axis_mask]
            y = np.asarray(chosen_bin_values)[on_axis_mask]
            plot_ax.scatter( x, y, marker="o", color="black", label="Datasets (on-axis)" )

            # Plot the hyperplane
            # Generate as bunch of values along the sys param axis to make the plot
            # Then calculate the hyperplane value at each point, using the nominal values for all other sys params
            x_plot = np.linspace( np.nanmin(x), np.nanmax(x), num=100 )
            sys_params_for_plot = { sys_param.name : x_plot, }
            for p in hyperplane.sys_params.values() :
                if p.name != sys_param.name :
                    sys_params_for_plot[p.name] = np.full_like(x_plot,hyperplane.sys_params_nominal_values[p.name])
            y_plot = hyperplane.evaluate(sys_params_for_plot,bin_idx=bin_idx)
            plot_ax.plot( x_plot, y_plot, color="red", label="Fit" )

            # Mark the nominal value
            plot_ax.axvline( x=sys_param.nominal_value, color="blue", alpha=0.5, linestyle="--", label="Nominal" )

            # Format ax
            plot_ax.set_xlabel(sys_param.name)
            plot_ax.grid(True)
            plot_ax.legend()

        # Format fig
        fig.tight_layout()

        return fig

 
    fig = plot_bin_fits(
        hyperplane=hyperplane,
        bin_idx=(0,),
    )

    fig_file_path = "hyperplane.pdf"
    fig.savefig(fig_file_path)
    print("Figure saved : %s" % fig_file_path)

