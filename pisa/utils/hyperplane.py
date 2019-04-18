import numpy as np
import collections, copy
from scipy.optimize import curve_fit

import inspect
import numba
from pisa import FTYPE, TARGET, ureg
from pisa.utils import vectorizer, jsons
from pisa.utils.jsons import from_json, to_json
from pisa.core.pipeline import Pipeline
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map
from numba import guvectorize, int32, float64

from uncertainties import correlated_values


'''
Helper functions
'''

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


'''
Hyperplane functional forms
'''

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



'''
Core hyperplane classes
'''

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

    def __init__(self,params,initial_intercept=None) :

        # Store args
        self.initial_intercept = initial_intercept

        # Store params as dict for ease of lookup
        self.params = collections.OrderedDict()
        for param in params :
            assert param.name not in self.params, "Duplicate param name found : %s" % param.name
            self.params[param.name] = param

        # Internal state
        self._initialized = False

        # Containers for storing fitting information
        self.fit_complete = False
        self.fit_maps_norm = None
        self.fit_maps_raw = None
        self.fit_chi2 = None
        self.fit_cov_mat = None
        self.fit_method = None

        # Serialization
        self._serializable_state = None


    def _init(self,binning,nominal_param_values) :
        '''
        Actually initialise the hyperplane
        '''

        #
        # Binning
        #

        # Store the binning
        self.binning = binning

        # Set a default initial intercept value if none provided
        if self.initial_intercept is None :
            self.initial_intercept = 0.

        # Create the fit coefficient arrays
        # Have one fit per bin
        self.intercept = np.full(self.binning.shape,self.initial_intercept,dtype=FTYPE)
        self.intercept_sigma = np.full_like(self.intercept,np.NaN)
        for param in self.params.values() :
            param.init_fit_coefft_arrays(self.binning)


        #
        # Nominal values
        #

        # Store the nominal param values
        #TODO better checks, including not already set
        for param in self.params.values() :
            param.nominal_value = nominal_param_values[param.name]


        #
        # Done
        #

        self._initialized = True


    @property
    def param_names(self) :
        return self.params.keys()


    def evaluate(self,param_values,bin_idx=None) :
        '''
        Evaluate the hyperplane, using the systematic parameter values provided
        Uses the current internal values for all functional form parameters
        '''

        assert self._initialized


        #
        # Check inputs
        #

        # Determine number of sys param values (per sys param)
        # This will be >1 when fitting, and == 1 when evaluating the hyperplane within the stage
        num_param_values = np.asarray(param_values.values()[0]).size

        # Check same number of values for all sys params
        for k,v in param_values.items() :
            n = np.asarray(v).size
            assert n == num_param_values, "All sys params must have the same number of values"

        # Determine whether using single bin or not
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
            assert num_param_values == 1, "Can only provide one value per sys param when evaluating all bins simultaneously"
            for v in param_values.values() :
                assert np.isscalar(v), "sys param values must be a scalar when evaluating all bins simultaneously"
            out_shape = self.binning.shape
            bin_idx = Ellipsis

        else :
            # Case 2 : Calculating for multiple sys param values, but only a single bin
            #          Use case is fitting the hyperplanes fucntional form fit params
            out_shape = (num_param_values,)

        # Create the output array
        out = np.full(out_shape,np.NaN,dtype=FTYPE)


        #
        # Evaluate the hyperplane
        #

        # Start with the intercept
        for i in range(num_param_values) :
            if single_bin_mode :
                out[i] = self.intercept[bin_idx]
            else :
                np.copyto( src=self.intercept[bin_idx], dst=out[bin_idx] )

        # Evaluate each individual parameter
        for k,p in self.params.items() :
            p.evaluate(param_values[k],out=out,bin_idx=bin_idx)

        return out


    @property
    def nominal_values(self) :
        return collections.OrderedDict([ (name,param.nominal_value) for name,param in self.params.items() ])

    @property
    def fit_param_values(self) :
        return collections.OrderedDict([ (name,param.fit_param_values) for name,param in self.params.items() ])

    @property
    def num_fit_sets(self) :
        return self.params.values()[0].num_fit_sets


    def fit(self,nominal_map,nominal_param_values,sys_maps,sys_param_values,norm=True,method=None) :
        '''
        Fit the function/shape parameters 
        Writes the results directly into this data structure
        '''

        #
        # Check inputs
        #

        # Check all maps
        #TODO

        # Check the systematic parameter values
        #TODO
        # assert isinstance(param_values,collections.Mapping)
        # assert set(param_values.keys()) == set(self.param_names), "`param_values` keys do not match the hyperplane's systematic params"
        # num_datasets = len(param_values.values()[0])
        # assert np.all(np.array([ len(x) for x in param_values.values()]) == num_datasets), "Each systematic parameter must have one value per dataset"

        # Check the maps
        #TODO number, binning, ...


        #
        # Format things before getting started
        #

        # Default fit method
        # Choosing one that produces covariance matrix results reliably
        self.fit_method = method
        if self.fit_method is None :
            self.fit_method = "lm"  # lm, trf, dogbox

        # Initialise hyperplane using nominal dataset
        self._init(binning=nominal_map.binning,nominal_param_values=nominal_param_values)

        # Combine nominal and sys sets
        maps = [nominal_map] + sys_maps
        param_values = [nominal_param_values] + sys_param_values

        # Store raw maps
        self.fit_maps_raw = maps

        # Convert params valus from `list of dicts` to `dict of lists`
        param_values_dict = { name:np.array([ p[name] for p in param_values ]) for name in param_values[0].keys() }

        # Save the param values used for fitting in the param objects (useful for plotting later)
        for name,values in param_values_dict.items() :
            self.params[name].fit_param_values = values

        # Format the fit `x` values : [ [sys param 0 values], [sys param 1 values], ... ]
        x = np.asarray( param_values_dict.values(), dtype=FTYPE )

        # Normalise bin values, if requested
        if norm :
            nominal_map = maps[0] #TOOD More general, check has the expected niminal values, etc
            normed_maps = [ m/nominal_map for m in maps ]
            self.fit_maps_norm = normed_maps

        # Prepare covariance matrix array
        self.fit_cov_mat = np.full( list(self.binning.shape)+[self.num_fit_coeffts,self.num_fit_coeffts] ,np.NaN )


        #
        # Loop over bins
        #

        for bin_idx in np.ndindex(self.binning.shape) : #TODO grab from input map


            #
            # Format this bin's data for fitting
            #

            # Format the fit `y` values : [ bin value 0, bin_value 1, ... ]
            # Also get the corresonding uncertainty
            y = np.asarray([ m.nominal_values[bin_idx] for m in self.fit_maps ], dtype=FTYPE)
            y_sigma = np.asarray([ m.std_devs[bin_idx] for m in self.fit_maps ], dtype=FTYPE)

            # Checks
            assert x.shape[0] == len(self.params)
            assert x.shape[1] == y.size

            # Get flat list of the fit param guesses
            p0 = np.array( [self.intercept[bin_idx]] + [ param.get_fit_coefft(bin_idx=bin_idx,coefft_idx=i_cft) for param in self.params.values() for i_cft in range(param.num_fit_coeffts) ], dtype=FTYPE )

            # Define a callback function for use with `curve_fit`
            #   x : sys params
            #   p : func/shape params
            def callback(x,*p) :

                #TODO Once only? What about bin_idx?

                # Unflatten list of the func/shape params, and write them to the hyperplane structure
                self.intercept[bin_idx] = p[0]
                i = 1
                for param in self.params.values() :
                    for j in range(param.num_fit_coeffts) :
                        bin_fit_idx = tuple( list(bin_idx) + [j] )
                        param.fit_coeffts[bin_fit_idx] = p[i]
                        i += 1

                # Unflatten sys param values
                params_unflattened = collections.OrderedDict()
                for i in range(len(self.params)) :
                    param_name = self.params.keys()[i]
                    params_unflattened[param_name] = x[i]

                return self.evaluate(params_unflattened,bin_idx=bin_idx)


            #
            # Fit
            #

            # Define the EPS (step length) used by the fitter
            # Need to take care with floating type precision, don't want to go smaller than the FTYPE being used by PISA can handle
            eps = np.finfo(FTYPE).eps

            #TODO finite mask

            #TODO if no values in bins, skip and add NaNs to fit params
 
            # Perform fit
            #TODO rescale all params to [0,1] as we do for minimizers?
            popt, pcov = curve_fit(
                callback,
                x,
                y,
                p0=p0,
                sigma=y_sigma,
                absolute_sigma=True, #TODO check this
                maxfev=1000, #TODO arg?
                epsfcn=eps,
                method=self.fit_method,
            )

            # Check the fit was successful
            #TODO

            #TODO uncertainties, empty bins, etc

            # Use covariance matrix to get uncertainty in fit parameters
            # Using uncertainties.correlated_values, and will extract the std shortly
            #TODO diectly store the uarray?
            corr_vals = correlated_values(popt,pcov)

            # Write the fitted param results back to the hyperplane structure
            self.intercept[bin_idx] = popt[0]
            self.intercept_sigma[bin_idx] = corr_vals[0].std_dev
            i = 1
            for param in self.params.values() :
                for j in range(param.num_fit_coeffts) :
                    idx = param.get_fit_coefft_idx(bin_idx=bin_idx,coefft_idx=j)
                    param.fit_coeffts[idx] = popt[i]
                    param.fit_coeffts_sigma[idx] = corr_vals[i].std_dev
                    i += 1

            # Store the covariance matrix
            self.fit_cov_mat[bin_idx] = pcov #TODO copyto?


        #
        # chi2
        #

        # Compare the result of the fitted hyperplane function with the actual data points used for fitting
        # Compute the resulting chi2 to have an estimate of the fit quality

        self.fit_chi2 = []

        # Loop over datasets
        for i_set in range(self.num_fit_sets) :

            # Get expected bin values according tohyperplane value
            predicted = self.evaluate({ name:values[i_set] for name,values in param_values_dict.items() })

            # Get the observed value
            observed = self.fit_maps[i_set].nominal_values
            sigma = self.fit_maps[i_set].std_devs

            # Compute chi2
            chi2 = ((predicted - observed) / sigma) ** 2

            # Add to container
            self.fit_chi2.append(chi2)

        # Combine into single array
        self.fit_chi2 = np.stack(self.fit_chi2,axis=-1).astype(FTYPE)


        #
        # Done
        #

        # Record some provenance info about the fits
        self.fit_complete = True


    def get_on_axis_mask(self,param_name) :
        '''
        TODO
        '''

        assert param_name in self.param_names

        num_fitting_datasets = self.params.values()[0].fit_param_values.size
        on_axis_mask = np.ones((num_fitting_datasets,),dtype=bool)

        # Loop over sys params
        for param in self.params.values() :

            # Ignore the chosen param
            if param.name  != param_name :

                # Define a "nominal" mask
                on_axis_mask = on_axis_mask & np.isclose(param.fit_param_values,param.nominal_value) 

        return on_axis_mask


    def report(self,bin_idx=None) :
        '''
        String version of the hyperplane contents
        '''

        # Fit results
        print(">>>>>> Fit coefficients >>>>>>")
        bin_indices = np.ndindex(self.binning.shape) if bin_idx is None else [bin_idx]
        for bin_idx in bin_indices :
            print("  Bin %s :" % (bin_idx,) )
            print("     Intercept : %0.3g" % (self.intercept[bin_idx],) )
            for param in self.params.values() :
                print("     %s : %s" % ( param.name, ", ".join([ "%0.3g"%param.get_fit_coefft(bin_idx=bin_idx,coefft_idx=cft_idx) for cft_idx in range(param.num_fit_coeffts) ])) )
        print("<<<<<< Fit coefficients <<<<<<")



    @property
    def fit_maps(self) :
        # assert self.fit_complete
        return self.fit_maps_raw if self.fit_maps_norm is None else self.fit_maps_norm


    @property
    def num_fit_sets(self) :
        return len(self.fit_param_values.values()[0])


    @property
    def num_fit_coeffts(self) :
        '''
        Return the total number of coefficients to fit
        This is the overall intercept, plus the coefficients for each individual param
        '''
        return int( 1 + np.sum([ param.num_fit_coeffts for param in self.params.values() ]) )


    @property
    def fit_coeffts(self) :
        '''
        Return all coefficients, in all bins, as a single array
        This is the overall intercept, plus the coefficients for each individual param
        Dimensions are: [binning ..., fit coeffts]
        '''
        
        array = [self.intercept]
        for param in self.params.values() :
            for i in range(param.num_fit_coeffts) :
                array.append( param.get_fit_coefft(coefft_idx=i) )
        array = np.stack(array,axis=-1)
        return array


    @property
    def fit_coefft_labels(self) :
        '''
        Return labels for each fit coefficient
        '''
        return ["intercept"] + [ "%s p%i"%(param.name,i) for param in self.params.values() for i in range(param.num_fit_coeffts) ]


    @property
    def serializable_state(self):
        """OrderedDict containing savable state attributes"""

        if self._serializable_state is None: #TODO always redo?

            state = collections.OrderedDict()

            state["_initialized"] = self._initialized
            state["binning"] = self.binning.serializable_state
            state["initial_intercept"] = self.initial_intercept
            state["intercept"] = self.intercept
            state["intercept_sigma"] = self.intercept_sigma
            state["fit_complete"] = self.fit_complete
            state["fit_maps_norm"] = self.fit_maps_norm
            state["fit_maps_raw"] = self.fit_maps_raw
            state["fit_chi2"] = self.fit_chi2
            state["fit_cov_mat"] = self.fit_cov_mat
            state["fit_method"] = self.fit_method

            state["params"] = collections.OrderedDict()
            for name,param in self.params.items() :
                state["params"][name] = param.serializable_state

            self._serializable_state = state

        return self._serializable_state 


    @classmethod
    def from_state(cls, state):
        """Instantiate a new object from the contents of a serialized state dict
        Parameters
        ----------
        resource : dict
            A dict
        See Also
        --------
        to_json
        """

        #
        # Get the state
        #

        # If it is not already a a state, alternativey try to load it in case a JSON file was passed
        if not isinstance(state,collections.Mapping) :
            try :
                state = jsons.from_json(state)
            except:
                raise IOError("Could not load state")


        #
        # Create params
        #

        params = []

        # Loop through params in the state        
        params_state = state.pop("params")
        for param_name,param_state in params_state.items() :

            # Create the param
            param = HyperplaneParam(
                name=param_state.pop("name"),
                func_name=param_state.pop("func_name"),
                initial_fit_coeffts=param_state.pop("initial_fit_coeffts"),
            )

            # Define rest of state
            for k in param_state.keys() :
                setattr(param,k,param_state.pop(k))
                # print param.name,k,type(getattr(param,k)),getattr(param,k)

            # Store
            params.append(param)


        #
        # Create hyperplane
        #

        # Instantiate
        hyperplane = cls(
            params=params,
            initial_intercept=state.pop("initial_intercept"),
        )

        # Add binning
        hyperplane.binning = MultiDimBinning(**state.pop("binning"))

        # Add maps
        hyperplane.fit_maps_raw = [ Map(**map_state) for map_state in state.pop("fit_maps_raw") ]
        fit_maps_norm = state.pop("fit_maps_norm")
        hyperplane.fit_maps_norm = None if fit_maps_norm is None else [ Map(**map_state) for map_state in fit_maps_norm ]

        # Define rest of state
        for k in state.keys() :
            setattr(hyperplane,k,state.pop(k))
            # print k,type(getattr(hyperplane,k)),getattr(hyperplane,k)

        return hyperplane


    def smooth(self,method="gauss") :
        '''
        Apply smoothing between bins for hyperplane coefficients
        '''

        pass #TODO implement

        #TODO see work done for Upgrade oscillations analsis along these lines

        # # Smooth the params across neighbouring bins
        # if smooth == 'gauss':
        #     hyperplanes["hyperplanes"][map_name]["fit_params_smooth"] = np.full_like(hyperplanes["hyperplanes"][map_name]["fit_params"],np.NaN) 
        #     for i_fit_param in range(num_params) :
        #         finite_mask = np.isfinite(hyperplanes["hyperplanes"][map_name]["fit_params"][...,i_fit_param])
        #         hyperplanes["hyperplanes"][map_name]["fit_params_smooth"][...,i_fit_param][finite_mask] = gaussian_filter(hyperplanes["hyperplanes"][map_name]["fit_params"][...,i_fit_param][finite_mask],sigma=1.)


class HyperplaneParam(object) :
    '''
    A class defining the systematic parameter in the hyperplane
    Use constructs this by passing the functional form (as a function)
    '''

    def __init__(self,name,func_name,initial_fit_coeffts=None) :

        # Store basic members
        self.name = name

        # Handle functional form fit parameters
        self.fit_coeffts = None # Fit params container, not yet populated
        self.fit_coeffts_sigma = None # Fit param sigma container, not yet populated
        self.initial_fit_coeffts = initial_fit_coeffts # The initial values for the fit parameters

        # Record information relating to the fitting
        self.fitted = False # Flag indicating whether fit has been performed
        self.fit_param_values = None # The values of this sys param in each of the fitting datasets

        # Placeholder for nominal value
        self.nominal_value = None

        # Serialization
        self._serializable_state = None


        #
        # Init the functional form
        #

        # Get the function
        self.func_name = func_name
        self._func = self.get_func(self.func_name)

        # Get the number of functional form parameters
        # This is the functional form function parameters, excluding the systematic paramater and the output object
        #TODO Does this support the GPU case?
        self.num_fit_coeffts = get_num_args(self._func) - 2

        # Check and init the fit param initial values
        #TODO Add support for per bin values?
        if initial_fit_coeffts is None :
            # No values provided, use 0 for all
            self.initial_fit_coeffts = np.zeros(self.num_fit_coeffts,dtype=FTYPE)
        else :
            # Use the provided initial values
            self.initial_fit_coeffts = np.array(self.initial_fit_coeffts)
            assert self.initial_fit_coeffts.size == self.num_fit_coeffts, "'initial_fit_coeffts' should have %i values, found %i" % (self.num_fit_coeffts,self.initial_fit_coeffts.size)


    def get_func(self,func_name) :
        '''
        Find the function defining the hyperplane functional form.

        User specifies this by it's string name, which must correspond to one 
        of the pre-defined functions.
        '''

        assert isinstance(func_name,basestring), "'func_name' must be a string"

        # Form the expected function name
        hyperplane_func_suffix = "_hyperplane_func"
        fullfunc_name = func_name + hyperplane_func_suffix

        # Find all functions
        all_hyperplane_functions = { k:v for k,v in globals().items() if k.endswith(hyperplane_func_suffix) }
        assert fullfunc_name in all_hyperplane_functions, "Cannot find hyperplane function '%s', choose from %s" % (func_name,[f.split(hyperplane_func_suffix)[0] for f in all_hyperplane_functions])
        return all_hyperplane_functions[fullfunc_name]


    def init_fit_coefft_arrays(self,binning) :
        '''
        Create the arrays for storing the fit parameters
        Have one fit per bin, for each parameter
        The shape of the `self.fit_coeffts` arrays is: (binning shape ..., num fit params )
        '''

        arrays = []

        self.binning_shape = binning.shape

        for fit_coefft_initial_value in self.initial_fit_coeffts :

            fit_coefft_array = np.full(self.binning_shape,fit_coefft_initial_value,dtype=FTYPE)
            arrays.append(fit_coefft_array)

        self.fit_coeffts = np.stack(arrays,axis=-1)
        self.fit_coeffts_sigma = np.full_like(self.fit_coeffts,np.NaN)


    def evaluate(self,param,out,bin_idx=None) :
        '''
        Evaluate the functional form for the given `param` values.
        Uses the current values of the fit parameters.
        By default evaluates all bins, but optionally can specify a particular bin (used when fitting).
        '''

        #TODO properly use SmartArrays

        # Create an array to file with this contorubtion
        this_out = np.full_like(out,np.NaN,dtype=FTYPE)

        # Form the arguments to pass to the functional form
        # Need to be flexible in terms of the number of fit parameters
        args = [param]
        for cft_idx in range(self.num_fit_coeffts) :
            # idx = tuple(list(bin_idx) + [cft_idx])
            # args += [self.fit_coeffts[idx]]
            args += [self.get_fit_coefft(bin_idx=bin_idx,coefft_idx=cft_idx)]
        args += [this_out]

        # Call the function
        self._func(*args)

        # Add to overall hyperplane result
        out += this_out


    def get_fit_coefft_idx(self,bin_idx=None,coefft_idx=None) :
        '''
        Indexing the fit_coefft matrix is a bit of a pain
        This helper function eases things
        TODO can probably do this more cleverly with numpy indexing, but works for now...
        '''

        # Indexing based on the bin
        if (bin_idx is Ellipsis) or (bin_idx is None) :
            idx = [Ellipsis]
        else :
            idx = list(bin_idx)

        # Indexing based on the coefficent
        if isinstance(coefft_idx,slice) :
            idx.append(coefft_idx)
        elif coefft_idx is None :
            idx.append(slice(0,-1))
        else :
            idx.append(coefft_idx)

        # Put it all together
        idx = tuple(idx)
        return idx


    def get_fit_coefft(self,*args,**kwargs) :
        '''
        Get a fit coefficient values from the matrix
        Basically just wrapping the indexing function
        '''
        idx = self.get_fit_coefft_idx(*args,**kwargs)
        return self.fit_coeffts[idx]


    @property
    def serializable_state(self):
        """OrderedDict containing savable state attributes"""

        if self._serializable_state is None: #TODO always redo?

            state = collections.OrderedDict()
            state["name"] = self.name
            state["func_name"] = self.func_name
            state["num_fit_coeffts"] = self.num_fit_coeffts
            state["fit_coeffts"] = self.fit_coeffts
            state["fit_coeffts_sigma"] = self.fit_coeffts_sigma
            state["initial_fit_coeffts"] = self.initial_fit_coeffts
            state["fitted"] = self.fitted
            state["fit_param_values"] = self.fit_param_values
            state["binning_shape"] = self.binning_shape
            state["nominal_value"] = self.nominal_value

            self._serializable_state = state

        return self._serializable_state 


'''
Hyperplane fitting and loading
'''

def fit_hyperplanes(nominal_dataset,sys_datasets,params,output_file,combine_regex=None) :
    '''
    Function for fitting hyperplanes to simulation dataset
    '''

    #TODO proper docs

    #
    # Check inputs
    #

    #TODO


    #
    # Run all pipelines
    #

    # Run the nominal and systematics pipelines
    #TODO maybe DistributionMaker
    nominal_dataset["pipeline"] = Pipeline(nominal_dataset["pipeline"])
    nominal_dataset["mapset"] = nominal_dataset["pipeline"].get_outputs() #return_sum=False)
    for sys_dataset in sys_datasets :
        sys_dataset["pipeline"] = Pipeline(sys_dataset["pipeline"])
        sys_dataset["mapset"] = sys_dataset["pipeline"].get_outputs() #return_sum=False)

    # Merge maps according to the combine regex, is one was provided
    if combine_regex is not None :
        nominal_dataset["mapset"] = nominal_dataset["mapset"].combine_re(combine_regex)
        for sys_dataset in sys_datasets :
            sys_dataset["mapset"] = sys_dataset["mapset"].combine_re(combine_regex)

    #TODO check every mapset has the same elements


    #
    # Loop over maps
    #

    # Create the container to fill
    hyperplanes = collections.OrderedDict()

    # Loop over maps
    for map_name in nominal_dataset["mapset"].names :


        #
        # Prepare data for fit
        #

        nominal_map = nominal_dataset["mapset"][map_name]
        nominal_param_values = nominal_dataset["sys_params"]

        sys_maps = [ sys_dataset["mapset"][map_name] for sys_dataset in sys_datasets   ]
        sys_param_values = [ sys_dataset["sys_params"] for sys_dataset in sys_datasets   ]


        #
        # Fit the hyperplane
        #

        # Create the hyperplane
        hyperplane = Hyperplane( 
            params=copy.deepcopy(params),
            initial_intercept=0., # Initial value for intercept
        )

        # Perform fit
        hyperplane.fit(
            nominal_map=nominal_map,
            nominal_param_values=nominal_param_values,
            sys_maps=sys_maps,
            sys_param_values=sys_param_values,
            norm=False,
        )

        # Report the results
        print("\nFitted hyperplane report:")
        hyperplane.report()

        # Store for later write to disk
        hyperplanes[map_name] = hyperplane


    #
    # Store results
    #

    # Write to a json file
    to_json(hyperplanes,output_file)



def load_hyperplanes(input_file) :
    '''
    Function to load file containing hyperplane fits
    Can be multiple hyperplanes assosicated with different map keys
    '''

    #TODO backwards compatibility

    # Load the file
    hyperplane_states = from_json(input_file)
    assert isinstance(hyperplane_states,collections.Mapping)

    # Loop over hyperplane states and load them
    hyperlanes = collections.OrderedDict()
    for map_name,hyperplane_state in hyperplane_states.items() :
        hyperlanes[map_name] = Hyperplane.from_state(hyperplane_state)

    return hyperlanes



'''
Plotting
'''

def plot_bin_fits(ax,hyperplane,bin_idx,param_name,color=None,label=None,show_nominal=True) :

    import matplotlib.pyplot as plt

    # Get the param
    param = hyperplane.params[param_name]

    # Default color
    if color is None :
        color = "red"

    # Check bin index
    assert len(bin_idx) == len(hyperplane.binning.shape)

    # Get bin values for this bin only
    chosen_bin_values = [ m.nominal_values[bin_idx] for m in hyperplane.fit_maps ]
    chosen_bin_sigma = [ m.std_devs[bin_idx] for m in hyperplane.fit_maps ]

    # Define a mask for selecting on-axis points only
    on_axis_mask = hyperplane.get_on_axis_mask(param.name)

    # Plot the points from the datasets used for fitting
    x = np.asarray(param.fit_param_values)[on_axis_mask]
    y = np.asarray(chosen_bin_values)[on_axis_mask]
    yerr = np.asarray(chosen_bin_sigma)[on_axis_mask]
    ax.errorbar( x=x, y=y, yerr=yerr, marker="o", color=color, linestyle="None", label=label )

    # Plot the hyperplane
    # Generate as bunch of values along the sys param axis to make the plot
    # Then calculate the hyperplane value at each point, using the nominal values for all other sys params
    x_plot = np.linspace( np.nanmin(x), np.nanmax(x), num=100 )
    params_for_plot = { param.name : x_plot, }
    for p in hyperplane.params.values() :
        if p.name != param.name :
            params_for_plot[p.name] = np.full_like(x_plot,hyperplane.nominal_values[p.name])
    y_plot = hyperplane.evaluate(params_for_plot,bin_idx=bin_idx)
    ax.plot( x_plot, y_plot, color=color )

    #TODO Add fit uncertainty. Problem using uarrays with np.exp at the minute, may need to shift to bin-wise calc...
    # ax.fill_between( curve_x[i,:], unp.nominal_values(y_opt)-unp.std_devs(y_opt), unp.nominal_values(y_opt)+unp.std_devs(y_opt), color='red', alpha=0.2 )

    # # Optonal : For testing, overlay the uncertainty one would find under the assumption fit parameters are uncorrelated
    # # Typically straight line fit parameters are strongly correlated, so expect this to be a large overestimation
    # if False :
    #     cov_mat = fit_results["hyperplanes"][map_name]["cov_matrices"][:,:,zind][idx]
    #     fit_params_uncorr = unp.uarray( unp.nominal_values(fit_params) , np.sqrt(np.diag(cov_mat)) )
    #     y_opt_uncorr = hyperplane_fun(curve_x, *fit_params_uncorr)
    #     ax.fill_between( curve_x[i,:], unp.nominal_values(y_opt_uncorr)-unp.std_devs(y_opt_uncorr), unp.nominal_values(y_opt_uncorr)+unp.std_devs(y_opt_uncorr), color='blue', alpha=0.5 )

    # Mark the nominal value
    if show_nominal :
        ax.axvline( x=param.nominal_value, color="blue", alpha=0.5, linestyle="--", label="Nominal" )

    # Format ax
    ax.set_xlabel(param.name)
    ax.grid(True)
    ax.legend()


if __name__ == "__main__" : 

    import sys

    #TODO turn this into a PASS/FAIL test, and add more detailed test of specific functions

    #
    # Create hyperplane
    #

    # Define the various systematic parameter sin the hyperplane
    params = [
        HyperplaneParam( name="foo", func_name="linear", initial_fit_coeffts=[1.], ),
        HyperplaneParam( name="bar", func_name="exponential", initial_fit_coeffts=[1.,-1.], ),
    ]

    # Create the hyperplane
    hyperplane = Hyperplane( 
        params=params, # Specify the systematic parameters
        initial_intercept=0., # Intercept value (or first guess for fit)
    )


    #
    # Create fake datasets
    #

    from pisa.core.map import Map, MapSet

    # Just doing something quick here for demonstration purposes
    # Here I'm only assigning a single value per dataset, e.g. one bin, for simplicity, but idea extends to realistic binning

    # Define binning
    from pisa.core.binning import OneDimBinning, MultiDimBinning
    binning = MultiDimBinning([OneDimBinning(name="reco_energy",domain=[0.,10.],num_bins=3,units=ureg.GeV,is_lin=True)])
    # binning = MultiDimBinning([OneDimBinning(name="reco_energy",domain=[0.,10.],num_bins=2,units=ureg.GeV,is_lin=True),OneDimBinning(name="reco_coszen",domain=[-1.,1.],num_bins=3,is_lin=True)])

    # Define the values for the parameters for each dataset
    nom_param_values = {
        "foo" : 0.,
        "bar" : 10.,
    }
    sys_param_values_dict = {
        "foo" : [ 0., 0.,  0.,-1.,+1.],
        "bar" : [20.,30.,-10.,10.,10.],
    }

    # Get number of datasets
    num_sys_datasets = len(sys_param_values_dict.values()[0])

    # Only consider one particle type for simplicity
    particle_key = "nue_cc"

    # Create a dummy "true" hyperplane that can be used to generate some fake bin values for the dataset 
    true_hyperplane = copy.deepcopy(hyperplane)
    true_hyperplane._init(binning=binning,nominal_param_values=nom_param_values)
    true_hyperplane.intercept.fill(3.)
    if "foo" in true_hyperplane.params :
        true_hyperplane.params["foo"].fit_coeffts[...,0].fill(2.)
    if "bar" in true_hyperplane.params :
        true_hyperplane.params["bar"].fit_coeffts[...,0].fill(1.)
        true_hyperplane.params["bar"].fit_coeffts[...,1].fill(-0.1)

    print("\nTruth hyperplane report:")
    print true_hyperplane.report()

    # Create each dataset, e.g. set the systematic parameter values, calculate a bin count
    hist = true_hyperplane.evaluate(nom_param_values)
    nom_map = Map(name=particle_key,binning=binning,hist=hist,error_hist=np.sqrt(hist))
    sys_maps = []
    sys_param_values = []
    for i in range(num_sys_datasets) :
        sys_param_values.append( { name:sys_param_values_dict[name][i] for name in true_hyperplane.params.keys() } )
        hist = true_hyperplane.evaluate(sys_param_values[-1])
        sys_maps.append( Map(name=particle_key,binning=binning,hist=hist,error_hist=np.sqrt(hist)) )


    #
    # Fit hyperplanes
    #

    # Perform fit
    hyperplane.fit(
        nominal_map=nom_map,
        nominal_param_values=nom_param_values,
        sys_maps=sys_maps,
        sys_param_values=sys_param_values,
        norm=False,
    )

    # Report the results
    print("\nFitted hyperplane report:")
    hyperplane.report()

    # Check the fitted parameter values match the truth
    print("\nChecking fit recovered truth...")
    assert np.allclose( hyperplane.intercept, true_hyperplane.intercept )
    for param_name in hyperplane.param_names :
        assert np.allclose( hyperplane.params[param_name].fit_coeffts, true_hyperplane.params[param_name].fit_coeffts )
    print("... fit was successful!\n")


    #
    # Save/load
    #

    # Save
    file_path = "hyperplane.json.bz2"
    to_json(hyperplane,file_path)

    # Re-load
    reloaded_hyperplane = Hyperplane.from_state(file_path)

    # Test
    #TODO

    # Done
    hyperplane = reloaded_hyperplane


    #
    # Plot
    #

    import matplotlib.pyplot as plt

    # Create the figure
    fig,ax = plt.subplots(1,len(hyperplane.params))

    # Choose an arbitrary bin for plotting
    bin_idx = tuple([ 0 for i in range(hyperplane.binning.num_dims) ])

    # Plot each param
    for i,param in enumerate(hyperplane.params.values()) :

        plot_ax = ax if len(hyperplane.params) == 1 else ax[i]

        plot_bin_fits(
            ax=plot_ax,
            hyperplane=hyperplane,
            bin_idx=bin_idx,
            param_name=param.name,
        )

    # Format
    fig.tight_layout()

    # Save
    fig_file_path = "hyperplane.pdf"
    fig.savefig(fig_file_path)
    print("Figure saved : %s" % fig_file_path)

