"""
Oscillation probability statge, using the DEIMOS solver
"""

from pisa.core.stage import Stage
from pisa.utils.profiler import profile, line_profile
from pisa import ureg, FTYPE

from deimos.wrapper.osc_calculator import *

__all__ = ["deimos"]

__author__ = "T. Stuttard"


class deimos(Stage):
    """
    TODO
    """

    def __init__(
        self,
        detector_depth,
        prop_height,
        **std_kwargs,
    ):

        # Store args
        self.detector_depth = detector_depth.m_as("km")
        self.prop_height = prop_height.m_as("km")

        # Define standard params
        expected_params = [
            "theta12",
            "theta13",
            "theta23",
            "deltam21",
            "deltam31",
            "deltacp",
        ]

        # # Add decoherence parameters
        # if self.use_decoherence:
        #     # Use derived nuSQuIDS classes
        #     import nuSQUIDSDecohPy

        #     self.nusquids_layers_class = nuSQUIDSDecohPy.nuSQUIDSDecohLayers
        #     # Checks
        #     assert (
        #         self.num_neutrinos == 3
        #     ), "Decoherence only supports 3 neutrinos currently"
        #     # Add decoherence params
        #     expected_params.extend(["gamma0"])
        #     expected_params.extend(["n"])
        #     expected_params.extend(["E0"])

        # Init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )


    def setup_function(self):

        # Create calculator
        self.calculator = OscCalculator(
            tool="deimos",
            atmospheric=True,
            num_neutrinos=3,
            detector_depth_km=self.detector_depth,#.m_as("km"),#TODO add units
            production_height_km=self.prop_height,#.m_as("km"),#TODO add units
        )

        # Define matter
        self.calculator.set_matter("vacuum") #TODO options for others

        # Init arrays
        for container in self.data:
            container['prob_e'] = np.empty((container.size), dtype=FTYPE)
            container['prob_mu'] = np.empty((container.size), dtype=FTYPE)
            # container['prob_tau'] = np.empty((container.size), dtype=FTYPE)


    def compute_function(self):

        # Set data representation
        self.data.representation = self.calc_mode

        #
        # Set params
        #

        theta12 = self.params.theta12.value.m_as('rad')
        theta13 = self.params.theta13.value.m_as('rad')
        theta23 = self.params.theta23.value.m_as('rad')
        dm21 = self.params.deltam21.value.m_as('eV**2')
        dm31 = self.params.deltam31.value.m_as('eV**2')
        deltacp = self.params.deltacp.value.m_as('rad')

        self.calculator.set_mass_splittings(dm21, dm31)
        self.calculator.set_mixing_angles(theta12=theta12, theta13=theta13, theta23=theta23, deltacp=deltacp)


        #
        # Compute oscillations
        #

        # Loop over containers
        for container in self.data:

            # Get flavor and nu/nubar
            nubar = container["nubar"] < 0
            flav = container["flav"]

            # Loop over events/grid points  #TODO make this more efficient
            for i in range(container.size) :

                # Calc osc probs
                osc_probs = self.calculator.calc_osc_prob( #TODO try adding jit to speed up
                    initial_flavor=flav,
                    nubar=nubar,
                    energy_GeV=container["true_energy"][i],
                    coszen=container["true_coszen"][i],
                )

                # Store osc probs
                container['prob_e'][i] = osc_probs[0, 0, 0]
                container['prob_mu'][i] = osc_probs[0, 0, 1]

            # Done
            container.mark_changed('prob_e')
            container.mark_changed('prob_mu')

    @profile
    def apply_function(self):

        # Update the outputted weights
        for container in self.data:
            container['weights'] *= (container['nu_flux'][:,0] * container['prob_e']) + (container['nu_flux'][:,1] * container['prob_mu'])

