from astropy import units as u


class circuitParameters:
    """
    circuitParameters is a class that encapsulates all the attributes required to configure and run an event-based sensor circuitry model.
    Attributes are populated via user input, with support for type checking, value bounds, and physical units (using astropy units). The class provides methods to prompt the user for each parameter, ensuring correct types and valid ranges.

    Notes
    -----
    - This class is designed for interactive use, requiring user input for each parameter.
    - Physical units are handled via the astropy.units package.
    - Input validation is performed for type and value bounds.
    - Alternatively, the class can be initialized with nominal (default) parameters if the user does not provide their own. These nominal parameters are reasonable placeholders and can be overridden by passing keyword arguments to the constructor.
    """

    def __init__(self, **kwargs):
        # Nominal (default) parameters as reasonable placeholders
        self.theta_on = kwargs.get("theta_on", 0.2)
        self.theta_off = kwargs.get("theta_off", 0.2)
        self.theta_on_var = kwargs.get("theta_on_var", 0.03)
        self.theta_off_var = kwargs.get("theta_off_var", 0.03)
        self.eta = kwargs.get("eta", 0.27)
        self.eta_lum = kwargs.get("eta_lum", 0.27)
        self.I_dark = kwargs.get("I_dark", 7e-16 * u.A)
        self.I_dark_ref = kwargs.get("I_dark_ref", 3.4e-16 * u.A)
        self.T_dark_ref = kwargs.get("T_dark_ref", 296.15 * u.K)
        self.Ea_dark = kwargs.get("Ea_dark", 0.6 * u.eV)
        self.photodiode_3db_freq = kwargs.get("photodiode_3db_freq", 1 * u.Hz)
        self.photodiode_3db_freq_sigma = kwargs.get(
            "photodiode_3db_freq_sigma", 0.01 * u.Hz
        )
        self.source_follower_3db_freq = kwargs.get(
            "source_follower_3db_freq", 3e3 * u.Hz
        )
        self.source_follower_3db_freq_sigma = kwargs.get(
            "source_follower_3db_freq_sigma", 0.0 * u.Hz
        )
        self.high_freq_std_fit = kwargs.get(
            "high_freq_std_fit", (-0.0113066, 0.273553, -14.1891)
        )
        # self.white_sigma = kwargs.get('white_sigma', 0.01)
        # self.white_noise_threshold = kwargs.get('white_noise_threshold', 0.0 * u.A)
        self.R_leak = kwargs.get("R_leak", 0.1 * u.Hz)
        self.P_leak = kwargs.get("P_leak", 0.1 * u.Hz / u.W)
        # self.R_shot = kwargs.get('R_shot', 1e3 * u.Hz)
        self.T_refr = kwargs.get("T_refr", 3e-3 * u.s)
        self.T_refr_sigma = kwargs.get("T_refr_sigma", 1e-4 * u.s)
        self.T = kwargs.get("T", 273.15 * u.K)
        self.recording_freq = kwargs.get("recording_freq", 3e6 * u.Hz)

    def __str__(self):
        """
        This class defines an object with all the attributes required to run the event-
        based sensor circuitry model.
        """

    def define_circuitry_parameters(self):
        """
        Prompt the user for the inputs to properly populate the object

        Returns
        -------
        None.

        """

        self.theta_on = self.__prompt_user_input(
            "on threshold between 0 and 1",
            float,
            upper_bound=1,
            lower_bound=0,
            bounds=True,
        )
        self.theta_off = self.__prompt_user_input(
            "off threshold between 0 and 1",
            float,
            upper_bound=1,
            lower_bound=0,
            bounds=True,
        )
        self.theta_on_var = self.__prompt_user_input(
            "variance of the on threshold", float
        )
        self.theta_off_var = self.__prompt_user_input(
            "variance of the off threshold", float
        )
        self.eta = self.__prompt_user_input(
            "quantum efficiency between 0 and 1. This value can also factor in the transmission factor",
            float,
            upper_bound=1,
            lower_bound=0,
            bounds=True,
        )
        self.eta_lum = self.__prompt_user_input(
            "luminous efficiency between 0 and 1",
            float,
            upper_bound=1,
            lower_bound=0,
            bounds=True,
        )
        self.I_dark = self.__prompt_user_input(
            "nominal dark current",
            float,
            variable_units=u.A,
        )
        self.I_dark_ref = self.__prompt_user_input(
            "reference dark current that is produced at a set temperature",
            float,
            variable_units=u.A,
        )
        self.T_dark_ref = self.__prompt_user_input(
            "reference temperature of sensor that produces the reference dark current",
            float,
            variable_units=u.K,
        )
        self.Ea_dark = self.__prompt_user_input(
            "dark current activation energy", float, variable_units=u.eV
        )
        self.photodiode_3db_freq = self.__prompt_user_input(
            "minimum cutoff frequency of the photodiode", float, variable_units=u.Hz
        )
        self.photodiode_3db_freq_sigma = self.__prompt_user_input(
            "photodiode cutoff frequency's 1 standard deviation. Set equal to 0 if this cutoff frequency is not dominant. The standard deviation is ",
            float,
            variable_units=u.Hz,
        )
        self.source_follower_3db_freq = self.__prompt_user_input(
            "cutoff frequency of the source follower", float, variable_units=u.Hz
        )
        self.source_follower_3db_freq_sigma = self.__prompt_user_input(
            "source follower cutoff frequency's 1 standard deviation. Set equal to 0 if this cutoff frequency is not dominant. The standard deviation is ",
            float,
            variable_units=u.Hz,
        )
        # Prompt for high_freq_std_fit tuple (3 variables)
        high_freq_std_fit_0 = self.__prompt_user_input(
            "Standard deviation fit of the high frequency noise quadratic term, (x)^2",
            float,
        )
        high_freq_std_fit_1 = self.__prompt_user_input(
            "Standard deviation fit of the high frequency noise linear term, x", float
        )
        high_freq_std_fit_2 = self.__prompt_user_input(
            "Standard deviation fit of the high frequency noise bais, b", float
        )
        self.high_freq_std_fit = (
            high_freq_std_fit_0,
            high_freq_std_fit_1,
            high_freq_std_fit_2,
        )
        # self.white_sigma = self.__prompt_user_input(
        #     "gaussian white noise 1 standard deviation value between 0 and 1",
        #     float,
        #     upper_bound=1,
        #     lower_bound=0,
        #     bounds=True,
        # )
        # self.white_noise_threshold = self.__prompt_user_input(
        #     "gaussian white noise threshold. Set equal to 0 if all pixels regardless of current should have white noise applied. Otherwise the threshold cutoff between shot and white noise is ",
        #     float,
        #     u.A,
        # )
        self.R_leak = self.__prompt_user_input(
            "nominal leak rate to the logarithmic scale memorized current",
            float,
            variable_units=u.Hz * u.A,
        )
        self.P_leak = self.__prompt_user_input(
            "parasitic leak rate to the logarithmic scale memorized current",
            float,
            variable_units=u.Hz * u.A / u.W,
        )
        # self.R_shot = self.__prompt_user_input(
        #     "shot noise rate", float, variable_units=u.Hz
        # )
        self.T_refr = self.__prompt_user_input(
            "refractory period mean", float, variable_units=u.s
        )
        self.T_refr_sigma = self.__prompt_user_input(
            "1 standard deviation in refractory period", float, variable_units=u.s
        )
        self.T = self.__prompt_user_input(
            "operating temperature of sensor", float, variable_units=u.K
        )
        self.recording_freq = self.__prompt_user_input(
            "maximum arbiter recording frequency", float, variable_units=u.Hz
        )

    def __prompt_user_input(
        self,
        variable_name,
        variable_type,
        variable_units=0,
        upper_bound=0,
        lower_bound=0,
        bounds=False,
    ):
        """
        Prompt the user to input variables for the circuit object definition.

        Parameters
        ----------
        variable_name : sting
            Name of the variable the user will be prompted to enter.
        variable_type : string
            The expected type of variable the user will enter.
        variable_units : astropy units, optional
            The units the input is expected to be in if applicable. The default is 0.
        upper_bound : float, optional
            Maximum value of the user input.
        lower_bound : float, optional
            Minimum value of the user input.
        bounds : bool, optional
            The units the input is expected to be in if applicable. The default is 0.

        Returns
        -------
        user_input : TYPE
            DESCRIPTION.

        """
        user_input = variable_name
        if variable_units != 0:
            user_input = variable_name
            input_check = variable_name
            while not isinstance(input_check, variable_type):
                user_input = input(
                    "Please input a {} for the sensor {} in units of {}: ".format(
                        str(variable_type), variable_name, str(variable_units)
                    )
                )
                try:
                    if variable_type == int:
                        input_check = int(user_input)
                        user_input = int(user_input) * variable_units
                    elif variable_type == float:
                        input_check = float(user_input)
                        user_input = float(user_input) * variable_units
                    if bounds:
                        if (
                            user_input.value <= upper_bound
                            and user_input.value >= lower_bound
                        ):
                            break
                        else:
                            print(
                                "User input is not between {} and {}. Please enter a new value.".format(
                                    lower_bound, upper_bound
                                )
                            )
                            input_check = variable_name
                except:
                    print(
                        "Input was not a {}. Please try input again.".format(
                            str(variable_type)
                        )
                    )
        else:
            user_input = variable_name
            while not isinstance(user_input, variable_type):
                user_input = input(
                    "Please input a {} for the sensor {}: ".format(
                        str(variable_type), variable_name
                    )
                )
                try:
                    if variable_type == int:
                        user_input = int(user_input)
                    elif variable_type == float:
                        user_input = float(user_input)
                    elif variable_type == bool:
                        user_input = bool(user_input)
                    if bounds:
                        if user_input <= upper_bound and user_input >= lower_bound:
                            break
                        else:
                            print(
                                "User input is not between {} and {}. Please enter a new value.".format(
                                    lower_bound, upper_bound
                                )
                            )
                            user_input = variable_name
                except:
                    print(
                        "Input was not a {}. Please try input again.".format(
                            str(variable_type)
                        )
                    )
        return user_input
