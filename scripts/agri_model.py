import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pvlib
import warnings
from pvfactors.geometry import OrderedPVArray
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Length and width of CS BiKu MODULE (doesn't account for frame)
def agri_model(len_module=1.972, width_module=0.963, pvrow_height=2, n_pvrows=4, sur_tilt=30, sur_azi=180,
               gcr=0.4, with_solar=True, grid_connected=True):

    # latitude, longitude, name, altitude, timezone
    coordinates = [
        12.028, 37.847, 'Lake Tana Ethiopia', 1809, 'Etc/UTC+3',
    ]

    # get the bi-facial module and inverter specifications from CEC DB
    cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')

    cec_inverter = pvlib.pvsystem.retrieve_sam('CECInverter')

    module = cec_modules['Canadian_Solar_Inc__CS3U_380MB_AG']

    inverter = cec_inverter['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    module_area = len_module * width_module  # m2
    no_modules_per_row = 50

    pvfactors_parameters = {
        'n_pvrows': n_pvrows,  # number of pv rows
        'pvrow_height': pvrow_height,  # height of pvrows (measured at center / torque tube)
        'pvrow_width': width_module,  # width of pvrows
        'axis_azimuth': 0.,  # azimuth angle of rotation axis
        'gcr': gcr,  # ground coverage ratio
        'albedo': 0.25,  # albedo of the ground
        'rho_front_pvrow': 0.03,  # Front surface reflectivity of PV rows
        'rho_back_pvrow': 0.05,  # Back surface reflectivity of PV rows
        'surface_tilt': sur_tilt,
        'surface_azimuth': sur_azi
    }

    latitude, longitude, name, altitude, timezone = coordinates
    # Date & time (UTC for normal CSV, local timezone time for the EPW format)
    # T2m [°C] - Dry bulb (air) temperature.
    # RH [%] - Relative Humidity.
    # G(h) [W/m2] - Global horizontal irradiance.
    # Gb(n) [W/m2] - Direct (beam) irradiance.
    # Gd(h) [W/m2] - Diffuse horizontal irradiance.
    # IR(h) [W/m2] - Infrared radiation downwards.
    # WS10m [m/s] - Windspeed.
    # WD10m [°] - Wind direction.
    # SP [Pa] - Surface (air) pressure.
    weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude, startyear=2006, endyear=2016, usehorizon=False)[0]
    weather.index.name = "utc_time"
    weather.reset_index(inplace=True)

    # Get the solar position (azimuth and zenith) over the time period
    solpos = pvlib.solarposition.get_solarposition(
        time=weather['utc_time'],
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        temperature=weather["T2m"],
        pressure=pvlib.atmosphere.alt2pres(altitude),
    )

    solpos.reset_index(inplace=True)
    solar_df = pd.merge(weather, solpos, left_on='utc_time', right_on='utc_time')

    # Determine clear sky GHI using the Haurwitz model.
    solar_df["ghi"] = pvlib.clearsky.haurwitz(apparent_zenith=solar_df['apparent_zenith'])

    # Create month number to group by
    solar_df['month'] = solar_df['utc_time'].dt.month
    solar_df_monthly = solar_df.groupby(['month'])['ghi'].mean().reset_index()

    # To account for local variation of GHI caused by cloudiness and altitude, we scale the integrated GHI over time to
    # match the satellite-derived 22-year monthly average GHI data from the NASA Surface meteorology
    # All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)
    ssdr = pd.read_csv("/Agrivoltaic/Data/POWER_Point_Monthly_Timeseries_2000_2020_012d0280N_037d8470E_LST.csv")
    ssdr_df = pd.DataFrame(ssdr)
    ssdr_df.set_index('YEAR', drop=True, inplace=True)
    ssdr_df_mean = ssdr_df.mean(axis=0).reset_index()
    ssdr_df_mean.columns = ['month', 'avg_ghi_per_day']
    ssdr_df_mean['days_in_month'] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ssdr_df_mean['avg_ghi_per_month'] = ssdr_df_mean['avg_ghi_per_day'] * ssdr_df_mean['days_in_month']
    ssdr_df_mean['ratio_clearsky'] = ssdr_df_mean['avg_ghi_per_month'] / solar_df_monthly['ghi']
    ssdr_df_mean['month'] = ssdr_df_mean['month'].astype(str).astype(int)
    ssdr_df_mean.drop(['avg_ghi_per_day', 'days_in_month', 'avg_ghi_per_month'], axis=1, inplace=True)

    # Scale according to the monthly average of the ghi
    solar_df = pd.merge(solar_df, ssdr_df_mean, left_on='month', right_on='month')
    solar_df['ghi_scaled'] = solar_df['ratio_clearsky'] * solar_df['ghi']
    solar_df.set_index('utc_time', drop=True, inplace=True)

    # We need to decompose GHI to DNI and the amount of DHI. We will use the DISC Model
    dni_dhi_df = pvlib.irradiance.dirint(ghi=solar_df['ghi_scaled'], solar_zenith=solar_df['zenith'],
                                         times=solar_df.index, pressure=101325.0, use_delta_kt_prime=True,
                                         temp_dew=None, min_cos_zenith=min(np.cos(solar_df['zenith'])),
                                         max_zenith=solar_df['zenith'].max())
    solar_df = pd.merge(solar_df, dni_dhi_df, left_index=True, right_index=True)
    solar_df['dhi'] = solar_df['ghi_scaled'] - (solar_df['dni'] * np.cos(solar_df['apparent_zenith']))

    # Calculate front and back surface plane-of-array irradiance on a fixed tilt or single-axis tracker PV array
    # configuration, and using the open-source “pvfactors” package.
    pv_parameters = {
        'n_pvrows': pvfactors_parameters['n_pvrows'],  # number of pv rows
        'pvrow_height': pvfactors_parameters['pvrow_height'],  # height of pvrows (measured at center / torque tube)
        'pvrow_width': pvfactors_parameters['pvrow_width'],  # width of pvrows
        'axis_azimuth': 0.,  # azimuth angle of rotation axis
        'surface_tilt': pvfactors_parameters['surface_tilt'],  # tilt of the pv rows
        'surface_azimuth': pvfactors_parameters['surface_azimuth'],  # azimuth of the pv rows front surface
        'solar_zenith': 40.,  # solar zenith angle
        'solar_azimuth': 150.,  # solar azimuth angle
        'gcr': pvfactors_parameters['gcr'],  # ground coverage ratio
    }

    bifacial_df = pvlib.bifacial.pvfactors_timeseries(solar_azimuth=solar_df['azimuth'],
                                                      solar_zenith=solar_df['apparent_zenith'],
                                                      surface_azimuth=pvfactors_parameters['surface_azimuth'],
                                                      surface_tilt=pd.Series(index=solar_df.index,
                                                                             data=pvfactors_parameters['surface_tilt']),
                                                      axis_azimuth=pvfactors_parameters['axis_azimuth'],
                                                      timestamps=solar_df.index,
                                                      dni=solar_df['dni'],
                                                      dhi=solar_df['dhi'],
                                                      gcr=pvfactors_parameters['gcr'],
                                                      pvrow_height=pvfactors_parameters['pvrow_height'],
                                                      pvrow_width=pvfactors_parameters['pvrow_width'],
                                                      albedo=pvfactors_parameters['albedo'],
                                                      n_pvrows=pvfactors_parameters['n_pvrows'],
                                                      rho_front_pvrow=pvfactors_parameters['rho_front_pvrow'],
                                                      rho_back_pvrow=pvfactors_parameters['rho_back_pvrow'])

    # Calculate the total power generation (kW) using the absorbed irradiance (this accounts for reflection losses)
    front_eff = 0.1894
    back_eff = front_eff * 0.98
    solar_df['front_power'] = module_area * bifacial_df[2] * pvfactors_parameters['n_pvrows'] * no_modules_per_row \
                              * front_eff / 1000
    solar_df['back_power'] = module_area * bifacial_df[3] * pvfactors_parameters['n_pvrows'] * no_modules_per_row \
                             * back_eff / 1000
    solar_df['total_power'] = solar_df['front_power'] + solar_df['back_power']

    print("The maximum front side generation (kW) is: ", round(solar_df['front_power'].max(), 0), '\n',
          "The maximum back side generation (kW) is: ", round(solar_df['back_power'].max(), 0))
    print("The annual front side generation (kWh) is: ", round(solar_df['front_power'].sum(), 0), '\n',
          "The annual back side generation (kWh) is: ", round(solar_df['back_power'].sum(), 0), '\n',
          "The total annual generation (kWh) is: ", round(solar_df['total_power'].sum(), 0), "at a row-to-row pitch"
          " over panel height (p/h) of",
          round((len_module / pvfactors_parameters['gcr']) / pvfactors_parameters['pvrow_height'], 2))

    pv_plot = OrderedPVArray.fit_from_dict_of_scalars(pv_parameters)
    print("The total irradiance (W/m2) on the array is: ", round(bifacial_df[0].sum(), 0))

    # Plot the total power generation by month of the year
    data = solar_df.copy()
    data.reset_index(inplace=True)
    data['month'] = data['utc_time'].dt.month
    data['hour'] = data['utc_time'].dt.hour
    # data.groupby(['month', 'hour'])['total_power'].mean().unstack(0). \
    #     plot(subplots=True, layout=(8, 3), figsize=(20, 4 * 3), legend=True,
    #          colormap='viridis')

    # Get the total global irradiance (W/m2) that the ground would be exposed to if there was no solar array above
    ground_parameters = {
        'n_pvrows': 1,  # number of pv rows
        'pvrow_height': 0.120,  # height of lettuce leaves
        'pvrow_width': (pvfactors_parameters['n_pvrows'] * pvfactors_parameters['pvrow_width'] /
                        pvfactors_parameters['gcr'] * np.cos(np.deg2rad(pvfactors_parameters['surface_tilt']))),  # width of pvrows
        'axis_azimuth': 0.,  # azimuth angle of rotation axis
        'surface_tilt': 0,  # tilt of the pv rows
        'surface_azimuth': 90.,  # azimuth of the pv rows front surface
        'solar_zenith': 40.,  # solar zenith angle
        'solar_azimuth': 150.,  # solar azimuth angle
        'gcr': 1,  # ground coverage ratio
    }
    ground_df = pvlib.bifacial.pvfactors_timeseries(solar_azimuth=solar_df['azimuth'],
                                                    solar_zenith=solar_df['apparent_zenith'],
                                                    surface_azimuth=pvfactors_parameters['surface_azimuth'],
                                                    surface_tilt=pd.Series(index=solar_df.index, data=0),
                                                    axis_azimuth=pvfactors_parameters['axis_azimuth'],
                                                    timestamps=solar_df.index,
                                                    dni=solar_df['dni'],
                                                    dhi=solar_df['dhi'],
                                                    gcr=1,
                                                    pvrow_height=ground_parameters['pvrow_height'],
                                                    pvrow_width=pvfactors_parameters['pvrow_width'],
                                                    albedo=0,
                                                    n_pvrows=pvfactors_parameters['n_pvrows'],
                                                    rho_front_pvrow=pvfactors_parameters['rho_front_pvrow'],
                                                    rho_back_pvrow=pvfactors_parameters['rho_back_pvrow'])

    ground_plot = OrderedPVArray.fit_from_dict_of_scalars(ground_parameters)
    print("The total irradiance (W/m2) on the ground without an array is: ", round(ground_df[0].sum(), 0))

    # Plot the array and the ground view
    f, ax = plt.subplots(ncols=1, nrows=2, figsize=(16, 10))
    pv_plot.plot_at_idx(0, ax[0])
    ground_plot.plot_at_idx(0, ax[1])

    # Create a lettuce growth model
    # Import time series data: this includes ambient temperature and irradiance (Wm-2) reaching the ground. This data
    # Convert to J/m2h and multiply by ≈ 0.51 which is the ratio of the integrated PAR (400nm – 700nm) to the
    # integrated AM1.5 sunlight spectrum.
    if with_solar:
        u_par_watt = 0.51 * ((ground_df[0] + ground_df[1]) - (bifacial_df[0] + bifacial_df[1]))      # Units W/m2
        u_par = 0.51 * (3600 * ((ground_df[0] + ground_df[1]) - (bifacial_df[0] + bifacial_df[1])))    # Units J/m2h
        u_t = solar_df['T2m'] - 1.65      # Estimate of temperature reduction under panels (Brecht Willock et al. 2020)
    else:
        u_par_watt = 0.51 * (ground_df[0] + ground_df[1])     # Units W/m2
        u_par = 0.51 * (3600 * (ground_df[0] + ground_df[1]))
        u_t = solar_df['T2m']

    # for df in (u_par, u_par_watt):
    u_par = np.where(u_par < 0, 0, u_par)
    u_par = np.nan_to_num(u_par)
    u_par_watt = np.where(u_par_watt < 0, 0, u_par_watt)
    u_par_watt = np.nan_to_num(u_par_watt)

    # create grouped
    par_df = solar_df.copy()
    par_df.reset_index(inplace=True)
    par_df['month'] = par_df['utc_time'].dt.month_name()
    par_df['hour'] = par_df['utc_time'].dt.hour
    par_df['u_par'] = u_par_watt
    par_df['bifacial_rad'] = np.array(bifacial_df[0]) + np.array(bifacial_df[1])
    par_df['bifacial_rad'] = np.nan_to_num(par_df['bifacial_rad'])
    sort_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                  'November', 'December']
    par_df.index = pd.CategoricalIndex(par_df['month'], categories=sort_order, ordered=True)
    par_df.sort_index().reset_index(drop=True)
    par_df.drop(labels='month', axis=1, inplace=True)
    par_df.reset_index()
    par_df = par_df.groupby(['month', 'hour'])['bifacial_rad', 'u_par'].mean().unstack(0)

    # Plot the Solar Array Incident Irradiance (Yellow) and PAR (Black) (W/m2)
    plot1 = par_df['bifacial_rad'].plot(subplots=True, layout=(4, 3), color='gold', xlim=(0, 23), figsize=(16, 10))
    par_df['u_par'].plot(subplots=True, layout=(4, 3), color='black', ax=plot1, legend=False,
                         title="Average Daily Solar Array Incident Irradiance (Yellow) and Ground PAR (Black) (W/m2)")

    # Create linear interpolators for u_t(t) and u_par(t) to feed into the ODE
    time_range = np.arange(u_t.size)
    u_t_func = interp1d(time_range, u_t, bounds_error=False, fill_value="extrapolate")
    u_par_func = interp1d(time_range, u_par, bounds_error=False, fill_value="extrapolate")
    u_t_pred = u_t_func(time_range)
    u_par_pred = u_par_func(time_range)

    f3, ax = plt.subplots(2, 1, figsize=(16, 10))
    ax[0].plot(pd.date_range(start=pd.datetime(2021, 1, 1), periods=8760, freq='1H'), u_t_pred, label='Annual Temperature')
    ax[1].plot(pd.date_range(start=pd.datetime(2021, 1, 1), periods=8760, freq='1H'), u_par_pred/3600, linewidth=3,
               label='Incident Photosynthetically Active Radiation (PAR) (W/m2)', zorder=5, color='dodgerblue')
    ax[1].plot(pd.date_range(start=pd.datetime(2021, 1, 1), periods=8760, freq='1H'), (bifacial_df[0] + bifacial_df[1]),
               linewidth=3, label='Incident irradiance on the solar array (W/m2)', zorder=2, color='indianred')
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')

    # Define event function where once the fresh weight per head=660g/head (this is equivalent to 4620g/m2 of fresh
    # weight at a density of 7 head/m2), once reached the ode restarts from the initial conditions
    # this event function feeds into the ODE and represents the lettuce harvesting
    harvest_fw = 610
    head_density = 7  # We will assume that the density of the patch is 7 head of lettuce per m2
    c_fw_to_dw = 0.045  # this is the conversion ratio from dry weight to fresh weight
    f_phot_list = []
    inc_rad = []
    req_fresh_weight = harvest_fw * head_density

    def event(t, X):
        # The fresh weight of the lettuce has to reach 250 g/head before harvesting
        return (X[0] + X[1]) / c_fw_to_dw - req_fresh_weight

    event.terminal = True

    def lettuce_ode(t, X):
        """
        Parameters:
        c_alpha : was the ratio of the molecular weights of CH2O to CO2 (30/44)
        f_phot : was the gross canopy photosynthesis
        r_gr : was the specific growth rate
        f_resp : was the maintenance respiration
        c_beta : was the coefficient for the respiratory and synthesis losses of non-structural material due to growth
        DW : was the final dry weight at harvest per unit area (g/m2)
        FW : was the final fresh weight per unit area (g/m2)
        c_fw_to_dw : was the conversion ratio from DW to FW
        c_gr_max : saturation growth rate at 20 degrees celcius
        c_q10_gr : for every temperature increase of 10C, the specific growth rate increases by a factor of c_q10_gr
        u_t : this is the ambient temperature (celcius)
        c_resp_sht : maintenance respiration coefficients for the shoot
        c_resp_rt : maintenance respiration coefficients for the roots
        c_mu : ratio of dry root weight to the total crop dry weight
        c_q10_resp : Q10 factor of the maintenance respiration
        c_k : this is the extinction coefficient
        c_lar : shoot structural leaf area ratio
        f_phot_max : canopy photosynthesis
        c_e : light use efficiency
        u_par : incident photosynthetically active radiation
        sigma : leaf conductance to CO2 diffusion
        u_co2 : CO2 concentration
        """

        Xnsdw = X[0]
        Xsdw = X[1]

        # Parameters
        c_alpha = 30 / 44
        c_beta = 0.80
        c_gr_max = 5e-6 * 3600
        c_q10_gr = 1.6
        c_mu = 0.15  # Increased from 0.07 in Van Henten to 0.15
        c_resp_sht = 3.47e-7 * 3600
        c_resp_rt = 1.16e-7 * 3600
        c_q10_resp = 2.0
        c_k = 0.3  # Originally 0.9 in Van Henten, however there is a reduced density (head/m2) outside a greenhouse
        c_lar = 75e-3
        c_e = 2.5e-6
        c_R = 40
        c_q10_R = 2.0
        c_stm = 0.02 * 3600
        c_bnd = 0.004 * 3600
        car_1 = -1.32e-5 * 3600
        car_2 = 5.94e-4 * 3600
        car_3 = -2.64e-3 * 3600
        u_co2 = 400  # ppm

        # Equations
        sigma_car = car_1 * (u_t_func(t)) ** 2 + car_2 * u_t_func(t) + car_3
        sigma = 1 / (1 / c_stm + 1 / c_bnd + 1 / sigma_car)
        R = c_R * c_q10_R ** ((u_t_func(t) - 20) / 10)
        epsilon = c_e * (u_co2 - R) / (u_co2 + 2 * R)
        f_phot_max = (epsilon * u_par_func(t) * sigma * (u_co2 - R)) / (epsilon * u_par_func(t) + sigma * (u_co2 - R))
        f_phot = (1 - np.exp(- c_k * c_lar * (1 - c_mu) * Xsdw)) * f_phot_max
        f_resp = (c_resp_sht * (1 - c_mu) * Xsdw + c_resp_rt * c_mu * Xsdw) * c_q10_resp ** ((u_t_func(t) - 25) / 10)
        r_gr = c_gr_max * Xnsdw / (Xsdw + Xnsdw) * c_q10_gr ** ((u_t_func(t) - 20) / 10)

        f_phot_list.append(f_phot)
        inc_rad.append(u_par_func(t))

        # Rate Equations
        dXsdw_dt = r_gr * Xsdw
        dXnsdw_dt = c_alpha * f_phot - r_gr * Xsdw - f_resp - ((1 - c_beta) / c_beta) * r_gr * Xsdw

        return [dXnsdw_dt, dXsdw_dt]

    # Initial Conditions
    X0 = [0.675, 2.025]  # (gm-2)

    ts, X_nsdw, X_sdw = [], [], []
    harvest_count = 0
    t = time_range[0] + 5088  # This takes the start date to the start of August for the start of the planting season
    tend = 8760
    while True:
        sol = solve_ivp(lettuce_ode, t_span=(t, tend), y0=X0, events=event)
        ts.append(sol.t)
        X_nsdw.append(sol.y[0, :])
        X_sdw.append(sol.y[1, :])
        if sol.status == 1:  # Event was hit (harvest time!)
            harvest_count += 1  # This counts the number of harvests over the year
            # New start time for integration
            t = sol.t[-1] + 72  # assume three days to begin the next cultivation (72 hours)
            # Reset initial state
            X = sol.y[:, [-2, -1]].copy()
            X[0] = 0.675  # reset to initial conditions
            X[1] = 2.025
        else:
            break

    # We have to stitch together the separate simulation results for plotting
    lettuce_time = np.concatenate(ts)
    X_nsdw_conc = np.concatenate(X_nsdw)
    X_sdw_conc = np.concatenate(X_sdw)
    inc_rad = [float(i) for i in inc_rad]

    # Calculate the total fresh weight of the lettuce
    d = {'NSDW': X_nsdw_conc, 'SDW': X_sdw_conc}
    lettuce_df = pd.DataFrame(data=d, index=lettuce_time)
    lettuce_df['dry_weight'] = lettuce_df['NSDW'] + lettuce_df['SDW']
    lettuce_df['fresh_weight'] = lettuce_df['dry_weight'] / c_fw_to_dw

    f4, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    sns.scatterplot(lettuce_time, X_nsdw_conc, color='dodgerblue', label='Non-Structural Dry Weight', ax=ax[0])
    sns.scatterplot(lettuce_time, X_sdw_conc, color='dodgerblue', label='Structural Dry Weight', ax=ax[1])
    sns.scatterplot(x=lettuce_df.index, y=lettuce_df['fresh_weight'], label='Fresh weight', color='dodgerblue', ax=ax[2])
    ax[0].set_xlim(lettuce_df.index.min(), lettuce_df.index.max())
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    ax[2].legend(loc='best')
    ax[0].set_ylabel("$g/m^2$")
    ax[1].set_ylabel("$g/m^2$")
    ax[2].set_ylabel("$g/m^2$")
    ax[2].set_xlabel("Hour")

    # Lettuce sales Calculation
    total_area = ground_parameters['pvrow_width'] * no_modules_per_row * len_module
    print("The total ground area used for lettuce is: ", round(total_area, 0), "m2 or", round(total_area / 4046.86, 2),
          "acres")
    cost_per_head = 0.45                # Cost per head of lettuce ($/head) in Ethiopia
    perc_marketable = 0.80              # Fraction of the lettuce that is marketable
    total_head = harvest_count * head_density * total_area
    total_yield = harvest_fw / 1e6 * head_density * harvest_count * 10000       # 10000 m2 per hectare
    total_sales = harvest_count * head_density * total_area * cost_per_head * perc_marketable
    print(f"The total head of lettuce before the marketable fraction is applied: ", round(total_head, 0))
    print("The total yield per hectare is: ", round(total_yield/harvest_count, 0), "(t/ha)")

    # Variable costs estimate for this region
    var_cost_per_acre = 3000
    # Fixed costs
    fixed_cost_per_acre = 1150

    # Net profit
    lettuce_net = total_sales - total_area / 4046.86 * (var_cost_per_acre + fixed_cost_per_acre)
    print("The total annual profit selling lettuce is: ", round(lettuce_net, 0), "$/year\n")

    # PV generation sales
    solar_kWp = 0.38 * no_modules_per_row * pvfactors_parameters['n_pvrows']
    mean_power = solar_df['total_power'].mean()
    inverter_eff = 0.95

    # O&M costs
    solar_om_costs = 10  # $/kW/year
    solar_costs = solar_om_costs * solar_kWp
    print("The total annual solar O&M costs are: ", round(solar_costs, 0), "$/year")

    if grid_connected:
        elec_price = 0.05  # Electricity export tariff price in Ethiopia ($/kWh) for similar scale projects
        solar_revenue = elec_price * solar_df['total_power'].sum() * inverter_eff
        print("The total annual solar generation revenue for a", round(solar_kWp, 0), "kWp array is: ",
              round(solar_revenue, 0), "$/year")
    else:
        solar_revenue = 0
        no_iphone = mean_power / 0.005
        no_tvs = mean_power / 0.0586
        no_irrigation_valves = mean_power / 0.12
        print(f"Can power a total of {int(no_iphone)} iphones or {int(no_tvs)} tvs or {int(no_irrigation_valves)} "
              f"irrigation valves at average power generation")

    pv_gen_net = solar_revenue - solar_costs
    print("The net solar profit is: ", round(pv_gen_net, 0), "$/year")

    total_profit = pv_gen_net + lettuce_net
    print("The total profit is: ", round(total_profit, 0), "$/year")

    return lettuce_net, pv_gen_net, total_profit, f_phot_list, inc_rad


lettuce_net, pv_gen_net, profit, f_phot_list, inc_rad = agri_model(pvrow_height=2,
                                                                   sur_tilt=7,
                                                                   sur_azi=173, with_solar=True, grid_connected=True)

# Plot gross canopy photosynthesis rate
# f2, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
# sns.scatterplot(x=inc_rad, y=f_phot_list, color='dodgerblue', ax=ax)

# Sensitivity Analysis
tilt_range = [i for i in range(1, 50, 2)]
lettuce_profit = []
solarpv_profit = []
total_profit = []
run_count = 1

# for i in tilt_range:
#     lettuce_net, pv_gen_net, profit, f_phot_list, inc_rad = agri_model(sur_tilt=i, with_solar=True)
#     lettuce_profit.append(lettuce_net)
#     solarpv_profit.append(pv_gen_net)
#     total_profit = np.array(lettuce_profit) + np.array(solarpv_profit)
#     print(f"Appended {run_count} iterations at a tilt angle of {int(i)} degrees\n\n")
#     run_count += 1

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# f5, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 10))
# sns.set_style("ticks")
# sns.axes_style("darkgrid")
# sns.lineplot(x=tilt_range, y=lettuce_profit, ax=ax, label='Lettuce Profit', color='dodgerblue', linewidth=3)
# sns.lineplot(x=tilt_range, y=solarpv_profit, ax=ax, label='Solar PV Profit', color='indianred', linewidth=3)
# sns.lineplot(x=tilt_range, y=total_profit, ax=ax, label='Total Profit', color='gold', linewidth=3)
# ax.set_xlim(tilt_range[0], tilt_range[-1])
# ax.set_ylabel("Profit ($/year)")
# ax.set_xlabel("Array Tilt Angle (degrees)")

pv_height = np.linspace(start=1.6, stop=4.6, num=20)
pv_azimuth = np.linspace(start=100, stop=300, num=26)
gcr_range = np.linspace(start=0.1, stop=0.8, num=12)
#
# for i in pv_azimuth:
#     lettuce_net, pv_gen_net, profit, f_phot_list, inc_rad = agri_model(sur_azi=i, with_solar=True)
#     lettuce_profit.append(lettuce_net)
#     solarpv_profit.append(pv_gen_net)
#     total_profit = np.array(lettuce_profit) + np.array(solarpv_profit)
#     print(f"Appended {run_count} iterations at an azimuth of {int(i)} degrees\n\n")
#     run_count += 1

# f5, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 10))
# sns.set_style("ticks")
# sns.axes_style("darkgrid")
# sns.lineplot(x=pv_azimuth, y=lettuce_profit, ax=ax, label='Lettuce Profit', color='dodgerblue', linewidth=3)
# sns.lineplot(x=pv_azimuth, y=solarpv_profit, ax=ax, label='Solar PV Profit', color='indianred', linewidth=3)
# sns.lineplot(x=pv_azimuth, y=total_profit, ax=ax, label='Total Profit', color='gold', linewidth=3)
# ax.set_xlim(pv_azimuth[0], pv_azimuth[-1])
# ax.set_ylabel("Profit ($/year)")
# ax.set_xlabel("PV Array Azimuth (degrees)")
#
# for i in pv_height:
#     lettuce_net, pv_gen_net, profit, f_phot_list, inc_rad = agri_model(pvrow_height=i, with_solar=True)
#     lettuce_profit.append(lettuce_net)
#     solarpv_profit.append(pv_gen_net)
#     total_profit = np.array(lettuce_profit) + np.array(solarpv_profit)
#     print(f"Appended {run_count} iterations at a height of {round(i,2)} meters\n\n")
#     run_count += 1
#
# f5, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 10))
# sns.set_style("ticks")
# sns.axes_style("darkgrid")
# sns.lineplot(x=pv_height, y=lettuce_profit, ax=ax, label='Lettuce Profit', color='dodgerblue', linewidth=3)
# sns.lineplot(x=pv_height, y=solarpv_profit, ax=ax, label='Solar PV Profit', color='indianred', linewidth=3)
# sns.lineplot(x=pv_height, y=total_profit, ax=ax, label='Total Profit', color='gold', linewidth=3)
# ax.set_xlim(pv_height[0], pv_height[-1])
# ax.set_ylabel("Profit ($/year)")
# ax.set_xlabel("PV Array Height (m)")


plt.show()


