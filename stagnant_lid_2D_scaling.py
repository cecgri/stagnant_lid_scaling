#!/usr/local/bin/python3
#
__author__ = "cecile.grigne@univ-brest.fr"

"""
Version October 15, 2024.

Script that computes the internal temperature and the heat flux 
for a given viscosity law and set of parameters for convection.

======================================================================================
| IMPORTANT: the scaling approach is appropriate only for a real stagnant-lid regime |
======================================================================================

-----------
Inputs: Parameters must be given as a .xml file (input.xml given as an example).

Outputs: value of internal temperature and heat flux (Nusselt number).
-----------
Results are given in a dimensionless form and/or with dimensions.

Two forms of viscosity laws are implemented : 
Arrhenius et Frank-Kamenetskii approximation
------------------------------
If dimensionless = True is set, the input parameters that are used are :

- the Rayleigh number 'rayleigh' 
    (taken at the bottom, for dimensionless temperature T*=1)

- For the Arrhenius law: the dimensionless activation energy 'Ea' and
    the dimensionless temperature offset 'ToffAdim'

- For the FK approximation: the dimensionless FK parameter 'theta'
------------------------------
If dimensionless = False is set (i.e. WITH dimensions):

- the Rayleigh number is computed from the lines kappa to T_ref.

- the reference viscosity eta_ref at temperature T_ref is used to compute
    the viscosity at the bottom.

- For the Arrhenius law:
    Ea is taken in kJ/mol.
    ToffAdim in the computations will be used, and computed as Ttop / (Tbot-Ttop) 
    An additional offset of temperature ToffDim can be given
     (for instance, to have the equivalent, with dimensions, of
        ToffAdim=1, one should choose ToffDim so that Ttop + ToffDim = Tbot).

- For the FK approximation: theta (dimensionless) is used as the total
    viscosity contrast (etatop / etabot). 
----------------------------------- 
"""

import numpy as np
import xml.etree.ElementTree as ET


""" =================================== """


def get_params(elem, params):
    """ in the dict params, write the items defined
    by <param name='name' value='value'> """
    for child in elem:
        if child.tag == 'param':
            if 'name' in child.attrib and 'value' in child.attrib:
                params[child.attrib['name']] = string_to_type(child.attrib['value'])
    return


""" =================================== """


def isfloat(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError) as e:
        return False


""" =================================== """


def string_to_type(st):
    if st.lower() == 'true' or st.lower() == 'false':
        """ logical """
        return True if st.lower() == 'true' else False
    elif isfloat(st):
        """ float or int"""
        if "." in st or "E" in st or st[0:3] == 'inf':
            return float(st)
        else:
            return int(st)
    else:
        """ string """
        return st


""" =================================== """


def read_input_xml(filename):
    """ Read the xml file with the input parameters,
     and returns three dicts """
    root = ET.parse(filename).getroot()
    if root.tag != "Stagnant_Lid_Scaling":
        print("ERROR: " + filename + " not an input file for " + __file__)
        exit(0)

    physics = dict()
    for elem in root:
        if elem.tag == "Physics":
            get_params(elem, physics)

    scaling = dict()
    for elem in root:
        if elem.tag == "Scaling":
            get_params(elem, scaling)

    return physics, scaling


""" =================================== """


def predict_ti_arrhenius(scaling):
    """
    Predicts Ti from (Ti-Tc)/(1-Ti) = (eta(Tc)/eta(Ti))^beta.
    Using Newton's method. """

    """ First: compute all possible couples (Tc, Ti) """
    tcmin, tcmax = scaling['Tcmin'], 1.0

    Tc = np.arange(tcmin, tcmax + scaling['precision'], scaling['precision'])

    Ti = []
    for tc in Tc:
        ti = compute_ti_from_tc(tc, scaling)
        Ti.append(ti)

    Ti = np.asarray(Ti)

    """ Ti such that 1-Ti = coeffTi * max(1-Ti) """
    Timin = np.min(Ti)
    Tipred = 1.0 - scaling['coeffTi'] * (1.0 - Timin)

    return Tipred


""" =================================== """


def compute_ti_from_tc(tc, scaling):
    """
    Computes Ti such that (Ti-Tc)/(1-Ti) = (eta(Tc)/eta(Ti))^beta.
    Using Newton's method.
    """

    iloop = 100
    eps = scaling['error']

    ti = 1.0  # initial guess
    if law == "FK":
        for i in range(iloop + 1):
            buf = np.exp(beta * theta * (ti - tc))
            f = ti - tc - (1.0 - ti) * buf
            fp = 1.0 + buf * (1.0 - beta * theta * (1.0 - ti))

            ti -= f / fp
            if np.abs(f) < eps:
                break

            if i == iloop and np.abs(f) > eps:
                print("WARNING: no solution found in compute_ti_from_tc : \n"
                      "for tc = {:.3f} -> f = {:.3e} and ti = {:.3f}".format(tc, f, ti))

    elif law == "Arrhenius":
        for i in range(iloop + 1):
            buf = np.exp(beta * EaAdim * (1.0 / (tc + ToffAdim) - 1.0 / (ti + ToffAdim)))
            f = ti - tc - (1.0 - ti) * buf
            fp = 1.0 + buf * (1.0 - beta * EaAdim * (1.0 - ti) / (ti + ToffAdim)**2)

            ti -= f / fp
            if np.abs(f) < eps:
                break

            if i == iloop and np.abs(f) > eps:
                print("WARNING: no solution found in compute_ti_from_tc : \n"
                      "for tc = {:.3f} -> f = {:.3e} and ti = {:.3f}".format(tc, f, ti))
    return ti


""" ===================================== """


def compute_tc_from_ti(ti, scaling):
    """
    Computes Tc such that (Ti-Tc)/(1-Ti) = (eta(Tc)/eta(Ti))^beta.
    Using Newton's method.
    """

    iloop = 100
    eps = scaling['error']

    tc = 2.0 * ti - 1.0  # initial guess
    if law == "FK":
        for i in range(iloop + 1):
            buf = np.exp(beta * theta * (ti - tc))
            f = ti - tc - (1.0 - ti) * buf
            fp = -1.0 + buf * beta * theta * (1.0 - ti)

            tc -= f / fp
            if np.abs(f) < eps:
                break

            if i == iloop and np.abs(f) > eps:
                print("WARNING: no solution found in compute_tc_from_ti : \n"
                      "for ti = {:.3f} -> f = {:.3e} and tc = {:.3f}".format(ti, f, tc))

    elif law == "Arrhenius":
        for i in range(iloop + 1):
            buf = np.exp(beta * EaAdim * (1.0 / (tc + ToffAdim) - 1.0 / (ti + ToffAdim)))
            f = ti - tc - (1.0 - ti) * buf
            fp = -1.0 + buf * beta * EaAdim * (1.0 - ti) / (tc + ToffAdim)**2

            tc -= f / fp
            if np.abs(f) < eps:
                break

            if i == iloop and np.abs(f) > eps:
                print("WARNING: no solution found in compute_tc_from_ti : \n"
                      "for ti = {:.3f} -> f = {:.3e} and tc = {:.3f}".format(ti, f, tc))

    return tc


""" =================================== """


def predict_ti_tc(scaling):
    """ predict Ti and Tc from (Ti-Tc)/(1-Ti) = (eta(Tc)/eta(Ti))^0.25 """

    coeffTi = scaling['coeffTi']

    Tipred = np.nan
    if law == "Arrhenius":
        Tipred = predict_ti_arrhenius(scaling)
    elif law == "FK":
        Tipred = 1.0 - coeffTi * np.exp(-1.0) / (theta * beta)  # beta defined in constants

    Tcpred = compute_tc_from_ti(Tipred, scaling)

    return Tipred, Tcpred


""" =================================== """


def compute_tc_over_nu(ti, tc, etai, scaling):
    """
    Computes y = Tc/Nu, which verifies (Tc/y-1)^3 = b / (1-y + 1/(1-y)^3),
    with b = coeffNu^3 * Ra * (1-Ti)^4/etai
    Newton's method """

    iloop = 100
    eps = scaling['error']

    b = scaling['coeffNu']**3 * Rab * (1.0 - ti)**4 / etai

    y = 0.0001  # initial guess
    for i in range(iloop + 1):
        f = (tc / y - 1.0)**3 - b / (1.0 - y + 1.0 / (1.0 - y)**3)
        fp = -3.0 * tc / y**2 * (tc / y - 1.0)**2 - \
             b * (1.0 - 3.0 / (1.0 - y)**4) / (1.0 - y + 1.0 / (1.0 - y)**3)**2

        y -= f / fp

        if np.abs(f) < eps:
            break

        if i == iloop and np.abs(f) > eps:
            print("WARNING: no solution found in compute_tc_over_nu \n"
                  "for ti, tc = {:.3f}, {:.3f} -> f = {:.3e} and y = {:.3f}".format(ti, tc, f, y))

    return y


""" =================================== """


def compute_viscosity(t):
    """
    Computes dimensionless viscosity, with input being dimensionless
    parameters.
    Considers only temperatures such that T=0 at the surface
    and T=1 at the bottom, for eta=1 at the bottom.
    """

    eta = 1.0
    if law == "Arrhenius":
        eta = np.exp(EaAdim * (1.0 / (t + ToffAdim) - 1.0 / (1.0 + ToffAdim)))
    elif law == "FK":
        eta = np.exp(theta * (1.0 - t))

    return eta


""" =================================== """


def predict(scaling):
    """ predict the internal temperature Ti, the temperature
    below the stagnant lid Tc and the Nusselt number Nu.
    The dict() scaling has the parameters for the computation. """

    """ compute the internal temperature Ti and the temperature Tc
    below the stagnant lid"""
    Tipred, Tcpred = predict_ti_tc(scaling)

    """  internal viscosity, at temperature Ti"""
    etai = compute_viscosity(Tipred)

    """ compute the thickness l = Tc/Nu"""
    Tc_Nu = compute_tc_over_nu(Tipred, Tcpred, etai, scaling)

    """ Nusselt number: """
    Nupred = Tcpred / Tc_Nu

    return Tipred, Tcpred, Nupred


""" =================================== """


def compute_visco_param(physics):
    """
    computes Rab, EaAdim, ToffAdim (for Arrhenius) and theta (for FK)
    input: dict 'physics' with the useful parameters.
    """

    global Rab, law, DT, ToffAdim, EaAdim, theta, dimensionless

    dimensionless = physics['dimensionless']

    DT = physics['Tbot'] - physics['Ttop']

    """
     VISCOSITY PARAMETERS: 
    """
    EaAdim = 0.0
    theta = 0.0
    etatop, etabot = 1.0, 1.0
    gamma = 1.0

    law = "Arrhenius"
    if physics['viscosity'][0:3].lower() == "arr":
        """ 
        Arrhenius 
        """
        EaAdim = physics['Ea']
        if dimensionless:
            ToffAdim = physics['ToffAdim']
        else:
            EaAdim *= 1.0E3 / (gas_constant * DT)
            ToffAdim = (physics['Ttop'] + physics['ToffDim']) / DT
        gamma = EaAdim / (ToffAdim * (1.0 + ToffAdim))
    elif physics['viscosity'].lower() == "fk" or physics['viscosity'][0:5].lower() == "frank":
        """ 
        Frank-Kamenetskii
        """
        law = "FK"
        theta = physics['theta']
        gamma = theta
    else:
        print("ERROR: viscosity law = " + physics['viscosity'] + " not recognized")
        exit(1)

    """
     RAYLEIGH NUMBER (defined at the bottom, for T*=1)
    """
    if dimensionless:
        Rab = physics['rayleigh']
    else:
        if law == "Arrhenius":
            c = physics['Ea'] * 1.0E3 / gas_constant
            etatop = physics['eta_ref'] * np.exp(c * (1.0 / physics['Ttop'] - 1.0 / physics['T_ref']))
            etabot = physics['eta_ref'] * np.exp(c * (1.0 / physics['Tbot'] - 1.0 / physics['T_ref']))
        elif law == "FK":
            etabot = physics['eta_ref'] * np.exp(-theta * (physics['Tbot'] - physics['T_ref']) / DT)
            etatop = etabot * np.exp(theta)

        coeffRa = physics['alpha'] * physics['rho'] * physics['g'] * DT * physics['D']**3 / physics['kappa']
        Rab = coeffRa / etabot

    """ OUTPUT """
    print("=" * 40)
    print("Viscosity law = ", law)
    if law == "Arrhenius":
        if not dimensionless:
            print("\tTbot, Ttop = {:.2f} K, {:.2f} K; DT = {:.2f} K".format(physics['Ttop'], physics['Tbot'], DT))
            print("\ttop and bottom viscosities: {:.4e} Pa.s and {:.4e} Pa.s".format(etatop, etabot))
            print("\tEa = {:.4f} kJ/mol".format(physics["Ea"]))
        print("\tEaAdim =   {:.4f}".format(EaAdim))
        print("\tToffAdim = {:.4f}".format(ToffAdim))
        print("\tTotal viscosity contrast (gamma = log(etatop/etabot)) = {:.5f}".format(gamma))
    else:
        if not dimensionless:
            print("\tTbot, Ttop = {:.2f} K, {:.2f} K; DT = {:.2f} K".format(physics['Ttop'], physics['Tbot'], DT))
            print("\ttop and bottom viscosities: {:.4e} Pa.s and {:.4e} Pa.s".format(etatop, etabot))

        print("\ttheta = {:.4f}".format(theta))
    print("=" * 40)
    print("Bottom Rayleigh number = {:.5e}".format(Rab))
    print("=" * 40)

    return


""" =================================== """


def compute_temperature_nusselt(physics, scaling):
    """
    computes the internal temperature, the stagnant lid's
    thickness and the heat flux (Nusselt number).
    scaling and io are dict() with the input parameters.
    """

    Ti, Tc, Nu = predict(scaling)

    print("="*40)
    print("SOLUTION: ")

    """ temperature """
    print("\tInternal temperature Ti* = {:.5f}".format(Ti))
    if not dimensionless:
        TiDim = Ti * DT + physics['Ttop']
        print("\t                     Ti = {:.2f} K".format(TiDim))

    """ heat flux """
    print("\tNusselt number Nu = {:.5f}".format(Nu))
    if not dimensionless:
        q = physics['k'] * DT / physics['D'] * Nu * 1.0E3  # in mW/m^2
        print("\tHeat flux       q = {:.5f} mW/m^2".format(q))
    print("=" * 40)

    """ others """
    print("\tTemperature below stagnant lid Tc* = {:.5f}".format(Tc))
    if not dimensionless:
        TcDim = Tc * DT + physics['Ttop']
        print("\t                     Tc = {:.2f} K".format(TcDim))

    l = Tc / Nu
    print("\tStagnant lid thickness l*={:.5f}".format(l))
    if not dimensionless:
        lDim = l * physics["D"] / 1.0E3
        print("\t                     l = {:.2f} km".format(lDim))
        
    etai = compute_viscosity(Ti)
    etac = compute_viscosity(Tc)
    print("\tViscosity(Ti)* = {:.5e}\n\tViscosity(Tc)* = {:.5e}".format(etai, etac))
    print("=" * 40) 
    
    return

""" =================================== """


def main():
    """ see comments at the top of the file """

    """ Read the input xml file """
    physics, scaling = read_input_xml("input_scaling.xml")

    """ viscosity and convection settings """
    global Rab, law, EaAdim, ToffAdim, theta, DT, dimensionless
    compute_visco_param(physics)  # defines the parameters Rab, EaAdim... given above as global

    """ COMPUTATION """
    compute_temperature_nusselt(physics, scaling)

    return


""" =================================== """

""" CONSTANT VALUES """
gas_constant = 8.314  # J/(K.mol)
beta = 0.25  # exponent in (Ti-Tc)/(1-Ti) = (eta(Tc)/eta(Ti))^beta -> should be 1/4

if __name__ == "__main__":
    main()
