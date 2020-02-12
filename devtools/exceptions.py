# coding: utf8
# Copyright 2014-2019 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module containing customized exceptions**

:Authors: **Simon Albright**
'''


# =======================
# BLonD Common exceptions
# =======================

class BLonD_Exception(Exception):
    pass


class InputError(BLonD_Exception):
    pass


class DataDefinitionError(BLonD_Exception):
    pass


# ===============
# Beam Exceptions
# ===============

class MassError(BLonD_Exception):
    pass


class AllParticlesLost(BLonD_Exception):
    pass


class ParticleAdditionError(BLonD_Exception):
    pass


# =================
# Bucket Exceptions
# =================

class BunchSizeError(BLonD_Exception, ValueError):
    pass


# ==================================
# Distribution Generation Exceptions
# ==================================

class DistributionError(BLonD_Exception):
    pass


class GenerationError(BLonD_Exception):
    pass


# ==================
# Profile Exceptions
# ==================

class CutError(BLonD_Exception):
    pass


class ProfileDerivativeError(BLonD_Exception):
    pass


# ====================
# Impedance Exceptions
# ====================

class WakeLengthError(BLonD_Exception):
    pass


class FrequencyResolutionError(BLonD_Exception):
    pass


class ResonatorError(BLonD_Exception):
    pass


class WrongCalcError(BLonD_Exception):
    pass


class MissingParameterError(BLonD_Exception):
    pass


# ===========================
# Input Parameters Exceptions
# ===========================

class MomentumError(BLonD_Exception):
    pass


# ===============
# LLRF Exceptions
# ===============

class PhaseLoopError(BLonD_Exception):
    pass


class PhaseNoiseError(BLonD_Exception):
    pass


class FeedbackError(BLonD_Exception):
    pass


class ImpulseError(BLonD_Exception):
    pass


# ==================
# Toolbox Exceptions
# ==================

class PhaseSpaceError(BLonD_Exception):
    pass


class NoiseDiffusionError(BLonD_Exception):
    pass


# ==================
# Tracker Exceptions
# ==================

class PotentialWellError(BLonD_Exception):
    pass


class SolverError(BLonD_Exception):
    pass


class PeriodicityError(BLonD_Exception):
    pass


class ProfileError(BLonD_Exception):
    pass


class SynchrotronMotionError(BLonD_Exception):
    pass


# ===============
# Util Exceptions
# ===============

class ConvolutionError(BLonD_Exception):
    pass


class IntegrationError(BLonD_Exception):
    pass


class SortError(BLonD_Exception):
    pass


# =================
# Global Exceptions
# =================

class InterpolationError(BLonD_Exception):
    pass


class InputDataError(BLonD_Exception):
    pass
