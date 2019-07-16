Copyright 2019 CERN. This software is distributed under the terms of the
GNU General Public Licence version 3 (GPL Version 3), copied verbatim in
the file LICENCE.txt. In applying this licence, CERN does not waive the
privileges and immunities granted to it by virtue of its status as an
Intergovernmental Organization or submit itself to any jurisdiction.

CODE NAME
=========

BLonD_common (Beam Longitudinal Dynamics - Common dependencies)

DESCRIPTION
===========

Collection of common functions and interfaces for codes dedicated to longitudinal
beam dynamics studies in synchrotrons

LINKS
=====

Repository: https://github.com/blond-org/blond_common

Documentation: To be included

Project website: http://blond.web.cern.ch

INSTALL
=======


Requirements
------------

1. An Anaconda distribution (Python 3 recommended).

2. That's all!


Install Steps
-------------


* The blond_common module is initially meant to be used as a submodule to other code developments. If you would like to add the blond_common module into your own development, you need to:
    .. code-block:: bash

        $ git submodule add https://github.com/blond-org/blond_common.git your_project

* If the blond_common module is included as a submodule into another project, remember to initialize the module using (NB: this will update the submodule to the version defined by the project):
    .. code-block:: bash

        $ git submodule init
        $ git submodule update

* The code can also be used standalone (e.g. to use fitting functions)

  1. Clone the repository from github or download and extract the zip from here.

  2. Add the module to your PYTHONPATH


CURRENT DEVELOPERS
==================

* Simon Albright (simon.albright (at) cern.ch)
* Theodoros Argyropoulos (theodoros.argyropoulos (at) cern.ch)
* Konstantinos Iliakis (konstantinos.iliakis (at) cern.ch)
* Ivan Karpov (ivan.karpov (at) cern.ch)
* Alexandre Lasheen (alexandre.lasheen (at) cern.ch)
* Helga Timko (Helga.Timko (at) cern.ch)

PREVIOUS DEVELOPERS
===================

Juan Esteban Muller
Danilo Quartullo
Joel Repond


STRUCTURE
=========

To be described


VERSION CONTENTS
================

No stable version at the moment
