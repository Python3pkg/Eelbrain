.. Building: $ make html
   If you get ValueError: unknown locale: UTF-8, run 
   export LC_ALL=en_US.UTF-8 
   before make (http://readthedocs.org/docs/flightdeck/en/latest/installation.html).

Eelbrain
========

.. image:: https://zenodo.org/badge/3651023.svg
   :target: https://zenodo.org/badge/latestdoi/3651023

Eelbrain is an open-source `Python <https://www.python.org>`_ package for
accessible statistical analysis of MEG and EEG data.
It is maintained by
`Christian Brodbeck <http://loop.frontiersin.org/people/120185>`_
at the `Computational sensorimotor systems lab
<http://www.isr.umd.edu/Labs/CSSL/simonlab/Home.html>`_
at University of Maryland, College Park.

If you use Eelbrain in work that is published, please acknowledge it by citing
it with the appropriate DOI for the version you used.


Manual
------

.. toctree::
   :maxdepth: 1

   installing
   getting_started
   intro
   changes
   development

.. toctree::
   :maxdepth: 2

   reference
   recipes
   experiment


.. seealso:: Eelbrain on `GitHub <http://github.com/christianbrodbeck/Eelbrain>`_,
    the `Python Package Index <https://pypi.python.org/pypi/eelbrain>`_,
    and `example scripts
    <https://github.com/christianbrodbeck/Eelbrain/tree/master/examples>`_

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Eelbrain relies on
`NumPy <http://www.numpy.org>`_,
`SciPy <http://scipy.org>`_,
`Matplotlib <http://matplotlib.org>`_,
`MNE-Python <http://martinos.org/mne/stable/index.html>`_,
`PySurfer <http://pysurfer.github.io>`_,
`WxPython <http://wxpython.org>`_ and
`Cython <http://cython.org>`_.


----

Current funding: `National Institutes of Health (NIH) <https://www.nih.gov>`_
grant R01-DC-014085 (since 2016).
Past funding: `NYU Abu Dhabi Institute
<http://nyuad.nyu.edu/en/research/nyuad-institute.html>`_ grant G1001
(2011-2016).
