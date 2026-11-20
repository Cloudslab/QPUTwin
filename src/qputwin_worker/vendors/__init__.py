"""One module per emulated QPU; each provides SPEC, SIMULATOR_CLASS and app.

Usage: from qputwin_worker.vendors import VENDORS

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

VENDORS = ("quantinuum_helios", "ionq_forte", "rigetti_ankaa_3", "iqm_emerald", "ibm_heron", "ibm_nighthawk",
           "google_willow")
