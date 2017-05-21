# dataset: mne_sample
"""
This example demonstrates how to define a function to load data form a specific
experiment.

Since there is usually just one event structure per experiment, it makes
sense to write a function for labeling events that can be re-used in
different analyses. Such a function is defined in this script. In order to use
it from another within another script, this file has to be in the same folder
as the other script. if this is the case, the function can be used like::

    >>> import mne_sample_loader
    >>> path = "path/to/my-raw.fif"
    >>> ds = mne_sample_laoder.load_evts(path)

"""

import eelbrain as eel


def load_evts(path):
    """Load events from the mne sample data as dataset

    Parameters
    ----------
    path : str
        Path to the raw file.

    Returns
    -------
    ds : dataset
        Events from the raw file as dataset.
    """
    # load the events in the raw file as a dataset
    ds = eel.load.fiff.events(path, stim_channel='STI 014')

    # get the trigger variable form the dataset for eaier access
    trigger = ds['trigger']

    # use trigger to add various labels to the dataset
    ds['condition'] = eel.Factor(trigger, labels={1:'LA', 2:'RA', 3:'LV', 4:'RV',
                                                  5:'smiley', 32:'button'})
    ds['side'] = eel.Factor(trigger, labels={1: 'L', 2:'R', 3:'L', 4:'R',
                                             5:'None', 32:'None'})
    ds['modality'] = eel.Factor(trigger, labels={1: 'A', 2:'A', 3:'V', 4:'V',
                                                 5:'None', 32:'None'})

    return ds


if __name__ == '__main__':
    # Use the function to load the events and plot data from a specific condition
    import os
    import mne
    datapath = mne.datasets.sample.data_path()
    raw_path = os.path.join(datapath, 'MEG', 'sample',
                            'sample_audvis_filt-0-40_raw.fif')

    ds = load_evts(raw_path)
    print(eel.table.frequencies('condition', ds=ds))
    ds = ds.sub('modality == "A"')

    ds = eel.load.fiff.add_epochs(ds, tmin=-0.1, tmax=0.3, baseline=(None, 0),
                                  proj=False, data='mag', reject=2e-12,
                                  name='meg', sysname='neuromag306mag')

    p = plot.TopoButterfly('meg', 'side', ds=ds)
    p.set_vlim(1e-12)
