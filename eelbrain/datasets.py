"""Some basic example datasets for testing."""
import os

import mne
from mne import minimum_norm as mn
import numpy as np

from . import load
from ._colorspaces import eeg_info
from ._data_obj import Dataset, Factor, Var, NDVar, Sensor, UTS, Ordered
from ._design import permute


def _apply_kernel(x, h, out=None):
    """Predict ``y`` by applying kernel ``h`` to ``x``

    x.shape is (n_stims, n_samples)
    h.shape is (n_stims, n_trf_samples)
    """
    if out is None:
        out = np.zeros(x.shape[1])
    else:
        out.fill(0)

    for ind in range(len(h)):
        out += np.convolve(h[ind], x[ind])[:len(out)]

    return out


def _get_continuous(n_samples=100, seed=0):
    """Generate continuous data for reverse correlation

    Parameters
    ----------
    n_samples : int
        Number of samples to simulate.
    seed : int
        Seed for :func:`numpy.random.seed` (``None`` to skip seeding random
        state; default us 0).

    Returns
    -------
    data : dict
        {:class:`str`: :class:`NDVar`}`` dictionary with simulated data:

         - ``x1``: random time series
         - ``x2``: two random time series
         - ``h1`` and ``h2``: Kernels corresponding to ``x1`` and ``x2``
         - ``y``: convolution of ``(x1 * h1) + (x2 * h2)``
    """
    if seed is not None:
        np.random.seed(seed)
    time = UTS(0, 0.1, n_samples)
    h_time = UTS(0, 0.1, 10)
    xdim = Ordered('xdim', [0, 1])

    x1 = NDVar(np.random.normal(0, 1, (n_samples,)), (time,), name='x1')
    h1 = NDVar(np.array([0, 0, 1, 3, 0, 0, 0, 0, 2, 3]), (h_time,), name='h1')

    x2 = NDVar(np.random.normal(0, 1, (2, n_samples,)),
               (xdim, time), name='x2')
    h2 = NDVar(np.array([[0, 0, 0, 0, 0, 0, -1, -3, 0, 0],
                         [0, 0, 2, 2, 0, 0, 0, 0, 0, 0]]),
               (xdim, h_time), name='h2')

    y = _apply_kernel(x1.x[np.newaxis], h1.x[np.newaxis])
    y += _apply_kernel(x2.x, h2.x)
    y = NDVar(y, (time,), name='y')
    return {'y': y, 'x1': x1, 'h1': h1, 'x2': x2, 'h2': h2}


def get_loftus_masson_1994():
    "Dataset used for illustration purposes by Loftus and Masson (1994)"
    ds = Dataset()
    ds['subject'] = Factor(list(range(1, 11)), tile=3, random=True)
    ds['exposure'] = Var([1, 2, 5], repeat=10)
    ds['n_recalled'] = Var([10, 6, 11, 22, 16, 15, 1, 12, 9, 8,
                            13, 8, 14, 23, 18, 17, 1, 15, 12, 9,
                            13, 8, 14, 25, 20, 17, 4, 17, 12, 12])
    return ds


def get_mne_epochs():
    """MNE-Python Epochs"""
    data_path = mne.datasets.sample.data_path()
    raw_path = os.path.join(data_path, 'MEG', 'sample',
                            'sample_audvis_raw.fif')
    events_path = os.path.join(data_path, 'MEG', 'sample',
                               'sample_audvis_raw-eve.fif')
    raw = mne.io.Raw(raw_path, add_eeg_ref=False)
    events = mne.read_events(events_path)
    epochs = mne.Epochs(raw, events, 32, -0.1, 0.4, preload=True,
                        add_eeg_ref=False)
    return epochs


def get_mne_evoked(ndvar=False):
    """MNE-Python Evoked

    Parameters
    ----------
    ndvar : bool
        Convert to NDVar (default False).
    """
    data_path = mne.datasets.sample.data_path()
    evoked_path = os.path.join(data_path, 'MEG', 'sample',
                               'sample_audvis-ave.fif')
    evoked = mne.Evoked(evoked_path, "Left Auditory")
    if ndvar:
        return load.fiff.evoked_ndvar(evoked)
    else:
        return evoked


def get_mne_stc(ndvar=False):
    """MNE-Python SourceEstimate

    Parameters
    ----------
    ndvar : bool
        Convert to NDVar (default False; src="ico-4" is false, but it works as
        long as the source space is not accessed).
    """
    data_path = mne.datasets.testing.data_path()
    stc_path = os.path.join(data_path, 'MEG', 'sample', 'fsaverage_audvis_trunc-meg')
    if not ndvar:
        return mne.read_source_estimate(stc_path, 'sample')
    subjects_dir = os.path.join(data_path, 'subjects')
    return load.fiff.stc_ndvar(stc_path, 'fsaverage', 'ico-5', subjects_dir)


def _mne_source_space(subject, src_tag, subjects_dir):
    """Load mne source space

    Parameters
    ----------
    subject : str
        Subejct
    src_tag : str
        Spacing (e.g., 'ico-4').
    """
    src_file = os.path.join(subjects_dir, subject, 'bem',
                            '%s-%s-src.fif' % (subject, src_tag))
    src, spacing = src_tag.split('-')
    if os.path.exists(src_file):
        return mne.read_source_spaces(src_file, False)
    elif src == 'ico':
        return mne.setup_source_space(subject, src_file, src + spacing,
                                      subjects_dir=subjects_dir, add_dist=True)
    elif src == 'vol':
        mri_file = os.path.join(subjects_dir, subject, 'mri', 'orig.mgz')
        bem_file = os.path.join(subjects_dir, subject, 'bem',
                                'sample-5120-5120-5120-bem-sol.fif')
        return mne.setup_volume_source_space(subject, src_file, float(spacing),
                                             mri=mri_file, bem=bem_file,
                                             mindist=0., exclude=0.,
                                             subjects_dir=subjects_dir)
    else:
        raise ValueError("src_tag=%s" % repr(src_tag))


def get_mne_sample(tmin=-0.1, tmax=0.4, baseline=(None, 0), sns=False,
                   src=None, sub="modality=='A'", fixed=False, snr=2,
                   method='dSPM', rm=False, stc=False):
    """Load events and epochs from the MNE sample data

    Parameters
    ----------
    tmin : scalar
        Relative time of the first sample of the epoch.
    tmax : scalar
        Relative time of the last sample of the epoch.
    baseline : {None, tuple of 2 {scalar, None}}
        Period for baseline correction.
    sns : bool
        Add sensor space data as NDVar as ``ds['meg']`` (default ``False``).
    src : False | 'ico' | 'vol'
        Add source space data as NDVar as ``ds['src']`` (default ``False``).
    sub : str | list | None
        Expresion for subset of events to load. For a very small dataset use e.g.
        ``[0,1]``.
    fixed : bool
        MNE inverse parameter.
    snr : scalar
        MNE inverse parameter.
    method : str
        MNE inverse parameter.
    rm : bool
        Pretend to be a repeated measures dataset (adds 'subject' variable).
    stc : bool
        Add mne SourceEstimate for source space data as ``ds['stc']`` (default
        ``False``).

    Returns
    -------
    ds : Dataset
        Dataset with epochs from the MNE sample dataset in ``ds['epochs']``.
    """
    data_dir = mne.datasets.sample.data_path()
    meg_dir = os.path.join(data_dir, 'MEG', 'sample')
    raw_file = os.path.join(meg_dir, 'sample_audvis_filt-0-40_raw.fif')
    event_file = os.path.join(meg_dir, 'sample_audvis_filt-0-40-eve.fif')
    subjects_dir = os.path.join(data_dir, 'subjects')
    subject = 'sample'
    label_path = os.path.join(subjects_dir, subject, 'label', '%s.label')

    if not os.path.exists(event_file):
        raw = mne.io.Raw(raw_file)
        events = mne.find_events(raw, stim_channel='STI 014')
        mne.write_events(event_file, events)
    ds = load.fiff.events(raw_file, events=event_file)
    ds.index()
    ds.info['subjects_dir'] = subjects_dir
    ds.info['subject'] = subject
    ds.info['label'] = label_path

    # get the trigger variable form the dataset for eaier access
    trigger = ds['trigger']

    # use trigger to add various labels to the dataset
    ds['condition'] = Factor(trigger, labels={
        1: 'LA', 2: 'RA', 3: 'LV', 4: 'RV', 5: 'smiley', 32: 'button'})
    ds['side'] = Factor(trigger, labels={
        1: 'L', 2: 'R', 3: 'L', 4: 'R', 5: 'None', 32: 'None'})
    ds['modality'] = Factor(trigger, labels={
        1: 'A', 2: 'A', 3: 'V', 4: 'V', 5: 'None', 32: 'None'})

    if rm:
        ds = ds.sub('trigger < 5')
        ds = ds.equalize_counts('side % modality')
        subject_f = ds.eval('side % modality').enumerate_cells()
        ds['subject'] = subject_f.as_factor('s%r', random=True)

    if sub:
        ds = ds.sub(sub)

    load.fiff.add_mne_epochs(ds, tmin, tmax, baseline)
    if sns:
        ds['meg'] = load.fiff.epochs_ndvar(ds['epochs'], data='mag',
                                           sysname='neuromag306mag')

    if not src:
        return ds
    elif src == 'ico':
        src_tag = 'ico-4'
    elif src == 'vol':
        src_tag = 'vol-10'
    else:
        raise ValueError("src = %r" % src)
    epochs = ds['epochs']

    # get inverse operator
    inv_file = os.path.join(meg_dir, 'sample_eelbrain_%s-inv.fif' % src_tag)
    if os.path.exists(inv_file):
        inv = mne.minimum_norm.read_inverse_operator(inv_file)
    else:
        fwd_file = os.path.join(meg_dir, 'sample-%s-fwd.fif' % src_tag)
        bem_dir = os.path.join(subjects_dir, subject, 'bem')
        bem_file = os.path.join(bem_dir, 'sample-5120-5120-5120-bem-sol.fif')
        trans_file = os.path.join(meg_dir, 'sample_audvis_raw-trans.fif')

        if os.path.exists(fwd_file):
            fwd = mne.read_forward_solution(fwd_file)
        else:
            src_ = _mne_source_space(subject, src_tag, subjects_dir)
            fwd = mne.make_forward_solution(epochs.info, trans_file, src_,
                                            bem_file, fwd_file)

        cov_file = os.path.join(meg_dir, 'sample_audvis-cov.fif')
        cov = mne.read_cov(cov_file)
        inv = mn.make_inverse_operator(epochs.info, fwd, cov, None, None,
                                       fixed)
        mne.minimum_norm.write_inverse_operator(inv_file, inv)
    ds.info['inv'] = inv

    stcs = mn.apply_inverse_epochs(epochs, inv, 1. / (snr ** 2), method)
    ds['src'] = load.fiff.stc_ndvar(stcs, subject, src_tag, subjects_dir,
                                    method, fixed)
    if stc:
        ds['stc'] = stcs

    return ds


def get_uts(utsnd=False, seed=0, nrm=False):
    """Create a sample Dataset with 60 cases and random data.

    Parameters
    ----------
    utsnd : bool
        Add a sensor by time NDVar (called 'utsnd').
    seed : None | int
        If not None, call ``numpy.random.seed(seed)`` to ensure replicability.
    nrm : bool
        Create nested random effect Factor "nrm".

    Returns
    -------
    ds : Dataset
        Datasets with data from random distributions.
    """
    if seed is not None:
        np.random.seed(seed)

    ds = Dataset()

    # add a model
    ds['A'] = Factor(['a0', 'a1'], repeat=30)
    ds['B'] = Factor(['b0', 'b1'], repeat=15, tile=2)
    ds['rm'] = Factor(('R%.2i' % i for i in range(15)), tile=4, random=True)
    ds['ind'] = Factor(('R%.2i' % i for i in range(60)), random=True)

    # add dependent variables
    rm_var = np.tile(np.random.normal(size=15), 4)
    y = np.hstack((np.random.normal(size=45), np.random.normal(1, size=15)))
    y += rm_var
    ds['Y'] = Var(y)
    ybin = np.random.randint(0, 2, size=60)
    ds['YBin'] = Factor(ybin, labels={0: 'c1', 1: 'c2'})
    ycat = np.random.randint(0, 3, size=60)
    ds['YCat'] = Factor(ycat, labels={0: 'c1', 1: 'c2', 2: 'c3'})

    # add a uts NDVar
    time = UTS(-.2, .01, 100)
    y = np.random.normal(0, .5, (60, len(time)))
    y += rm_var[:, None]
    y[:15, 20:60] += np.hanning(40) * 1  # interaction
    y[:30, 50:80] += np.hanning(30) * 1  # main effect
    ds['uts'] = NDVar(y, dims=('case', time))

    # add sensor NDVar
    if utsnd:
        locs = np.array([[-1.0,  0.0, 0.0],
                         [ 0.0,  1.0, 0.0],
                         [ 1.0,  0.0, 0.0],
                         [ 0.0, -1.0, 0.0],
                         [ 0.0,  0.0, 1.0]])
        sensor = Sensor(locs, sysname='test_sens')
        sensor.set_connectivity(connect_dist=1.75)

        y = np.random.normal(0, 1, (60, 5, len(time)))
        y += rm_var[:, None, None]
        # add interaction
        win = np.hanning(50)
        y[:15, 0, 50:] += win * 3
        y[:15, 1, 50:] += win * 2
        y[:15, 4, 50:] += win
        # add main effect
        y[30:, 2, 25:75] += win * 2.5
        y[30:, 3, 25:75] += win * 1.5
        y[30:, 4, 25:75] += win
        # add spectral effect
        freq = 15.0  # >= 2
        x = np.sin(time.times * freq * 2 * np.pi)
        for i in range(30):
            shift = np.random.randint(0, 100 / freq)
            y[i, 2, 25:75] += 1.1 * win * x[shift: 50 + shift]
            y[i, 3, 25:75] += 1.5 * win * x[shift: 50 + shift]
            y[i, 4, 25:75] += 0.5 * win * x[shift: 50 + shift]

        dims = ('case', sensor, time)
        ds['utsnd'] = NDVar(y, dims, eeg_info())

    # nested random effect
    if nrm:
        ds['nrm'] = Factor([a + '%02i' % i for a in 'AB' for _ in range(2) for
                            i in range(15)], random=True)

    return ds


def get_uv(seed=0, nrm=False):
    """Dataset with random univariate data

    Parameters
    ----------
    seed : None | int
        Seed the numpy random state before generating random data.
    nrm : bool
        Add a nested random-effects variable (default False).
    """
    if seed is not None:
        np.random.seed(seed)

    ds = permute([('A', ('a1', 'a2')),
                  ('B', ('b1', 'b2')),
                  ('rm', ['s%03i' % i for i in range(20)])])
    ds['rm'].random = True
    ds['intvar'] = Var(np.random.randint(5, 15, 80))
    ds['intvar'][:20] += 3
    ds['fltvar'] = Var(np.random.normal(0, 1, 80))
    ds['fltvar'][:40] += 1.
    ds['fltvar2'] = Var(np.random.normal(0, 1, 80))
    ds['fltvar2'][40:] += ds['fltvar'][40:].x
    ds['index'] = Var(np.repeat([True, False], 40))
    if nrm:
        ds['nrm'] = Factor(['s%03i' % i for i in range(40)], tile=2, random=True)
    return ds


def setup_samples_experiment(dst, n_subjects=3, n_segments=4, n_sessions=1):
    """Setup up file structure for the SampleExperiment class

    Parameters
    ----------
    dst : str
        Path. ``dst`` should exist, a new folder called ``SampleExperiment``
        will be created within ``dst``.
    n_subjects : int
        Number of subjects.
    n_segments : int
        Number of data segments to include in each file.
    n_sessions : int
        Number of sessions.
    """
    data_path = mne.datasets.sample.data_path()
    raw_path = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
    raw = mne.io.read_raw_fif(raw_path, add_eeg_ref=False)
    raw.info['bads'] = []
    sfreq = raw.info['sfreq']

    # find segmentation points
    events = mne.find_events(raw)
    events[:, 0] -= raw.first_samp
    segs = []
    n = 0
    t_start = 0
    for sample, _, trigger in events:
        if trigger == 5:  # smiley
            n += 1
        if n == n_segments:
            t = sample / sfreq
            segs.append((t_start, t))
            if len(segs) == n_subjects * n_sessions:
                break
            t_start = t
            n = 0
    else:
        raise ValueError("Not enough data in sample raw. Try smaller ns.")
    dst = os.path.realpath(os.path.expanduser(dst))
    root = os.path.join(dst, 'SampleExperiment')
    meg_sdir = os.path.join(root, 'meg')
    meg_dir = os.path.join(meg_sdir, '{subject}')
    raw_file = os.path.join(meg_dir, '{subject}_{session}-raw.fif')

    os.mkdir(root)
    os.mkdir(meg_sdir)

    if n_sessions == 1:
        sessions = ['sample']
    else:
        sessions = ['sample%i' % (i + 1) for i in range(n_sessions)]

    for s_id in range(n_subjects):
        subject = 'R%04i' % s_id
        os.mkdir(meg_dir.format(subject=subject))
        for session in sessions:
            start, stop = segs.pop()
            raw_ = raw.copy().crop(start, stop)
            raw_.load_data()
            raw_.pick_types('mag', stim=True, exclude=[])
            raw_.save(raw_file.format(subject=subject, session=session))
