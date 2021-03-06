# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Pre-processing operations based on NDVars"""
from os import mkdir, remove
from os.path import dirname, exists, getmtime

import mne
from mne.io import read_raw_fif

from .. import load
from .._data_obj import NDVar
from ..mne_fixes import CaptureLog


class RawPipe(object):

    def __init__(self, name, path, log):
        self.name = name
        self.path = path
        self.log = log

    def as_dict(self):
        return {'type': self.__class__.__name__, 'name': self.name}

    def load(self, subject, session, add_bads=True, preload=False):
        path = self.path.format(subject=subject, session=session)
        raw = read_raw_fif(path, preload=preload)
        if add_bads:
            raw.info['bads'] = self.load_bad_channels(subject, session)
        else:
            raw.info['bads'] = []
        return raw

    def load_bad_channels(self, subject, session):
        raise NotImplementedError

    def make_bad_channels(self, subject, session, bad_chs, redo):
        raise NotImplementedError

    def make_bad_channels_auto(self, subject, session, flat):
        raise NotImplementedError

    def mtime(self, subject, session, bad_chs=True):
        "Modification time of anything influencing the output of load"
        raise NotImplementedError


class RawSource(RawPipe):
    "Raw data source"
    def __init__(self, name, path, bads_path, log):
        RawPipe.__init__(self, name, path, log)
        self.bads_path = bads_path

    def load_bad_channels(self, subject, session):
        path = self.bads_path.format(subject=subject, session=session)
        if not exists(path):
            # need to create one to know mtime after user deletes the file
            self.log.info("Generating bad_channels file for %s %s",
                          subject, session)
            self.make_bad_channels_auto(subject, session)
        with open(path) as fid:
            return [l for l in fid.read().splitlines() if l]

    def make_bad_channels(self, subject, session, bad_chs, redo):
        path = self.bads_path.format(subject=subject, session=session)
        if exists(path):
            old_bads = self.load_bad_channels(subject, session)
        else:
            old_bads = None
        # find new bad channels
        if isinstance(bad_chs, (str, int)):
            bad_chs = (bad_chs,)
        raw = self.load(subject, session, add_bads=False)
        sensor = load.fiff.sensor_dim(raw)
        new_bads = sensor._normalize_sensor_names(bad_chs)
        # update with old bad channels
        if old_bads is not None and not redo:
            new_bads = sorted(set(old_bads).union(new_bads))
        # print change
        if old_bads is None:
            print(("-> %s" % new_bads))
        else:
            print(("%s -> %s" % (old_bads, new_bads)))
        # write new bad channels
        text = '\n'.join(new_bads)
        with open(path, 'w') as fid:
            fid.write(text)

    def make_bad_channels_auto(self, subject, session, flat=1e-14):
        if not flat:
            return
        raw = self.load(subject, session, preload=True, add_bads=False)
        raw = load.fiff.raw_ndvar(raw)
        bad_chs = raw.sensor.names[raw.std('time') < flat]
        self.make_bad_channels(subject, session, bad_chs, False)

    def mtime(self, subject, session, bad_chs=True):
        path = self.path.format(subject=subject, session=session)
        if exists(path):
            mtime = getmtime(path)
            if not bad_chs:
                return mtime
            path = self.bads_path.format(subject=subject, session=session)
            if exists(path):
                return max(mtime, getmtime(path))


class CachedRawPipe(RawPipe):

    _bad_chs_affect_cache = False

    def __init__(self, name, source, path, log):
        assert isinstance(source, RawPipe)
        path = path.format(raw=name, subject='{subject}', session='{session}')
        RawPipe.__init__(self, name, path, log)
        self.source = source

    def as_dict(self):
        out = RawPipe.as_dict(self)
        out['source'] = self.source.name
        return out

    def cache(self, subject, session):
        "Make sure the cache is up to date"
        path = self.path.format(subject=subject, session=session)
        if (not exists(path) or getmtime(path) <
                self.mtime(subject, session, self._bad_chs_affect_cache)):
            dir_path = dirname(path)
            if not exists(dir_path):
                mkdir(dir_path)
            with CaptureLog(path[:-3] + 'log'):
                raw = self._make(subject, session)
            raw.save(path, overwrite=True)

    def load(self, subject, session, add_bads=True, preload=False):
        self.cache(subject, session)
        return RawPipe.load(self, subject, session, add_bads, preload)

    def load_bad_channels(self, subject, session):
        return self.source.load_bad_channels(subject, session)

    def _make(self, subject, session):
        raise NotImplementedError

    def make_bad_channels(self, subject, session, bad_chs, redo):
        self.source.make_bad_channels(subject, session, bad_chs, redo)

    def make_bad_channels_auto(self, *args, **kwargs):
        self.source.make_bad_channels_auto(*args, **kwargs)

    def mtime(self, subject, session, bad_chs=True):
        return self.source.mtime(subject, session, bad_chs)


class RawFilter(CachedRawPipe):

    def __init__(self, name, source, path, log, args, kwargs):
        CachedRawPipe.__init__(self, name, source, path, log)
        self.args = args
        self.kwargs = kwargs

    def as_dict(self):
        out = CachedRawPipe.as_dict(self)
        out['args'] = self.args
        out['kwargs'] = self.kwargs
        return out

    def filter_ndvar(self, ndvar):
        axis = ndvar.get_axis('time')
        sfreq = 1. / ndvar.time.tstep
        x = ndvar.x.swapaxes(axis, 0) if axis else ndvar.x
        x = mne.filter.filter_data(x, sfreq, *self.args, **self.kwargs)
        if axis:
            x = x.swapaxes(axis, 0)
        return NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)

    def _make(self, subject, session):
        raw = self.source.load(subject, session, preload=True)
        self.log.debug("Raw %s: filtering for %s/%s...", self.name, subject,
                       session)
        raw.filter(*self.args, **self.kwargs)
        return raw


class RawICA(CachedRawPipe):
    """ICA raw pipe

    Notes
    -----
    This pipe merges bad channels from its source raw pipes.

    When checking whether the ICA file is up to date, the ICA does not check
    raw source mtime. However, if bad channels change the ICA is automatically
    recomputed.
    """

    def __init__(self, name, source, path, ica_path, log, session, kwargs):
        CachedRawPipe.__init__(self, name, source, path, log)
        if isinstance(session, str):
            self.session = (session,)
        else:
            assert isinstance(session, tuple)
        self.ica_path = ica_path
        self.session = session
        self.kwargs = kwargs

    def as_dict(self):
        out = CachedRawPipe.as_dict(self)
        out['session'] = self.session
        out['kwargs'] = self.kwargs
        return out

    def load_bad_channels(self, subject, session=None):
        bad_chs = set()
        for session in self.session:
            bad_chs.update(self.source.load_bad_channels(subject, session))
        return sorted(bad_chs)

    def load_ica(self, subject):
        path = self.ica_path.format(subject=subject)
        if not exists(path):
            raise RuntimeError("ICA file does not exist for raw=%r, "
                               "subject=%r. Run e.make_ica_selection() to "
                               "create it." % (self.name, subject))
        return mne.preprocessing.read_ica(path)

    def make_ica(self, subject):
        path = self.ica_path.format(subject=subject)
        raw = self.source.load(subject, self.session[0], add_bads=False)
        bad_channels = self.load_bad_channels(subject)
        raw.info['bads'] = bad_channels
        if exists(path):
            ica = mne.preprocessing.read_ica(path)
            picks = mne.pick_types(raw.info, ref_meg=False)
            if ica.ch_names == [raw.ch_names[i] for i in picks]:
                return path
            self.log.info("Raw %s: ICA outdated due to change in bad channels "
                          "for %s", self.name, subject)

        for session in self.session[1:]:
            raw_ = self.source.load(subject, session, False)
            raw_.info['bads'] = bad_channels
            raw.append(raw_)

        self.log.debug("Raw %s: computing ICA decomposition for %s", self.name,
                       subject)
        ica = mne.preprocessing.ICA(max_iter=256, **self.kwargs)
        # reject presets from meeg-preprocessing
        ica.fit(raw, reject={'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6})
        ica.save(path)
        return path

    def _make(self, subject, session):
        raw = self.source.load(subject, session, preload=True)
        ica = self.load_ica(subject)
        ica.apply(raw)
        return raw


class RawMaxwell(CachedRawPipe):
    "Maxwell filter raw pipe"

    _bad_chs_affect_cache = True

    def __init__(self, name, source, path, log, kwargs):
        CachedRawPipe.__init__(self, name, source, path, log)
        self.kwargs = kwargs

    def as_dict(self):
        out = CachedRawPipe.as_dict(self)
        out['kwargs'] = self.kwargs
        return out

    def _make(self, subject, session):
        raw = self.source.load(subject, session)
        self.log.debug("Raw %s: computing Maxwell filter for %s/%s", self.name,
                       subject, session)
        return mne.preprocessing.maxwell_filter(raw, **self.kwargs)


def assemble_pipeline(raw_dict, raw_path, bads_path, cache_path, ica_path, log):
    "Assemble preprocessing pipeline form a definition in a dict"
    raw = {}
    unassigned = raw_dict.copy()
    has_source = False
    while unassigned:
        n_unassigned = len(unassigned)
        for name in tuple(unassigned):
            params = unassigned[name]
            source = params.get('source')
            if source is None:
                if has_source:
                    raise NotImplementedError("Preprocessing pipeline with "
                                              "more than one raw source")
                raw[name] = RawSource(name, raw_path, bads_path, log)
                has_source = name or True
                del unassigned[name]
            elif source in raw:
                if params['type'] == 'filter':
                    raw[name] = RawFilter(name, raw[source], cache_path, log,
                                          params['args'],
                                          params.get('kwargs', {}))
                elif params['type'] == 'ica':
                    raw[name] = RawICA(name, raw[source], cache_path,
                                       ica_path.replace('{raw}', name), log,
                                       params['session'], params['kwargs'])
                elif params['type'] == 'maxwell_filter':
                    raw[name] = RawMaxwell(name, raw[source], cache_path, log,
                                           params['kwargs'])
                else:
                    raise ValueError("unknonw raw pipe type=%s" %
                                     repr(params['type']))
                del unassigned[name]

        if len(unassigned) == n_unassigned:
            raise RuntimeError("unable to resolve preprocessing pipeline "
                               "definition: %s" % unassigned)

    if not has_source:
        raise ValueError("Preprocssing pipeline has not raw source")
    elif has_source != 'raw':
        raise NotImplementedError("The raw source must be called 'raw'")
    return raw


###############################################################################
# Comparing pipelines
######################


def pipeline_dict(pipeline):
    return {k: v.as_dict() for k, v in pipeline.items()}


def compare_pipelines(old, new):
    """Return a tuple of raw keys for which definitions changed

    Parameters
    ----------
    old : {str: dict}
        A {name: params} dict for the previous preprocessing pipeline.
    new : {str: dict}
        Current pipeline.

    Returns
    -------
    bad_raw : {str: str}
        ``{pipe_name: status}`` dictionary. Status can be 'new', 'removed' or
        'changed'.
    bad_ica : {str: str}
        Same as ``bad_raw`` but only for RawICA pipes (for which ICA files
        might have to be removed).
    """
    out = {k: 'new' for k in new if k not in old}
    out.update({k: 'removed' for k in old if k not in new})
    out['raw'] = 'good'

    # parameter changes
    to_check = set(new) - set(out)
    for key in tuple(to_check):
        if new[key] != old[key]:
            out[key] = 'changed'
            to_check.remove(key)

    # secondary changes
    while to_check:
        n = len(to_check)
        for key in tuple(to_check):
            parent = new[key]['source']
            if parent in out:
                out[key] = out[parent]
                to_check.remove(key)
        if len(to_check) == n:
            raise RuntimeError("Queue not decreasing")

    bad_raw = {k: v for k, v in out.items() if v != 'good'}
    bad_ica = {k: v for k, v in bad_raw.items() if
               new.get(k, old.get(k))['type'] == 'RawICA'}
    return bad_raw, bad_ica


def ask_to_delete_ica_files(raw, status, filenames):
    "Ask whether outdated ICA files should be removed and act accordingly"
    if status == 'new':
        msg = ("The definition for raw=%r has been added, but ICA-files "
               "already exist. These files might not correspond to the new "
               "settings and should probably be deleted." % (raw,))
    elif status == 'removed':
        msg = ("The definition for raw=%r has been removed. The corresponsing "
               "ICA files should probably be deleted:" % (raw,))
    elif status == 'changed':
        msg = ("The definition for raw=%r has changed. The corresponding ICA "
               "files should probably be deleted." % (raw,))
    else:
        raise RuntimeError("status=%r" % (status,))
    msg += (" Delete %i files? Abort to fix the raw definition and try again. "
            "If you choose to ignore, you will not be warned again." %
            len(filenames))
    print(msg)

    command = ''
    while command not in ('abort', 'delete', 'ignore'):
        command = input("type delete/abort/ignore: ").lower()

    if command == 'delete':
        for filename in filenames:
            remove(filename)
    elif command == 'abort':
        raise RuntimeError("User abort")
    elif command != 'ignore':
        raise RuntimeError("command=%r" % (command,))
