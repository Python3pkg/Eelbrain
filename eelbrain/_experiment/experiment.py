# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>


from collections import defaultdict
from glob import glob
import inspect
from itertools import chain, product
import os
import re
import shutil
import subprocess
from time import localtime, strftime
import traceback

import numpy as np

from .. import fmtxt
from .._utils import LazyProperty
from .._utils.com import Notifier, NotNotifier


def _etree_expand(node, state):
    for tk, tv in node.items():
        if tk == '.':
            continue

        for k, v in state.items():
            name = '{%s}' % tk
            if str(v).startswith(name):
                tv[k] = {'.': v.replace(name, '')}
        if len(tv) > 1:
            _etree_expand(tv, state)


def _etree_node_repr(node, name, indent=0):
    head = ' ' * indent
    out = [(name, head + node['.'])]
    for k, v in node.items():
        if k == '.':
            continue

        out.extend(_etree_node_repr(v, k, indent=indent + 3))
    return out


class LayeredDict(dict):
    """Dictionary which can store and restore states"""
    def __init__(self):
        self._states = []
        dict.__init__(self)

    def __repr__(self):
        rep = ("<LayeredDict with %i stored states:\n"
               "%r>" % (len(self._states), dict.__repr__(self)))
        return rep

    def get_stored(self, key, level, default=None):
        """Retrieve a field value from any level

        Parameters
        ----------
        key : str
            the field name (dictionary key).
        level : int
            The level from which to retrieve the value. -1 = the current level.
        """
        return self._states[level].get(key, default)

    def restore_state(self, state=-1, discard_tip=True):
        """Restore a previously stored state

        Parameters
        ----------
        state : int | dict
            Index of the state which to restore (specified as index into a
            list of stored states, i.e., negative values access recently
            stored states).
        discard_tip : bool
            Discard the relevant state after restoring it. All states stored
            later are discarded either way.

        See Also
        --------
        .get_stored(): Retrieve a stored value without losing stored states
        """
        if isinstance(state, int):
            index = state
            state = self._states[index]
            if discard_tip:
                del self._states[index:]
            elif index != -1:
                del self._states[index + 1:]
        elif not isinstance(state, dict):
            raise TypeError("state needs to be either int or dict, got %r" %
                            (state,))

        self.clear()
        self.update(state)

    def store_state(self):
        "Store the current state"
        self._states.append(self.copy())


class _TempStateController(object):
    def __init__(self, experiment):
        self.experiment = experiment

    def __enter__(self):
        self.experiment.store_state()

    def __exit__(self, exc_type, exc_value, traceback):
        self.experiment.restore_state()


class TreeModel(object):
    """
    A hierarchical collection of format strings and field values

    Notes
    -----
    Any subclass should make sure to call the ``.store_state()`` method at the
    end of initialization.
    """
    owner = None  # email address as string (for notification)
    _auto_debug = False  # in notification block

    _fmt_pattern = re.compile('\{([\w-]+)\}')

    # a dictionary of static templates (i.e., templates that do not have any hooks)
    _templates = {}
    defaults = {}

    exclude = {}  # field_values to exclude (e.g. subjects)

    _repr_args = ()
    _repr_kwargs = ()

    def __init__(self, **state):
        self.exclude = self.exclude.copy()

        # scaffold for state
        self._fields = LayeredDict()
        self._field_values = LayeredDict()
        self._params = LayeredDict()
        self._user_fields = []

        # scaffold for hooks
        self._compound_members = {}
        self._compounds = defaultdict(list)
        self._eval_handlers = defaultdict(list)
        self._post_set_handlers = defaultdict(list)
        self._set_handlers = {}
        self._slave_fields = defaultdict(list)
        self._slave_handlers = {}

        # construct initial state: make all defaults available, then set as
        # many values as we can
        self._defaults = dict(self.defaults)
        self._defaults.update(state)
        for k, v in self._templates.items():
            if v is None or isinstance(v, str):
                self._register_constant(k, v)
            elif isinstance(v, tuple):
                self._register_field(k, v, v[0])
            else:
                err = ("Invalid templates field value: %r. Need None, tuple "
                       "or string" % v)
                raise TypeError(err)

        if self.owner:
            task = self.__class__.__name__
            self.notification = Notifier(self.owner, task, self._crash_report,
                                         self._auto_debug)
        else:
            self.notification = NotNotifier()

    def __repr__(self):
        args = [repr(self._fields[arg]) for arg in self._repr_args]
        kwargs = [(arg, repr(self._fields[arg])) for arg in self._repr_kwargs]

        no_initial_state = len(self._fields._states) == 0
        for k in sorted(self._fields):
            if k in self._repr_args or k in self._repr_kwargs:
                continue
            elif k in self._compound_members:
                continue
            elif '{' in self._fields[k]:
                continue

            v = self._fields[k]
            if no_initial_state or v != self._fields.get_stored(k, level=0):
                kwargs.append((k, repr(v)))

        args.extend('='.join(pair) for pair in kwargs)
        args = ', '.join(args)
        return "%s(%s)" % (self.__class__.__name__, args)

    def _bind_eval(self, key, handler):
        self._eval_handlers[key].append(handler)

    def _bind_post_set(self, key, handler):
        handlers = self._post_set_handlers[key]
        if handler not in handlers:
            handlers.append(handler)

    def _bind_set(self, key, handler):
        if key in self._set_handlers:
            raise KeyError("set-handler for %r already set" % key)
        self._set_handlers[key] = handler

    def _crash_report(self):
        try:
            source = inspect.getsource(self.__class__)
        except Exception as e:
            source = "Failed to retrieve source:\n" + traceback.format_exc(e)

        try:
            tree = str(self.show_state())
        except Exception as e:
            tree = "Failed to retrieve state:\n" + traceback.format_exc(e)

        # package versions
        from .. import __version__
        import mne
        import scipy
        versions = '\n'.join(("Eelbrain %s" % __version__,
                              "mne-python %s" % mne.__version__,
                              "SciPy %s" % scipy.__version__,
                              "NumPy %s" % np.__version__))

        return source, tree, versions

    def _find_missing_fields(self):
        """Check that all field names occurring in templates are valid entries

        Raises
        ------
        KeyError
            If any field names occurring in templates are not registered fields.
        """
        # find field names occurring in field values but not as fields
        missing = set()
        for temp in list(self._fields.values()):
            for field in self._fmt_pattern.findall(temp):
                if field not in self._fields:
                    missing.add(field)
        if missing:
            raise KeyError("The following fields occur in templates but "
                           "are undefined: %s" % ', '.join(sorted(missing)))

    def _register_compound(self, key, elements):
        """Register a field that is composed out of other fields

        The compound always reflects ``' '.join(elements)`` including only
        elements that are not empty.

        Parameters
        ----------
        key : str
            The name of the compound field.
        elements : tuple of str
            The field names of the elements.
        """
        self._compound_members[key] = elements
        for e in elements:
            self._compounds[e].append(key)
            self._bind_post_set(e, self._update_compounds)
        self._fields[key] = None
        self._update_compound(key)

    def _register_constant(self, key, value):
        value = self._defaults.get(key, value)
        if value is None:
            raise ValueError("The %r field needs to be set as default" % key)
        self._fields[key] = value

    def _register_field(self, key, values=None, default=None, set_handler=None,
                        eval_handler=None, post_set_handler=None,
                        depends_on=None, slave_handler=None):
        """Register an iterable field

        Parameters
        ----------
        key : str
            Name of the field.
        values : None | sequence of str
            Possible values for this field, if known.
        default : None | str
            Set the default value (if None, the first element in values).
        set_handler : None | callable
            Function to call instead of updating the state value. The return
            value of the set_handler is sent to the post_set_handler.
        eval_handler : None | callable
            Function to use for evaluating a value before setting. Can be
            called without actually setting the value; any parameter changes
            need to be evaluated in post_set_handlers.
        post_set_handler : None | callable
            Function to call after the value is changed. Needs to be able to
            handle non-existing values for ``e.set(..., vmatch=False)`` calls.
        depends_on : str | sequence of str
            Slave fields: Fields in depends_on trigger change in ``key``.
        handler : func
            Slave fields: Function that determines the new value of ``key``.
        """
        if key in self._fields:
            raise KeyError("Field already exists: %r" % key)

        if depends_on is not None:
            if (set_handler is not None or eval_handler is not None or
                    post_set_handler is not None):
                raise RuntimeError("Slave values can't have other handlers")
            elif slave_handler is None:
                raise RuntimeError("Slave value requires slave_handler")
            self._register_slave_field(key, depends_on, slave_handler)
            if default is None:
                default = slave_handler(self._fields)
        if set_handler is not None:
            self._bind_set(key, set_handler)
        if eval_handler is not None:
            self._bind_eval(key, eval_handler)
        if post_set_handler is not None:
            self._bind_post_set(key, post_set_handler)

        default = self._defaults.get(key, default)

        if values is not None:
            if default is None:
                if len(values):
                    default = values[0]
                else:
                    msg = "Values for %r empty, can`t set default" % key
                    raise RuntimeError(msg)
            elif default not in values:
                err = ("Default %r for %r not in values "
                       "%s" % (default, key, str(values)))
                raise ValueError(err)

            self._field_values[key] = tuple(values)

        self._user_fields.append(key)
        self._fields[key] = ''
        if default is not None:
            self.set(**{key: default})

    def _register_slave_field(self, key, depends_on, handler):
        """Register a field that strictly depends on one or more other fields

        Parameters
        ----------
        key : str
            Field name.
        depends_on : str | sequence of str
            Fields that trigger change.
        handler : func
            Function that determines the new value.

        Notes
        -----
        Restrictions:

        - Slave fields can not have any other handlers
        - Slave fields can not depend on other slave fields
        """
        if isinstance(depends_on, str):
            depends_on = (depends_on,)
        for dep in depends_on:
            self._slave_fields[dep].append(key)
        self._slave_handlers[key] = handler
        self._fields[key] = handler(self._fields)

    def expand_template(self, temp, keep=()):
        """Expand all constant variables in a template

        Parameters
        ----------
        temp : str
            Template or name of the template which should be expanded.
        keep : container (implements __contains__)
            Names of the variables which should not be expanded.

        Returns
        -------
        formatted_temp : str
            Template with all variables replaced by their values, except
            variables which have entries in field_values or in ``keep``.
        """
        temp = self._fields.get(temp, temp)

        while True:
            stop = True
            for name in self._fmt_pattern.findall(temp):
                if (name in keep) or (self._field_values.get(name, False)):
                    pass
                else:
                    temp = temp.replace('{%s}' % name, self._fields[name])
                    stop = False

            if stop:
                break

        return temp

    def find_keys(self, temp, root=True):
        """Find all terminal field names that are relevant for a template.

        Parameters
        ----------
        temp : str
            Template (or field name) for which to find terminal field names.
        root : bool
            Include "root" if present (default True).

        Returns
        -------
        keys : set
            All terminal field names that are relevant for formatting ``temp``.
        """
        keys = set()

        if temp in self._compound_members:
            temporary_keys = list(self._compound_members[temp])
        else:
            temp = self._fields.get(temp, temp)
            temporary_keys = self._fmt_pattern.findall(temp)

        while temporary_keys:
            key = temporary_keys.pop()

            if key == 'root':
                if root:
                    keys.add('root')
            # are there sub-fields?
            elif (key in self._compound_members or
                  self._fmt_pattern.findall(self._fields[key])):
                keys.update(self.find_keys(key, root))
            else:
                keys.add(key)

        return keys

    def format(self, string, vmatch=True, **kwargs):
        """Format a string (i.e., replace any '{xxx}' fields with their values)

        Parameters
        ----------
        string : str
            Template string.
        vmatch : bool
            For fields with known names, only allow existing field names.
        others :
            State parameters.

        Returns
        -------
        formatted_string : str
            The template temp formatted with current state values.
        """
        self.set(match=vmatch, **kwargs)

        while self._fmt_pattern.search(string):
            string = string.format(**self._fields)

        return string

    def get(self, temp, **state):
        return self.format('{%s}' % temp, **state)

    def _get_rel(self, temp, start):
        "Get the path of ``temp`` relative to ``start`` (both field names)"
        abs_ = self.get(temp)
        start_ = self.get(start)
        return os.path.relpath(abs_, start_)

    def get_field_values(self, field, exclude=True):
        """Find values for a field taking into account exclusion

        Parameters
        ----------
        field : str
            Field for which to find values.
        exclude : bool | list of str | str
            Exclude values. If True, exclude values based on ``self.exclude``.
        """
        values = self._field_values[field]
        if exclude is True:
            exclude = self.exclude.get(field, None)
        elif isinstance(exclude, str):
            exclude = (exclude,)

        if exclude:
            values = [v for v in values if v not in exclude]
        else:
            values = list(values)

        return values

    def iter(self, fields, exclude=True, values={}, mail=False, prog=False,
             **constants):
        """
        Cycle the experiment's state through all values on the given fields

        Parameters
        ----------
        fields : sequence | str
            Field(s) over which should be iterated.
        exclude : bool | dict  {str: bool, str: str, str: iterator over str}
            Exclude values from iteration. Boolean specifies whether to apply
            standard exclusion (``self.exclude``). A ``dict`` can be used to
            customize the exclusion per field with one of {field: bool,
            field: value, field: (sequence of values, )}. If only some fields
            are specified in a dict, True is assumed for absent fields.
        values : dict  {str: iterator over str}
            Fields with custom values to iterate over (instead of the
            corresponding field values) with {name: (sequence of values)}
            entries.
        *others* :
            Fields with constant values throughout the iteration.
        """
        # set constants
        self.set(**constants)

        if isinstance(fields, str):
            fields = (fields,)
            yield_str = True
        else:
            yield_str = False
        iter_fields = tuple(f for f in chain(fields, values) if f not in constants)

        # gather possible values to iterate over
        field_values = {}
        for field in iter_fields:
            if field in values:
                field_values[field] = values[field]
            else:
                if isinstance(exclude, bool):
                    exclude_ = exclude
                elif field in exclude:
                    exclude_ = exclude[field]
                else:
                    exclude_ = True
                field_values[field] = self.get_field_values(field, exclude_)

        # pick out the fields to iterate, but drop excluded cases:
        v_lists = []
        for field in iter_fields:
            v_lists.append(field_values[field])

        if len(v_lists):
            with self._temporary_state:
                for v_list in product(*v_lists):
                    self.restore_state(discard_tip=False)
                    self.set(**dict(list(zip(iter_fields, v_list))))

                    if yield_str:
                        yield self.get(fields[0])
                    else:
                        yield tuple(self.get(f) for f in fields)
        else:
            yield ()

    def iter_temp(self, temp, exclude=True, values={}, **constants):
        """
        Iterate through all paths conforming to a template given in ``temp``.

        Parameters
        ----------
        temp : str
            Name of a template in the MneExperiment.templates dictionary, or
            a path template with variables indicated as in ``'{var_name}'``
        """
        # if the name is an existing template, retrieve it
        temp = self.expand_template(temp, list(values.keys()))

        # find variables for iteration
        variables = set(self._fmt_pattern.findall(temp))
        variables.difference_update(constants)

        for _ in self.iter(variables, exclude, values, **constants):
            path = temp.format(**self._fields)
            yield path

    def _partial(self, temp, skip=()):
        "Format a template while leaving some slots unfilled"
        skip = set(skip)
        fields = self._fields.copy()
        fields.update({k: '{%s}' % k for k in skip})
        string = '{%s}' % temp

        while set(self._fmt_pattern.findall(string)).difference(skip):
            string = string.format(**fields)

        return string

    def _copy_state(self):
        """Copy of the state that can be used with ``.restore_state()``"""
        return self._fields.copy(), self._field_values.copy(), self._params.copy()

    def restore_state(self, state=-1, discard_tip=True):
        """Restore a previously stored state

        Parameters
        ----------
        state : int
            Index of the state which to restore (specified as index into a
            list of stored states, i.e., negative values access recently
            stored states).
        discard_tip : bool
            Discard the relevant state after restoring it. All states stored
            later are discarded either way.
        """
        if isinstance(state, int):
            s1 = s2 = s3 = state
        else:
            s1, s2, s3 = state
        self._fields.restore_state(s1, discard_tip)
        self._field_values.restore_state(s2, discard_tip)
        self._params.restore_state(s3, discard_tip)

    def set(self, match=True, allow_asterisk=False, **state):
        """Set the value of one or more fields.

        Parameters
        ----------
        match : bool
            For fields with stored values, only allow valid values.
        allow_asterisk : bool
            If a value contains '*', set the value without the normal value
            evaluation and checking mechanism.
        kwargs :
            Fields and values to set. Invalid fields raise a KeyError. Unless
            match == False, Invalid values raise a ValueError.
        """
        if not state:
            return

        # fields with special set handlers
        handled_state = {}
        for k in list(state.keys()):
            if k in self._set_handlers:
                handled_state[k] = self._set_handlers[k](state.pop(k))

        # make sure only valid fields are set
        for k in state:
            if k not in self._fields:
                raise KeyError("No template named %r" % k)

        # eval all values
        for k in list(state.keys()):
            eval_handlers = self._eval_handlers[k]
            v = state[k]
            if not isinstance(v, str):
                msg = "Values have to be strings, got %s=%r" % (k, v)
                raise TypeError(msg)
            elif '*' in v and allow_asterisk:
                pass
            elif eval_handlers:
                for handler in eval_handlers:
                    try:
                        v = handler(v)
                    except ValueError:
                        if match:
                            raise

                    if not isinstance(v, str):
                        err = "Invalid conversion: %s=%r" % (k, v)
                        raise RuntimeError(err)
                    state[k] = v
            elif not match:
                pass
            elif k not in self._field_values:
                pass
            elif v not in self.get_field_values(k, False):
                err = ("Variable {k!r} has no value {v!r}. In order to "
                       "see valid values use e.show_fields(); In order to "
                       "set a non-existent value, use e.set({k!s}={v!r}, "
                       "match=False).".format(k=k, v=v))
                raise ValueError(err)

        self._fields.update(state)

        # fields depending on changes in other fields
        slave_state = {}
        for state_key in set(state).union(handled_state).intersection(self._slave_fields):
            for slave_key in self._slave_fields[state_key]:
                if slave_key not in slave_state:
                    v = self._slave_handlers[slave_key](self._fields)
                    if v is not None:
                        slave_state[slave_key] = v
        self._fields.update(slave_state)

        # call post_set handlers
        for k, v in chain(iter(state.items()), iter(handled_state.items()),
                          iter(slave_state.items())):
            for handler in self._post_set_handlers[k]:
                handler(k, v)

    def show_fields(self, str_out=False):
        """
        Generate a table for all iterable fields and ther values.

        Parameters
        ----------
        str_out : bool
            Return the table as a string (instead of printing it).
        """
        lines = []
        for key in self._field_values:
            values = list(self.get_field_values(key))
            line = '%s:' % key
            head_len = len(line) + 1
            while values:
                v = repr(values.pop(0))
                if values:
                    v += ','
                if len(v) < 80 - head_len:
                    line += ' ' + v
                else:
                    lines.append(line)
                    line = ' ' * head_len + v

                if not values:
                    lines.append(line)

        table = '\n'.join(lines)
        if str_out:
            return table
        else:
            print(table)

    def show_state(self, temp=None, empty=False, hide=()):
        """List all top-level fields and their values

        (Top-level fields are fields whose values do not contain templates)

        Parameters
        ----------
        temp : None | str
            Only show variables relevant to this template.
        empty : bool
            Show empty variables (items whose value is the empty string '').
        hide : collection of str
            State variables to hide.

        Returns
        -------
        state : Table
            Table of (relevant) variables and their values.
        """
        table = fmtxt.Table('lll')
        table.cells('Key', '*', 'Value')
        table.caption('*: Value is modified from initialization state.')
        table.midrule()

        if temp is None:
            keys = self._user_fields
        else:
            keys = self.find_keys(temp)

        for k in sorted(keys):
            if k in hide:
                continue

            v = self._fields[k]
            if v != self._fields.get_stored(k, level=0):
                mod = '*'
            else:
                mod = ''

            if empty or mod or v:
                table.cells(k, mod, repr(v))

        return table

    def show_tree(self, root='root', fields=None):
        """
        Print a tree of the filehierarchy implicit in the templates

        Parameters
        ----------
        root : str
            Name of the root template (e.g., 'besa-root').
        fields : list of str
            Which fields to include in the tree (default is all).
        """
        if fields is None:
            fields = self._fields
        else:
            # find all implied fields
            new_fields = set(fields)
            fields = {}
            while new_fields:
                k = new_fields.pop()
                fields[k] = v = self._fields[k]
                new_fields.update([f for f in self._fmt_pattern.findall(v) if
                                   f not in fields])

        tree = {'.': self.get(root)}
        root_temp = '{%s}' % root
        for k, v in fields.items():
            if str(v).startswith(root_temp):
                tree[k] = {'.': v.replace(root_temp, '')}
        _etree_expand(tree, fields)
        nodes = _etree_node_repr(tree, root)
        name_len = max(len(n) for n, _ in nodes)
        path_len = max(len(p) for _, p in nodes)
        pad = ' ' * (80 - name_len - path_len)
        print('\n'.join(n.ljust(name_len) + pad + p.ljust(path_len) for
                        n, p in nodes))

    def store_state(self):
        """Store the current state

        See also
        --------
        .restore_state() : restore a previously stored state
        """
        self._fields.store_state()
        self._field_values.store_state()
        self._params.store_state()

    @LazyProperty
    def _temporary_state(self):
        return _TempStateController(self)

    def _update_compound(self, key):
        compound = ''
        for item_key in self._compound_members[key]:
            value = self.get(item_key)
            if value == '*':
                compound += '*'
            elif value:
                if compound and not compound.endswith('*'):
                    compound += ' '
                compound += value
        self.set(**{key: compound})

    def _update_compounds(self, key, _):
        for compound in self._compounds[key]:
            self._update_compound(compound)


class FileTree(TreeModel):
    """:class:`TreeModel` subclass for a file system hierarchy"""
    _repr_args = ('root',)

    def __init__(self, **state):
        TreeModel.__init__(self, **state)
        self._make_handlers = {}
        self._cache_handlers = {}
        self._register_field('root', eval_handler=self._eval_root)

    def _bind_cache(self, key, handler):
        """Bind a cache function to a ``*-file`` key

        The cache function is called every time the file name is retrieved and
        should recreate the file if it is outdated.
        """
        if key in self._cache_handlers:
            raise RuntimeError("Cache handler for %r already defined." % key)
        elif key in self._make_handlers:
            raise RuntimeError("Make handler for %r already defined." % key)
        self._cache_handlers[key] = handler

    def _bind_make(self, key, handler):
        """Bind a make function to a ``*-file`` key

        The make function is called only when the file name is retrieved and
        the file does not exist.
        """
        if key in self._cache_handlers:
            raise RuntimeError("Cache handler for %r already defined." % key)
        elif key in self._make_handlers:
            raise RuntimeError("Make handler for %r already defined." % key)
        self._make_handlers[key] = handler

    @staticmethod
    def _eval_root(root):
        root = os.path.abspath(os.path.expanduser(root))
        if root != '':
            root = os.path.normpath(root)
        return root

    def get(self, temp, fmatch=False, vmatch=True, match=True, mkdir=False,
            make=False, **kwargs):
        """
        Retrieve a formatted template

        With match=True, '*' are expanded to match a file,
        and if there is not a unique match, an error is raised. With
        mkdir=True, the directory containing the file is created if it does not
        exist.

        Parameters
        ----------
        temp : str
            Name of the requested template.
        fmatch : bool
            "File-match": If the template contains asterisk ('*'), use glob to
            fill it in. An IOError is raised if the pattern does not match
            exactly one file.
        vmatch : bool
            "Value match": Require existence of the assigned value (only
            applies for fields with stored values).
        match : bool
            Do any matching (i.e., match=False sets fmatch as well as vmatch
            to False).
        mkdir : bool
            If the directory containing the file does not exist, create it.
        make : bool
            If a requested file does not exists, make it if possible.
        kwargs :
            Set any state values.
        """
        if not match:
            fmatch = vmatch = False

        path = TreeModel.get(self, temp, vmatch=vmatch, **kwargs)
        path = os.path.expanduser(path)

        # assert the presence of the file
        if fmatch and ('*' in path):
            paths = glob(path)
            if len(paths) == 0 and make and temp in self._make_handlers:
                self._make_handlers[temp]()
                paths = glob(path)

            if len(paths) == 1:
                path = paths[0]
            elif len(paths) > 1:
                err = "More than one files match %r: %r" % (path, paths)
                raise IOError(err)
            else:
                raise IOError("No file found for %r" % path)

        # create the directory
        if mkdir:
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                root = self.get('root')
                if root == '':
                    raise IOError("Prevented from creating directories because "
                                  "root is not set. Use root='.' for a "
                                  "relative root.")
                elif os.path.exists(root):
                    os.makedirs(dirname)
                else:
                    raise IOError("Prevented from creating directories because "
                                  "Root does not exist: %r" % root)

        # make the file
        if make:
            if temp in self._cache_handlers:
                self._cache_handlers[temp]()
            elif not os.path.exists(path):
                if temp in self._make_handlers:
                    with self._temporary_state:
                        self._make_handlers[temp]()
                elif temp.endswith('-dir'):
                    os.makedirs(path)
                else:
                    raise RuntimeError("No make handler for %r." % temp)

        return path

    def glob(self, temp, **state):
        """Find all files matching a certain pattern

        Parameters
        ----------
        temp : str
            Name of the path template for which to find files.

        Notes
        -----
        State parameters can include an asterisk ('*') to match multiple files.
        Uses :func:`glob.glob`.
        """
        with self._temporary_state:
            pattern = self.get(temp, allow_asterisk=True, **state)
        file_paths = glob(pattern)
        return file_paths

    def show_file_status(self, temp, row, col=None, count=True, present='time',
                         absent='-', **kwargs):
        """Compile a table about the existence of files

        Parameters
        ----------
        temp : str
            The name of the path template for the files to examine.
        row : str
            Field over which to alternate rows.
        col : None | str
            Field over which to alternate columns (default is a single column).
        count : bool
            Add a column with a number for each line (default True).
        present : 'time' | 'date' | str
            String to display when a given file is present. 'time' to use last
            modification date and time (default); 'date' for date only.
        absent : str
            String to display when a given file is absent (default '-').
        others :
            ``self.iter()`` kwargs.
        """
        if col is None:
            col_v = (None,)
            ncol = 1
        else:
            col_v = self.get_field_values(col)
            ncol = len(col_v)

        # table header
        table = fmtxt.Table('r' * bool(count) + 'l' * (ncol + 1))
        if count:
            table.cell()
        table.cell(row)
        if col is None:
            table.cell(temp)
        else:
            for name in col_v:
                table.cell(name)
        table.midrule()

        # body
        for i, row_v in enumerate(self.iter(row, **kwargs)):
            if count:
                table.cell(i)
            table.cell(row_v)
            for v in col_v:
                if v is None:
                    path = self.get(temp)
                else:
                    path = self.get(temp, **{col: v})

                if os.path.exists(path):
                    if present == 'time':
                        r = strftime('%x %X', localtime(os.path.getmtime(path)))
                    elif present == 'date':
                        r = strftime('%x', localtime(os.path.getmtime(path)))
                    else:
                        r = present
                else:
                    r = absent
                table.cell(r)

        return table

    def show_file_status_mult(self, files, fields, count=True, present='X',
                              absent='-', **kwargs):
        """
        Compile a table about the existence of multiple files

        Parameters
        ----------
        files : str | list of str
            The names of the path templates whose existence to list.
        fields : str | list of str
            The names of the variables for which to list files (i.e., for each
            unique combination of ``fields``, list ``files``).
        count : bool
            Add a column with a number for each subject.
        present : str
            String to display when a given file is present.
        absent : str
            String to display when a given file is absent.

        Examples
        --------
        >>> e.show_file_status_mult(['raw-file', 'trans-file', 'fwd-file'],
        ... 'subject')
             Subject   Raw-file   Trans-file   Fwd-file
        -----------------------------------------------
         0   AD001     X          X            X
         1   AD002     X          X            X
         2   AD003     X          X            X
        ...
        """
        if not isinstance(files, (list, tuple)):
            files = [files]
        if not isinstance(fields, (list, tuple)):
            fields = [fields]

        ncol = (len(fields) + len(files))
        table = fmtxt.Table('r' * bool(count) + 'l' * ncol)
        if count:
            table.cell()
        for name in fields + files:
            table.cell(name.capitalize())
        table.midrule()

        for i, _ in enumerate(self.iter(fields, **kwargs)):
            if count:
                table.cell(i)

            for field in fields:
                table.cell(self.get(field))

            for temp in files:
                path = self.get(temp)
                if os.path.exists(path):
                    table.cell(present)
                else:
                    table.cell(absent)

        return table

    def show_in_finder(self, temp, **kwargs):
        "Reveal the file corresponding to the ``temp`` template in the Finder."
        fname = self.get(temp, **kwargs)
        subprocess.call(["open", "-R", fname])

    def push(self, dst_root, names, overwrite=False, exclude=False, **kwargs):
        """Copy files to another experiment root folder.

        Before copying any files the user is asked for confirmation.

        Parameters
        ----------
        dst_root : str
            Path to the root to which the files should be copied.
        names : str | sequence of str
            Name(s) of the template(s) of the files that should be copied.
        overwrite : bool
            What to do if the target file already exists.
        others :
            Update experiment state.

        Notes
        -----
        Use ``e.show_tree()`` to find out which element(s) to copy.
        """
        if isinstance(names, str):
            names = [names]

        # find files
        files = []
        for name in names:
            for src in self.iter_temp(name, exclude=exclude, **kwargs):
                if '*' in src:
                    raise NotImplementedError("Can't fnmatch here yet")

                if os.path.exists(src):
                    dst = self.get(name, root=dst_root)
                    if src == dst:
                        raise ValueError("Source == destination (%r)" % src)

                    if os.path.exists(dst):
                        flag = 'o' if overwrite else 'e'
                    else:
                        flag = ' '
                else:
                    dst = None
                    flag = 'm'
                files.append((src, dst, flag))

        # prompt for confirmation
        root = self.get('root')
        n_root = len(root)
        for src, dst, flag in files:
            if src.startswith(root):
                src = src[n_root:]
            print(' '.join((flag, src[-78:])))
        print("Flags: o=overwrite, e=skip, it exists, m=skip, source is "
              "missing")
        msg = "Proceed? (confirm with 'yes'): "
        if input(msg) != 'yes':
            return

        # copy the files
        for src, dst, flag in files:
            if flag in ('e', 'm'):
                continue

            dirpath = os.path.dirname(dst)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

            if os.path.isdir(src):
                if flag == 'o':
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)

    def rename(self, old, new, exclude=False):
        """Rename all files corresponding to a pattern (or template)

        Parameters
        ----------
        old : str
            Template for the files to be renamed. Can interpret '*', but will
            raise an error in cases where more than one file fit the pattern.
        new : str
            Template for the new names.

        Examples
        --------
        The following command will collect a specific file for each subject and
        place it in a common folder:

        >>> e.rename('info-file', '/some_other_place/{subject}_info.txt')
        """
        new = self.expand_template(new)
        files = []
        for old_name in self.iter_temp(old, exclude):
            if '*' in old_name:
                matches = glob(old_name)
                if len(matches) == 1:
                    old_name = matches[0]
                elif len(matches) > 1:
                    err = ("Several files fit the pattern %r" % old_name)
                    raise ValueError(err)

            if os.path.exists(old_name):
                new_name = self.format(new)
                files.append((old_name, new_name))

        if not files:
            print("No files found for %r" % old)
            return

        old_pf = os.path.commonprefix([pair[0] for pair in files])
        new_pf = os.path.commonprefix([pair[1] for pair in files])
        n_pf_old = len(old_pf)
        n_pf_new = len(new_pf)

        table = fmtxt.Table('lll')
        table.cells('Old', '', 'New')
        table.midrule()
        table.caption("%s -> %s" % (old_pf, new_pf))
        for old, new in files:
            table.cells(old[n_pf_old:], '->', new[n_pf_new:])

        print(table)

        msg = "Rename %s files (confirm with 'yes')? " % len(files)
        if input(msg) == 'yes':
            for old, new in files:
                dirname = os.path.dirname(new)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                os.rename(old, new)

    def rename_field(self, temp, field, old, new, exclude=False, **kwargs):
        """Change the value of one field in paths corresponding to a template

        Parameters
        ----------
        temp : str
            Template name.
        field : str
            Field to change.
        old : str
            Old value.
        new : str
            New value.
        kwargs :
            ``self.iter_temp`` arguments.
        """
        items = []  # (tag, src, dst)
        kwargs[field] = old
        dst_kwa = {field: new}
        for src in self.iter_temp(temp, exclude, ** kwargs):
            dst = self.get(temp, **dst_kwa)
            if os.path.exists(src):
                if os.path.exists(dst):
                    tag = 'o'
                else:
                    tag = ' '
            else:
                tag = 'm'
            items.append((tag, src, dst))

        src_prefix = os.path.commonprefix(tuple(item[1] for item in items))
        dst_prefix = os.path.commonprefix(tuple(item[2] for item in items))
        src_crop = len(src_prefix)
        dst_crop = len(dst_prefix)

        # print info
        if src_prefix == dst_prefix:
            lines = ['in ' + src_prefix, '']
        else:
            lines = [src_prefix, '->' + dst_prefix, '']

        for tag, src, dst in items:
            lines.append('%s %s -> %s' % (tag, src[src_crop:], dst[dst_crop:]))
        lines.append('')
        msg = 'Legend  m: source is missing;  o: will overwite a file'
        lines.append(msg)
        print('\n'.join(lines))
        rename = tuple(item for item in items if item[0] == ' ')
        if not rename:
            return

        msg = "Rename %i files (confirm with 'yes')? " % len(rename)
        if input(msg) != 'yes':
            return

        for _, src, dst in rename:
            os.rename(src, dst)
        print("Done")

    def rm(self, temp, confirm=False, **constants):
        """
        Remove all files corresponding to a template

        Asks for confirmation before deleting anything. Uses glob, so
        individual templates can be set to '*'.

        Parameters
        ----------
        temp : str
            Name of the path template for which to find and delete files.
        confirm : bool
            Confirm removal of the selected files. If False (default) the user
            is prompted for confirmation with a list of files; if True, the
            files are removed immediately.
        **others** :
            Set field values (values can be '*' to match all).
        """
        files = self.glob(temp, **constants)
        if files:
            root = self.get('root')
            print("root: %s\n" % root)
            root_len = len(root)
            for name in files:
                if name.startswith(root):
                    print(name[root_len:])
                else:
                    print(name)
            msg = ("Delete %i files or directories (confirm with 'yes')? "
                   % len(files))
            if confirm or input(msg) == 'yes':
                print('deleting...')
                for path in files:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            else:
                print('aborting...')
        else:
            print("No files found for %r" % temp)
