# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from inspect import getargspec

from .._utils.parse import find_variables


class DefinitionError(Exception):
    "MneExperiment definition error"


def assert_dict_has_args(d, cls, kind, name, n_internal=0):
    "Make sure the dictionary ``d`` has all keys required by ``cls``"
    argspec = getargspec(cls.__init__)
    required = argspec.args[1 + n_internal: -len(argspec.defaults)]
    missing = set(required).difference(d)
    if missing:
        raise DefinitionError(
            "%s definition %s is missing the following parameters: %s" %
            (kind, name, ', '.join(missing)))


def find_epoch_vars(params):
    "Find variables used in a primary epoch definition"
    out = ()
    if params.get('sel'):
        out += find_variables(params['sel'])
    if 'trigger_shift' in params and isinstance(params['trigger_shift'], str):
        out += (params['trigger_shift'],)
    if 'post_baseline_trigger_shift' in params:
        out += (params['post_baseline_trigger_shift'],)
    return out


def find_epochs_vars(epochs):
    "Find variables used in all epochs"
    todo = list(epochs)
    out = {}
    while todo:
        for e in tuple(todo):
            p = epochs[e]
            if 'sel_epoch' in p:
                if p['sel_epoch'] in out:
                    out[e] = out[p['sel_epoch']] + find_epoch_vars(p)
                    todo.remove(e)
            elif 'sub_epochs' in p:
                if all(se in out for se in p['sub_epochs']):
                    out[e] = sum((out[se] for se in p['sub_epochs']),
                                 find_epoch_vars(p))
                    todo.remove(e)
            else:
                out[e] = find_epoch_vars(p)
                todo.remove(e)
    return out


def find_dependent_epochs(epoch, epochs):
    "Find all epochs whise definition depends on epoch"
    todo = set(epochs).difference(epoch)
    out = [epoch]
    while todo:
        last_len = len(todo)
        for e in tuple(todo):
            p = epochs[e]
            if 'sel_epoch' in p:
                if p['sel_epoch'] in out:
                    out.append(e)
                    todo.remove(e)
            elif 'sub_epochs' in p:
                if any(se in out for se in p['sub_epochs']):
                    out.append(e)
                    todo.remove(e)
            else:
                todo.remove(e)
        if len(todo) == last_len:
            break
    return out[1:]


def find_test_vars(params):
    "Find variables used in a test definition"
    if 'model' in params:
        vs = set(find_variables(params['model']))
    else:
        vs = set()

    if params['kind'] == 'two-stage':
        vs.update(find_variables(params['stage 1']))

    if 'vars' in params:
        vardef = params['vars']
        if isinstance(vardef, dict):
            vardef = iter(vardef.items())
        elif isinstance(vardef, tuple):
            vardef = (list(map(str.strip, v.split('=', 1))) for v in vardef)
        else:
            raise TypeError("vardef=%r" % (vardef,))

        for name, definition in vardef:
            if name in vs:
                vs.remove(name)
                if isinstance(definition, tuple):
                    definition = definition[0]
                vs.update(find_variables(definition))
    return vs

find_test_vars.__test__ = False
