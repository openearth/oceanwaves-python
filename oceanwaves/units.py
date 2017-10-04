import re
import six
import logging
import numpy as np


# regular expressions
RE_OPERATORS = r'\s*([ \*\/\+\-\^])\s*'
RE_TERMS = r'([^\^])([\+\-])'
RE_GROUPS = r'\/?\(([^\(\)]+?)\)((\^[\+\-\d\.]+)*)'
RE_NUMBER = r'[\+\-\d\.]+'
RE_VARIABLE = r'[a-zA-Z]+'
RE_EXPONENTS = r'([^a-zA-Z])?(%s(\^[\+\-\d\.]+)*)'
RE_EXPONENTS_MISSING = r'([a-zA-Z])([\-\+\d\.]+)'

# initialize logger
logger = logging.getLogger(__name__)


def simplify(units):
    '''Simplify the notation of units

    Parameters
    ----------
    units : str
        Unit specification

    Returns
    -------
    units : str
        Simplified unit specification

    See Also
    --------
    parse
    format
    
    '''

    def simplify_group(m):
        group = m.groups()[0]
        parts = parse(group)
        if m.group().startswith('/'):
            parts = [(u, -e) for u, e in parts]
        if m.groups()[1]:
            exp = np.prod([float(e) for e in m.groups()[1].split('^') if e])
            parts = [(u, e * exp) for u, e in parts]
        return ' %s' % format(parts)
        

    # only continue in case of string or unicode input
    if type(units) not in six.string_types + (six.text_type, str):
        return units
    
    # remove spaces around operators (space itself is also a multiplication operator)
    units = re.sub(RE_OPERATORS, r'\1', units)

    # encapsulate terms to be treated separately
    units = '(%s)' % re.sub(RE_TERMS, r'\1) \2(', units)
        
    # treat groups seprately
    while re.search(RE_GROUPS, units) is not None:
        units = re.sub(RE_GROUPS, simplify_group, units)
    parts = parse(units)
    
    # prevent odd units
    parts = prevent_odd_units(parts)
    
    return format(parts).strip()


def format(parts, order=['kg','m','s','Hz']):
    '''Format unit parts into string

    Parameters
    ----------
    parts : list of 2-tuples
        List of 2-tuples containing pairs of unit names and exponents
    order : list, optional
        Preferred order of units in formatted string (default: kg, m, s, Hz)

    Returns
    -------
    units : str
        Formatted unit specification

    See Also
    --------
    parse

    '''

    # order units
    parts = sorted(parts, key=lambda x: order.index(x[0])
                   if x[0] in order else np.inf)
    
    # format individual units
    for i, (u, e) in enumerate(parts):
        if e == 0:
            parts[i] = ''
        elif e == 1:
            parts[i] = u
        elif np.mod(e, 1.) == 0.:
            parts[i] = '%s^%d' % (u, e)
        else:
            parts[i] = '%s^%0.1f' % (u, e)

    # join units
    return ' '.join(parts)

    
def parse(units):
    '''Parse unit string into parts

    Parameters
    ----------
    units : str
        Unit specification

    Returns
    -------
    parts : list of 2-tuples
        List of 2-tuples containing pairs of unit names and exponents

    See Also
    --------
    format

    '''

    # multiple terms not supported, return as is
    if re.search(RE_TERMS, units):
        return [(units, 1.)]

    # add missing exponents
    units = re.sub(RE_EXPONENTS_MISSING, r'\1^\2', units)
            
    # loop over unique units
    parts = []
    for u in set(re.findall(RE_VARIABLE, units)):
            
        # expand exponents
        m = re.findall(RE_EXPONENTS % u, units)
            
        n = 0.
        for prefix, string, exp in m:
                
            # group exponents
            if len(exp) > 0:
                exp = np.prod([float(x) for x in re.findall(RE_NUMBER, string)])
            else:
                exp = 1.
                
            if prefix == '/':
                # negate exponents
                exp = -exp
            elif prefix == '^':
                # abort when exponent is a variable
                return [(units, 1.)]
                
            n += exp

        parts.append((u, n))

    return parts


def prevent_odd_units(parts):

    # replace per-hertz by seconds
    if len(parts) == 1:
        p = parts[0]
        if p[0].upper() == 'HZ' and p[1] == -1.:
            parts = [('s', 1.)]

    # replace degr with deg
    for i, p in enumerate(parts):
        if p[0].upper() == 'DEGR':
            parts[i] = ('deg', p[1])

    return parts
