import re
import numpy as np


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
        return format(parts)
        
        
    # remove spaces
    units.replace(' ','')
        
    # treat groups seprately
    while '(' in units and ')' in units:
        units = re.sub(r'\/?\(([^\(\)]+?)\)', simplify_group, units)

    return format(parse(units))


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

    # prevent odd units
    parts = prevent_odd_units(parts)
    
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
    if re.search(r'[^\^][\+\-]', units):
        return units

    # add missing exponents
    units = re.sub(r'([a-zA-Z])([\-\+\d\.]+)', r'\1^\2', units)
            
    # loop over unique units
    parts = []
    for u in set(re.findall(r'[a-zA-Z]+', units)):
            
        # expand exponents
        m = re.findall(r'(\/)?(%s(\^[\+\-\d\.]+)*)' % u, units)
            
        n = 0.
        for prefix, string, exp in m:
                
            # group exponents
            if len(exp) > 0:
                exp = np.prod([float(x) for x in re.findall(r'[\+\-\d\.]+', string)])
            else:
                exp = 1.
                
            # negate exponents
            if prefix == '/':
                exp = -exp
                
            n += exp
                
        # replace Hz
        #if u.upper() == 'HZ' and n < 0.:
        #    u, n = 's', -n

        if n != 0:
            parts.append((u, n))

    return parts


def prevent_odd_units(parts):

    # replace per-hertz by seconds
    if len(parts) == 1:
        p = parts[0]
        if p[0].upper() == 'HZ' and p[1] == -1.:
            parts = [('s', 1.)]

    return parts
