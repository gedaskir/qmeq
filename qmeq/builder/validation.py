"""Module containing methods for validation of input parameters."""


def validate_kerntype(kerntype):
    if isinstance(kerntype, str):
        if kerntype not in {'Pauli', 'Lindblad', 'Redfield', '1vN', '2vN', 'pyPauli',
                    'pyLindblad', 'pyRedfield', 'py1vN', 'py2vN', 'pyRTD', 'RTD'}:
            print("WARNING: Allowed kerntype values are: " +
                  "\'Pauli\', \'Lindblad\', \'Redfield\', \'1vN\', \'2vN\', " +
                  "\'pyPauli\', \'pyLindblad\', \'pyRedfield\', \'py1vN\', \'py2vN\', \'RTD\'. " +
                  "Using default kerntype=\'Pauli\'.")
            kerntype = 'Pauli'
    return kerntype


def validate_itype(itype, kerntype):
    if kerntype not in ('RTD', 'pyRTD'):
        if itype not in {0, 1, 2, 3}:
            print("WARNING: itype needs to be 0, 1, 2, or 3. Using default itype=0.")
            itype = 0
    else:
        if itype != 1:
            print("WARNING: only itype=1 is supported by the RTD approach. Using itype=1.")
            itype = 1
    return itype


def validate_itype_ph(itype_ph):
    if itype_ph not in {0, 2}:
        print("WARNING: itype_ph needs to be 0, or 2. Using default itype=0.")
        itype_ph = 0
    return itype_ph

def validate_mfreeq(kerntype, mfreeq):
    if mfreeq and kerntype in {'RTD', 'pyRTD'}:
        print("WARNING: mfreeq=True is not supported by the RTD approach. Using default mfreeq=False.")
        mfreeq = False
    return mfreeq

def validate_indexing(indexing, symmetry, kerntype):
    if indexing is None:
        if symmetry == 'spin' and kerntype in {'pyRTD', 'RTD'}:
            print("WARNING: symmetry=\'spin\' is not supported by the RTD approach. " +
                  "Using default indexing=\'charge\'.")
            indexing = 'charge'
            symmetry = None
        elif symmetry == 'spin' and kerntype not in {'py2vN', '2vN'}:
            indexing = 'ssq'
        else:
            indexing = 'charge'

    if indexing not in {'Lin', 'charge', 'sz', 'ssq'}:
        print("WARNING: Allowed indexing values are: \'Lin\', \'charge\', \'sz\', \'ssq\'. " +
              "Using default indexing=\'charge\'.")
        indexing = 'charge'

    if indexing not in {'Lin', 'charge'} and kerntype in {'py2vN', '2vN'}:
        print("WARNING: For the 2vN approach indexing needs to be \'Lin\' or \'charge\'. " +
              "Using indexing=\'charge\' as a default.")
        indexing = 'charge'

    if indexing != 'charge' and kerntype in {'pyRTD', 'RTD'}:
        print("WARNING: For the RTD approach indexing needs to be \'charge\'. " +
              "Using indexing=\'charge\' as a default.")
        indexing = 'charge'

    return indexing, symmetry
