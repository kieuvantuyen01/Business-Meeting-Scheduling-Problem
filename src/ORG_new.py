"""Standalone MaxSAT encoding for B2B scheduling with gap-slot fairness.

The feasibility encoding is inherited from the original all-in-one maxsat.py.
New objective:
    minimize max_p G_p - min_p G_p
where G_p is the number of empty time slots strictly between participant p's
first and last meetings. The old hard fairness bound d=2 is intentionally removed.
"""


import sys
from pysat.formula import CNF, WCNF
from pysat.card import CardEnc, EncType, ITotalizer
from pysat.examples.rc2 import RC2
from math import inf, sqrt
import subprocess
import time
import os
import glob

# Get all input files
input_files = sorted(glob.glob('./input/*.dzn'))
# print(f"Found {len(input_files)} input files to process")

test_counter = 0
for input_file in input_files:
    test_counter += 1
    # Already solved these tests
    if test_counter <= 0:
        continue    
    # Get base filename
    base_name = os.path.basename(input_file)
    output_file = f'./example_output_gap_fairness/{base_name}'
    
    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")
    print(f"Test number: {test_counter}")
    print(f"{'='*60}")
    
    in_path = input_file
    out_path = output_file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if 'original' not in in_path: # and 'prec15' not in in_path and 'prec25' not in in_path:
        continue
    # Reset variables for each input file
    cnf = CNF()
    wcnf = WCNF()
    variable_size = 0
    
    def read_input():
        with open(in_path) as f:
            lines = f.readlines()
        
        # Read the initial variables
        nBusiness = int(lines[0].split('=')[1].strip().rstrip(';'))
        nMeetings = int(lines[1].split('=')[1].strip().rstrip(';'))
        nTables = int(lines[2].split('=')[1].strip().rstrip(';'))
        nTotalSlots = int(lines[3].split('=')[1].strip().rstrip(';'))
        nMorningSlots = int(lines[4].split('=')[1].strip().rstrip(';'))
        
        # Parse the requested array
        requested = [[0, 0, 0]]
        i = 6  # Start after the blank line
        while i < len(lines):
            line = lines[i].strip()
            if line == '|];':
                i += 1
                break
            elif 'requested' in line and '[|' in line:
                # Handle the header line with data
                content = line.split('[|', 1)[1].rstrip(',')
                parts = content.split(',')
                if len(parts) >= 3:
                    requested.append([int(parts[0].strip()), 
                                    int(parts[1].strip()), 
                                    int(parts[2].strip())])
            elif line.startswith('|'):
                # Remove '|' and trailing comma
                parts = line.lstrip('|').rstrip(',').split(',')
                if len(parts) >= 3:
                    requested.append([int(parts[0].strip()), 
                                    int(parts[1].strip()), 
                                    int(parts[2].strip())])
            i += 1
        
        # Parse meetingsxBusiness if needed
        meetingsxBusiness = [[]]
        while i < len(lines):
            line = lines[i].strip()
            # Skip empty lines
            if not line:
                i += 1
                continue
            # Skip the header line "meetingsxBusiness = [" and extract first set if present
            if 'meetingsxBusiness' in line and '[' in line:
                # Extract everything after the '['
                content = line.split('[', 1)[1]
                if content.startswith('{') and ',' in content:
                    # First set is on the same line
                    first_set = content.lstrip('{').rstrip(',}')
                    numbers = [int(x.strip()) - 1 for x in first_set.split(',') if x.strip()]
                    numbers = numbers[1:]
                    meetingsxBusiness.append(numbers)
            elif line.startswith('{'):
                # Regular set line
                should_break = False
                if line.endswith('},'):
                    numbers_str = line.strip('{},')
                elif line.endswith('};') or line.endswith('}];'):
                    numbers_str = line.strip('{};]')
                    should_break = True
                else:
                    i += 1
                    continue
                numbers = [int(x.strip()) - 1 for x in numbers_str.split(',') if x.strip()]
                numbers = numbers[1:]
                meetingsxBusiness.append(numbers)
                if should_break:
                    i += 1
                    break
            elif line == '];' or line == '};':
                i += 1
                break
            i += 1
        
        # Parse nMeetingsBusiness
        nMeetingsBusiness = []
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if 'nMeetingsBusiness' in line and '[' in line:
                # Extract the array content
                content = line.split('[', 1)[1].rstrip('];')
                nMeetingsBusiness = [0] + [int(x.strip()) for x in content.split(',') if x.strip()]
                i += 1
                break
            i += 1
        
        # Parse forbidden (array of sets)
        forbidden = [[]]
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if 'forbidden' in line and '[' in line:
                # Extract first set if on same line
                content = line.split('[', 1)[1]
                if content.startswith('{'):
                    first_set = content.lstrip('{').rstrip(',}')
                    numbers = [int(x.strip()) for x in first_set.split(',') if x.strip()]
                    numbers = [n for n in numbers if n != 0]  # {0} means empty, keep nonzero
                    forbidden.append(numbers)
            elif line.startswith('{'):
                should_break = False
                if line.endswith('},'):
                    numbers_str = line.strip('{},') 
                elif line.endswith('};') or line.endswith('}];'):
                    numbers_str = line.strip('{};]')
                    should_break = True
                else:
                    i += 1
                    continue
                numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip()]
                numbers = [n for n in numbers if n != 0]  # {0} means empty, keep nonzero
                forbidden.append(numbers)
                if should_break:
                    i += 1
                    break
            elif line == '];' or line == '};':
                i += 1
                break
            i += 1
        
        # Parse fixed (simple array of integers)
        fixed = []
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if 'fixed' in line and '[' in line:
                # Extract the array content
                content = line.split('[', 1)[1].rstrip('];')
                fixed = [0] + [int(x.strip()) for x in content.split(',') if x.strip()]
                i += 1
                break
            i += 1
        
        # Parse precedences (array of sets)
        precedences = [[]]
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if 'precedences' in line and '[' in line:
                # Extract first set if on same line
                content = line.split('[', 1)[1]
                if content.startswith('{'):
                    first_set = content.lstrip('{').rstrip(',}')
                    if first_set:  # Not empty
                        numbers = [int(x.strip()) for x in first_set.split(',') if x.strip()]
                        precedences.append(numbers)
                    else:
                        precedences.append([])
            elif line.startswith('{'):
                should_break = False
                if line.endswith('},'):
                    numbers_str = line.strip('{},') 
                elif line.endswith('};') or line.endswith('}];'):
                    numbers_str = line.strip('{};]')
                    should_break = True
                else:
                    i += 1
                    continue
                if numbers_str:  # Not empty
                    numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip()]
                    precedences.append(numbers)
                else:
                    precedences.append([])
                if should_break:
                    i += 1
                    break
            elif line == '];' or line == '};':
                i += 1
                break
            i += 1
        
        return nBusiness, nMeetings, nTables, nTotalSlots, nMorningSlots, requested, meetingsxBusiness, nMeetingsBusiness, forbidden, fixed, precedences

    def pairwise_AMO(lits):
        clauses = []
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                clauses.append([-lits[i], -lits[j]])
        return clauses
    
    def ALO(lits):
        clauses = [lits]
        return clauses

    # Exactly 1
    def commander_EO(lits):
        global variable_size
        sz = len(lits)
        if sz == 0:
            return
        if sz == 1:
            cnf.append([lits[0]])
            return
        # Base case: small enough to encode directly with pairwise AMO + ALO
        if sz <= 4:
            cnf.extend(pairwise_AMO(lits))
            cnf.extend(ALO(lits))
            return

        group_size = max(2, int(sqrt(sz)) + (1 if int(sqrt(sz)) ** 2 < sz else 0))
        commanders = []
        for i in range(0, sz, group_size):
            group = lits[i:min(sz, i + group_size)]
            c = variable_size + 1
            variable_size += 1
            commanders.append(c)
            # AMO within the group
            cnf.extend(pairwise_AMO(group))
            # c_i -> ALO(group)
            cnf.append([-c] + group)
            # X_j -> c_i:
            for x in group:
                cnf.append([-x, c])

        # Exactly one commander is true
        cnf.extend(pairwise_AMO(commanders))
        cnf.extend(ALO(commanders))

    def add_comparator(left, right, high, low):
        """Exact Boolean comparator: high=left OR right, low=left AND right."""
        cnf.append([-left, high])
        cnf.append([-right, high])
        cnf.append([left, right, -high])
        cnf.append([left, -low])
        cnf.append([right, -low])
        cnf.append([-left, -right, low])

    def sort_descending(lits):
        """Return exact unary sorting outputs in descending Boolean order."""
        global variable_size
        outputs = []
        for lit in lits:
            carry = lit
            next_outputs = []
            for existing in outputs:
                high = variable_size + 1
                variable_size += 1
                low = variable_size + 1
                variable_size += 1
                add_comparator(carry, existing, high, low)
                next_outputs.append(high)
                carry = low
            next_outputs.append(carry)
            outputs = next_outputs
        return outputs


    start_time = time.time()
    nBusiness, nMeetings, nTables, nTotalSlots, nMorningSlots, requested, meetingsxBusiness, nMeetingsBusiness, forbidden, fixed, precedences = read_input()
    input_time = time.time()
    print(f"Input parsing completed in {input_time - start_time:.4f} seconds")

    # print(nBusiness, nMeetings, nTables, nTotalSlots, nMorningSlots)
    # print("Requested:", requested)
    # print("MeetingsxBusiness:", meetingsxBusiness)
    # print("nMeetingsBusiness:", nMeetingsBusiness)
    # print("Forbidden:", forbidden)
    # print("Fixed:", fixed)
    # print("Precedences:", precedences)

    # HARD CONSTRAINTS: keep the original meeting-centered encoding.
    # Only the break/fairness semantics and objective are replaced below.

    # x[m][t] = 1 iff meeting m is scheduled at time slot t.
    x = [[0 for _ in range(nTotalSlots + 1)] for _ in range(nMeetings + 1)]

    # y[p][t] = 1 iff participant p has a meeting at time slot t.
    y = [[0 for _ in range(nTotalSlots + 1)] for _ in range(nBusiness + 1)]

    # prefix[p][t] = 1 iff p has a meeting at or before slot t.
    prefix = [[0 for _ in range(nTotalSlots + 1)] for _ in range(nBusiness + 1)]

    # suffix[p][t] = 1 iff p has a meeting at or after slot t.
    suffix = [[0 for _ in range(nTotalSlots + 1)] for _ in range(nBusiness + 1)]

    # gap_slot[p][t] = 1 iff slot t is empty and lies strictly between
    # p's first and last meetings. Only internal slots 2..T-1 can be gaps.
    gap_slot = [[0 for _ in range(nTotalSlots + 1)] for _ in range(nBusiness + 1)]

    for m in range(1, nMeetings + 1):
        for t in range(1, nTotalSlots + 1):
            x[m][t] = variable_size + 1
            variable_size += 1

    for p in range(1, nBusiness + 1):
        for t in range(1, nTotalSlots + 1):
            y[p][t] = variable_size + 1
            variable_size += 1

    for p in range(1, nBusiness + 1):
        for t in range(1, nTotalSlots + 1):
            prefix[p][t] = variable_size + 1
            variable_size += 1
            suffix[p][t] = variable_size + 1
            variable_size += 1

        for t in range(2, nTotalSlots):
            gap_slot[p][t] = variable_size + 1
            variable_size += 1

    # At most one meeting involving the same participant is scheduled at each time slot (19)
    for p in range(1, nBusiness + 1):
        for t in range(1, nTotalSlots + 1):
            lits = [x[m][t] for m in meetingsxBusiness[p]]
            if len(lits) > 1:
                # atmost_one = CardEnc.atmost(
                #     lits=lits, bound=1, encoding=EncType.seqcounter, top_id=variable_size
                # )
                atmost_one = pairwise_AMO(lits)
                cnf.extend(atmost_one)

    # Each meeting happened exactly once (20), (22), (24)
    for m in range(1, nMeetings + 1):
        if requested[m][2] == 3: # No time restriction
            lits = [x[m][t] for t in range(1, nTotalSlots + 1)]
            # clauses = CardEnc.equals(lits=lits, bound=1, encoding=EncType.seqcounter, top_id=variable_size)
            commander_EO(lits)
        elif requested[m][2] == 1: # Morning
            lits = [x[m][t] for t in range(1, nMorningSlots + 1)]
            # clauses = CardEnc.equals(lits=lits, bound=1, encoding=EncType.seqcounter, top_id=variable_size)
            commander_EO(lits)
        else: # Afternoon
            lits = [x[m][t] for t in range(nMorningSlots + 1, nTotalSlots + 1)]
            # clauses = CardEnc.equals(lits=lits, bound=1, encoding=EncType.seqcounter, top_id=variable_size)
            commander_EO(lits)

    # At most nTables meetings can happen at the same time (21)
    for t in range(1, nTotalSlots + 1):
        lits = [x[m][t] for m in range(1, nMeetings + 1)]
        if len(lits) > nTables:
            atmost_tables = CardEnc.atmost(
                lits=lits, bound=nTables, encoding=EncType.seqcounter, top_id=variable_size
            )
            variable_size = max(variable_size, atmost_tables.nv)
            cnf.extend(atmost_tables.clauses)
    # Handle AM/PM restrictions (23), (25)
    for m in range(1, nMeetings + 1):
        if requested[m][2] == 1: # Morning
            for t in range(nMorningSlots + 1, nTotalSlots + 1):
                cnf.append([-x[m][t]])
        elif requested[m][2] == 2: # Afternoon
            for t in range(1, nMorningSlots + 1):
                cnf.append([-x[m][t]])
    # Handle fixed meetings (26)
    for m in range(1, nMeetings + 1):
        if fixed[m] != 0:
            t = fixed[m]
            cnf.append([x[m][t]])
    
    # Handle forbidden time slots (27)
    for p in range(1, nBusiness + 1):
        for t in forbidden[p]:
            cnf.append([-y[p][t]])
    
    # Handle precedence constraints (28)
    for m in range(1, nMeetings + 1):
        for prec in precedences[m]:
            # Add staircase constraints (meeting prec must be scheduled before meeting m)
            sfx = [0 for _ in range(nTotalSlots + 1)]
            sfx[nTotalSlots] = x[prec][nTotalSlots]
            for t in range(nTotalSlots - 1, 0, -1):
                sfx[t] = variable_size + 1
                variable_size += 1
                cnf.append([-x[prec][t], sfx[t]])  # x[prec][t] => sfx[t]
                cnf.append([-sfx[t + 1], sfx[t]])  # sfx[t + 1] => sfx[t]
                cnf.append([x[prec][t], sfx[t + 1], -sfx[t]])  # not x[prec][t] and not sfx[t + 1] => not sfx[t]
            # Strict precedence: prec must be at slot < t (not at t or later)
            for t in range(1, nTotalSlots + 1):
                cnf.append([-x[m][t], -sfx[t]])
            
    # If a meeting is scheduled at time slot t then y[p1][t] and y[p2][t] must be true (29)
    # => x[m][t] -> y[p1][t] and y[p2][t]
    for m in range(1, nMeetings + 1):
        p1 = requested[m][0]
        p2 = requested[m][1]
        for t in range(1, nTotalSlots + 1):
            cnf.append([-x[m][t], y[p1][t]])
            cnf.append([-x[m][t], y[p2][t]])
    # If a time slot is used by business p then one of the meetings involving p must be scheduled at that time slot (30)
    # => y[p][t] -> OR_{m in meetingsxBusiness[p]} x[m][t]
    for p in range(1, nBusiness + 1):
        for t in range(1, nTotalSlots + 1):
            lits = [x[m][t] for m in meetingsxBusiness[p]]
            cnf.append([-y[p][t]] + lits)

    # ------------------------------------------------------------------
    # NEW BREAK SEMANTICS
    # total_gap_slots[p] = number of empty slots strictly between the
    # first and last meetings of participant p.
    # ------------------------------------------------------------------

    # prefix[p][t] <-> OR(y[p][1], ..., y[p][t]).
    for p in range(1, nBusiness + 1):
        if nTotalSlots >= 1:
            cnf.append([y[p][1], -prefix[p][1]])
            cnf.append([-y[p][1], prefix[p][1]])
        for t in range(2, nTotalSlots + 1):
            cnf.append([-y[p][t], prefix[p][t]])
            cnf.append([-prefix[p][t - 1], prefix[p][t]])
            cnf.append([y[p][t], prefix[p][t - 1], -prefix[p][t]])

    # suffix[p][t] <-> OR(y[p][t], ..., y[p][T]).
    for p in range(1, nBusiness + 1):
        if nTotalSlots >= 1:
            cnf.append([y[p][nTotalSlots], -suffix[p][nTotalSlots]])
            cnf.append([-y[p][nTotalSlots], suffix[p][nTotalSlots]])
        for t in range(nTotalSlots - 1, 0, -1):
            cnf.append([-y[p][t], suffix[p][t]])
            cnf.append([-suffix[p][t + 1], suffix[p][t]])
            cnf.append([y[p][t], suffix[p][t + 1], -suffix[p][t]])

    # gap_slot[p][t] <-> prefix[p][t-1] AND not y[p][t]
    #                                  AND suffix[p][t+1].
    for p in range(1, nBusiness + 1):
        for t in range(2, nTotalSlots):
            g = gap_slot[p][t]
            cnf.append([-g, prefix[p][t - 1]])
            cnf.append([-g, -y[p][t]])
            cnf.append([-g, suffix[p][t + 1]])
            cnf.append([-prefix[p][t - 1], y[p][t], -suffix[p][t + 1], g])

    # Unary representation sortedGap[p][j]: participant p has at least j
    # internal gap slots. The largest possible value for p is T-|M_p|
    # when p has at least two meetings; otherwise it is zero.
    participant_gap_upper = [0 for _ in range(nBusiness + 1)]
    for p in range(1, nBusiness + 1):
        if nMeetingsBusiness[p] >= 2:
            participant_gap_upper[p] = max(0, nTotalSlots - nMeetingsBusiness[p])

    max_gap_slots = max(participant_gap_upper, default=0)
    sortedGap = [[0 for _ in range(max_gap_slots + 1)] for _ in range(nBusiness + 1)]

    for p in range(1, nBusiness + 1):
        for j in range(1, max_gap_slots + 1):
            sortedGap[p][j] = variable_size + 1
            variable_size += 1

    for p in range(1, nBusiness + 1):
        gap_lits = [gap_slot[p][t] for t in range(2, nTotalSlots)]
        upper = min(participant_gap_upper[p], len(gap_lits))

        if upper > 0:
            unary_outputs = sort_descending(gap_lits)
            for j in range(1, upper + 1):
                s = sortedGap[p][j]
                output = unary_outputs[j - 1]
                cnf.append([-s, output])
                cnf.append([-output, s])

        # Thresholds above the participant-specific upper bound are false.
        for j in range(upper + 1, max_gap_slots + 1):
            cnf.append([-sortedGap[p][j]])

    # Exact unary maximum, minimum, and their difference. For each j:
    # maxGap[j] = OR_p sortedGap[p][j]
    # minGap[j] = AND_p sortedGap[p][j]
    # difGap[j] = maxGap[j] AND not minGap[j]
    # Therefore sum_j difGap[j] = max_p G_p - min_p G_p.
    maxGap = [0 for _ in range(max_gap_slots + 1)]
    minGap = [0 for _ in range(max_gap_slots + 1)]
    difGap = [0 for _ in range(max_gap_slots + 1)]

    for j in range(1, max_gap_slots + 1):
        maxGap[j] = variable_size + 1
        variable_size += 1
        minGap[j] = variable_size + 1
        variable_size += 1
        difGap[j] = variable_size + 1
        variable_size += 1

        threshold_lits = [sortedGap[p][j] for p in range(1, nBusiness + 1)]
        if threshold_lits:
            # maxGap[j] <-> OR(threshold_lits)
            for lit in threshold_lits:
                cnf.append([-lit, maxGap[j]])
            cnf.append([-maxGap[j]] + threshold_lits)

            # minGap[j] <-> AND(threshold_lits)
            for lit in threshold_lits:
                cnf.append([-minGap[j], lit])
            cnf.append([-lit for lit in threshold_lits] + [minGap[j]])
        else:
            cnf.append([-maxGap[j]])
            cnf.append([minGap[j]])

        # difGap[j] <-> maxGap[j] AND not minGap[j]
        cnf.append([-difGap[j], maxGap[j]])
        cnf.append([-difGap[j], -minGap[j]])
        cnf.append([-maxGap[j], minGap[j], difGap[j]])

    # Helpful unary monotonicity (already implied, but propagation-friendly).
    for j in range(1, max_gap_slots):
        cnf.append([-maxGap[j + 1], maxGap[j]])
        cnf.append([-minGap[j + 1], minGap[j]])

    # Imp1: The number of meetings of a participant p must equal nMeetingsBusiness[p] (43)
    for p in range(1, nBusiness + 1):
        lits = [y[p][t] for t in range(1, nTotalSlots + 1)]
        clauses = CardEnc.equals(lits=lits, bound=nMeetingsBusiness[p], encoding=EncType.cardnetwrk, top_id=variable_size)
        cnf.extend(clauses)
        variable_size = max(variable_size, clauses.nv)
    
    # Imp2: The number of participants having a meeting in a given time slot is bounded by twice the number of available locations (44)
    # for t in range(1, nTotalSlots + 1):
    #     lits = [y[p][t] for p in range(1, nBusiness + 1)]
    #     clauses = CardEnc.atmost(lits=lits, bound=2*nTables, encoding=EncType.cardnetwrk, top_id=variable_size)
    #     cnf.extend(clauses)
    #     variable_size = max(variable_size, clauses.nv)

    # Apply Itotalizer to (44)
    for t in range(1, nTotalSlots + 1):
        lits = [y[p][t] for p in range(1, nBusiness + 1)]
        if len(lits) > 2*nTables:
            itotalizer = ITotalizer(lits=lits, ubound=2*nTables + 1, top_id=variable_size)
            cnf.extend(itotalizer.cnf)
            variable_size = max(variable_size, itotalizer.cnf.nv)
            # Enforce at-most 2*nTables: forbid the (2*nTables+1)-th output being true
            cnf.append([-itotalizer.rhs[2*nTables]])
            for i in range(0, 2*nTables, 2):
                if i + 1 < 2 * nTables:
                    cnf.append([-itotalizer.rhs[i], itotalizer.rhs[i + 1]])

    
    # Add hard clauses 
    for clause in cnf.clauses:
        wcnf.append(clause)  # Default weight is top (hard)

    # SOFT CONSTRAINTS:
    # Minimize the fairness gap, not the sum of participant breaks.
    # One violated soft clause corresponds to one unit of
    # max(total_gap_slots) - min(total_gap_slots).
    for j in range(1, max_gap_slots + 1):
        wcnf.append([-difGap[j]], weight=1)

    constraint_time = time.time()
    print(f"Constraint building completed in {constraint_time - input_time:.4f} seconds")
    print(f"Total variables: {variable_size}")
    print(f"Total hard clauses: {len(cnf.clauses)}")
    print(f"Total soft clauses: {max_gap_slots}")

    def solve_maxsat():
        local_bin = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'uwrmaxsat', 'build', 'release', 'bin', 'uwrmaxsat'
        )
        UWRMAXSAT_BIN = os.environ.get('UWRMAXSAT_BIN', local_bin)
        WCNF_FILE = 'maxHS_gap_fairness.wcnf'
        TIMEOUT = 3600  # 1 hour

        if os.path.isfile(UWRMAXSAT_BIN) and os.access(UWRMAXSAT_BIN, os.X_OK):
            try:
                result = subprocess.run(
                    [UWRMAXSAT_BIN, '-m', WCNF_FILE],
                    capture_output=True, text=True, timeout=TIMEOUT
                )
                output = result.stdout

                model = []
                solution_cost = None
                status = None

                for line in output.splitlines():
                    if line.startswith('s '):
                        status = line[2:].strip()
                    elif line.startswith('o '):
                        solution_cost = int(line[2:].strip())
                    elif line.startswith('v '):
                        model.extend(int(lit) for lit in line[2:].split())

                if status == 'OPTIMUM FOUND' and model:
                    print(f"UWrMaxSAT optimum fairness gap: {solution_cost}")
                    return model, solution_cost, 'UWrMaxSAT'

                print(f"UWrMaxSAT status: {status}")
                return None, None, 'UWrMaxSAT'

            except subprocess.TimeoutExpired:
                print(f"TIMEOUT: UWrMaxSAT timeout after {TIMEOUT} seconds")
                return None, None, 'UWrMaxSAT'

        # Portable fallback: same WCNF encoding, solved by PySAT RC2.
        print('UWrMaxSAT binary not found; using PySAT RC2 fallback.')
        with RC2(wcnf) as solver:
            model = solver.compute()
            if model is None:
                return None, None, 'RC2'
            return model, solver.cost, 'RC2'

    # Write the WCNF to a file
    wcnf.to_file('maxHS_gap_fairness.wcnf')

    # Solve
    solve_start = time.time()
    assignment, solver_cost, solver_used = solve_maxsat()

    # Independently recompute the new objective from the decoded schedule.
    participant_gap_slots = []
    fairness_gap = None
    if assignment:
        positive_assignment = {lit for lit in assignment if lit > 0}
        for p in range(1, nBusiness + 1):
            busy_slots = [
                t for t in range(1, nTotalSlots + 1)
                if y[p][t] in positive_assignment
            ]
            if len(busy_slots) <= 1:
                participant_gap_slots.append(0)
            else:
                first_slot = min(busy_slots)
                last_slot = max(busy_slots)
                busy_set = set(busy_slots)
                participant_gap_slots.append(sum(
                    1 for t in range(first_slot + 1, last_slot)
                    if t not in busy_set
                ))

        fairness_gap = (
            max(participant_gap_slots, default=0)
            - min(participant_gap_slots, default=0)
        )
        if solver_cost is not None:
            assert solver_cost == fairness_gap, (
                f"Objective mismatch: solver_cost={solver_cost}, "
                f"recomputed_fairness_gap={fairness_gap}"
            )
    solve_time = time.time()
    print(f"MaxSAT solving completed in {solve_time - solve_start:.4f} seconds")

    # print(assignment)

    end_time = time.time()
    total_time = end_time - start_time

    # Output the schedule based on the assignment

    # if assignment:
    #     for m in range(1, nMeetings + 1):
    #         for t in range(1, nTotalSlots + 1):
    #             if x[m][t] in assignment:
    #                 print(f"Meeting {m} → Time slot {t}")

    print(f"\n{'='*60}")
    print(f"TOTAL RUNTIME: {total_time:.4f} seconds ({total_time:.2f}s)")
    print(f"{'='*60}")

    # Write output to file
    with open(out_path, 'w') as f:
        f.write(f"Input: {base_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total Runtime: {total_time:.4f} seconds\n")
        f.write(f"Input parsing: {input_time - start_time:.4f} seconds\n")
        f.write(f"Constraint building: {constraint_time - input_time:.4f} seconds\n")
        f.write(f"MaxSAT solving: {solve_time - solve_start:.4f} seconds\n")
        f.write(f"{'='*60}\n\n")
        f.write("Total variables: {}\n".format(variable_size))
        f.write("Total clauses: {}\n".format(len(cnf.clauses)))
        f.write("Solver: {}\n".format(solver_used))
        f.write("Objective (max gap slots - min gap slots): {}\n".format(
            fairness_gap if fairness_gap is not None else "N/A"
        ))
        f.write("Participant gap-slot totals: {}\n".format(participant_gap_slots))
        if assignment:
            # Checker
            f.write("SCHEDULE:\n")
            for m in range(1, nMeetings + 1):
                for t in range(1, nTotalSlots + 1):
                    if x[m][t] in assignment:
                        f.write(f"Meeting {m} → Time slot {t}\n")
            # Create a mapping of variable numbers to their assigned values
            var_assignment = {abs(var): (var > 0) for var in assignment}
            f.write("Checking hard constraints...\n")

            # Helper: variable truth in assignment
            def is_true(var_id):
                return var_assignment.get(var_id, False)

            # Check hard constraints
            # Each meeting happens exactly once
            for m in range(1, nMeetings + 1):
                count = sum(is_true(x[m][t]) for t in range(1, nTotalSlots + 1))
                assert count == 1, f"Meeting {m} does not happen exactly once (count={count})"
            
            # No more than nTables meetings at the same time
            for t in range(1, nTotalSlots + 1):
                count = sum(is_true(x[m][t]) for m in range(1, nMeetings + 1))
                assert count <= nTables, f"More than {nTables} meetings at time slot {t} (count={count})"
            
            # At most one meeting at moment t for the same business
            for p in range(1, nBusiness + 1):
                for t in range(1, nTotalSlots + 1):
                    count = sum(is_true(x[m][t]) for m in meetingsxBusiness[p])
                    assert count <= 1, f"More than one meeting for business {p} at time slot {t} (count={count})"
            
            # Handle time session
            for m in range(1, nMeetings + 1):
                if requested[m][2] == 3: # No time restriction
                    continue
                elif requested[m][2] == 1: # Morning
                    for t in range(nMorningSlots + 1, nTotalSlots + 1):
                        assert not is_true(x[m][t]), f"Meeting {m} should be in the morning but is scheduled at time slot {t}"
                else: # Afternoon
                    for t in range(1, nMorningSlots + 1):
                        assert not is_true(x[m][t]), f"Meeting {m} should be in the afternoon but is scheduled at time slot {t}"

            # y constraints: exact count and x <-> y channeling.
            for p in range(1, nBusiness + 1):
                y_count = sum(is_true(y[p][t]) for t in range(1, nTotalSlots + 1))
                assert y_count == nMeetingsBusiness[p], (
                    f"Business {p} has wrong number of occupied slots in y "
                    f"(got={y_count}, expected={nMeetingsBusiness[p]})"
                )
                for t in range(1, nTotalSlots + 1):
                    has_x = any(is_true(x[m][t]) for m in meetingsxBusiness[p])
                    assert is_true(y[p][t]) == has_x, (
                        f"y[{p}][{t}] inconsistent with x variables "
                        f"(y={is_true(y[p][t])}, has_x={has_x})"
                    )

            # Exact prefix/suffix and internal gap-slot semantics.
            for p in range(1, nBusiness + 1):
                busy = [is_true(y[p][t]) for t in range(1, nTotalSlots + 1)]
                for t in range(1, nTotalSlots + 1):
                    expected_prefix = any(busy[:t])
                    expected_suffix = any(busy[t - 1:])
                    assert is_true(prefix[p][t]) == expected_prefix, (
                        f"prefix[{p}][{t}] inconsistent"
                    )
                    assert is_true(suffix[p][t]) == expected_suffix, (
                        f"suffix[{p}][{t}] inconsistent"
                    )

                expected_count = 0
                for t in range(2, nTotalSlots):
                    expected_gap = any(busy[:t - 1]) and (not busy[t - 1]) and any(busy[t:])
                    assert is_true(gap_slot[p][t]) == expected_gap, (
                        f"gap_slot[{p}][{t}] inconsistent"
                    )
                    expected_count += int(expected_gap)

                assert expected_count == participant_gap_slots[p - 1], (
                    f"Business {p} gap total mismatch: "
                    f"encoded={expected_count}, recomputed={participant_gap_slots[p - 1]}"
                )

                for j in range(1, max_gap_slots + 1):
                    assert is_true(sortedGap[p][j]) == (expected_count >= j), (
                        f"sortedGap[{p}][{j}] inconsistent with gap total {expected_count}"
                    )

            # Exact maximum/minimum unary values and fairness-difference objective.
            for j in range(1, max_gap_slots + 1):
                threshold_values = [
                    is_true(sortedGap[p][j]) for p in range(1, nBusiness + 1)
                ]
                expected_max = any(threshold_values)
                expected_min = all(threshold_values) if threshold_values else True
                expected_dif = expected_max and (not expected_min)
                assert is_true(maxGap[j]) == expected_max, f"maxGap[{j}] inconsistent"
                assert is_true(minGap[j]) == expected_min, f"minGap[{j}] inconsistent"
                assert is_true(difGap[j]) == expected_dif, f"difGap[{j}] inconsistent"

            encoded_objective = sum(
                is_true(difGap[j]) for j in range(1, max_gap_slots + 1)
            )
            assert encoded_objective == fairness_gap, (
                f"Encoded objective={encoded_objective}, fairness_gap={fairness_gap}"
            )

            # Participant load bound per slot
            for t in range(1, nTotalSlots + 1):
                participants_at_t = sum(is_true(y[p][t]) for p in range(1, nBusiness + 1))
                assert participants_at_t <= 2 * nTables, (
                    f"Too many participants at slot {t} (count={participants_at_t}, bound={2*nTables})"
                )

            # Check forbidden time slots
            for p in range(1, nBusiness + 1):
                for t in forbidden[p]:
                    assert not is_true(y[p][t]), f"Business {p} has a meeting at forbidden time slot {t}"
            
            # Check fixed meetings
            for m in range(1, nMeetings + 1):
                if fixed[m] != 0:
                    t = fixed[m]
                    assert is_true(x[m][t]), f"Meeting {m} should be scheduled at time slot {t} but is not"
            
            # Check precedence constraints
            for m in range(1, nMeetings + 1):
                for prec in precedences[m]:
                    prec_time = None
                    m_time = None
                    for t in range(1, nTotalSlots + 1):
                        if is_true(x[prec][t]):
                            prec_time = t
                        if is_true(x[m][t]):
                            m_time = t
                    assert prec_time is not None and m_time is not None, f"Precedence constraint between meeting {prec} and {m} is not satisfied (prec_time={prec_time}, m_time={m_time})"
                    assert prec_time < m_time, f"Meeting {prec} should be scheduled before meeting {m} (prec_time={prec_time}, m_time={m_time})"
            f.write("All hard constraints are satisfied\n")
        else:
            f.write("NO SOLUTION FOUND\n")
    
    print(f"Output written to: {out_path}")
