# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import random
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple
import numpy as np
cimport numpy as np
import progressbar
import numba
from clifford_circuits.permutation import Permutation, split_regions_by_permutation

cdef long pack_bits((char, char, char, char) bits):
    return (( (<long>bits[0]) << 1 | (<long>bits[1])) << 1 | (<long>bits[2])) << 1 | (<long>bits[3])

# cdef (char, char, char, char) unpack_bits(long bits):
#     return <char> ((bits & 0b1000) != 0), <char> ((bits & 0b100) != 0), <char> ((bits & 0b10) != 0), <char> ((bits & 1) != 0)

cdef list cnot_mapping = [(False, False, False, False), (False, True, False, True), 
                          (False, False, True, False), (False, True, True, True), 
                          (False, True, False, False), (False, False, False, True), 
                          (False, True, True, False), (False, False, True, True), 
                          (True, False, True, False), (True, True, True, True), 
                          (True, False, False, False), (True, True, False, True), 
                          (True, True, True, False), (True, False, True, True), 
                          (True, True, False, False), (True, False, False, True)]
    
cdef list swap_mapping = [(False, False, False, False), (False, True, False, False), 
                          (True, False, False, False), (True, True, False, False), 
                          (False, False, False, True), (False, True, False, True), 
                          (True, False, False, True), (True, True, False, True), 
                          (False, False, True, False), (False, True, True, False), 
                          (True, False, True, False), (True, True, True, False), 
                          (False, False, True, True), (False, True, True, True), 
                          (True, False, True, True), (True, True, True, True)]

cdef list iswap_mapping = [(False, False, False, False), (False, True, False, False), 
                           (True, True, False, True), (True, False, False, True), 
                           (False, False, False, True), (False, True, False, True), 
                           (True, True, False, False), (True, False, False, False), 
                           (False, True, True, True), (False, False, True, True), 
                           (True, False, True, False), (True, True, True, False), 
                           (False, True, True, False), (False, False, True, False), 
                           (True, False, True, True), (True, True, True, True)]

cdef class PauliString:
    cdef public char [:] z_string
    cdef public char [:] x_string

    def __init__(self, x_string, z_string) -> None:
        # assert isinstance(x_string, np.ndarray)
        # assert isinstance(z_string, np.ndarray)
        assert len(z_string) == len(x_string)
        # assert x_string.dtype == z_string.dtype and x_string.dtype == bool

        self.z_string = z_string.copy()
        self.x_string = x_string.copy()
        

    def __matmul__(self, other):
        if not isinstance(other, PauliString):
            raise TypeError("Can only multiply two stabilizers")
        
        if len(self.z_string) != len(other.z_string) or len(self.x_string) != len(other.x_string):
            raise ValueError("The system sizes of the two stabilizers must be the same for the product to work")
        
        return PauliString(np.logical_xor(self.x_string, other.x_string), np.logical_xor(self.z_string, other.z_string))
    
    cpdef bint anticommutes(self, other: "PauliString"):
        if not isinstance(other, PauliString):
            raise TypeError("Can only anticommute two stabilizers")
        
        if len(self.z_string) != len(other.z_string) or len(self.x_string) != len(other.x_string):
            raise ValueError("The system sizes of the two stabilizers must be the same for the anticommutaiton to work")


        # l_and = np.logical_and
        # l_not = np.logical_not
        # l_or = np.logical_or

        # A = self.x_string
        # B = self.z_string
        # C = other.x_string
        # D = other.z_string

        # anti_witness_1 = l_and(l_not(A), l_and(B, C))
        # anti_witness_2 = l_and(B, l_and(C, l_not(D)))
        # anti_witness_3 = l_and(A, l_and(l_not(B), D))
        # anti_witness_4 = l_and(A, l_and(l_not(C), D))

        # witness = l_or(anti_witness_1, l_or(anti_witness_2, l_or(anti_witness_3, anti_witness_4)))

        # witness = bool(np.logical_xor.reduce(witness))

        cdef char witness = 0
        cdef char anti_witness_1, anti_witness_2, anti_witness_3, anti_witness_4, A, B, C, D
        cdef long i

        for i in range(len(self.x_string)):
            A = self.x_string[i]
            B = self.z_string[i]
            C = other.x_string[i]
            D = other.z_string[i]

            anti_witness_1 = (not A) and (B and C)
            anti_witness_2 = B and (C and (not D))
            anti_witness_3 = (not B) and D
            anti_witness_4 = (not C) and D

            witness = witness != (anti_witness_1 or anti_witness_2 or (A and (anti_witness_3 or anti_witness_4)))
            

        return witness
    
    cpdef PauliString copy(self):
        return PauliString(self.x_string, self.z_string)
    
    cpdef void x_2(self, long location):
        self.x_string[location] ^= self.z_string[location]
    
    cpdef void y_2(self, long location):
        self.x_string[location], self.z_string[location] = self.z_string[location], self.x_string[location]
    
    cpdef void z_2(self, long location):
        self.z_string[location] ^= self.x_string[location]

    cpdef void cnot(self, long control_loc, long controlled_loc):
        """
        this location is the controlling qbit, and the next qbit is the controlled one
        """
        cdef (char, char, char, char) in_bits = (self.x_string[control_loc], self.z_string[control_loc], self.x_string[controlled_loc], self.z_string[controlled_loc])
        cdef char in_index = pack_bits(in_bits)
        out_bits = cnot_mapping[in_index]
        self.x_string[control_loc] = out_bits[0]
        self.z_string[control_loc] = out_bits[1]
        self.x_string[controlled_loc] = out_bits[2]
        self.z_string[controlled_loc] = out_bits[3]

    cpdef void swap(self, long control_loc, long controlled_loc):
        in_bits = (self.x_string[control_loc], self.z_string[control_loc], self.x_string[controlled_loc], self.z_string[controlled_loc])
        in_index = pack_bits(in_bits)
        out_bits = swap_mapping[in_index]
        self.x_string[control_loc] = out_bits[0]
        self.z_string[control_loc] = out_bits[1]
        self.x_string[controlled_loc] = out_bits[2]
        self.z_string[controlled_loc] = out_bits[3]
    
    cpdef void iswap(self, long control_loc, long controlled_loc):
        in_bits = (self.x_string[control_loc], self.z_string[control_loc], self.x_string[controlled_loc], self.z_string[controlled_loc])
        in_index = pack_bits(in_bits)
        out_bits = iswap_mapping[in_index]
        self.x_string[control_loc] = out_bits[0]
        self.z_string[control_loc] = out_bits[1]
        self.x_string[controlled_loc] = out_bits[2]
        self.z_string[controlled_loc] = out_bits[3]
    
    cpdef void identity(self, long control_loc, long controlled_loc):
        pass
    
    cpdef single_qbit(self, long location, (char, char) x_to, (char, char) z_to):
        # if there is an x in the current location, then add what x goes to. Similarly for z
        cdef char x_new = (self.x_string[location] & x_to[0]) ^ (self.z_string[location] & z_to[0])
        cdef char z_new = (self.x_string[location] & x_to[1]) ^ (self.z_string[location] & z_to[1])

        self.x_string[location], self.z_string[location] = x_new, z_new
    
    cpdef clone_from(self, PauliString other):
        assert len(other) == len(self)
        cdef long i
        for i in range(len(self.x_string)):
            self.x_string[i] = other.x_string[i]
            self.z_string[i] = other.z_string[i]
        

    def __eq__(self, other: "PauliString") -> bool:
        if not isinstance(other, PauliString):
            return False

        return np.array_equal(self.x_string, other.x_string) and np.array_equal(self.z_string, other.z_string)
    
    def __len__(self) -> int:
        return len(self.x_string)
    
    def __getitem__(self, long index) -> bool:
        if index % 2:
            return self.z_string[index // 2]
        else:
            return self.x_string[index // 2]
    
    def __str__(self):
        builder = []
        for i, (x, z) in enumerate(zip(self.x_string, self.z_string)):
            if not x and not z:
                builder.append(' ' * len(f"Z_{i}"))
            elif not x and z:
                builder.append(f"Z_{i}")
            elif x and not z:
                builder.append(f"X_{i}")
            elif x and z:
                builder.append(f"Y_{i}")
        
        return ' '.join(builder)

    cpdef long left(self):
        for i in range(len(self.x_string)):
            if self.x_string[i] or self.z_string[i]:
                return i
        
        raise ValueError("This is the identity pauli string")

    cpdef long right(self):
        for i in reversed(range(len(self.x_string))):
            if self.x_string[i] or self.z_string[i]:
                return i
        
        raise ValueError("This is the identity pauli string")
    
    cpdef PauliString subsystem(self, long i, long j, copy_to: PauliString):
        assert i < j
        assert self.right() < j
        assert self.left() >= i
        assert j - i == len(copy_to)

        cdef long k
        for k in range(i, j):
            copy_to.x_string[k - i] = self.x_string[k]
            copy_to.z_string[k - i] = self.z_string[k]

        return copy_to

    cdef long stabilizer_length(self):
        return self.right() - self.left() + 1

    def __reduce__(self):
        return (PauliString, (np.array(self.x_string), np.array(self.z_string)), None, None, None)


    cpdef permute(self, permutation: Permutation):
        """
        Permutes this pauli string in-place.
        """

        cdef list perm = permutation.permutation

        cdef char x_val, z_val
        cdef int[:] done = np.zeros(len(self), dtype=np.int32)
        
        cdef int i = 0
        cdef int j = 0

        for i in range(len(self)):
            if not done[i]:
                x_val = self.x_string[i]
                z_val = self.z_string[i]
                j = i
                while not done[j]:
                    done[j] = True
                    j = perm[j]
                    x_val, self.x_string[j] = self.x_string[j], x_val
                    z_val, self.z_string[j] = self.z_string[j], z_val

                    
        
        

def x(n: int, length: int) -> PauliString:
    x_str = np.zeros(length, dtype=bool)
    z_str = np.zeros(length, dtype=bool)
    x_str[n] = True
    return PauliString(x_str, z_str)
    
def y(n: int, length: int) -> PauliString:
    x_str = np.zeros(length, dtype=bool)
    z_str = np.zeros(length, dtype=bool)
    x_str[n] = True
    z_str[n] = True
    return PauliString(x_str, z_str)

def z(n: int, length: int) -> PauliString:
    x_str = np.zeros(length, dtype=bool)
    z_str = np.zeros(length, dtype=bool)
    z_str[n] = True
    return PauliString(x_str, z_str)

    

cdef class DensityMatrix:
    cdef public list stabilizers
    cdef public long system_size
    def __init__(self, stabilizers: List[PauliString]) -> None:
        self.stabilizers = [i.copy() for i in stabilizers]
        self.system_size = len(self.stabilizers[0])
        assert self.system_size == len(self.stabilizers)
    
    cpdef void measure(self, operator: PauliString):
        anticommuting_indices = [i for i, stabilizer in enumerate(self.stabilizers) if stabilizer.anticommutes(operator)]
        if len(anticommuting_indices) > 0:
            representative = self.stabilizers[anticommuting_indices[0]]
            
            for fixable_index in anticommuting_indices[1:]:
                self.stabilizers[fixable_index] @= representative
            
            self.stabilizers[anticommuting_indices[0]].clone_from(operator)
        elif len(self.stabilizers) < self.system_size:
            self.stabilizers.append(operator)
        
    def x_2(self, location: int) -> None:
        for stabilizer in self.stabilizers:
            stabilizer.x_2(location)
    
    def y_2(self, location: int) -> None:
        for stabilizer in self.stabilizers:
            stabilizer.y_2(location)
    
    def z_2(self, location: int) -> None:
        for stabilizer in self.stabilizers:
            stabilizer.z_2(location)
        
    def cnot(self, control_loc: int, controlled_loc: int) -> None:
        assert (control_loc % len(self)) != (controlled_loc % len(self))
        for stabilizer in self.stabilizers:
            stabilizer.cnot(control_loc, controlled_loc)
    
    def swap(self, control_loc: int, controlled_loc: int) -> None:
        assert (control_loc % len(self)) != (controlled_loc % len(self))
        for stabilizer in self.stabilizers:
            stabilizer.swap(control_loc, controlled_loc)

    def iswap(self, control_loc: int, controlled_loc: int) -> None:
        assert (control_loc % len(self)) != (controlled_loc % len(self))
        for stabilizer in self.stabilizers:
            stabilizer.iswap(control_loc, controlled_loc)
    
    def identity(self, *args, **kwargs):
        pass
        
    cpdef void single_qbit(self, long location, (char, char) x_to, (char, char) z_to):
        #they cannot be equal
        # assert x_to != z_to

        # #they cannot map to the identity
        # assert x_to != (False, False)
        # assert z_to != (False, False)

        # assert location < len(self)
        cdef PauliString stabilizer

        for stabilizer in self.stabilizers:
            stabilizer.single_qbit(location, x_to, z_to)
    
    cpdef void permute(self, perm: Permutation):
        for string in self.stabilizers:
            string.permute(perm)
    
    def random_single_qbit(self, location: int):
        possibilities = (False, True), (True, False), (True, True)
        self.single_qbit(location, *random.sample(possibilities, 2))


    def copy(self) -> "DensityMatrix":
        return self.__class__(self.stabilizers)
    
    def clone_from(self, other: "DensityMatrix") -> None:
        assert len(self) == len(other)
        for my_stabilizer, other_stabilizer in zip(self.stabilizers, other.stabilizers):
            my_stabilizer.clone_from(other_stabilizer)
    
    def get_substate(self, region_start: int, region_end: int, copy_to: "DensityMatrix") -> "DensityMatrix":
        self.clip_gauge()

        for stabilizer in self.stabilizers[:region_start]:
            assert stabilizer.right() < region_start
        
        for stabilizer in self.stabilizers[region_end:]:
            if stabilizer.left() < region_end:
                print(stabilizer)
            assert stabilizer.left() >= region_end
        
        assert len(copy_to) == region_end - region_start

        for stabilizer, copy_to_stabilizer in zip(self.stabilizers[region_start:region_end], copy_to.stabilizers):
            stabilizer.subsystem(region_start, region_end, copy_to_stabilizer)


    def clip_gauge(self) -> None:
        # this algorithm is specialized for this case 
        # so as not to deal with edge cases right now. 
        assert len(self.stabilizers) ==  self.system_size

        # journals.aps.org/prb/pdf/10.1103/PhysRevB.100.134306
        # this is the preclipping phase we fix the left endpoints
        stabilizers_found = 0

        for location in range(self.system_size * 2):
            # this is the location we are fixing

            # first find an operator which we can use for fixing
            # and put it in the right spot
            for i in range(stabilizers_found, len(self.stabilizers)):
                if self.stabilizers[i][location]:
                    self.stabilizers[stabilizers_found], self.stabilizers[i] = self.stabilizers[i], self.stabilizers[stabilizers_found]
                    stabilizers_found += 1
                    break
            else:
                continue # no worries we just move on to the next location
            
            #now fix all the others
            for i in range(stabilizers_found, len(self.stabilizers)):
                if self.stabilizers[i][location]:
                    self.stabilizers[i] = self.stabilizers[i] @ self.stabilizers[stabilizers_found - 1]
        
        # now let us do the reverse
        reverse_stabilizers_found = 0
        reverse_locations_used = set()

        for location in reversed(range(self.system_size * 2)):
            # see if there is a stabilizer with 
            # nonzero value from the right
            for i in reversed(range(len(self.stabilizers))):
                if self.stabilizers[i][location] and i not in reverse_locations_used:
                    remover = self.stabilizers[i]
                    remover_location = i
                    reverse_stabilizers_found += 1
                    reverse_locations_used.add(i)
                    break
            else:
                continue
        
            # now eliminate all the others with something in that location
            for i in reversed(range(remover_location)):
                if self.stabilizers[i][location]:
                    self.stabilizers[i] = self.stabilizers[i] @ remover

    def entropies_of_contiguous_regions(self, regions: List[Tuple[int]]) -> List[int]:
        """
        Will give the entropies of every region listed in regions. Each region is demarcated by 
        its endpoints which are includsive on the left and non-inclusive on the right. [left, right)
        """
        self.clip_gauge()
        entropies = [0] * len(regions)

        endpoints = []
        for stabilizer in self.stabilizers:
            endpoints.append((stabilizer.left(), stabilizer.right()))
        
        for i, (region_left, region_right) in enumerate(regions):
            region = range(region_left, region_right)
            for left, right in endpoints:
                if (left in region and right not in region) or (right in region and left not in region):
                    entropies[i] += 1
        
        for i in entropies:
            assert i % 2 == 0

        return [i // 2 for i in entropies]
    
    def entropies_of_permutably_contiguous_regions(self, regions: List[List[int]], perms: List[Permutation], division: List[List[Tuple[int, List[int], Tuple[int, int]]]], workhorse_state: "DensityMatrix") -> List[int]:
        # division = split_regions_by_permutation(regions, perms)
        output = [None] * len(regions)
        assert len(self) == len(workhorse_state)
        
        for subset, perm in zip(division, perms):
            assert len(workhorse_state) == perm.size
            workhorse_state.clone_from(self)
            workhorse_state.permute(perm)

            region_subset = [endpoints for (region_index, region, endpoints) in subset]
            indices = [region_index for (region_index, region, endpoints) in subset]

            

            entropies = workhorse_state.entropies_of_contiguous_regions(region_subset)

            for idx, H in zip(indices, entropies):
                output[idx] = H
        
        return output

                

    def contiguous_entropy(self, left: int, right: int) -> int:
        self.clip_gauge()
        S = 0
        for stabilizer in self.stabilizers:
            left_end = stabilizer.left()
            right_end = stabilizer.right()
            if (left <= left_end < right and right <= right_end) or (left <= right_end < right and left_end < left):
                S += 1
        
        assert S % 2 == 0
        return S // 2
    
    def entropy(self, region: List[int]) -> int:
        # this function should compute the entropy by row-reducing the stabilizer matrix after projection
        # and write tests for it as well
        # the entropy is given by the difference in system size and the rank of the projected stabilizer
        # matrix

        region_size = len(region)
        assert len(set(region)) == region_size
        assert len(region) <= len(self)

        projected_stabilizers = np.zeros((len(self), len(region) * 2), dtype=bool)

        for i, stabilizer in enumerate(self.stabilizers):
            for location in range(len(region)):
                projected_stabilizers[i, location * 2] = stabilizer[region[location] * 2]
                projected_stabilizers[i, location * 2 + 1] = stabilizer[region[location] * 2 + 1]


        current_row = 0
        for location in range(region_size * 2):
            for row in range(current_row, len(self.stabilizers)):
                #if we find a 1 in the column move it to the first spot
                if projected_stabilizers[row, location]:
                    projected_stabilizers[[current_row, row]] = projected_stabilizers[[row, current_row]]

                    #now we should use the current row to zero out all the stuff in the lower rows
                    for row in range(current_row + 1, len(self.stabilizers)):
                        if projected_stabilizers[row, location]:
                            projected_stabilizers[row] = np.logical_xor(projected_stabilizers[row], projected_stabilizers[current_row])

                
                    current_row += 1

                    break
        
        return current_row - region_size # this will tell us how many rows there are below which there are all zeros

    def random_unitary(self, i_min: int, i_max: int, length: int) -> List[int]:
        """
        Applies a random unitary and then rturns a list with the information to apply it again.
        """
        assert i_min < i_max <= len(self)

        choices = []

        for step in range(length):
            for loc in range(i_min, i_max):
                choices.append(random.randrange(4))
                (self.x_2, self.y_2, self.z_2, self.identity)[choices[-1]](loc)
            
            for loc in range(i_min + step % 2, i_max, 2):
                if loc + 1 < i_max:
                    choices.append(random.randrange(3))
                    [self.swap, self.iswap, self.cnot][choices[-1]](loc, loc + 1)
        
        return choices

    def unitary(self, i_min: int, i_max: int, length: int, choices: List[int]) -> None:
        """
        Executes the unitary given by the list of choices
        """
        assert i_min < i_max <= len(self)

        choices = iter(choices)

        for step in range(length):
            for loc in range(i_min, i_max):
                (self.x_2, self.y_2, self.z_2, self.identity)[next(choices)](loc)

            for loc in range(i_min + step % 2, i_max, 2):
                if loc + 1 < i_max:
                    [self.swap, self.iswap, self.cnot][next(choices)](loc, loc + 1)

    def inverse_unitary(self, i_min: int, i_max: int, length: int, choices: List[int]) -> None:
        """
        Executes the inverse unitary given by the list of choices
        """
        assert i_min < i_max <= len(self)

        reversed_choices = reversed(choices)

        for step in reversed(range(length)):
            for loc in reversed(range(i_min + step % 2, i_max, 2)):
                if loc + 1 < i_max:
                    # notice that the inverse of iswap is not iswap, but it is true up to a
                    # sign which we do not keep track of here
                    [self.swap, self.iswap, self.cnot][next(reversed_choices)](loc, loc + 1)
            
            for loc in reversed(range(i_min, i_max)):
                (self.x_2, self.y_2, self.z_2, self.identity)[next(reversed_choices)](loc)
    
    def __str__(self):
        return '\n'.join(['-' * (len(self.stabilizers[0]) * 4)] + [str(ps) for ps in self.stabilizers] + ['*' * (len(self.stabilizers[0]) * 4)])
            
    def __len__(self):
        return self.system_size

    def __reduce__(self):
        return (DensityMatrix, (self.stabilizers, ), None, None, None)

class ChunkedState(DensityMatrix):
    def __init__(self, stabilizers: List[PauliString]) -> None:
        super().__init__(stabilizers)

        self.z_operators = [z(i, len(self)) for i in range(len(self))]

    def chunked_unitary(self, i_min: int, i_max: int, layers: int=-1) -> None:
        assert i_min < i_max <= len(self)
        length = i_max - i_min
        if layers < 0:
            layers = length

        for step in range(layers):
            for loc in range(i_min, i_max):
                (self.x_2, self.y_2, self.z_2)[(step + loc) % 3](loc)
            
            for loc in range(i_min + step % 2, i_max, 2):
                if loc + 1 < i_max:
                    self.cnot(loc, loc + 1)

    def chunked_random_unitary(self, i_min: int, i_max: int) -> List[int]:
        assert i_min < i_max <= len(self)
        length = i_max - i_min

        choices = []

        for step in range(length):
            for loc in range(i_min, i_max):
                choices.append(random.randrange(4))
                (self.x_2, self.y_2, self.z_2, self.identity)[choices[-1]](loc)
            
            for loc in range(i_min + step % 2, i_max, 2):
                if loc + 1 < i_max:
                    choices.append(random.randrange(3))
                    [self.swap, self.iswap, self.cnot][choices[-1]](loc, loc + 1)
        
        return choices

    def inverse_unitary(self, i_min: int, i_max: int, choices: List[int]) -> None:
        assert i_min < i_max <= len(self)
        length = i_max - i_min

        reversed_choices = reversed(choices)

        for step in reversed(range(length)):
            for loc in reversed(range(i_min + step % 2, i_max, 2)):
                if loc + 1 < i_max:
                    [self.swap, self.iswap, self.cnot][next(reversed_choices)](loc, loc + 1)
                    
            
            for loc in reversed(range(i_min, i_max)):
                (self.x_2, self.y_2, self.z_2, self.identity)[next(reversed_choices)](loc)
                


    def chunked_measure(self, i_min: int, i_max: int) -> None:
        assert i_min < i_max <= len(self)
        for op in self.z_operators[i_min: i_max]:
            self.measure(op)
    
    def chunked_random_measure(self, i_min: int, i_max: int) -> None:
        choices = self.chunked_random_unitary(i_min, i_max)
        self.chunked_measure(i_min, i_max)
        self.inverse_unitary(i_min, i_max, choices)


def test():
    x_ = PauliString(np.array([True], dtype=bool), np.array([False], dtype=bool))
    z_ = PauliString(np.array([False], dtype=bool), np.array([True], dtype=bool))
    y_ = PauliString(np.array([True], dtype=bool), np.array([True], dtype=bool))
    I_ = PauliString(np.array([False], dtype=bool), np.array([False], dtype=bool))

    assert x_ @ I_ == x_
    assert x_ @ x_ == I_
    assert x_ @ y_ == z_
    assert x_ @ z_ == y_

    assert y_ @ I_ == y_
    assert y_ @ x_ == z_
    assert y_ @ y_ == I_
    assert y_ @ z_ == x_
    
    assert z_ @ I_ == z_
    assert z_ @ x_ == y_
    assert z_ @ y_ == x_
    assert z_ @ z_ == I_

    assert I_ @ x_ == x_
    assert I_ @ y_ == y_
    assert I_ @ z_ == z_
    assert I_ @ I_ == I_
    print("Test Passed!\n")

    assert not I_.anticommutes(I_)
    assert not I_.anticommutes(x_)
    assert not I_.anticommutes(y_)
    assert not I_.anticommutes(z_)
    assert not x_.anticommutes(I_)
    assert not x_.anticommutes(x_)
    assert x_.anticommutes(y_)
    assert x_.anticommutes(z_)
    assert not y_.anticommutes(I_)
    assert y_.anticommutes(x_)
    assert not y_.anticommutes(y_)
    assert y_.anticommutes(z_)
    assert not z_.anticommutes(I_)
    assert z_.anticommutes(x_)
    assert z_.anticommutes(y_)
    assert not z_.anticommutes(z_)
    print("Test Passed!\n")


    #I'm just praying that all that works....
    
    L = 5
    stabilizers = [x(i, L) for i in range(L)]
    rho = DensityMatrix(stabilizers)

    print(rho)
    rho.measure(z(0, L) @ z(1, L))
    print()
    print(rho)
    rho.measure(z(-1, L) @ z(-2, L))
    print()
    print(rho)

    rho.measure(y(-1, L))
    print()
    print(rho)

    rho.clip_gauge()
    print()
    print(rho)
    

    L = 10
    import random
    random.seed(1)
    rho = DensityMatrix([x(i, L) for i in range(L)])
    print(rho)
    for _ in range(100):
        rho.measure(random.choice((x, y, z))(random.randrange(L), L) @ random.choice((x, y, z))(random.randrange(L), L))
    
    print()
    print(rho)
    rho.clip_gauge()
    print(rho)


    rho = ChunkedState([x(i, L) for i in range(L)])
    print("Testing reversability")
    print(rho)
    choices = rho.chunked_random_unitary(0, L)
    rho.clip_gauge()
    print(rho)
    rho.inverse_unitary(0, L, choices)
    rho.clip_gauge()
    print(rho)

    rho = DensityMatrix([x(i, L) for i in range(L)])
    choices = rho.random_unitary(0, L, L)
    rho2 = DensityMatrix([x(i, L) for i in range(L)])
    rho2.unitary(0, L, L, choices)
    assert str(rho) == str(rho2)
    rho2.inverse_unitary(0, L, L, choices)

    rho2.clip_gauge()
    rho3 = DensityMatrix([x(i, L) for i in range(L)])
    assert str(rho2) == str(rho3)

    L=20
    region_start = 5
    region_end = 15
    rho = ChunkedState([z(i, L) for i in range(L)])
    rho.chunked_random_unitary(0, L)
    #now measure out the ends

    outrho = DensityMatrix([z(i, region_end - region_start) for i in range(region_end - region_start)])
    
    try:
        rho.get_substate(region_start, region_end, outrho)
    except AssertionError:
        pass
    else:
        raise AssertionError("Test Failed to detect something wrong with the subregion")

    for i in range(region_start):
        rho.measure(z(i, L))
    for i in range(region_end, L):
        rho.measure(z(i, L))
    
    rho.get_substate(region_start, region_end, outrho)

    print(rho)
    print(outrho)

    print("Passed new test!")


    L = 20
    rho = ChunkedState([x(i, L) for i in range(L)])
    
    rho.chunked_random_unitary(0, L) # do some random unitary
    for i in range(L):
        for j in range(i,L):
            if rho.contiguous_entropy(i, j) != rho.entropy(list(range(i, j))):
                print((i, j),":", rho.contiguous_entropy(i, j), rho.entropy(list(range(i, j))))

    rho = ChunkedState([x(0, 3) @ x(1, 3) @ x(2, 3), z(0, 3) @ z(1, 3), z(1, 3) @ z(2, 3)])
    assert rho.contiguous_entropy(0, 1) == rho.entropy(list(range(1)))
    assert rho.contiguous_entropy(0, 2) == rho.entropy(list(range(2)))
    assert rho.contiguous_entropy(1, 2) == rho.entropy(list(range(1,2)))
    assert rho.contiguous_entropy(0, 3) == rho.entropy(list(range(3)))

    assert rho.entropies_of_contiguous_regions([(0, 1), (0, 2), (1, 2), (0, 3)]) == [rho.contiguous_entropy(0, 1), rho.contiguous_entropy(0, 2), rho.contiguous_entropy(1, 2), rho.contiguous_entropy(0, 3)]

    pauli = x(0, 1)
    pauli.single_qbit(0, (True, False), (False, True))
    assert pauli == x(0, 1)
    pauli.single_qbit(0, (True, True), (False, True))
    assert pauli == y(0, 1)
    pauli.single_qbit(0, (True, True), (False, True))
    assert pauli == x(0, 1)
    pauli.single_qbit(0, (False, True), (True, False))
    assert pauli == z(0, 1)

    print("All tests passed")



def stabilizer_lengths(L, p):
    def z(n, length):
        x_str = np.zeros(length, dtype=bool)
        z_str = np.zeros(length, dtype=bool)
        z_str[n] = True
        return PauliString(x_str, z_str)

    z_operators = [z(i, L) for i in range(L)]

    rho = DensityMatrix([z(i, L) for i in range(L)])

    for i in progressbar.progressbar(range(L)):
        for loc_1 in range(i % 2, L, 2):
            loc_2 = (loc_1 + 1) % L
            random.choice((rho.x_2, rho.y_2, rho.z_2))(loc_1)
            random.choice((rho.x_2, rho.y_2, rho.z_2))(loc_2)
            random.choice((rho.cnot, rho.swap, rho.iswap))(loc_1, loc_2)
            random.choice((rho.x_2, rho.y_2, rho.z_2))(loc_1)
            random.choice((rho.x_2, rho.y_2, rho.z_2))(loc_2)

        for loc in range(L):
            if random.random() < p:
                rho.measure(z_operators[loc])

    lengths = []

    rho.clip_gauge()

    for stabilizer in rho.stabilizers:
        lengths.append(1 + stabilizer.right() - stabilizer.left())
    
    return lengths, rho

def time():
    def round(state: DensityMatrix, p: float):
        #for each point do a random single site
        for i in range(len(state)):
            state.random_single_qbit(i)
            if random.random() < p:
                state.measure(z(i, len(state)))
        
        #then do brickwork random two site dual unitary
        for i in range(0, len(state), 2):
            if random.random() < 1/9:
                state.swap(i, (i + 1) % len(state))
            else:
                state.iswap(i, (i + 1) % len(state))
        
        #for each point do a random single site
        for i in range(len(state)):
            if random.random() < p:
                state.measure(z(i, len(state)))
            state.random_single_qbit(i)
        
        #then do the other brickwork random two site dual unitary
        for i in range(1, len(state), 2):
            if random.random() < 1/9:
                state.swap(i, (i + 1) % len(state))
        

    def I3(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], state: DensityMatrix) -> int:
        Ha, Hb, Hc = state.entropies_of_contiguous_regions([A, B, C])

        A = list(range(A[0], A[1]))
        B = list(range(B[0], B[1]))
        C = list(range(C[0], C[1]))

        Hab = state.entropy(A + B)
        Hac = state.entropy(A + C)
        Hbc = state.entropy(B + C)
        Habc = state.entropy(A + B + C)

        return Ha + Hb + Hc - Hab - Hac - Hbc + Habc



    def run(L: int, p):
        stabilizers = [z(i, L) for i in range(L)]
        state = DensityMatrix(stabilizers)

        for i in progressbar.progressbar(range(L * 3)):
            round(state, p)

        A = (0, L // 4)
        B = (L // 3, L * 7 // 12)
        C = (L * 2 // 3, L * 11 // 12)
        
        return I3(A, B, C, state)

    for p in [.01, .05, .1]:
        print(f"{p:.4e}  I_3 = {run(512, p)}")


if __name__ == '__main__':
    test()
    """
    random.seed(1)
    # L = 128
    # pmin = 0
    # pmax = .25
    # ps = np.linspace(pmin, pmax, 26)

    # entropies = []
    # rhos = []
    # for probability in ps:
    #     print(f'------------ p = {probability:.2e} ------------')
    #     lengths, rho = stabilizer_lengths(L, probability)

    #     rhos.append(rho)
    #     entropies.append(np.mean(np.array(lengths)**2 / L))
        
    
    # with open(f'./density_matrices_{L=}_{pmin=}_{pmax=}.dat', 'wb') as f:
    #     pickle.dump(rhos, f)

    # plt.plot(ps, entropies)
    # plt.show()

    ps = np.linspace(.1, .2, 11)
    Ls = [8, 16, 32, 64, 128, 256]
    TRIALS = 20
    rhos = {}

    for trial in range(TRIALS):
        for L in Ls:
            for probability in ps:        
                print(f"{L = }  {probability = }")
                lengths, rho = stabilizer_lengths(L, probability)
                rhos[(L, probability, trial)] = rho
                print(np.sum(np.array(lengths) / L))

        
                with open('./about_the_transition.dat', 'wb') as f:
                    pickle.dump(rhos, f)"""
        
