from typing import List, Tuple

class Permutation:
    def __init__(self, size: int, permutation: List[int]):
        """
        permutation tells me where each index is mapped to, so that
        permutation[index] -> index is the mapping
        """
        self.permutation = permutation.copy()
        self.size = size

        assert len(set(permutation)) == len(permutation) == size
        assert min(permutation) == 0
        assert max(permutation) == len(permutation) - 1

    
    def is_contiguous_after_permutation(self, region: List[int]) -> bool:
        """
        Returns [left_endpoint, right_endpoint) of the region after the new permutation
        so that the entropy of the region is equal to the entropy of the region range(left_endpoint, right_endpoint)
        after the permtuation. If it is not contiguous return False.
        """
        if len(region) == 0:
            # empty regions are always contiguous 
            return True

        permuted_region = self.apply(region)
        permuted_region.sort()
        assert (0 <= min(region) <= max(region) < self.size)

        # find the first break in the region
        # if there is no break then just return the endpoints
        for (i, j) in zip(permuted_region[:-1], permuted_region[1:]):
            if i + 1 != j:
                break
        
        else:
            return (min(permuted_region), max(permuted_region) + 1)
        
        # otherwise we shift the region and test if it's contiguous
        shifted = [(j - i - 1) % self.size for j in permuted_region]
        
        l_end = (min(shifted) + i + 1) % self.size
        r_end = (max(shifted) + i + 1) % self.size

        # add plus one to the min and not the max because 
        # the region is inverted
        if max(shifted) - min(shifted) == len(region) - 1:
            return (min(l_end, r_end) + 1, max(l_end, r_end))
        
        return False

    def inverse(self) -> "Permutation":
        out = [None] * self.size
        for i in range(self.size):
            out[self.permutation[i]] = i
        
        return Permutation(self.size, out)
    
    def apply(self, region: List[int]) -> List[int]:
        return [self.permutation[i] for i in region]

    def __repr__(self):
        return f"Permutation({self.size}, {self.permutation})"


def split_regions_by_permutation(regions: List[List[int]], perms: List[Permutation]) -> List[List[Tuple[int, List[int], Tuple[int, int]]]]:
    """
    Returns a partition of the regions into subsets such that each subset corresponds
    to a permutation which will convert the regions it contains into contiguous regions.
    Each region is also given with its new endpoints after the permutation
    out = [subset1, subset2, ...]

    subset_i = [(original_region_index, region, (start, stop)), ...]
    """
    region_division = [[] for _ in perms]

    for region_index, region in enumerate(regions):
        for i, permutation in enumerate(perms):
            new_endpoints = permutation.is_contiguous_after_permutation(region)
            if new_endpoints:
                region_division[i].append((region_index, region, new_endpoints))
                break
        else:
            print(region)
            raise LookupError("Did not find a permutation in the set that brought a region into normal form")

    return region_division