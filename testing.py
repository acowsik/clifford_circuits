from clifford_circuits import clifford
from clifford_circuits.permutation import Permutation, split_regions_by_permutation

def test():
    import random
    L = 16

    random_perm = list(range(L))
    random.shuffle(random_perm)
    
    random_perm = Permutation(L, random_perm)

    hopefully_identity = random_perm.apply(random_perm.inverse().apply(list(range(L))))
    assert hopefully_identity == list(range(L))

    hopefully_identity = random_perm.inverse().apply(random_perm.apply(list(range(L))))
    assert hopefully_identity == list(range(L))

    id_perm = Permutation(L, list(range(L)))
    assert id_perm.apply([13, 14, 15, 0]) == [13, 14, 15, 0]
    assert id_perm.is_contiguous_after_permutation([13, 14, 15, 0])

    print("Basic tests passed.")

    regions = [[(start + i) % L for i in range(L // 4)] for start in range(L)]
    regions.extend([region + [L] for region in regions])
    regions.append(list(range(L + 1)))

    permutations = [Permutation(L + 1, list(range(L + 1)))]

    for ancilla_location in range(0, L , L // 4):
        perm = list(range(ancilla_location)) + [L] + list(range(ancilla_location, L))
        permutations.append(Permutation(L + 1, perm).inverse())


    for perm, regions in zip(permutations, split_regions_by_permutation(regions, permutations)):
        print("Permutation :", perm)
        for region_idx, region, new_endpoints in regions:
            print(region_idx, region, new_endpoints)

            new_region = perm.apply(region)
            assert set(new_region) == set(range(*new_endpoints)) or (set(new_region).intersection(range(*new_endpoints)) == set() and set(new_region + list(range(*new_endpoints))) == set(range(L + 1)))


        print('\n\n')

    import random
    import time

    
    
    L = 256

    empty_state = clifford.DensityMatrix([clifford.z(i, L) for i in range(L)])

    
    # see if permutations preserve the entropy
    for trial in range(3):
        state = empty_state.copy()
        state.random_unitary(0, L, L // 4)
        permutation = list(range(L))
        random.shuffle(permutation)
        permutation = Permutation(L, permutation)
        region = random.sample(range(L), 32)

        before_entropy = state.entropy(region)
        new_region = permutation.apply(region)
        state.permute(permutation)
        after_entropy = state.entropy(new_region)

        assert before_entropy == after_entropy
    

    # see if the splitting generates the right intervals by making it all trivial
    permutation = Permutation(L, list(range(L)))
    regions = [(random.randrange(0, L), random.randrange(0, L)) for _ in range(40)]
    regions = map(lambda x: (min(x), max(x)), regions)
    regions = filter(lambda x: x[0] != x[1], regions)
    endpoints = list(regions)
    regions = map(lambda x: list(range(*x)), endpoints)
    regions = list(regions)

    breakups = split_regions_by_permutation(regions, [permutation])
    assert len(breakups) == 1

    indices = [i[0] for i in breakups[0]]
    derived_regions = [i[1] for i in breakups[0]]
    derived_endpoints = [i[2] for i in breakups[0]]

    assert indices == list(range(len(indices)))
    assert derived_regions == regions
    assert derived_endpoints == endpoints

    entropies = [state.entropy(region) for region in regions]
    derived_entropies = state.entropies_of_permutably_contiguous_regions(regions, [permutation], breakups, empty_state)

    assert entropies == derived_entropies

    

    # now check regions which wrap around
    regions = [(random.randrange(0, L), random.randrange(0, L)) for _ in range(40)]
    regions = map(lambda x: (min(x), max(x)), regions)
    regions = filter(lambda x: x[0] + 2 < x[1], regions)
    endpoints = list(regions)
    regions = map(lambda x: [i for i in range(L) if i not in range(*x)], endpoints)
    regions = list(regions)

    breakups = split_regions_by_permutation(regions, [permutation])
    assert len(breakups) == 1


    indices = [i[0] for i in breakups[0]]
    derived_regions = [i[1] for i in breakups[0]]
    derived_endpoints = [i[2] for i in breakups[0]]

    assert indices == list(range(len(indices)))
    assert derived_regions == regions

    assert derived_endpoints == endpoints

    entropies = [state.entropy(region) for region in regions]
    derived_entropies = state.entropies_of_permutably_contiguous_regions(regions, [permutation], breakups, empty_state)


    # now check with randomly permuted regions and a custom permutation to fix that region
    permutations = [Permutation(L, random.sample(range(L), L)) for _ in range(40)]
    regions = [perm.apply(list(range(L // 4))) for perm in permutations]
    permutations = [perm.inverse() for perm in permutations]

    breakups = split_regions_by_permutation(regions, permutations)
    indices = [i[0][0] for i in breakups]
    derived_regions = [i[0][1] for i in breakups]
    derived_endpoints = [i[0][2] for i in breakups]

    assert indices == list(range(len(indices)))
    print(derived_endpoints)
    assert derived_endpoints == [(0, L // 4)] * len(derived_endpoints)

    print()
    print(derived_regions[0])
    assert derived_regions == regions
    

    entropies = [state.entropy(region) for region in regions]
    derived_entropies = state.entropies_of_permutably_contiguous_regions(regions, permutations, breakups, empty_state)

    assert entropies == derived_entropies

    print("New test done")

    regions = [[(start + i) % L for i in range(L // 4)] for start in range(L)]
    regions.extend([region + [L] for region in regions])
    regions.append(list(range(L + 1)))

    permutations = [Permutation(L + 1, list(range(L + 1)))]

    for ancilla_location in range(0, L , L // 4):
        perm = list(range(ancilla_location)) + [L] + list(range(ancilla_location, L))
        permutations.append(Permutation(L + 1, perm).inverse())

    L = L + 1

    state = clifford.DensityMatrix([clifford.z(i, L) for i in range(L)])
    state.random_unitary(0, L, L // 4)

    breakups = split_regions_by_permutation(regions, permutations)


    stime = time.time()
    entropies_correct = [state.entropy(region) for region in regions]
    print(f"Time part 1 took {time.time() - stime}")
    stime = time.time()
    entropies_test = state.entropies_of_permutably_contiguous_regions(regions, permutations, breakups, state.copy())
    print(f"Time part 2 took {time.time() - stime}")

    print('test_ent')
    print(entropies_test)

    assert sorted(entropies_correct) == sorted(entropies_test)
    assert entropies_correct == entropies_test

    state.clip_gauge()

    copy_state = clifford.DensityMatrix([clifford.z(i, L) for i in range(L)])
    copy_state.clone_from(state)

    permutation = list(range(L)) 
    random.shuffle(permutation)
    permutation = Permutation(L, permutation)

    print(permutation)

    copy_state.permute(permutation) 
    copy_state.permute(permutation.inverse())
    
    for s1, s2 in zip(copy_state.stabilizers, state.stabilizers):
        assert s1 == s2
    


    print("All tests passed!")

print('test')
clifford.test()
test()
clifford.time()
