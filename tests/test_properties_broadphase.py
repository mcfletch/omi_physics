"""Properties the broad phase and the contact islands hold for any scene.

The broad phase decides what the rest of the step is even asked about. A pair it
drops is a collision that never happens -- a character through a wall, a car
through the track -- and it is the hardest kind of defect to see, because
nothing reports an error and the scene merely behaves as though the geometry
were not there.

Two claims are made in the module and are checked here against the thing they
claim to equal: the sweep-and-prune yields the same pairs as testing every body
against every other, and the tree answers a box query with the leaves that
overlap it.
"""
import numpy as np
import pytest

# Hypothesis is a CPython-only test dependency; see test_properties_math.py.
pytest.importorskip('hypothesis', reason='hypothesis has no wheel for this interpreter')

from hypothesis import given
from hypothesis import strategies as st
from property_strategies import SETTINGS, vectors, worlds

from omi_physics import solver
from omi_physics.broadphase import BroadPhase, DynamicAABBTree, brute_force_pairs
from omi_physics.collide import Contact


def boxes() -> st.SearchStrategy:
    """A ``(lo, hi)`` pair, flat and zero-size boxes included."""
    return st.tuples(vectors(50.0), vectors(10.0)).map(
        lambda p: (p[0], p[0] + np.abs(p[1])))


def overlapping(a_lo, a_hi, b_lo, b_hi) -> bool:
    """True when two closed boxes share any point."""
    return bool(np.all(a_lo <= b_hi) and np.all(b_lo <= a_hi))


def check_tree(tree: DynamicAABBTree) -> None:
    """Every structural invariant the tree relies on to answer a query.

    A tree that has lost these still answers -- with the wrong leaves, or with
    the same leaf twice -- so they are asserted rather than inferred from the
    answers being plausible.
    """
    seen = []

    def walk(node, parent):
        assert node.parent is parent
        if node.is_leaf:
            seen.append(node)
            return
        assert node.child1 is not None and node.child2 is not None
        for child in (node.child1, node.child2):
            walk(child, node)
            # A parent's box holds its children's, or a query stops descending
            # before it reaches the leaf that would have answered it.
            assert np.all(np.asarray(node.lo) <= np.asarray(child.lo) + 1e-9)
            assert np.all(np.asarray(node.hi) >= np.asarray(child.hi) - 1e-9)
        assert node.height == 1 + max(node.child1.height, node.child2.height)

    if tree.root is not None:
        walk(tree.root, None)
    assert sorted(n.obj for n in seen) == sorted(tree.leaves)


class TestTheTreeStaysATree:
    @given(items=st.lists(st.tuples(st.integers(0, 12), boxes()),
                          min_size=1, max_size=20))
    @SETTINGS
    def test_inserting_and_refitting_leaves_it_consistent(self, items):
        tree = DynamicAABBTree()
        for obj, (lo, hi) in items:
            tree.update(obj, lo, hi)
            check_tree(tree)

    @given(items=st.lists(st.tuples(st.integers(0, 8), boxes()),
                          min_size=1, max_size=16),
           removals=st.lists(st.integers(0, 8), max_size=8))
    @SETTINGS
    def test_removing_leaves_it_consistent(self, items, removals):
        tree = DynamicAABBTree()
        for obj, (lo, hi) in items:
            tree.update(obj, lo, hi)
        for obj in removals:
            tree.remove(obj)
            check_tree(tree)

    @given(items=st.lists(st.tuples(st.integers(0, 8), boxes()),
                          min_size=1, max_size=16))
    @SETTINGS
    def test_removing_everything_empties_it(self, items):
        tree = DynamicAABBTree()
        for obj, (lo, hi) in items:
            tree.update(obj, lo, hi)
        for obj in {obj for obj, _ in items}:
            tree.remove(obj)
        assert tree.leaves == {}
        assert tree.root is None


class TestTheTreeAnswersWhatItHolds:
    @given(items=st.lists(st.tuples(st.integers(0, 12), boxes()),
                          min_size=1, max_size=20),
           probe=boxes())
    @SETTINGS
    def test_a_query_finds_exactly_the_overlapping_leaves(self, items, probe):
        tree = DynamicAABBTree()
        for obj, (lo, hi) in items:
            tree.update(obj, lo, hi)
        lo, hi = probe
        found = tree.query(lo, hi)
        assert len(found) == len(set(found))
        expected = {leaf.obj for leaf in tree.all_leaves()
                    if overlapping(np.asarray(leaf.lo), np.asarray(leaf.hi), lo, hi)}
        assert set(found) == expected

    @given(items=st.lists(st.tuples(st.integers(0, 12), boxes()),
                          min_size=1, max_size=20),
           probe=boxes(), skip=st.integers(0, 12))
    @SETTINGS
    def test_a_skipped_object_is_never_returned(self, items, probe, skip):
        tree = DynamicAABBTree()
        for obj, (lo, hi) in items:
            tree.update(obj, lo, hi)
        assert skip not in tree.query(probe[0], probe[1], skip=skip)


class TestTheSweepFindsWhatTheLoopWouldFind:
    """`BroadPhase.pairs` says it equals the brute-force set. Its sweep is a
    sort, a `searchsorted` and a stack of masks, none of which resembles the
    loop it replaces, so the two are compared over whatever scenes turn up."""

    @given(world=worlds(max_bodies=8))
    @SETTINGS
    def test_the_pairs_are_the_brute_force_pairs(self, world):
        assert set(BroadPhase().pairs(world)) == set(brute_force_pairs(world))

    @given(world=worlds(max_bodies=8, filters=True))
    @SETTINGS
    def test_collision_filters_do_not_change_that(self, world):
        assert set(BroadPhase().pairs(world)) == set(brute_force_pairs(world))

    @given(world=worlds(max_bodies=8))
    @SETTINGS
    def test_each_pair_appears_once_with_its_indices_in_order(self, world):
        pairs = BroadPhase().pairs(world)
        assert len(pairs) == len(set(pairs))
        assert all(a < b for a, b in pairs)

    @given(world=worlds(max_bodies=6))
    @SETTINGS
    def test_stepping_does_not_change_the_agreement(self, world):
        """The boxes are refit every step, so what matters is that the two stay
        equal as the scene moves rather than that they start equal."""
        broad = BroadPhase()
        for _ in range(5):
            world.step(1.0 / 120.0)
            assert set(broad.pairs(world)) == set(brute_force_pairs(world))


class TestIslandsPartitionTheContacts:
    """Islands are solved independently, so a contact in the wrong one is solved
    against the wrong bodies -- and a contact in none is not solved at all."""

    @staticmethod
    def contacts_from(world, count, rng):
        n = world.body_count
        return [Contact(int(rng[i] % n), int(rng[i + count] % n),
                        np.zeros(3), np.array([0.0, 1.0, 0.0]), 0.1)
                for i in range(count)]

    @given(world=worlds(min_bodies=2, max_bodies=6),
           picks=st.lists(st.integers(0, 30), min_size=2, max_size=24))
    @SETTINGS
    def test_every_contact_lands_in_exactly_one_island(self, world, picks):
        contacts = self.contacts_from(world, len(picks) // 2, picks)
        islands = solver.build_islands(world, contacts)
        placed = [c for island in islands for c in island]
        assert len(placed) == len(contacts)
        assert {id(c) for c in placed} == {id(c) for c in contacts}

    @given(world=worlds(min_bodies=2, max_bodies=6),
           picks=st.lists(st.integers(0, 30), min_size=2, max_size=24))
    @SETTINGS
    def test_no_dynamic_body_appears_in_two_islands(self, world, picks):
        """The point of an island: solving one cannot change another's answer."""
        contacts = self.contacts_from(world, len(picks) // 2, picks)
        islands = solver.build_islands(world, contacts)
        dynamic = world.motion_type == 2
        seen = {}
        for number, island in enumerate(islands):
            for c in island:
                for body in (c.a, c.b):
                    if not dynamic[body]:
                        continue
                    assert seen.setdefault(body, number) == number
