from owlready2 import *
import rdflib
import os

import numpy as np
import random
from copy import copy



def set_dict(_dict:dict, key:int, value:int):
    if key in _dict:
        _dict[key].append(value)
    else:
        _dict[key] = [value]


def get_edges(entities:dict, USE_UNDEFINED_EDGES:bool=True):
    # _edges will contain a list of tuples of all the edges in an ontology
    # _parents is a dictionary of all the parents of a given key
    # _children is a dictionary of all the children of a given key
    _edges    = []
    _parents  = {}
    _children = {}

    # loop over all of the entities in the dictionary of human entities
    for e in entities:
        _node1 = entities[e]

        # loop over the entity's 'is_a' relations
        for i in e.is_a:

            # this condition is for UNDEFINED edges
            if hasattr(i, 'value'):
                _node2 = entities[i.value]
            # this condition is for subClassOf relations
            else:
                _node2 = entities[i]

            # add the edge between two nodes to a list of edges
            _edges.append((_node1, _node2))

            # add the edge to a dictionary that contains all the parents and children
            set_dict(_parents, _node1, _node2)
            set_dict(_children, _node2, _node1)

    return _edges, _parents, _children


def get_siblings(parents:dict, children:dict, node:int):
    siblings = []

    # There should only be only one node that doesn't have any parents, the root node
    if node in parents:
        parents_of_node = parents[node]

        # Cycle through all possible parents of the given node
        for p in parents_of_node:

            # if the parent node has any children, add them to the siblings list
            if p in children:
                siblings = siblings + children[p]

        # remove the node from the siblings list - ambiguous
        # print(siblings)
        siblings.remove(node)

        # if there are any siblings, return the list of them
        if siblings:
            return siblings

#         tmp = copy(siblings.remove(node))
#         if tmp:
#             return tmp

        # if there are no siblings, return -1
        else:
            print("Given node does not have any siblings:", node)
            return -1

    # if the node does not have any parents, return -1
    else:
        print("Given node does not have any parents:", node)
        return -1
    # ----

def generate_true_neg_alignments(alignments:list, h_parents, h_children, m_parents, m_children, alignment_split:float=0.5, ratio:float=1.0):

    true_negatives = []
    numFailures = 0
    num_samples = int(len(alignments) * alignment_split * ratio)

    print(len(alignments), num_samples, alignment_split)

    while (len(true_negatives) < num_samples) and (numFailures < 100):
        # Select a random alignment within the list of all alignments
        rdm_align = random.choice(alignments)

        # Pick a node to alter within the randomly chosen alignment
        const_node = rdm_align[0]
        change_node = rdm_align[1]
        # const_node  = 2979
        # change_node = 1589

        # generate all siblings within the human ontology of the chosen node
        if change_node in h_parents:
            siblings = get_siblings(h_parents, h_children, change_node)

        # generate all siblings within the mouse ontology of the chosen node
        elif change_node in m_parents:
            siblings = get_siblings(m_parents, m_children, change_node)

        # This shouldn't be triggered -- every node should have a parent node
        # The only possible node that could trigger the below statement is the root node
        else:
            print("Node not found in either Ontology or does not have any parents")


        # This error will typically be thrown if the chosen node does not have any siblings
        if siblings == -1:
            print("Error thrown when retrieving siblings")

        else:
            # Choose some random siblings to be make the true negative
            negative_alignment = (const_node, random.choice(siblings))

            if negative_alignment in alignments:
                numFailures += 1
                print("Generated negative is an existing alignment:", negative_alignment, "OG random:", rdm_align, siblings)
                pass

            elif negative_alignment in true_negatives:
                numFailures += 1
                print("Generated negative already in true_negatives:", negative_alignment)
                pass

            # include this negative alignment in the true_negatives list
            else:
                true_negatives.append(negative_alignment)
                true_negatives.append((negative_alignment[1], const_node))
                numFailures = 0


    return true_negatives


# TODO: Generate a number of true negative alignments
# TODO: Make sure the human ontology has it's own root node (don't share 0 b/w the two)


def main():
    # PATH = "../datasets/anatomy/"
    PATH = "datasets/anatomy/"
    MOUSEFILE = 'mouse.owl'
    HUMANFILE = 'human.owl'
    REFFILE   = 'reference.rdf'

    mouse_onto = get_ontology(PATH+MOUSEFILE).load()
    human_onto = get_ontology(PATH+HUMANFILE).load()

    mc = list(mouse_onto.classes())
    hc = list(human_onto.classes())

    mouse_entities = {}
    human_entities = {}
    entities       = {}
    entities_names = {}

    # We need to save the object 'owl.Thing' in the entity dictionaries,
    # Find the owl.Thing object (typically in position 0) and save its position within the mouse classes
    for i, m in enumerate(mc):
        if m is owl.Thing:
            ot = i
            print("owl.Thing is located at position:", ot)
            break

    # insert the owl.Thing object as the first object within the dictionary
    mouse_entities[mc[ot]]      = 0
    human_entities[mc[ot]]      = 0
    entities[mc[ot]]            = 0
    entities_names[mc[ot].name] = 0

    for c in mc:
        if "MA" in c.name: # filter only the classes, prevent "OboInOwl" types being included in the dictionary
            if c not in entities:
                mouse_entities[c] = len(entities)
                entities[c] = len(entities)
                entities_names[c.name] = len(entities_names)

    for c in hc:
        if "NCI" in c.name: # filter only the classes, prevent "OboInOwl" types being included in the dictionary
            if c not in entities:
                human_entities[c] = len(entities)
                entities[c] = len(entities)
                entities_names[c.name] = len(entities_names)

    idx_to_entity = dict((v,k) for k,v in entities.items())
    idx_to_entity_names = dict((v,k) for k,v in entities_names.items())
    idx_to_mouse_entity = dict((v,k) for k,v in mouse_entities.items())
    idx_to_human_entity = dict((v,k) for k,v in human_entities.items())


    h_edges, h_parents, h_children = get_edges(human_entities)
    m_edges, m_parents, m_children = get_edges(mouse_entities)

    g = rdflib.Graph()
    g.load(PATH+REFFILE)

    # Create a list of triples that contain:
    # (Subject, Predicate, Object)
    SPOtriple = []

    for sub, pred, obj in g:
        SPOtriple.append((sub, pred, obj))

    alignment_subjects = {}
    alignments  = []

    # Make a count of how often subjects occur in this list of triples
    for t in SPOtriple:
        if t[0] in alignment_subjects:
            alignment_subjects[t[0]] += 1
        else:
            alignment_subjects[t[0]] = 1

    # Loop through every alignment subject (one alignment between two nodes)
    for i, key in enumerate(alignment_subjects):
        # Make a list to get the two aligned nodes
        alignment_pair = []

        # Search for the two objects that contain the same subject alignment number
        for t in SPOtriple:
            if key in t[0]:
                if "alignmententity" in t[1]:
                    # Save the index of the entity to the alignment pair
                    get_iri = human_onto.search(iri=t[2])[0]
                    alignment_pair.append(entities[get_iri])

        # There are some SPOtriples that do not actually contain an alignment, so we ignore those.
        if alignment_pair:
            # Save 'is_a' relations in both directions, as alignments are bi-directional
            alignments.append(tuple(alignment_pair))
            alignments.append(tuple(alignment_pair[::-1])) # saves the reverse of the string

    true_negatives = generate_true_neg_alignments(alignments, h_parents, h_children, m_parents, m_children)





if __name__ == "__main__":
    main()
