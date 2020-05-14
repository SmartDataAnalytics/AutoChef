# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Recipe class



# %%
import sys

import settings

import pycrfsuite

import json

import db.db_settings as db_settings
from db.database_connection import DatabaseConnection

from Tagging.conllu_generator import ConlluGenerator
from Tagging.crf_data_generator import *

from difflib import SequenceMatcher

import numpy as np

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)

from graphviz import Digraph

import itertools


import plotly.io as pio
pio.renderers.default = "jupyterlab"

from IPython.display import Markdown, HTML, display

# %% [markdown]
# * sequence similarity matcher

# %%
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# %%
def string_similarity(a,b):
    """
    does the same like `similar` but also compares single words of multi word tokens
    and returns the max similar value
    """
    
    tokens_a = a.split()
    tokens_b = b.split()
    
    max_similarity = -1
    max_a = None
    max_b = None
    
    for t_a in tokens_a:
        for t_b in tokens_b:
            s = similar(t_a, t_b)
            if s > max_similarity:
                max_similarity = s
                max_a = t_a,
                max_b = t_b,
    
    return max_similarity, max_a, max_b

# %% [markdown]
# * get vocabulary

# %%
import importlib.util
# loading ingredients:
spec = importlib.util.spec_from_file_location(
    "ingredients", "/".join(str(settings.__file__).split("/")[:-1]) + "/" + settings.ingredients_file)
ingredients = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingredients)

# loading actions:
spec = importlib.util.spec_from_file_location(
    "actions", "/".join(str(settings.__file__).split("/")[:-1]) + "/" + settings.actions_file)
actions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actions)

# loading containers
spec = importlib.util.spec_from_file_location(
    "containers", "/".join(str(settings.__file__).split("/")[:-1]) + "/" + settings.container_file)
containers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(containers)

# loading placeholders
spec = importlib.util.spec_from_file_location(
    "placeholders", "/".join(str(settings.__file__).split("/")[:-1]) + "/" + settings.placeholder_file)
placeholders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placeholders)


# %%
# helper function since the lemmatizer not always lemmatize in a meaningful way :shrug:
def check_ingredient(ing_token):
    form = ing_token['form'].lower()
    lemma = ing_token['lemma'].lower()
    
    if form in ingredients.ingredients:
        return True
    
    if lemma in ingredients.ingredients_stemmed:
        return True
    
    if lemma.endswith('s'):
        if lemma[:-1] in ingredients.ingredients_stemmed:
            return True
    
    else:
        if lemma + 's' in ingredients.ingredients_stemmed:
            return True
    
    return False


# %%
tagger = pycrfsuite.Tagger()
tagger.open("/".join(str(settings.__file__).split("/")[:-1]) + "/" + "/Tagging/test.crfsuite")


# %%
id_query = "select * from recipes where id like %s"


# %%
def escape_md_chars(s):
    s = s.replace("*", "\*")
    s = s.replace("(", "\(")
    s = s.replace(")", "\)")
    s = s.replace("[", "\[")
    s = s.replace("]", "\]")
    s = s.replace("_", "\_")
    
    return s


# %%
import json
class Ingredient(object):
    
    @staticmethod
    def from_json(j):
        d = json.loads(j)
        ing = Ingredient(d['base'])
        ing._action_set = set(d['actions'])
        return ing
    
    def __init__(self, base_ingredient, last_touched_instruction=0):
        self._base_ingredient = base_ingredient
        self._action_set = set()
        self._last_touched_instruction = last_touched_instruction
        self._is_mixed = False
    
    def apply_action(self, action, instruction_number=0, touch=True):
        if action in actions.mixing_cooking_verbs:
            self.mark_for_mixing()
        else:
            self._action_set.add(action)
        
        if touch:
            self._last_touched_instruction = instruction_number
        
        return self
    
    def similarity(self, ingredient, use_actions=False, action_factor = 0.5):
        sim,_,_ = string_similarity(self._base_ingredient, ingredient._base_ingredient)
        if not use_actions:
            return sim
        
        return (1 - action_factor) + action_factor * similar(list(self._action_set), list(ingredient._action_set))
    
    def mark_for_mixing(self):
        self._is_mixed = True
    
    def unmark_mixing(self):
        self._is_mixed = False
    
    def is_mixed(self):
        return self._is_mixed
    
    def most_similar_ingredient(self, ing_list, use_actions=False, action_factor=0.5):
        best_index = -1
        best_value = -1
        
        for i, ing in enumerate(ing_list):
            sim = self.similarity(ing, use_actions=use_actions, action_factor=action_factor)
            if sim > best_value:
                best_value = sim
                best_index = i
        return best_value, ing_list[best_index]
    
    def copy(self):
        result = Ingredient(self._base_ingredient, self._last_touched_instruction)
        result._action_set = self._action_set.copy()
        result._is_mixed = self._is_mixed
        
        return result
    
    def to_json(self):
        result = {}
        result['base'] = self._base_ingredient
        result['actions'] = list(self._action_set)
        return json.dumps(result)
    
    def __repr__(self):
        return f"{'|'.join(list(self._action_set))} 🠊 {self._base_ingredient} (last touched @ {self._last_touched_instruction})" 
    
    


# %%
class RecipeState(object):
    def __init__(self, initial_ingredients):
        self._ingredients = initial_ingredients
        self._seen_ingredients = set()
        self._seen_actions = set()
        
        self._mix_matrix = None
        self._mix_labels = None
        self._act_matrix = None
        self._act_labels = None
        self._ing_labels = None
        self._mat_need_update = True
        
        # set of (ing_a, ing_b) tuples
        self._seen_mixes = set()
        
        # set of (action, ing) tuples
        self._seen_applied_actions = set()
        
        for ing in self._ingredients:
            self._seen_ingredients.add(ing.to_json())
    
    def copy(self):
        return RecipeState([ing.copy() for ing in self._ingredients])
    
    def apply_action(self, action: str, ing: Ingredient, instruction_number=0, sim_threshold = 0.6, add_new_if_not_similar=True):
        # find most similar ingredient to the given one and apply action on it
        sim_val, best_ing = ing.most_similar_ingredient(self._ingredients)
        
        # if sim_val is good enough, we apply the action to the best ingredient, otherwise
        # we add a new ingredient to our set (and assume that it was not detected or listed in the
        # ingredient set before)
        
        self._mat_need_update = True
        
        if sim_val > sim_threshold:
            if action not in actions.stemmed_mixing_cooking_verbs:
                self._seen_actions.add(action)
                self._seen_applied_actions.add((action, best_ing.to_json()))
            best_ing.apply_action(action, instruction_number)
            self._seen_ingredients.add(best_ing.to_json())
        elif add_new_if_not_similar:
            self._ingredients.append(ing)
            if action not in actions.stemmed_mixing_cooking_verbs:
                self._seen_actions.add(action)
                self._seen_ingredients.add(ing.to_json())
                self._seen_applied_actions.add((action, ing.to_json()))
            ing.apply_action(action, instruction_number)
            self._seen_ingredients.add(ing.to_json())
    
    def apply_action_on_all(self, action, instruction_number=0, exclude_instruction_number=None):
        self._mat_need_update = True
        for ing in self._ingredients:
            if exclude_instruction_number is None or exclude_instruction_number != ing._last_touched_instruction:
                if action not in actions.stemmed_mixing_cooking_verbs:
                    self._seen_actions.add(action)
                    self._seen_applied_actions.add((action, ing.to_json()))
                ing.apply_action(action, instruction_number)
                self._seen_ingredients.add(ing.to_json())
    
    def apply_action_by_last_touched(action, last_touched_instruction, instruction_number=0):
        self._mat_need_update = True
        for ing in self.get_ingredients_touched_in_instruction(last_touched_instruction):
            if action not in actions.stemmed_mixing_cooking_verbs:
                self._seen_actions.add(action)
                self._seen_applied_actions.add((action, ing.to_json()))
            ing.apply_action(action, instruction_number)
            self._seen_ingredients.add(ing.to_json())
    
    def get_combined_ingredients(self):
        combined = []
        for ing in self._ingredients:
            if ing.is_mixed():
                combined.append(ing)
            ing.unmark_mixing()
        
        for x in combined:
            for y in combined:
                self._seen_mixes.add((x.to_json(), y.to_json()))
        
        self._mat_need_update = True
        return combined
    
    def _update_matrices(self):
        
        ing_list = list(self._seen_ingredients)
        idx = {}
        
        m = np.zeros((len(ing_list), len(ing_list)))
        
        for i,ing in enumerate(ing_list):
            idx[ing] = i
        
        for x,y in self._seen_mixes:
            m[idx[x], idx[y]] = 1
        
        self._mix_matrix = m
        self._mix_labels = [Ingredient.from_json(j) for j in ing_list]
        
        ing_list = list(self._seen_ingredients)
        idx_i = {}
        
        act_list = list(self._seen_actions)
        idx_a = {}
        
        for i,ing in enumerate(ing_list):
            idx_i[ing] = i
        
        for i,act in enumerate(act_list):
            idx_a[act] = i
        
        m = np.zeros((len(act_list), len(ing_list)))
        
        for act, ing in self._seen_applied_actions:
            m[idx_a[act], idx_i[ing]] = 1
        
        self._act_matrix = m
        self._act_labels = act_list
        self._ing_labels = [Ingredient.from_json(j) for j in ing_list]
        
        self._mat_need_update = False
        
    
    def get_mixing_matrix(self):    
        if self._mat_need_update:
            self._update_matrices()
        return self._mix_matrix, self._mix_labels

    
    def get_action_matrix(self):
        if self._mat_need_update:
            self._update_matrices()
        return self._act_matrix, self._act_labels, self._ing_labels
    
    
    def get_ingredients_touched_in_instruction(self, instruction_number = 0):
        ings = []
        for ing in self._ingredients:
            if ing._last_touched_instruction == instruction_number:
                ings.append(ing)
        return ings     
                
    
    def get_ingredients(self):
        return self._ingredients
    
    def __repr__(self):
        s = ""
        for ing in self._ingredients:
            s += f"• {str(ing)}\n"
        return s
        


# %%
class Node(object):
    def __init__(self, id, label, shape):
        self.id = id
        self.label = label
        self.shape = shape


# %%
class GraphWrapper(object):
    def __init__(self, comment="recipe graph"):
        self._comment = comment
        self._nodes = set()
        self._nodes_by_id = {}
        self._nodes_by_label = {}
        self._edges = set()
        self._to_node = {}
        self._from_node = {}
    
    def node(self, id, label, shape = None):
        assert id not in self._nodes_by_id
        n = Node(id, label, shape)
        self._nodes.add(n)
        self._nodes_by_id[id] = n
        if label not in self._nodes_by_label:
            self._nodes_by_label[label] = set()
        self._nodes_by_label[label].add(n)
        self._to_node[id] = set()
        self._from_node[id] = set()
    
    def edge(self, a, b):
        assert a in self._nodes_by_id and b in self._nodes_by_id
        self._edges.add((a,b))
        self._from_node[a].add(b)
        self._to_node[b].add(a)
    
    def remove_edge(self, a, b):
        self._edges.discard((a,b))
        if a in self._from_node:
            self._from_node[a].discard(b)
        if b in self._to_node:
            self._to_node[b].discard(a)
    
    def remove_node(self, id, redirect_edges=False):
        assert id in self._nodes_by_id
        
        if redirect_edges:
            f_set = self._from_node[id].copy()
            t_set = self._to_node[id].copy()
            
            self.remove_node(id)
            
            for a in t_set:
                for b in f_set:
                    self.edge(a,b)
            return
        
        # remove all edges
        b_set = self._from_node[id].copy()
        for b in b_set:
            self.remove_edge(id, b)

        a_set = self._to_node[id].copy()
        for a in a_set:
            self.remove_edge(a, id)
        
        # remove node itself
        n = self._nodes_by_id[id]
        self._nodes_by_label[n.label].remove(n)
        if len(self._nodes_by_label[n.label]) == 0:
            del(self._nodes_by_label[n.label])
        self._nodes.remove(n)
        del(self._nodes_by_id[id])
        del(self._from_node[id])
        del(self._to_node[id])
    
    def merge(self, a, b):
        """
        merge a with b and return id of merged node
        """
        assert a in self._nodes_by_id and b in self._nodes_by_id
        
        if (a,b) in self._edges:
            self.remove_edge(a,b)
        if (b,a) in self._edges:
            self.remove_edge(b,a)
        
        to_merged = set()
        from_merged = set()
        
        if a in self._from_node:
            from_merged = from_merged.union(self._from_node[a])
        if b in self._from_node:
            from_merged = from_merged.union(self._from_node[b])
        
        if a in self._to_node:
            to_merged = to_merged.union(self._to_node[a])
        if b in self._to_node:
            to_merged = to_merged.union(self._to_node[b])
        
        from_merged.discard(a)
        from_merged.discard(b)
        
        to_merged.discard(a)
        to_merged.discard(b)
        
        merged_node = self._nodes_by_id[a]
        
        self.remove_node(a)
        self.remove_node(b)
        
        self.node(merged_node.id, merged_node.label, merged_node.shape)
                
        for x in to_merged:
            self.edge(x, merged_node.id)
        
        for x in from_merged:
            self.edge(merged_node.id, x)
    
    def insert_before(self, node_id, insert_id, insert_label, insert_shape):
        assert insert_id not in self._nodes_by_id
        assert node_id in self._nodes_by_id
        to_node = self._to_node[node_id].copy()
        
        for a in to_node:
            self.remove_edge(a, node_id)
        
        self.node(insert_id, insert_label, insert_shape)
        
        for a in to_node:
            self.edge(a, insert_id)
        self.edge(insert_id, node_id)
    
    def merge_adjacent_with_label(self, label):
        """
        merge all adjacent nodes with given label
        """
        
        assert label in self._nodes_by_label
        
        node_set = self._nodes_by_label[label]
        mix_set = set()
        
        connected_clusters = {}
        
        for x in node_set:
            for y in node_set:
                if (x.id, y.id) in self._edges:
                    # mark for merge
                    mix_set.add(x.id)
                    mix_set.add(y.id)
                    
                    if x.id not in connected_clusters:
                        connected_clusters[x.id] = set()
                    if y.id not in connected_clusters:
                        connected_clusters[y.id] = set()
                    
                    u = connected_clusters[x.id].union(connected_clusters[y.id])
                    u.add(x.id)
                    u.add(y.id)
                    
                    for n in u:
                        connected_clusters[n] = u
        
        clusters = []
        while len(mix_set) > 0:
            arbitrary_node = mix_set.pop()
            # get cluster for node:
            c = connected_clusters[arbitrary_node]
            c_list = list(c)
            
            # merge all nodes:
            for i in range(len(c_list) - 1):
                # note: order matters since 'merge' keeps the id of the first node!
                self.merge(c_list[i + 1], c_list[i])
            
            # subtract cluster set from mix_set
            mix_set = mix_set.difference(c)
    
    def merge_sisters(self):
        sister_nodes = set()
        sisters = {}
        for label, node_set in self._nodes_by_label.items():
            for x in node_set:
                for y in node_set:
                    if x.id == y.id:
                        continue
                    if len(self._from_node[x.id].intersection(self._from_node[y.id])) > 0:
                        sister_nodes.add(x.id)
                        sister_nodes.add(y.id)
                        if x.id not in sisters:
                            sisters[x.id] = set()
                        if y.id not in sisters:
                            sisters[y.id] = set()
                        
                        u = sisters[x.id].union(sisters[y.id])
                        u.add(x.id)
                        u.add(y.id)
                    
                        for n in u:
                            sisters[n] = u
        
        if len(sister_nodes) <= 1:
            return False
        while len(sister_nodes) > 0:
            arbitrary_node = sister_nodes.pop()
            # get cluster for node:
            c = sisters[arbitrary_node]
            c_list = list(c)
            
            # merge all nodes:
            for i in range(len(c_list) - 1):
                # note: order matters since 'merge' keeps the id of the first node!
                self.merge(c_list[i + 1], c_list[i])
            
            i = 0
            mix_id = "mix0"
            while mix_id in self._nodes_by_id:
                i += 1
                mix_id = f"mix{i}"
            self.insert_before(c_list[-1], mix_id, "mix", "diamond")
            
            # subtract cluster set from mix_set
            sister_nodes = sister_nodes.difference(c)
        
        return True
    
    def get_paths(self):
        cluster = {}
        nodes = set()
        for a,b in self._edges:
            if len(self._from_node[a]) == 1 and len(self._to_node[b]) == 1:
                if a not in cluster:
                    cluster[a] = set()
                if b not in cluster:
                    cluster[b] = set()
                
                nodes.add(a)
                nodes.add(b)
                
                u = cluster[a].union(cluster[b])
                u.add(a)
                u.add(b)
                
                for n in u:
                    cluster[n] = u
        
        paths = []
        while len(nodes) > 0:
            
            arbitrary_node = nodes.pop()
            # get cluster for node:
            c = cluster[arbitrary_node]
            
            paths.append(c)
            
            nodes = nodes.difference(c)
        
        return paths
    
    def clean_paths(self):
        for path in self.get_paths():
            seen_labels = set()
            for n in path:
                l = self._nodes_by_id[n].label
                if l == "mix" and len(self._to_node[n]) == 1:
                    self.remove_node(n, redirect_edges=True)
                elif l in seen_labels:
                    self.remove_node(n, redirect_edges=True)
                else:
                    seen_labels.add(l)
                    
            
                
    def simplify(self):
        
        changed = True
        
        while changed:
        
            # merge all adjacent nodes with the same label
            for key in self._nodes_by_label:
                self.merge_adjacent_with_label(key)

            # and now merge all sister nodes with the same label
            # (just to make it more clean structured)

            changed = self.merge_sisters()
        
        self.clean_paths()
        
        
    
    def compile_graph(self, simplify = False):
        if simplify:
            self.simplify()
        dot = Digraph(self._comment)
        for n in self._nodes:
            dot.node(n.id, label=n.label, shape=n.shape)
        
        for e in self._edges:
            dot.edge(e[0], e[1])
        
        return dot


# %%
class RecipeGraph(object):
    def __init__(self, initial_ingreds=None):
        self._base_ing_nodes = set()
        self._dot = GraphWrapper(comment="recipe graph")
        self._ing_state_mapping = {}          # key: ingredient, value: state_id
        self._seen_actions = set()
        self._ings_connected_with_state = {}  # key: state_id, value: set of ingreds 
        
        self._seen_actions_for_ingredient = {}
        
        
        if initial_ingreds is not None:
            for ing in initial_ingreds:
                self.add_base_ingredient(ing)
    
    def add_base_ingredient(self, ingredient):
        if type(ingredient) == Ingredient:
            self.add_base_ingredient(ingredient._base_ingredient)
            return
        self._base_ing_nodes.add(ingredient)
        self._dot.node(ingredient, label=ingredient,shape="box")
        self._ing_state_mapping[ingredient] = ingredient
        self._ings_connected_with_state[ingredient] = set([ingredient])
        self._seen_actions_for_ingredient[ingredient] = set() 
    
    def add_action(self, action, ingredient):
        if type(ingredient) == Ingredient:
            return self.add_action(ingredient._base_ingredient)
        
        if ingredient not in self._seen_actions_for_ingredient:
            self._seen_actions_for_ingredient[ingredient] = set()
        
        if action in self._seen_actions_for_ingredient[ingredient]:
            return False
        
        self._seen_actions_for_ingredient[ingredient].add(action)
        
        action_id = action + "0"
        
        i = 0
        
        while action_id in self._seen_actions:
            i += 1
            action_id = action + str(i)
        
        self._seen_actions.add(action_id)
        
        self._dot.node(action_id, action)
        
        # get to the bottom of our tree (last known thing that happened to our ingredient)
        last_node = self._ing_state_mapping[ingredient]
        
        # update the reference of the last known state for all connected ingredients
        # (and for ourselve)
        
        connected_ingredients = self._ings_connected_with_state[last_node]
        
        for ing_id in connected_ingredients:
            self._ing_state_mapping[ing_id] = action_id
        
        # set ingredient set for new node
        self._ings_connected_with_state[action_id] = connected_ingredients.copy()
        
        # connect nodes with an edge
        self._dot.edge(last_node, action_id)
        
        return True
    
    def add_action_if_possible(self, action, ingredient):
        # extract actions for ingredient
        action_set = ingredient._action_set
        
        if action_set.issubset(self._seen_actions_for_ingredient[ingredient._base_ingredient]):
            return self.add_action(action, ingredient._base_ingredient)
        return False
    
    def mix_ingredients(self, ingredient_list):
        assert len(ingredient_list) > 0
        
        if type(ingredient_list[0]) == Ingredient:
            self.mix_ingredients([ing._base_ingredient for ing in ingredient_list])
            return
        
        last_nodes = set([self._ing_state_mapping[ing] for ing in ingredient_list])
        
        # create mixed ingredient set
        ing_set = set()
        
        for state in last_nodes:
            ing_set = ing_set.union(self._ings_connected_with_state[state])
        
        mix_action_id = "mix0"
        i = 0
        while mix_action_id in self._seen_actions:
            i += 1
            mix_action_id = f"mix{i}"
        
        self._seen_actions.add(mix_action_id)
        
        self._dot.node(mix_action_id, "mix", shape="diamond")
        
        self._ings_connected_with_state[mix_action_id] = ing_set.copy()
        
        for ing in ing_set:
            self._ing_state_mapping[ing] = mix_action_id
        
        for state in last_nodes:
            self._dot.edge(state, mix_action_id)
    
    def mix_if_possible(self, ingredient_list):
        assert len(ingredient_list) > 0
        assert type(ingredient_list[0]) == Ingredient
        
        # check whether ingredients are mixed already
        state_set = set(
            [self._ing_state_mapping[ing._base_ingredient] for ing in ingredient_list]
        )
        
        if len(state_set) <= 1:
            # all ingredients have the same last state → they're mixed already
            return False
        
        # check if action sets are matching the requirements
        for ing in ingredient_list:
            for act in ing._action_set:
                if act not in self._seen_actions_for_ingredient[ing._base_ingredient]:
                    return False
        
        # now we can mix the stuff:
        self.mix_ingredients(ingredient_list)
        return True
    
    @staticmethod
    def fromRecipeState(rec_state: RecipeState):
        # get all ingredients
        base_ingredients = set([ing._base_ingredient for ing in rec_state._ingredients])
        
        mix_m, mix_label = rec_state.get_mixing_matrix()
        act_m, act_a, act_i = rec_state.get_action_matrix()
        
        graph = RecipeGraph(base_ingredients)
        
        # create list of tuples: [action, ingredient]
        seen_actions = np.array(list(itertools.product(act_a,act_i))).reshape((len(act_a), len(act_i), 2))
        
        # create list of tuples [ingredient, ingredient]
        seen_mixes = np.array(list(itertools.product(mix_label,mix_label))).reshape((len(mix_label), len(mix_label), 2))
        
        seen_actions = seen_actions[act_m == 1]
        seen_mixes = seen_mixes[mix_m == 1]
        
        seen_actions = set([tuple(x) for x in seen_actions.tolist()])
        seen_mixes = set([tuple(x) for x in seen_mixes.tolist()])
        
        # for each ingredient get the list of unseen applied actions. (They were applied
        # before the first instruction)
        
        seen_actions_per_ingred = {}
        for act, json_ing in rec_state._seen_applied_actions:
            ing = Ingredient.from_json(json_ing)._base_ingredient
            if ing not in seen_actions_per_ingred:
                seen_actions_per_ingred[ing] = set()
            seen_actions_per_ingred[ing].add(act)
        
        unseen_actions_per_ingred = {}
        for ing in rec_state._ingredients:
            base = ing._base_ingredient
            if base not in seen_actions_per_ingred:
                unseen_actions_per_ingred[base] = ing._action_set.copy()
            else:
                unseen_actions_per_ingred[base] = ing._action_set.difference(seen_actions_per_ingred[base])
        
        # for each ingredient: apply unseen actions first
        for ing in rec_state._ingredients:
            base = ing._base_ingredient
            for act in unseen_actions_per_ingred[base]:
                graph.add_action(act, base)
        
        # iterate over all mixes and actions until the graph does not change anymore
        # TODO: there are more efficient ways to do that!
        changed = True
        while changed:
            changed = False
            changed_ingreds = True
            while changed_ingreds:
                changed_ingreds = False
                for mix in list(seen_mixes):
                    if graph.mix_if_possible([mix[0], mix[1]]):
                        changed = True
                        changed_ingreds = True
            changed_acts = True
            while changed_acts:
                changed_acts = False
                for act in list(seen_actions):
                    if graph.add_action_if_possible(act[0], act[1]):
                        changed = True
                        changed_acts = True
        
        return graph
        
        
        
    


# %%
class Recipe(object):
    def __init__(self, recipe_db_id = None):
        
        self._sentences = None
        self._title = None
        self._part = None
        self._ingredients = None
        self._recipe_id = recipe_db_id
        self._get_from_db()
        
        self._extracted_ingredients = None # TODO
        
        self.annotate_ingredients()
        self.annotate_sentences()
    
    def _get_from_db(self):
        result = DatabaseConnection.global_single_query(id_query, (self._recipe_id))
        assert len(result) > 0
        result = result[0]
        self._title = result['title']
        self._part = result['part']
        
        raw_sentences = json.loads(result['instructions'])
        raw_ingredients = json.loads(result['ingredients'])
        
        # throwing the raw data through our connlu generator to annotate them right
        cg_sents = ConlluGenerator(["\n".join(raw_sentences)])
        cg_ings = ConlluGenerator(["\n".join(raw_ingredients)])
        
        cg_sents.tokenize()
        cg_sents.pos_tagging_and_lemmatization()
        
        cg_ings.tokenize()
        cg_ings.pos_tagging_and_lemmatization()
        
        # TODO
        self._sentences = cg_sents.get_conllu_elements()[0]
        self._ingredients = cg_ings.get_conllu_elements()[0]
        #self._sentences = json.loads(result['instructions'])
        #self._ingredients = json.loads(result['ingredients'])
    
    def avg_sentence_length(self):
        return sum([len(s) for s in self._sentences])/len(self._sentences)
    
    def n_instructions(self):
        return len(self._sentences)
    
    def max_sentence_length(self):
        return max([len(s) for s in self._sentences])
    
    def keyword_ratio(self):
        sentence_ratios = []
        for sent in self._sentences:
            # FIXME: only works if there are no other misc annotations!
            sentence_ratios.append(sum([token['misc'] is not None for token in sent]))
        return sum(sentence_ratios) / len(sentence_ratios)
    
    def predict_labels(self):
        features = [sent2features(sent) for sent in self._sentences]
        labels = [tagger.tag(feat) for feat in features]
        return labels
    
    def predict_ingredient_labels(self):
        features = [sent2features(sent) for sent in self._ingredients]
        labels = [tagger.tag(feat) for feat in features]
        return labels
    
    def _annotate_sentences(self, sent_token_list, predictions):
        # test whether we predicted an label or found it in our label list
        for i, ing in enumerate(sent_token_list):
            for j, token in enumerate(ing):
                lemma = token['lemma']
                
                # check for labels
                if check_ingredient(token):
                    token.add_misc("food_type", "ingredient")
                    continue
                    
                if lemma in actions.stemmed_curated_cooking_verbs:
                    token.add_misc("food_type", "action")
                    continue
                
                if predictions[i][j] == 'ingredient':
                    token.add_misc("food_type", "ingredient")
                    continue
                
                #if predictions[i][j] == 'action':
                #    token.add_misc("food_type", "action")
                #    continue
                
                if lemma in containers.stemmed_containers:
                    token.add_misc("food_type", "container")
                    continue
                if predictions[i][j] == 'container':
                    token.add_misc("food_type", "container")
                    continue
                
                if lemma in placeholders.stemmed_placeholders:
                    token.add_misc("food_type", "placeholder")
                if predictions[i][j] == 'placeholder':
                    token.add_misc("food_type", "placeholder")
    
    def annotate_ingredients(self):
        self._annotate_sentences(self._ingredients, self.predict_ingredient_labels())
    
    def annotate_sentences(self):
        self._annotate_sentences(self._sentences, self.predict_labels())
    
    def recipe_id(self):
        return self._recipe_id
    
    '''
    # TODO: only conllu module compatible, and not with our own conllu classes
    def serialize(self):
        result = "# newdoc\n"
        if self._recipe_id is not None:
            result += f"# id: {self._recipe_id}\n"
        
        for sent in self._sentences:
            result += f"{sent.serialize()}"
        return result + "\n"
    '''
    
    def display_recipe(self):
        display(Markdown(f"## {self._title}\n({self._recipe_id})"))
        display(Markdown(f"### Ingredients"))
        display(Markdown("\n".join([f" * '{escape_md_chars(self.tokenlist2str(ing))}'" for ing in self._ingredients])))
        display(Markdown(f"### Instructions"))
        display(Markdown("\n".join([f" * {escape_md_chars(self.tokenlist2str(ins))}" for ins in self._sentences])))
        
    def tokenlist2str(self, tokenlist):
        return " ".join([token['form'] for token in tokenlist])
    
    def tokenarray2str(self, tokenarray):
        return "\n".join([self.tokenlist2str(tokenlist) for tokenlist in tokenarray])
    
    
    def __repr__(self):
        s = "recipe: " + (self._recipe_id if self._recipe_id else "") + "\n"
        s += "instructions: \n"
        for sent in self._sentences:
            s += " ".join([token['form'] for token in sent]) + "\n"
        
        s += "\nscores:\n"
        s += f"avg_sent_length: {self.avg_sentence_length()}\n"
        s += f"n_instructions: {self.n_instructions()}\n"
        s += f"keyword_ratio: {self.keyword_ratio()}\n\n\n"
        
        return s
    
    # --------------------------------------------------------------------------
    # functions for extracting ingredients
    
    def extract_ingredients(self):
        self._extracted_ingredients = []
        for ing in self._ingredients:
            entry_ing_tokens = []
            entry_act_tokens = []
            for token in ing:
                t_misc = token['misc']
                if t_misc is not None and "food_type" in t_misc:
                    ftype = t_misc['food_type']
                    if ftype == "ingredient":
                        entry_ing_tokens.append(token)
                    elif ftype == "action":
                        entry_act_tokens.append(token)
            
            # find max cluster of ingredients and merge them
            index_best = 0
            best_size = 0
            current_size = 0
            for i, ing_token in enumerate(entry_ing_tokens):
                if i == 0 or entry_ing_tokens[i - 1]['id'] + 1 == ing_token['id']:
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        index_best = i - current_size + 1
            
            if best_size == 0:
                # unfortunately, no ingredient is found :(
                continue
            
            ingredient = Ingredient(" ".join([entry['lemma'] for entry in entry_ing_tokens[index_best:index_best + best_size]]))
            
            # apply found actions:
            for action in entry_act_tokens:
                ingredient.apply_action(action['lemma'])
            
            self._extracted_ingredients.append(ingredient)
        
        return self._extracted_ingredients
    
    def apply_instructions(self, confidence_threshold = 0.4, max_dist_last_token = 4, debug=False):
        current_state = RecipeState(self._extracted_ingredients)
        self._recipe_state = current_state
        
        instruction_number = 0
        
        for sent in self._sentences:
            
            instruction_number += 1
            
            if debug:
                display(Markdown(f"----\n* **instruction {instruction_number}**:\n`" + escape_md_chars(self.tokenlist2str(sent)) + "`\n"))
            
            instruction_ing_tokens = []
            instruction_act_tokens = []
            
            ing_dist_last_token = []
            act_dist_last_token = []
            
            
            last_token = -1
            
            for i, token in enumerate(sent):
                t_misc = token['misc']
                if t_misc is not None and "food_type" in t_misc:
                    ftype = t_misc['food_type']
                    if ftype == "ingredient":
                        instruction_ing_tokens.append(token)
                        ing_dist_last_token.append(1000 if last_token < 0 else i - last_token)
                        last_token = i
                    elif ftype == "action":
                        instruction_act_tokens.append(token)
                        act_dist_last_token.append(1000 if last_token < 0 else i - last_token)
                        last_token = i
            
            # cluster ingredient tokens together and apply actions on it:
            clustered_ingredients = []
            clustered_conllu_ids = []
            clustered_last_tokens = []
            i = 0
            n = len(instruction_ing_tokens)
            
            current_token_start = 0
            while i < n:
                current_token_start = i
                clustered_conllu_ids.append(instruction_ing_tokens[i]['id'])
                clustered_last_tokens.append(ing_dist_last_token[i])
                ing_str = instruction_ing_tokens[i]['lemma']
                while i+1 < n and instruction_ing_tokens[i+1]['id'] - instruction_ing_tokens[i]['id'] == 1:
                    ing_str += " " + instruction_ing_tokens[i+1]['lemma']
                    i += 1
                clustered_ingredients.append(ing_str)
                i += 1
            
            def matching_action(ing_str, ing_id, action_token_list):
                
                action = None
                action_dists = [act['id'] - ing_id for act in action_token_list]
                
                # so far: simple heuristic by matching to next action to the left
                # (or first action to the right, if there is no one left to the ingredient)
                
                for i in range(len(action_token_list)):
                    if action_dists[i] < 0:
                        action = action_token_list[i]
                
                return action
            
            ingredients_used = set()
            actions_used = set()
            
            if debug:
                print("apply actions regular rule based:")
            
            for i, ing_str in enumerate(clustered_ingredients):
                
                ing = Ingredient(ing_str)
                
                # get matching action:
                action = matching_action(ing_str, clustered_conllu_ids[i], instruction_act_tokens)
                
                if clustered_last_tokens[i] < max_dist_last_token:
                    if action is not None:
                        actions_used.add(action['lemma'])
                        ingredients_used.add(ing_str)
                        # apply action on state
                        current_state.apply_action(action['lemma'], ing, instruction_number=instruction_number, add_new_if_not_similar=False)
                        if debug:
                            print(f"\tapply {action['lemma']} on {ing}")
            
            if debug:
                print("try to match unused actions:")
            # go throuh all actions. if we found an unused one, we assume it is applied either on the next right ingredient.
            
            
            for act_token in instruction_act_tokens:
                if act_token['lemma'] not in actions_used:
                    # fing next ingredient right to it
                    next_ing = None
                    for i, ing_str in enumerate(clustered_ingredients):
                        if clustered_conllu_ids[i] > act_token['id']:
                            actions_used.add(act_token['lemma'])
                            ingredients_used.add(ing_str)
                            ing = Ingredient(ing_str)
                            current_state.apply_action(act_token['lemma'], ing, instruction_number=instruction_number, add_new_if_not_similar=False)
                            if debug:
                                print(f"\tapply {act_token['lemma']} on {ing}")
                            break
                            
            
            actions_unused = []
            ingredients_unused = []
            
            
            for act_token in instruction_act_tokens:
                if act_token['lemma'] in actions_used:
                    continue
                actions_unused.append(act_token['lemma'])
            
            for ing_str in clustered_ingredients:
                if ing_str in ingredients_used:
                    continue
                ingredients_unused.append(ing_str)
            
            if debug:
                print(f"\nunused actions: {actions_unused} \nunused ings: {ingredients_unused}\n")
            
            if (instruction_number > 1):
                if debug:
                    print("mixing ingredients based on mixing actions with last instruction:")
                for ing in current_state.get_ingredients_touched_in_instruction(instruction_number -1):
                    ing.mark_for_mixing()

                for ing in current_state.get_combined_ingredients():
                    if debug:
                        print(f"\t* {ing}")
            
            if debug:
                print("mixing all ingredients in this instruction:")
            
            for ing_str in clustered_ingredients:
                current_state.apply_action("mix", Ingredient(ing_str), instruction_number=instruction_number, add_new_if_not_similar=False)
        
            for ing in current_state.get_combined_ingredients():
                if debug:
                    print(f"\t* {ing}")
                
            
            # if no ingredient is found, apply actions on all ingredients so far used
            
            if len(clustered_ingredients) == 0 and len(actions_unused) > 0:
                if debug:
                    print("\nno ingredients found. So apply actions on all ingredients that are touched so far:")
                for action in actions_unused:
                    current_state.apply_action_on_all(action, instruction_number, exclude_instruction_number=0)
                    
            if debug:
                print(f"\nstate after instruction {instruction_number}:")
                print(current_state)
                print("\n")
    
    def plot_matrices(self):
        if self._recipe_state is None:
            print("Error: no recipe state found")
            return
        
        mixings, mix_labels = self._recipe_state.get_mixing_matrix()
        
        x_labels = [f"{ing._base_ingredient} 🡸 ({' '.join([act for act in ing._action_set])})" for ing in mix_labels]
        y_labels = [f"({' '.join([act for act in ing._action_set])}) 🢂 {ing._base_ingredient}" for ing in mix_labels]
        

        fig = go.Figure(data=go.Heatmap(
                   z=mixings,
                   x=x_labels,
                   y=y_labels,
                   xgap = 1,
                   ygap = 1,))

        fig.update_layout(
            width=1024,
            height=1024,
            yaxis = dict(
              scaleanchor = "x",
              scaleratio = 1,
            )
        )
        fig.show()

        
        actions, act_labels, ing_labels = self._recipe_state.get_action_matrix()
        

        fig = go.Figure(data=go.Heatmap(
                   z=actions,
                   x=[f"{ing._base_ingredient} 🡸 ({' '.join([act for act in ing._action_set])})" for ing in ing_labels],
                   y=[str(a) for a in act_labels],
                   xgap = 1,
                   ygap = 1,))

        fig.update_layout(
            width=1024,
            height=1024,
            yaxis = dict(
              scaleanchor = "x",
              scaleratio = 1,
            )
        )
        fig.show()


               


# %%


