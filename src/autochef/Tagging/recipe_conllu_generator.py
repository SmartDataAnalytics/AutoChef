#!/usr/bin/env python3
# coding: utf-8

# # Recipe Conllu Generator

import sys
sys.path.insert(0, '..')

from conllu_generator import ConlluDict, ConlluElement, ConlluDocument, ConlluGenerator
import settings
import importlib.util
from json_buffered_reader import JSON_buffered_reader as JSON_br


# loading ingredients:
spec = importlib.util.spec_from_file_location(
    "ingredients", "../" + settings.ingredients_file)
ingredients = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingredients)

# loading actions:
spec = importlib.util.spec_from_file_location(
    "actions", "../" + settings.actions_file)
actions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actions)

# loading containers
spec = importlib.util.spec_from_file_location(
    "containers", "../" + settings.container_file)
containers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(containers)

# loading placeholders
spec = importlib.util.spec_from_file_location(
    "placeholders", "../" + settings.placeholder_file)
placeholders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placeholders)

# skipping recipes:
n_skipped_recipes = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print("start reading at recipe " + str(n_skipped_recipes))

# settings:
recipe_buffer_size = 1000
recipe_buffers_per_file = 5


# create reader
buffered_reader_1M = JSON_br("../" + settings.one_million_recipes_file)


def process_instructions(instructions: list, document_ids=None):

    if len(instructions) == 0:
        return

    conllu_input_docs = instructions

    cg = ConlluGenerator(
        conllu_input_docs, ingredients.multi_word_ingredients_stemmed, ids=document_ids)
    cg.tokenize()
    cg.pos_tagging_and_lemmatization()
    
    
    cg.add_misc_value_by_list("food_type", "ingredient", [w.replace(" ","_") for w in ingredients.multi_word_ingredients_stemmed] + ingredients.ingredients_stemmed)
    cg.add_misc_value_by_list("food_type", "action", actions.stemmed_cooking_verbs)
    cg.add_misc_value_by_list("food_type", "containers", containers.stemmed_containers)
    cg.add_misc_value_by_list("food_type", "placeholders", placeholders.stemmed_placeholders)

    savefile.write(str(cg))


i = 0
buffer_count = n_skipped_recipes % recipe_buffer_size
file_count = n_skipped_recipes // (recipe_buffer_size * recipe_buffers_per_file)

savefile = open(f"recipes{file_count}.conllu", 'w')
instructions = []
ids = []


for raw_recipe in buffered_reader_1M:

    i += 1

    if i > n_skipped_recipes:

        instruction = ""
        for item in raw_recipe['instructions']:
            instruction += item['text'] + '\n'
        ids.append(raw_recipe['id'])

        instructions.append(instruction)

        if i % recipe_buffer_size == 0:
            process_instructions(instructions, ids)
            print(f"processed {i} recipes")
            instructions = []
            ids = []
            buffer_count += 1
            if buffer_count % recipe_buffers_per_file == 0:
                savefile.close()
                file_count += 1
                savefile = open(f"recipes{file_count}.conllu", 'w')
    




process_instructions(instructions)
print(f"processed {i} recipes")

savefile.close()

