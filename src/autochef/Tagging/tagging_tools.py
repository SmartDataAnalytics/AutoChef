#!/usr/bin/env python3

from IPython.display import Markdown, display
import conllu

def print_visualized_tags(
        conllu_sentence,
        food_tags_and_colors={'ingredient': 'cyan', 'action': "orange"},
        upos_colors={'VERB': 'yellow'}):
    colorstr = "<span style='background-color:{}'>{}</span>"
    s = ""
    for tag in conllu_sentence:
        # print(tag)
        upos = tag['upostag']
        if tag['misc'] != None:
            for food_tag in food_tags_and_colors:
                if food_tag == tag['misc']['food_type']:
                    s += colorstr.format(
                        food_tags_and_colors[food_tag], tag['form']) + " "

        elif upos in upos_colors:
            s += colorstr.format(upos_colors[upos], tag['form']) + " "
        else:
            s += tag['form'] + " "

    display(Markdown(s))


