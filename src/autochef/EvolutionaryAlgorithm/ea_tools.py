#!/usr/bin/env python3
# coding: utf-8

# # Statistical Tools

import numpy as np
import scipy.stats


# * Helper function to calculate the wheel of fortune

def wheel_of_fortune(rank_i,n):
    return rank_i / (0.5 * n * (n + 1))


def wheel_of_fortune_weights(items:list, item_scores:list):
    rank = scipy.stats.rankdata(item_scores)

    n = len(items)

    return wheel_of_fortune(rank, n)


def wheel_of_fortune_selection(items: list, item_scores:list, num_choices=1):
    
    wheel_weights = wheel_of_fortune_weights(items, item_scores)
    
    n = min(len(items), num_choices)
    
    choice = np.random.choice(items, size=n, replace=False, p=wheel_weights)
    
    if num_choices == 1:
        return choice[0]

    return choice


def combined_wheel_of_fortune_selection(items_list:list, item_scores_list:list, num_choices=1):
    
    scores = {}
    
    for i in range(len(items_list)):
        items = items_list[i]
        item_scores = item_scores_list[i]
        
        w = wheel_of_fortune_weights(items, item_scores)
        #print(items, item_scores)
        #print(w)
        
        for j, item in enumerate(items):
            if item in scores:
                scores[item] += w[j]
            else:
                scores[item] = w[j]
        
    combined_items = []
    combined_scores = []
    
    for i,s in scores.items():
        combined_items.append(i)
        combined_scores.append(s)
    
    combined_scores = np.array(combined_scores)
    
    #print(combined_scores)
    #print(np.sum(combined_scores))
    
    combined_scores /= len(items_list)
    
    #print(combined_scores)
    
    #print(np.sum(combined_scores))
    
    n = min(len(combined_items), num_choices)
    
    return np.random.choice(combined_items, size=n, replace=False, p=combined_scores)
        
        

