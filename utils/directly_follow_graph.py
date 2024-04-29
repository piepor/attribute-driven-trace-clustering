""" 
This module contains functions related to the Directly Follow Graph visualization
"""
from collections import Counter
import numpy as np
import graphviz

def create_activities_freq(dfg, start_activities):
    """
    given the Directly Follow Graph and the start activities,
    computes how many time an activity is executed counting the incoming arcs 
    """
    freq_act = {}
    for tail, head in dfg:
        if not head in freq_act:
            freq_act[head] = dfg[(tail, head)]
        else:
            freq_act[head] += dfg[(tail, head)]
    for start_act in start_activities:
        if not start_act in freq_act:
            freq_act[start_act] = start_activities[start_act]
        else:
            freq_act[start_act] += start_activities[start_act]
    return freq_act

def check_in_out_activities(dfg, start_activities, end_activities):
    """
    given the Directly Follow Graph, start and end activities,
    checks if every activy has both incoming and outcoming arcs
    i.e. if there are no loose ends
    """
    tails = set()
    heads = set()
    acts_to_remove = []
    for tail, head in dfg:
        tails.add(tail)
        heads.add(head)
    no_heads = tails.difference(heads)
    tails_to_remove = no_heads.difference(set(start_activities)) 
    if len(tails_to_remove) > 0:
        acts_to_remove.extend(list(tails_to_remove))
    no_tails = heads.difference(tails)
    heads_to_remove = no_tails.difference(set(end_activities)) 
    if len(heads_to_remove) > 0:
        acts_to_remove.extend(list(heads_to_remove))
    return acts_to_remove

def check_start_act_to_include(dfg, start_activities):
    """
    given the Directly Follow Graph and start activities,
    checks if all the activities in start_activities have to be included.
    Needed for example if the dfg has been filtered and the new start
    activities have to be computed
    """
    start_to_include = {}
    for start_act in start_activities:
        for tail, _ in dfg:
            if tail == start_act:
                start_to_include[start_act] = start_activities[start_act]
                break
    return start_to_include

def remove_edges(dfg, acts_to_remove):
    """ 
    removes edges in acts_to_remove from the Directly Follow Graph
    """
    return Counter({(tail, head): dfg[(tail, head)] for tail, head in dfg if not tail in acts_to_remove and not head in acts_to_remove})

def remove_unconnected(dfg, start_activities, end_activities):
    """
    removes activities and edges from unconnected part of the dfg
    """
    acts_to_remove = check_in_out_activities(dfg, start_activities, end_activities)
    start_acts = {el: start_activities[el] for el in start_activities if not el in acts_to_remove}
    end_acts = {el: end_activities[el] for el in end_activities if not el in acts_to_remove}
    return remove_edges(dfg, acts_to_remove), start_acts, end_acts

def get_trans_freq_color(trans_count, min_trans_count, max_trans_count):
    """
    given a transition's count, computes the related color
    """
    trans_base_color = int(255 - 100 * (trans_count - min_trans_count) / (max_trans_count - min_trans_count + 0.00001))
    trans_base_color_hex = str(hex(trans_base_color))[2:].upper()
    return "#" + trans_base_color_hex + trans_base_color_hex + "FF"

def get_activities_color(activities_count):
    """
    assigns a color to every considered activity based on its frequency
    """
    activities_color = {}
    min_value = min(activities_count.values())
    max_value = max(activities_count.values())
    for act in activities_count:
        v0 = activities_count[act]
        """transBaseColor = int(
            255 - 100 * (v0 - min_value) / (max_value - min_value + 0.00001))
        transBaseColorHex = str(hex(transBaseColor))[2:].upper()
        v1 = "#" + transBaseColorHex + transBaseColorHex + "FF"""
        v1 = get_trans_freq_color(v0, min_value, max_value)
        activities_color[act] = v1
    return activities_color

def get_edge_size(edge_count, dfg, min_size=0.5, max_size=6):
    """
    computes the size of an edge based on its frequency
    """
    min_count = min(dfg.values())
    max_count = max(dfg.values())
    if min_count == max_count:
        min_count = 0
        max_count = edge_count
    return (edge_count - min_count) / (max_count - min_count) *(max_size - min_size) + min_size

def compute_edge_style_starts_ends(dfg, act, freq_acts, min_num):
    font_size = get_edge_size(freq_acts[act], dfg)
    # choose if balck solid or red dashed
    if freq_acts[act] > min_num:
        color = "#000000"
        style = 'solid'
    else:
        color = "#FF0000"
        style = 'dashed'
    return font_size, color, style

def compute_edge_style(dfg, tail, head, min_num):
    if dfg[(tail, head)] > min_num:
        color = "#000000"
        style = 'solid'
    else:
        color = "#FF0000"
        style = 'dashed'
    font_size = get_edge_size(dfg[(tail, head)], dfg)
    return font_size, color, style

def create_graphviz_delete_arcs(dfg, start_acts, end_acts, perc_filt=0.0):
    """
    creates the graphviz representation of the input Directly Follow Graph
    (dfg + start activities + end activities). Every arc below the filtering
    percentage is removed along with the activities that remain without incoming 
    or outcoming connections due to the filtering
    """
    # prepares the graph
    min_num = int(np.round(perc_filt*max(dfg.values())))
    dfg_filtered = Counter({el: dfg[el] for el in dfg if dfg[el] > min_num})
    start_acts = {el: start_acts[el] for el in start_acts if start_acts[el] > min_num}
    end_acts = {el: end_acts[el] for el in end_acts if end_acts[el] > min_num}
    dfg_filtered, start_acts, end_acts = remove_unconnected(dfg_filtered, start_acts, end_acts)
    start_acts = check_start_act_to_include(dfg_filtered, start_acts)
    # get the colors
    freq_act = create_activities_freq(dfg_filtered, start_acts)
    acts_color = get_activities_color(freq_act)
    # creates graphviz representation
    added_acts = []
    dot = graphviz.Digraph('dfg') 
    # start node has a circle inside another circle
    dot.node('start_node', "<&#9679;>", shape='circle')
    for start_act in start_acts:
        dot.node(start_act, f'{start_act} ({freq_act[start_act]})', shape='box', style='filled', fillcolor=acts_color[start_act])
        font_size = get_edge_size(start_acts[start_act], dfg_filtered)
        dot.edge('start_node', start_act, label=str(start_acts[start_act]), penwidth=str(font_size))
        added_acts.append(start_act)
    for tail, head in dfg_filtered:
        if tail not in added_acts:
            dot.node(tail, f'{tail} ({freq_act[tail]})', shape='box', style='filled', fillcolor=acts_color[tail])
        if head not in added_acts:
            dot.node(head, f'{head} ({freq_act[head]})', shape='box', style='filled', fillcolor=acts_color[head])
        font_size = get_edge_size(dfg_filtered[(tail, head)], dfg_filtered)
        dot.edge(tail, head, label=str(dfg_filtered[(tail, head)]), penwidth=str(font_size))
    # end node has a square inside a double circle
    dot.node('end_node', "<&#9632;>", shape='doublecircle')
    for end_act in end_acts:
        font_size = get_edge_size(end_acts[end_act], dfg_filtered)
        dot.edge(end_act, 'end_node', label=str(freq_act[end_act]), penwidth=str(font_size))
    return dot

def create_graphviz_dashed_arcs(dfg, start_acts, end_acts, perc_filt=0.0) -> graphviz.Digraph:
    """
    creates the graphviz representation of the input Directly Follow Graph
    (dfg + start activities + end activities). Every arc below the filtering
    percentage is drawn as a red dashed connection.
    """
    # get the colors
    min_num = int(np.round(perc_filt*max(dfg.values())))
    freq_act = create_activities_freq(dfg, start_acts)
    acts_color = get_activities_color(freq_act)
    # creates graphviz representation
    added_acts = []
    dot = graphviz.Digraph('dfg') 
    # start node has a circle inside another circle
    dot.node('start_node', "<&#9679;>", shape='circle')
    for start_act in start_acts:
        dot.node(start_act, f'{start_act} ({freq_act[start_act]})', shape='box', style='filled', fillcolor=acts_color[start_act])
        font_size, color, style = compute_edge_style_starts_ends(dfg, start_act, start_acts, min_num)
        dot.edge('start_node', start_act, label=str(start_acts[start_act]),
                 penwidth=str(font_size), style=style, color=color)
        added_acts.append(start_act)
    for tail, head in dfg:
        if tail not in added_acts:
            dot.node(tail, f'{tail} ({freq_act[tail]})', shape='box', style='filled', fillcolor=acts_color[tail])
        if head not in added_acts:
            dot.node(head, f'{head} ({freq_act[head]})', shape='box', style='filled', fillcolor=acts_color[head])
        # choose if balck solid or red dashed
        font_size, color, style = compute_edge_style(dfg, tail, head, min_num)
        dot.edge(tail, head, label=str(dfg[(tail, head)]),
                 penwidth=str(font_size), style=style, color=color)
    # end node has a square inside a double circle
    dot.node('end_node', "<&#9632;>", shape='doublecircle')
    for end_act in end_acts:
        font_size, color, style = compute_edge_style_starts_ends(dfg, end_act, end_acts, min_num)
        dot.edge(end_act, 'end_node', label=str(end_acts[end_act]),
                 penwidth=str(font_size), style=style, color=color)
    return dot

def create_variants_graphs(variants) -> graphviz.Digraph:
    # breakpoint()
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})
    for i, variant in enumerate(variants):
        dot.node(f'start_node_var{i}', f'{len(variants[variant])} traces', shape='plaintext')
        last_node = f'start_node_var{i}'
        for j, activity in enumerate(variant):
            new_node = f'{activity}_var{i}_step{j}' 
            dot.node(new_node, f'{activity}', shape='box', style='filled')
            if j != 0:
                dot.edge(last_node, new_node)
            last_node = new_node
        # dot.node(f'end_node_var{i}', "<&#9632;>", shape='doublecircle')
        # dot.edge(last_node, f'end_node_var{i}')
    # breakpoint()
    return dot
