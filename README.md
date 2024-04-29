# Attribute-Driven Trace Clustering

Code to reproduce the results of the chapter "*Attribute-Driven Trace Clustering*" of the thesis "*Machine Learning for Probabilistic and Attributes-Aware Process Mining*".
Requirements are in the file "*requirements.txt", to install with pip:

```
[~]$ pip install requirements.txt
```

Event logs available:
- Business Process Intelligence Challenge 2012 &rarr; bpic2012 
- Business Process Intelligence Challenge 2017 &rarr; bpic2017 
- Business Process Intelligence Challenge 2019 &rarr; bpic2019 
- Road Traffic Fine Management Process &rarr; fines 

## Visualize results
To visualize the results, use the mode "*vis*" with the desired dataset. For example for *bpic2012*:

```
[~]$ python main.py bpic2012 vis
```

## Clustering procedure
To create new clusters, the procedure is divide in different stages:
1) evaluate the elbow curve for the first level
```
[~]$ python main.py bpic2012 cluster --elbow_curve_first_level True
```
2) choose the number of first-level clusters, edit the "*main.py*" file and modify the "*N_CLUSTERS_FIRST_LEVEL*" variable. Then launch the clustering with:
```
[~]$ python main.py bpic2012 cluster --cluster_first_level True
```
3) evaluate the elbow curve for each first-level cluster obtained.
```
[~]$ python main.py bpic2012 cluster --elbow_curve_second_level True
```
4) choose the number of second-level clusters, edit the "*main.py*" file and modify the "*N_CLUSTERS_SECOND_LEVEL*" variable. Then launch the clustering with:
```
[~]$ python main.py bpic2012 cluster --cluster_second_level True
```
