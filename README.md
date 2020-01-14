# KGRecommendation

- data/
-- movie/
--- item_index2entity_id.txt: the mapping from item indices in the raw rating file to entity IDs in the KG;
--- kg.txt: knowledge graph file;
- src/: implementations of KGCN.

Dataset used is MovieLens-20m, which contains rating data for multiple movies by multiple users, as well as movie metadata information and user attribute information. Mainly four csv files are obtained: links.csv, movies.csv, ratings.csv, tags.csv. During pre-processing, the data with rating more than 4 is judged as a sample with a value of 1. The negative sampling method in the paper is to sample the same amount of data from the unrated data as a sample with a value of 0. The corresponding KG was constructed using Microsoft Satori. The following two types of information obtained after preprocessing are used as inputs:
- rating:    user_ID	item_ID	label
- kg:    head ID (entity ID is same as item ID it represents)     relationship ID    tail ID

```
$ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
$ unzip ml-20m.zip
$ mv ml-20m/ratings.csv data/movie/
$ cd src
$ python preprocess.py
$ python main.py
```
