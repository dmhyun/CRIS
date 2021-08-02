<h1 align="center">
<span> Collaborative Representation Learning with Interest Sustainability</span>
</h1>

<p align="center">
    <a href="http://icdm2020.bigke.org/" alt="Conference">
        <img src="https://img.shields.io/badge/ICDM'20-Regular%20paper-brightgreen" /></a>   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>   
</p>

<p align="center">
<span>Official implementation of ICDM'20 paper</span>
<a href="https://ieeexplore.ieee.org/document/9338423">Interest Sustainability-Aware Recommender System</a>
</p>

## Overview
### Interest Sustainablity Prediction 
We first predict the interest sustainability of each item, that is, how likely each item will be consumed in the future. 

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/en/9/95/Test_image.jpg" alt="graph" width="8%"></p>

### User Preference Learning with Interest Sustainability
Then, our goal is to make users closer to the items with high interest sustainability scores in the representation space than those with low interest sustainability scores.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/en/9/95/Test_image.jpg" alt="graph" width="8%"></p>

### Recommendation Performance

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/en/9/95/Test_image.jpg" alt="graph" width="8%"></p>


## Major Requirements
* Python
* Pytorch
* Numpy

## Preprocessing Data
1. Download user-item consumption data (and extract the compressed file) into `./data/`.
    * [Amazon](http://jmcauley.ucsd.edu/data/amazon/)
    * [Yelp](https://www.yelp.com/dataset)
    * [GoodReads](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph)
    * Other data you want

    :exclamation: *Please make sure your data in the same JSON format of Amazon data.*
 
 2. Split your data into training/validation/test data in `./data/`.
   <pre><code>python split_data.py your_decompressed_file.json</code></pre>
 
 3. Build a dataset for training a recommender syetem with using the splitted data.
 <pre><code>python build_recdata.py generated_directory </code></pre>
 
 ## Training 
1. Train a BILSTM network to obtain Interest sustainability score (ISS) for each item.
 <pre><code>python train_interest.py --dataset your_dataset --period 16 --binsize 8 --pos_weight 1e-2 --hidden_dim 64</code></pre>

   
2. Train the proposed recommender system (CRIS).
 <pre><code>python train_recommender.py --dataset your_dataset --lamb 0.2 --gamma 1.6 --K 50 --margin 0.6 --numneg 10 </code></pre>

## Citation
If you use this repository for your work, please consider citing our paper [Interest Sustainability-Aware Recommender System](https://ieeexplore.ieee.org/document/9338423):

<pre><code> @inproceedings{hyun2020interest,
  title={Interest Sustainability-Aware Recommender System},
  author={Hyun, Dongmin and Cho, Junsu and Park, Chanyoung and Yu, Hwanjo},
  booktitle={2020 IEEE International Conference on Data Mining (ICDM)},
  pages={192--201},
  year={2020},
  organization={IEEE}
}
</code></pre>
