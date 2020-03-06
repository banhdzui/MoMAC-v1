# MMAC: Multi-objective Optimization to combine Multiple Association Rules into an Interpretable Classification

MMAC is an interpretable classification model based on association rule mining and multi-objective optimization. The classification model itself is a rule list that makes a single prediction based on multiple rules. We exert association rule mining to generate all potential classification rules and then multi-objective optimization to learn an interestingness measure to prioritize these rules. The combination of these two techniques allows us to select an optimal rule list that is small in size but produces high accuracy in prediction. The interestingness measure is formulated as a function of basic probabilities related to the rules and it is learned from data.

## Requirements
* Python 3.7
* scikit-learn 
* platypus 

## How to use?
**Input data**: a text file, one sample in one line and the attribute-value pairs are separated by comma. You can find an example for data format in folder ```Data```

**Train MMAC classifiers**

Run script ```TestMMACS.py``` with parameters to train classifer

For example, the following command line is used to train a MMACNet for Breast Cancer dataset. Only the rules whose support is greater than 0.01 are considered. All potential solutions (or classifiers) are saved in the file ```breast.sol```. The parameter ```--class``` indicates the index of decision class in the input file. To train a MMACSig, we change the parameter ```--option``` from ```net``` to ```sigmoid```.

<pre><code class="language-python"> python TestMMACS.py --train data/breast/breast_cancer_w.csv.train.0 --test data/breast/breast_cancer_w.csv.test.0 --class 0 --minsup 0.01 --nloop 10000 --option net --out breast.sol</code></pre>

**Choose solution (or classifier) and test its performance**

Run script ```VisualizeMMAC.py</span>``` with parameters to choose solution (or classifer) among found solutions. 

<pre><code class="language-python">python VisualizeMMAC.py --train data/breast/breast_cancer_w.csv.train.0 --test data/breast/breast_cancer_w.csv.test.0 --minsup 0.01 --sol breast.sol.0 --option net --class 0</code></pre>
