# DP-NTK code for CelebA, CIFAR-10 & Tabular
For MNIST/FashionMNIST experiments, please refer to the `MNIST` folder.

Scripts `models/generators.py`, `data_loading.py`, `sythn_data_benchmark.py`, `util.py` (named `aux.py` originally),
the first two functions of `dp_ntk_gen.py` are taken from the repository of [DP-MERF](https://github.com/frhrdr/dp-merf/tree/main/code_balanced).

The script `all_aux_tab.py` (named `all_aux_files_tab_data.py` originally) is taken from the repository of [DP-HP](https://github.com/ParkLabML/DP-HP/tree/master/dp_mehp).

The script `fid_eval.py` (named `eval_fid.py` originally) is partially taken from the repository of [DP-MEPF](https://anonymous.4open.science/r/dp-gfmn/code/eval_fid.py).

## Data Download
In order to run tabular data experiments, please download the data from [DP-HP data folder](https://github.com/ParkLabML/DP-HP/tree/master/data) and place it into `./data/tab_data`

For the CelebA experiments, please download the [image_align_celeba](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?select=img_align_celeba) file from Kaggle and unzip it directly into `./data/`


## Code Structure
- `models/`
  - `ntk.py` gives access to all NTK networks used in the code.
  - `generators.py` definitions for all generators.
- `all_aux_tab.py` contains the auxiliary code for tabular data.
- `data_loading.py` contains the dataloader for mnist/fmnist.
- `dp_ntk.py` This is the main function to run for the cifar10 and tabular experiments, calls `dp_ntk_mean_emb_1.py` then `dp_ntk_gen.py` (see below “how to run”).
- `dp_ntk_mean_emb_1.py` generates and saves the DP/non-DP mean embedding of sensitive data.
- `dp_ntk_gen.py` generator training for DP-NTK.
- `dp_ntk_one_class.py` This is the main function to run for CelebA experiments, calls `dp_ntk_mean_emb_1_one_class.py` then `dp_ntk_gen_one_class.py` (see below “how to run”). Is also possible to run CIFAR-10 on this file, and images will be treated as 1 class.
- `dp_ntk_mean_emb_1_one_class.py` generates and saves the DP/non-DP mean embedding of sensitive data.
- `dp_ntk_gen_one_class.py` generator training for DP-NTK.
- `fid_eval.py` various utility functions.
- `synth_data_benchmark.py` functions used for evaluate the synthetic data.
- `util.py` various utility functions.
- `util_logging.py` utility functions specific to logging experiments.


## How to Run:

### 1) Notable Flags:
- `--seed`: value of random seed to use. 
- `--model-ntk`: here the ntk models considered are `fc_1l`, `fc_2l`, `cnn_1l` and `cnn_2l`.
- `--ntk-width`: first FC layer width of the specified NTK structure.
- `--ntk_width-2`: second FC layer width of the specified NTK structure.
- `--tgt-eps`: the desired DP constant epsilon. the necessary noise is computed and added to mean.
- `--tab_classifiers`: the tabular classifiers (named 0-11 for our parameters) considered are as follows:
  - `LogisticRegression` from `sklearn.linear_model`.
  - `GaussianNB` from `sklearn.naive_bayes`.
  - `BernoulliNB` from `sklearn.naive_bayes`.
  - `SVC` from `sklearn.svm`.
  - `DecisionTreeClassifier` from `sklearn.tree`.
  - `LinearDiscriminantAnalysis` from `sklearn.discriminant_analysis`.
  - `AdaBoostClassifier` from `sklearn.ensemble`.
  - `BaggingClassifier` from `sklearn.ensemble`.
  - `RandomForestClassifier` from `sklearn.ensemble`.
  - `GradientBoostingClassifier` from `sklearn.ensemble`.
  - `MLPClassifier` from `sklearn.neural_network`.
  - `XGBClassifier` from `xgboost`.
  Please put the number sequence of the preferred methods behind `--tab_classifiers` such as `--tab_classifiers 3 4`
  for `SVC` and `DecisionTreeClassifier` method.

### 2) Run via Command line

Below we only list one private run for each dataset. Other runs can be generated by changing `tgt_eps` arguments accordingly.

#### a) CelebA

`python3 dp_ntk_one_class.py --data celeba --n_iter 2000 -lr 1e-2 -bs 125 -dcode 100 --model-ntk fc_1l --ntk-width 800`

#### b) CIFAR-10

`python3 dp_ntk.py --data cifar10 --n_iter 2000 -lr 1e-2 -bs 1500 -dcode 100 --model-ntk cnn_2l --ntk-width 32 --ntk-width-2 256`

#### c) Tabular
The datasets we consider are "adult", "census", "cervical", "credit", "isolet", "epileptic", "intrusion" and "covtype".

`python3 dp_ntk.py --data adult --n_iter 50 -lr 1e-2 -bs 200 -dcode 11 --model-ntk cnn_2l --ntk-width 30 --ntk-width-2 200 --tgt-eps 1`

#### d) CIFAR-10 one class

`python3 dp_ntk_one_class.py --data cifar10 --n_iter 2000 -lr 1e-2 -bs 1500 -dcode 100 --model-ntk cnn_2l --ntk-width 32 --ntk-width-2 256`