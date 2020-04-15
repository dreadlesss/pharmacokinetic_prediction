import os
import sys
import math
from pathlib import Path
import time
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
# import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import warnings


sys.setrecursionlimit(5000)
warnings.filterwarnings("ignore")

def progress_bar(count, total, bar_len):
    percent = count / total
    sys.stdout.write('\rprogress: |{0:{2}}|{1:>4.0%}\n'.format('#'*int(bar_len*percent), percent, bar_len))
    sys.stdout.flush()
    
# set charge for N with valence of 4
def add_formal_charges(m):
    m.UpdatePropertyCache(strict=False)
    for at in m.GetAtoms():
        if at.GetAtomicNum() == 7 and at.GetExplicitValence()==4 and at.GetFormalCharge()==0: # N 
            at.SetFormalCharge(1)
            
def smiles_from_lib(lib_path, properties=['Name', 'Synonyms', 'CAS', 'ID']):
    '''Read in the molecular library such as '.sdf' file
    and extract the properties such as smiles, names, synonyms, cas, id.
    
    lib_path: the path of a '.sdf' file or a directory containing '.sdf' files.
    properties: a property list that wanted.
    '''
    # collect the library path
    path = Path(lib_path)
    path_list = []
    if path.is_dir():
        for file in path.iterdir():
            if file.suffix == '.sdf':
                path_list.append(file)
    else:
        if path.suffix == '.sdf':
            path_list.append(path)
    if not path_list:
        raise ValueError(f'no library found in {lib_path}')
    
    # combine_lib: collect all of the library
    combine_lib = []
    for library in path_list:
        lib_name = library.stem
        suppl = Chem.SDMolSupplier(library.absolute().as_posix(), sanitize=False)
        # molecules: collect all the molecule smiles and properties
        molecules = []
        for mol in suppl:
            # molecule: collect the smiles and properties of each molecule
            molecule = []
            add_formal_charges(mol)
            try:
                new_smiles = Chem.MolToSmiles(mol,isomericSmiles=True)
            except:
                new_smiles = None
                continue
            molecule.append(new_smiles)
            
            # extract properties
            for prop in properties:
                try:
                    mol_prop = mol.GetProp(prop)
                except:
                    mol_prop = None
                finally:
                    molecule.append(mol_prop)
            molecules.append(molecule)
        df = pd.DataFrame(molecules, columns=['SMILES']+properties)
        df['source'] = lib_name
        combine_lib.append(df)
    new_df = pd.concat(combine_lib, axis=0, ignore_index=True)
    new_df.to_csv('smiles_from_library.tsv', sep='\t')
            
def extract_features(path, smiles_col, active_col, force=False, predict=False):
    '''read the 'tsv' file containing SMILES and activity, which is used for feature extracting
    
    path: file path
    force: if True, try to find the feature data, skip feature extraction
           if False, use SMILES to extract features
    predict: if Ture, return only the features for predict task.
             if False, return the features and labels for train task.
    '''
    path = Path(path)
    # find the data exist
    if not force:
        for file in path.parent.iterdir():
            if predict:
                if file.name == 'predict_x.np':
                    data = np.loadtxt(file)
                    return data
            else:
                if file.name == 'train_x_y.np':
                    data = np.loadtxt(file)
                    return data[:,:-1], data[:,-1]

    # read the file and convert the smiles into rdkit.mol object
    df = pd.read_csv(path, sep='\t', index_col=0)
    df['rdmol'] = df[smiles_col].map(Chem.MolFromSmiles)
    # PandasTools.AddMoleculeColumnToFrame(df, smiles_col, 'rdmol')
    df_error = df[df['rdmol'] != df['rdmol']][smiles_col]

    # try to set charge for fail molecules of the train set
    print('\n\nset charge')
    if not predict:
        for index, smiles in df_error.items():
            m = Chem.MolFromSmiles(smiles,sanitize=False)
            try:
                new_smiles = Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True)
                df.loc[index, smiles_col] = new_smiles
                m = Chem.MolFromSmiles(new_smiles)
                df.loc[index, 'rdmol'] = m
            except:
                df.loc[index, 'rdmol'] = None

    # if NaN or None still exists: delete
    df_error = df[df.rdmol != df.rdmol].index.tolist()
    if df_error:
        df.drop(index=df_error, inplace=True)

    # calculate the morgan fingerprint, mfp
    df['rdmol'] = df['rdmol'].map(Chem.RemoveHs)
    mfp_list = []
    for mol in df['rdmol']:
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        mfp_list.append(mfp)
    df['mfp2'] = mfp_list
    df['mfp2'] = df['mfp2'].map(list)
    df['mfp2'] = df['mfp2'].apply(lambda x:str(x)[1:-1])
    df_mfp = df['mfp2'].str.split(',', expand=True)
    df_mfp.columns = [f'm{x}' for x in range(df_mfp.shape[1])]
    df.drop('mfp2', axis=1, inplace=True)
    
    # calculate the descriptors
    des_columns = [desc_name[0] for desc_name in Descriptors._descList]
#     des_columns = ['NumHAcceptors', 'NumHDonors']
    des_calc = MoleculeDescriptors.MolecularDescriptorCalculator(des_columns)
    df_descriptors = pd.DataFrame([des_calc.CalcDescriptors(mol) for mol in df.rdmol], columns=des_columns, index=df.index)
    
    # combine the df and drop where active_col is null
    df = pd.concat([df, df_mfp, df_descriptors], axis=1)
    if not predict:
        drop_index = df[df[active_col].isnull()].index.tolist()
        df.drop(drop_index, axis=0, inplace=True)
    
    # drop the test data that contains np.inf in the df
    if predict:
        df.drop(df[df.isin([np.inf, -np.inf]).any(1)].index.tolist(), axis=0, inplace=True)
    
    # if NaN in features: fill with the mean value
    fea_len = len(des_columns)+2048
    feature = df.iloc[:,-fea_len:].values.astype(float)
    feature = SimpleImputer(np.nan).fit_transform(feature)

    # Standardization
    stdScale = StandardScaler().fit(feature)
    feature_std = stdScale.transform(feature)
    print('feature shape:',feature_std.shape)
    
    # save the feature of x (and y)
    df.drop('rdmol', axis=1, inplace=True)
    out_path = path.parent / f'{path.stem}_clean.tsv'
    df.to_csv(out_path, sep='\t')
    
    if predict:
        np.savetxt(path.parent / 'predict_x.np', feature_std)
        return feature_std
    # training
    else:
        label = df[active_col].values
        np_save = np.concatenate((feature_std, label.reshape(len(label), 1)), axis=1)
        np.savetxt(path.parent / 'train_x_y.np', np_save)
        return feature_std, label

# stratified sampling
def stratified_split(x, y, portion=0.2, bins=[]):
    '''Stratified sampling for continuous variables.
    
    x: data
    y: label
    portion: percentage for the test set
    bins: a list used for segmentation of the continuous values
    '''
    # used for set the interval of the stratified split
    def cut_bins(y, bins):
        mapping = pd.cut(y, bins)
        for i, (interval, group) in enumerate(y.groupby(mapping)):
            if len(group) > 9:
                continue
            else:
                if i != len(bins)-2:
                    bins.remove(bins[i+1])
                    cut_bins(y, bins)
                    break
                else:
                    bins.remove(bins[i])
            break
    
    # generate the bins, if not provided
    y = pd.Series(y)
    if not bins:
        min_val = math.floor(y.min())
        max_val = math.ceil(y.max())
        bins = list(range(min_val-1, max_val+1, 1))
    cut_bins(y, bins)
            
    # segmentation
    mapping = pd.cut(y, bins)
    print(bins)
    # y.groupby(mapping).count()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=portion, stratify=mapping, shuffle=True, random_state=10)
    print(x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test

class auto_gbdt:
    '''Auto tuning the parameters of GBDT by using the preseted parameters.
    
    Examples
    --------
    Using test set as a validation set:
    >>> auto_tune = auto_gbdt(tune_params, gbdt_params, grid_params)
    >>> auto_tune.auto_search(x_train, y_train, x_test, y_test)
    
    Obtain the score only by cv:
    >>> auto_tune = auto_gbdt(tune_params, gbdt_params, grid_params)
    >>> auto_tune.auto_search(x_train, y_train)
    '''
    def __init__(self, tune_params, gbdt_params, grid_params):
        self.tune_params = tune_params
        self.gbdt_params = gbdt_params
        self.grid_params = grid_params
        self.history = {}
        
    def RMSD(self, y_true, y_predict):
        '''RMSD metics'''
        return np.sqrt((y_true - y_predict) ** 2).mean()
        
    def backup(self, i, gsearch):
        '''store the rank of each params into the dict'''
        params_list = gsearch.cv_results_['params']
        params_rank = gsearch.cv_results_['rank_test_score']
        record = {}
        record_data = [x for x in zip(map(int, list(params_rank)), params_list)]
        record_data.sort(key=lambda x:x[0], reverse=True)
        record_data = record_data[-100:]
        for each_rank, each_params in record_data:
            z = 0
            # if records have the same key, number the keys
            while record.get(each_rank):
                z += 1
                if z == 1:
                    each_rank = '{}_{}'.format(str(each_rank), z)
                else:
                    each_rank = '{}_{}'.format(each_rank.split('_')[0], z)
            record[each_rank] = each_params
        record['best_params_score'] = gsearch.best_score_
        if self.test:
            record['pred_rmsd_on_test'] = self.pred_rmsd
        self.history[i] = record
    
    def update_predict(self, i, gsearch):
        '''update the params and predict on the test set'''
        best_param = gsearch.best_params_
        best_score = gsearch.best_score_
        self.gbdt_params.update(best_param)
        print(f'round {i} best params: ', best_param, flush=True)
        print(f'round {i} best cv score: ', best_score, flush=True)
        
        # predict on test set using the best estimator
        if self.test:
            estimator = gsearch.best_estimator_
            y_pred = estimator.predict(self.x_test)
            self.pred_rmsd = self.RMSD(y_test, y_pred)
            print(f'round {i} best param\'s score on testset: ', self.pred_rmsd, flush=True)
            if self.pred_rmsd < self.best_pred_rmsd:
                self.best_pred_rmsd = self.pred_rmsd
                self.best_gbdt_params = self.gbdt_params
        
    def save(self, filename, model):
        '''save the results and model into files'''
        # save the results of each iteration
        self.history['best_params_on_test'] = self.best_gbdt_params
        self.history['best_rmsd_on_test'] = self.best_pred_rmsd
        results = json.dumps(self.history, indent=4)
        with open(filename,'w') as f:
            f.write(results)
            
        # train the model with all the features and labels
        if model:
            estimator = GradientBoostingRegressor(**self.best_gbdt_params)
            x_all = np.concatenate((self.x_train, self.x_test), axis=0)
            y_all = np.concatenate((self.y_train, self.y_test), axis=0)
            estimator.fit(x_all, y_all)
            joblib.dump(estimator, 'model_save.m')
    
    def feature_importances_plot(self, data='', model='model_save.m', top_n=20, sep='\t'):
        '''draw a hist plot for the importances of features in gbdt
        
        model: the estimator file
        data: a DataFrame containing the feature columns
        top_n: the number of most important features showed in plot
        sep: the seperator for the DataFrame
        '''
        # get the feature importances
        estimator = joblib.load(model)
        self.feature_importances = estimator.feature_importances_
        fea_len = len(self.feature_importances)
        
        # get the feature columns
        df_clean = pd.read_csv(data, sep=sep, index_col=0)
        fea_col = df_clean.columns.values[-fea_len:]
        print(fea_col)
        
        # draw a plot of importances for top features
        top_n_imp = pd.Series(self.feature_importances, fea_col).sort_values(ascending=False)[:top_n]
        sns.barplot(x=top_n_imp.keys().tolist(), y=top_n_imp.values.tolist(), color='blue')
        plt.xticks(rotation=90)
        plt.title('Feature Importance')
        plt.ylabel('Feature Importance Score')
        plt.tight_layout()
        save_path = 'feature_importance.png'
        plt.savefig(save_path)
        
    def auto_search(self, x_train, y_train, x_test=None, y_test=None, fine=False, output='auto_results_0.json', model=True):
        '''Iter the grid_parameters and auto tuning the params by GridSearchCV.
        After finishing each round, update the new parameters, predict on test set and make a backup.
        
        x: data
        y: label
        output: the file results stored
        model: refit the model with all the data then save the model
        '''
        self.x_train = x_train
        self.y_train = y_train
        self.best_pred_rmsd = np.inf
        self.best_gbdt_params = self.gbdt_params
        # check out the output file
        i = 1
        while Path(output).exists():
            output = f'{Path(output).stem[:-1]}{i}.json'
            i += 1
            
        if x_test is None:
            self.test = False
        else:
            if y_test is None:
                raise ValueError("y_test is None!")
            self.test = True
            self.x_test = x_test
            self.y_test = y_test
            
        sort_params = sorted(list(self.tune_params), key=lambda x:int(x[-2:].strip('d')))
        for i, tune_key in enumerate(sort_params):
            time1 = time.time()
            tune_param = self.tune_params[tune_key]
            print(f'round {i}: ', tune_param)
            # initialize and search
            estimator = GradientBoostingRegressor(**self.gbdt_params)
            print(estimator)
            progress_bar(i+1, len(sort_params), 30)
            gsearch = GridSearchCV(estimator, tune_param, **self.grid_params)
            gsearch.fit(x_train, y_train)
            
            # update (and predict on test set)
            self.update_predict(i, gsearch)
            
            # backup
            self.backup(i, gsearch)
                
            time2 = time.time()
            print(f'round {i} finish, using time {time2-time1} \n')
        # store the tuning results into json file.
        self.save(output, model)

def predict(features, model='model_save.m'):
    estimator = joblib.load(model)
    y_pred = estimator.predict(features)
    return y_pred
    
def get_params():
    '''There are three types of parameters used for auto tuning:

    tune_params: Used for tuning the GBDT. Just type the parameters name and values in it.
    gbdt_params: Initialized parameters in GBDT. It will be updated as the program executing.
    grid_params: Used for GridSearchCV.

    You can modify the parameters below and have a try.
    '''
    # the number of the key mush arranged in order
    tune_params = {'round0': {'n_estimators':range(100,700,20)},
                   'round1': {'max_depth':range(5,12,1), 
                              'min_samples_split':range(3,60,4)},
                   'round2': {'min_samples_leaf':range(3,60,4), 
                              'min_samples_split':range(3,60,4)},
                   'round3': {'min_samples_leaf':range(3,60,5), 
                              'min_samples_split':range(3,60,5)},
                   'round4': {'max_features':[0.15, 0.3, 0.45, 0.6, 0.75, 0.9]},
                   'round5': {'max_features':[0.07, 0.22, 0.37, 0.52, 0.67, 0.82, 0.97]},
                   'round6': {'subsample':[0.5, 0.6, 0.7, 0.8, 0.9]},
                   'round7': {'subsample':[0.55, 0.65, 0.75, 0.85, 0.95]},
                   'round8': {'n_estimators':range(1000,3000,200), 
                              'learning_rate':[0.01]},
                   'round9': {'n_estimators':range(1200,3500,200), 
                              'learning_rate':[0.009]},
                   'round10': {'n_estimators':range(1400,3500,200), 
                              'learning_rate':[0.008]},
                   'round11': {'n_estimators':range(1400,4000,200), 
                              'learning_rate':[0.007]},
                   'round12': {'n_estimators':range(1400,4000,200), 
                              'learning_rate':[0.006]},
                  }

    gbdt_params = {'n_estimators': 180,
                   'max_depth': 10,
                   'min_samples_split': 16, # 0.5-1% of data set
                   'min_samples_leaf': 8,
                   'subsample': 0.65,
                   'max_features': 0.1,
                   'learning_rate': 0.02,
                   'loss': 'ls',
                   'random_state': 11
                  }

    grid_params = {'scoring':'neg_mean_squared_error',
                   'iid':False,
                   'n_jobs':os.cpu_count(),
                   'cv':4
                  }
    return tune_params, gbdt_params, grid_params

# an example to fit the specified data
if __name__ == '__main__':
    act_list = ['human VDss (L/kg)', 'MRT (h)', 'terminal  t1/2 (h)', 'human CL (mL/min/kg)', 'fraction unbound in plasma (fu)']
    short_list = ['hum_vdss', 'hum_mrt', 'hum_t12', 'hum_clr', 'hum_fu']
    df = pd.read_excel('dataset.xlsx')
    cwd = os.getcwd()
    for act, short in zip(act_list, short_list):
        print(f'{short} begin')
        workspace = Path(cwd) / short
        workspace.mkdir(parents=True, exist_ok=True)
        os.chdir(workspace.as_posix())
        
        # transform
        if act != 'fraction unbound in plasma (fu)':
            df[short] = df[act].apply(lambda x:np.log10(x))
            df.to_csv(f'{short}.tsv', sep='\t')
        
        # training
        x, y = extract_features(f'{short}.tsv', 'SMILES', short, False)
        x_train, x_test, y_train, y_test = stratified_split(x, y, 0.3, bins = [round(x, 1) for x in np.arange(-3, 4, 0.1)])
        tune_params, gbdt_params, grid_params = get_params()
        auto_tune = auto_gbdt(tune_params, gbdt_params, grid_params)
        auto_tune.auto_search(x_train, y_train, x_test, y_test)
        auto_tune.feature_importances_plot(data=f'{short}_clean.tsv')