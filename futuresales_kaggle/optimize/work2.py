'''
Created on Aug. 2, 2021

@author: zollen
@url: https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c
'''
import optuna 
import joblib
from optuna.samplers import CmaEsSampler
import pandas as pd
import time
from os import path
import warnings


warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)


label = 'item_cnt_month'
base_features = ['date_block_num', 'shop_id', 'item_id', 
            'shop_category', 'shop_city', 
            'item_category_id', 'name2', 
            'name3', 'item_type', 'item_subtype', 'item_price', label]




train = pd.read_csv('../data/monthly_train.csv')
items = pd.read_csv('../data/monthly_items.csv')
cats = pd.read_csv('../data/monthly_cats.csv')
shops = pd.read_csv('../data/monthly_shops.csv')


'''
merge cats, shops and items
'''
items_cats = pd.merge(items, cats, how='left', on='item_category_id')
train_item_cats = pd.merge(train, items_cats, how='left', on='item_id')
train_item_cats_shops = pd.merge(train_item_cats, shops, how='left', on='shop_id')

data = train_item_cats_shops[base_features].values.tolist()




'''
clip values between 0 and 20
'''
train_item_cats_shops[label] = train_item_cats_shops[label].clip(0, 20)



'''
Optimization
'''
def evaluate(trial, data):
    
    p = []
    p.append(trial.suggest_float("p0", -10, 10))
    for i in range(1, 12):
        p.append(trial.suggest_float("p" + str(i), -2, 2))     
    
    return sum(abs(x[11] - (p[0] +      
                         p[1]  * x[0] +    
                         p[2]  * x[1] + 
                         p[3]  * x[2] + 
                         p[4]  * x[3] + 
                         p[5]  * x[4] + 
                         p[6]  * x[5] + 
                         p[7]  * x[6] + 
                         p[8]  * x[7] + 
                         p[9]  * x[8] +
                         p[10] * x[9] + 
                         p[11] * x[10])) for x in data)
    

start_st = time.time()

file = "futuresales.pkl"
# Create study that minimizes
if path.exists(file):
    study = joblib.load(file)
else:
    study = optuna.create_study(
                study_name='futuresales-study',
                direction="minimize", sampler=CmaEsSampler(seed=int(time.time())))

# Pass additional arguments inside another function
func = lambda trial: evaluate(trial, data)

# Start optimizing with 100 trials
study.optimize(func, n_trials=400)

end_st = time.time()

print(f"Best Score: {study.best_value:.4f} params: {study.best_params}")
print("TIME: ", end_st - start_st)

joblib.dump(study, file)

'''
TOTAL: 14242872
Score: 2250073130.0039  params: {'p0': -0.5474754044323396, 'p1': -0.22365215518548298, 'p2': -1.288031780721671, 'p3': -0.00726546261621994, 'p4': -0.37894192162338103, 'p5': 1.1902314904699083, 'p6': -0.39375544834483, 'p7': -0.0683183202926567, 'p8': -0.11570387695028395, 'p9': 0.9350971601602047, 'p10': -0.6414492712711383, 'p11': 0.06585529945306223}
Score: 1603687314.1622  params: {'p0': -0.26701899380834504, 'p1': 0.5192701508371832, 'p2': 0.0010182488657887675, 'p3': -0.014977711643763594, 'p4': 0.03285648564724977, 'p5': 0.10203659111934255, 'p6': 0.28920470992600295, 'p7': -0.7602955823756627, 'p8': 0.1361232979885994, 'p9': -1.6865537135981563, 'p10': 1.2379125214377276, 'p11': 0.01903338535365648}
Score: 1208362412.0784  params: {'p0': -0.30765495158226214, 'p1': -0.029031933525130782, 'p2': -0.30741734047186403, 'p3': 0.005358124757376518, 'p4': -0.017848373531018952, 'p5': 0.719662202065308, 'p6': 0.4838679424152582, 'p7': -1.12995252252948, 'p8': 6.16524535470564e-05, 'p9': -1.1581930974154977, 'p10': 0.8297081140081501, 'p11': -0.05102987682630354}
Score:  756355966.2265  params: {'p0': -0.2611470037911816, 'p1': 0.5744318035329472, 'p2': -1.0001012033257455, 'p3': 0.003366586514593309, 'p4': 0.5959202153015166, 'p5': 0.8275140843344919, 'p6': 0.45478096430222525, 'p7': -1.0102388979774857, 'p8': -0.05525752963978575, 'p9': -0.7109598704495426, 'p10': 0.6615630494039888, 'p11': -0.028849211620357168}
Score:  445973079.0594  params: {'p0': -0.5022641345851151, 'p1': 0.9639120679965139, 'p2': -1.1682852021751502, 'p3': -0.001567861196213311, 'p4': 0.7849135331643058, 'p5': 0.4581500317986383, 'p6': 0.5056857476084591, 'p7': -0.8864713088386362, 'p8': 0.019387114263768303, 'p9': -1.2218034245209133, 'p10': 0.5957139475984341, 'p11': 0.023665919315747054}
Score:  392083009.0393  params: {'p0': -0.5387194878036606, 'p1': 1.0444998440181419, 'p2': -1.0295171211814047, 'p3': -0.0011728871121430549, 'p4': 0.638089356581512, 'p5': 0.19695254295871314, 'p6': 0.14719174515689684, 'p7': -0.6944693224611554, 'p8': 0.010734660420002682, 'p9': -1.093239400018196, 'p10': 0.7040157621877511, 'p11': 0.008736949276336009}
Score:    9993428.9892  params: {'p0': 0.22263052594961716, 'p1': -0.0037574446777883367, 'p2': -0.21160089443082766, 'p3': 5.345092567586848e-06, 'p4': 0.1584767966669373, 'p5': 0.444078371995385, 'p6': 0.03614217131723978, 'p7': -0.0034997025703796406, 'p8': -5.3281510311061204e-05, 'p9': -0.4018596239024941, 'p10': 0.002584811461937474, 'p11': -4.143774467790806e-05}
Score:    5283001.9754  params: {'p0': 0.3995027651309915, 'p1': -0.005205458475763059, 'p2': -0.011827831329361518, 'p3': 1.3245932717327848e-06, 'p4': 0.06460644869825166, 'p5': 0.01409295492701945, 'p6': 0.035601820616450196, 'p7': -0.0007417129197036373, 'p8': -3.88454306929386e-05, 'p9': -0.3236454240278343, 'p10': -0.0037560178454344203, 'p11': 5.327291178231295e-06}
Score:    4585957.8220  params: {'p0': 0.3630337990884213, 'p1': -0.0019920671500559114, 'p2': 0.0002768197376257237, 'p3': -5.708091754501396e-07, 'p4': 0.0037273543927742146, 'p5': -0.001760081672170653, 'p6': 0.037002790364250325, 'p7': -0.00017971775156264624, 'p8': -7.117334574514825e-05, 'p9': -0.3308243207553824, 'p10': -0.00308233578016467, 'p11': -8.947824759849404e-07}
Score:    4499567.3107  params: {'p0': 0.3414917153213903, 'p1': -0.0007867477063044753, 'p2': 0.0005696528052444431, 'p3': -3.0241276731980586e-06, 'p4': -0.0036499804553888912, 'p5': -0.0022933769635819007, 'p6': 0.03674485643047441, 'p7': -0.0002504345114540311, 'p8': -4.738994173605679e-05, 'p9': -0.3220762554266675, 'p10': -0.004233390575115428, 'p11': 3.063646293141192e-06}
Score:    4471981.6582  params: {'p0': 0.3347085991307273, 'p1': -0.0007675979026359548, 'p2': 0.0015496759896976675, 'p3': -2.258348882203343e-06, 'p4': -0.005451181962567783, 'p5': -0.004241471948945454, 'p6': 0.036027936470967685, 'p7': -0.0004885430254316118, 'p8': -3.021437151532771e-05, 'p9': -0.3196848511163721, 'p10': -0.0037509102375263242, 'p11': 3.0235513341668553e-06}
Score:    4331106.1406  params: {'p0': 0.2264010315336057, 'p1': -0.0009047846491175149, 'p2': 0.0004007501824045904, 'p3': -3.173089792356543e-06, 'p4': -0.00041176728774440247, 'p5': -0.0018250855829772313, 'p6': 0.030152709765053978, 'p7': -8.709346174862865e-05, 'p8': -1.6614113032627461e-06, 'p9': -0.26167210968598364, 'p10': -0.0027686749122003106, 'p11': -7.085077801971091e-07}

'''


