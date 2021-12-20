import argparse

import numpy as np
from mlxtend.evaluate import mcnemar_table, mcnemar

import config
import main

def McNemar_main(model_1_path, use_gcn_1, use_le_1, model_2_path, use_gcn_2, use_le_2):
    assert(config.is_classification_task)
    config.model_name = model_1_path
    config.use_gcn = use_gcn_1
    config.use_le = use_le_1
    _, _, _, predictions_1, actuals_1 = main.choose_main()
    config.model_name = model_2_path
    config.use_gcn = use_gcn_2
    config.use_le = use_le_2
    _, _, _, predictions_2, actuals_2 = main.choose_main()
    assert(actuals_1 == actuals_2)
    actuals = actuals_1

    alpha = 0.05
    tb = mcnemar_table(y_target=np.array(actuals), y_model1=np.array(predictions_1), y_model2=np.array(predictions_2))
    chi2, p = mcnemar(ary=tb, corrected=True)
    print(f'chi-squared: {chi2}, p-value: {p}')
    if(p > alpha):
        print('No significant difference')
    else:
        print('Significant difference')

if __name__ == '__main__':
    assert(config.train == False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=True, default='McNemar')
    parser.add_argument('--model_1_path', required=True)
    parser.add_argument('--use_gcn_1', action='store_true')
    parser.add_argument('--use_le_1', action='store_true')
    parser.add_argument('--model_2_path', required=True)
    parser.add_argument('--use_gcn_2', action='store_true')
    parser.add_argument('--use_le_2', action='store_true')
    args = parser.parse_args()

    if(args.test == 'McNemar'):
        McNemar_main(model_1_path=args.model_1_path, use_gcn_1=args.use_gcn_1, use_le_1=args.use_le_1, model_2_path=args.model_2_path, use_gcn_2=args.use_gcn_2, use_le_2=args.use_le_2)
