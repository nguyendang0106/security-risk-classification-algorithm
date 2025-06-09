import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
import pickle
import pathlib
import itertools # To generate combinations
import sys

def evaluate_stage1_extension_thresholds(
    y_true_binary,
    y_scores,
    # --- Updated: Define candidate thresholds directly ---
    candidate_thresholds_b,
    candidate_thresholds_u,
    # --- End Update ---
    zero_day_recall_min=0.7
):
    """
    Evaluates specific combinations of threshold_b and threshold_u.

    Args:
        y_true_binary (np.array): True binary labels from validation set
                                  (e.g., 1 for Benign, -1 for Malicious/Anomaly).
        y_scores (np.array): Anomaly scores from the Stage 1 model (OCSVM) on the
                             validation set (lower scores mean more likely Benign).
        candidate_thresholds_b (list): List of specific threshold_b values to test.
        candidate_thresholds_u (list): List of specific threshold_u values to test.
        zero_day_recall_min (float): Minimum recall for the Malicious class (-1) required
                                     for the 'Best Overall' criterion.

    Returns:
        dict: A dictionary containing the best (threshold_b, threshold_u) pairs found
              for different evaluation criteria.
        pd.DataFrame: A DataFrame containing the calculated metrics for all evaluated
                      threshold combinations.
    """
    results = []

    # --- Use provided candidate thresholds directly ---
    # Ensure unique values and sort them
    candidate_thresholds_b = sorted(list(set(candidate_thresholds_b)))
    candidate_thresholds_u = sorted(list(set(candidate_thresholds_u)))

    print(f"Evaluating {len(candidate_thresholds_b)} specific candidates for threshold_b and {len(candidate_thresholds_u)} specific for threshold_u ({len(candidate_thresholds_b) * len(candidate_thresholds_u)} combinations)...")

    # --- Evaluation Loop ---
    # Use itertools.product for cleaner combination generation
    for t_b, t_u in itertools.product(candidate_thresholds_b, candidate_thresholds_u):

        # Optional constraint: Enforce t_u >= t_b if it makes logical sense
        # This might be important depending on how you interpret the thresholds
        # If t_u < t_b, a sample could pass stage 1 but fail stage 3, which might be unintended.
        # Uncomment the following lines if you want to enforce t_u >= t_b
        # if t_u < t_b:
        #     print(f"Skipping combination t_b={t_b:.6f}, t_u={t_u:.6f} because t_u < t_b")
        #     continue

        # --- Simulate Pipeline Logic (Stage 1 + Extension) ---
        # Stage 1: Classify Benign vs Potential_Fraud
        pred_stage1 = np.where(y_scores < t_b, 1, -1) # 1=Benign, -1=Potential_Fraud
        final_pred = pred_stage1.copy()

        # Identify samples marked as Potential_Fraud for the Extension Stage
        potential_fraud_indices = (final_pred == -1)

        # Extension Stage: Re-classify Potential_Fraud -> Benign or Unknown/Fraud
        if np.sum(potential_fraud_indices) > 0:
            # Apply threshold_u only to samples flagged by Stage 1
            pred_extension = np.where(y_scores[potential_fraud_indices] < t_u, 1, -1) # 1=Benign, -1=Remains Unknown/Fraud
            final_pred[potential_fraud_indices] = pred_extension
        # If no samples were flagged by Stage 1, final_pred remains pred_stage1

        # --- Calculate Metrics ---
        # Treat the final -1 label as the 'Malicious' or 'Anomaly' class for binary metrics
        try:
            if len(np.unique(y_true_binary)) < 2:
                 raise ValueError("y_true_binary must contain at least two classes.")

            acc = accuracy_score(y_true_binary, final_pred)
            if len(np.unique(final_pred)) < 2: # Handle cases where only one class is predicted
                 bacc = balanced_accuracy_score(y_true_binary, final_pred)
                 f1_macro = f1_score(y_true_binary, final_pred, average='macro', labels=[1, -1], zero_division=0)
                 f1_weighted = f1_score(y_true_binary, final_pred, average='weighted', labels=[1, -1], zero_division=0)
                 recall_malicious = recall_score(y_true_binary, final_pred, pos_label=-1, zero_division=0)
            else:
                 bacc = balanced_accuracy_score(y_true_binary, final_pred)
                 f1_macro = f1_score(y_true_binary, final_pred, average='macro', labels=[1, -1], zero_division=0)
                 f1_weighted = f1_score(y_true_binary, final_pred, average='weighted', labels=[1, -1], zero_division=0)
                 recall_malicious = recall_score(y_true_binary, final_pred, pos_label=-1, zero_division=0)

        except Exception as e:
            print(f"Warning: Could not calculate metrics for t_b={t_b}, t_u={t_u}. Error: {e}")
            acc, bacc, f1_macro, f1_weighted, recall_malicious = np.nan, np.nan, np.nan, np.nan, np.nan # Use NaN for errors

        results.append({
            "threshold_b": t_b,
            "threshold_u": t_u,
            "Accuracy": acc,
            "Balanced Accuracy": bacc,
            "F1 Macro": f1_macro,
            "F1 Weighted": f1_weighted,
            "Malicious Recall": recall_malicious # Proxy for Zero-Day Recall in this context
        })
    # --- End Evaluation Loop ---

    results_df = pd.DataFrame(results).dropna() # Drop rows where metrics failed
    print("Evaluation complete.")
    if results_df.empty:
        print("Warning: No valid results generated after dropping NaNs.")
        return {}, pd.DataFrame()
    print(f"Generated {len(results_df)} valid combinations.")


    # --- Find Best Thresholds Based on Criteria ---
    best_thresholds = {}
    try:
        # Best Accuracy
        best_acc_row = results_df.loc[results_df['Accuracy'].idxmax()]
        best_thresholds['Best Accuracy'] = (best_acc_row['threshold_b'], best_acc_row['threshold_u'])

        # Best Balanced Accuracy
        best_bacc_row = results_df.loc[results_df['Balanced Accuracy'].idxmax()]
        best_thresholds['Best Balanced Accuracy'] = (best_bacc_row['threshold_b'], best_bacc_row['threshold_u'])

        # Best F1 Weighted
        best_f1w_row = results_df.loc[results_df['F1 Weighted'].idxmax()]
        best_thresholds['Best F1 Weighted'] = (best_f1w_row['threshold_b'], best_f1w_row['threshold_u'])

        # Best Overall (High Malicious Recall, then max F1 Weighted)
        overall_candidates = results_df[results_df['Malicious Recall'] >= zero_day_recall_min]
        if not overall_candidates.empty:
            best_overall_row = overall_candidates.loc[overall_candidates['F1 Weighted'].idxmax()]
            best_thresholds[f'Best Overall (Recall >= {zero_day_recall_min})'] = (best_overall_row['threshold_b'], best_overall_row['threshold_u'])
        else:
            best_thresholds[f'Best Overall (Recall >= {zero_day_recall_min})'] = "No thresholds met criteria"
            print(f"Warning: No threshold combinations found with Malicious Recall >= {zero_day_recall_min}")

    except Exception as e:
        print(f"Error finding best thresholds: {e}")


    return best_thresholds, results_df

# --- Example Usage ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly

    results_dir = pathlib.Path(".") # Assuming script is in W10
    model_dir = results_dir / "models1" # Or wherever train.py saved models

    try:
        # --- Define the specific candidate thresholds ---
        candidate_thresholds_b = [
            -1.948011937091633,  # F1, F2
            -1.963997780965287,  # F3
            -1.965265219911375,  # F4, F5, F6, F7, F8
            -2.093237890622742   # F9
        ]
        candidate_thresholds_u = [
            -0.05087859980271059,  # 0.995
            -0.09133471475575776,  # 0.99
            -0.15782979628351096, # 0.975
            -0.2512931719589538,   # 0.95
            -0.3019786148202116,   # 0.935
            -0.3284944572372088,   # 0.925
            -0.3770408812652148,   # 0.9
            -0.45640522776052656,  # 0.85
            -0.4634258538702278,   # 0.845
            -0.46993522279665545,  # 0.84
            -0.47665639328004134,  # 0.835
            -0.48384136149362134,  # 0.83
            -0.4925321368877058,   # 0.825
            -0.5009604503864216,   # 0.82
            -0.5100464235105172,   # 0.815
            -0.5165369659431871,   # 0.81
            -0.5241690360438408,   # 0.805
            -0.5308593611107351,   # 0.8
            -0.6269508482599138,   # 0.75
            -0.7016076441016396,   # 0.7
            -0.7827844311060698,   # 0.65
            -0.9302507749410611,   # 0.6
            -1.0269408273712022,  # 0.55
            -1.1280812835148408,  # 0.5
            -1.2386972646948347,  # 0.45
            -1.4374062291834169,  # 0.4
            -1.6178625807267037,  # 0.35
            -1.8986482295841358,  # 0.3
            -2.032462394914112,   # 0.25
            -2.1314457188018845,  # 0.2
            -2.25895516898398,    # 0.15
            -2.312643858334508,   # 0.1
            -2.344586263096881    # 0.05
        ]
        # --- End Define specific candidates ---


        # --- Load necessary data from train.py output ---
        # You MUST ensure these files exist and contain the correct data
        # Modify train.py to save these if necessary.
        try:
            # Load the validation labels (1 for Benign, -1 for Malicious)
            with open(model_dir / "val_ae_y.pkl", "rb") as f:
                 y_val_binary = pickle.load(f)
            # Load the OCSVM scores on the validation set (using the 100k model)
            with open(model_dir / "score_val_100k.pkl", "rb") as f:
                 score_val_100k = pickle.load(f)
            print(f"Loaded validation labels (shape: {y_val_binary.shape}) and scores (shape: {score_val_100k.shape})")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure train.py saves 'val_ae_y.pkl' and 'score_val_100k.pkl' in the models directory.")
            print("Using dummy data as fallback - results will not be meaningful.")
            # Fallback to dummy data if files don't exist
            dummy_y_true_benign = np.ones(1500) # Example size
            dummy_y_true_malicious = -np.ones(1948) # Example size
            y_val_binary = np.concatenate((dummy_y_true_benign, dummy_y_true_malicious))
            # Dummy scores - replace with actual scores
            score_val_100k = np.random.normal(loc=0, scale=5, size=len(y_val_binary)) # Adjusted scale based on logs
            score_val_100k[y_val_binary == 1] -= 5 # Make benign scores lower
            score_val_100k[y_val_binary == -1] += 2 # Make malicious scores higher
            score_val_100k += np.random.normal(0, 1, len(y_val_binary)) # Add noise
        except Exception as e:
             print(f"An unexpected error occurred during data loading: {e}")
             sys.exit(1) # Exit if data loading fails unexpectedly
        # --- End Load Data ---


        # --- Run the evaluation using the specific thresholds ---
        best_threshold_pairs, all_results_df = evaluate_stage1_extension_thresholds(
            y_val_binary,
            score_val_100k,
            candidate_thresholds_b, # Pass the specific list
            candidate_thresholds_u, # Pass the specific list
            zero_day_recall_min=0.7 # Example minimum recall for 'Overall'
        )

        # --- Print Results ---
        print("\n--- Best Threshold Pairs Found (Based on 100k Validation) ---")
        if best_threshold_pairs:
            for criterion, thresholds in best_threshold_pairs.items():
                if isinstance(thresholds, tuple):
                    print(f"- {criterion}: threshold_b={thresholds[0]:.6f}, threshold_u={thresholds[1]:.6f}")
                else:
                    print(f"- {criterion}: {thresholds}")
        else:
            print("No best thresholds could be determined.")

        # --- Save Full Results (Optional) ---
        output_csv_path = results_dir / "stage1_extension_threshold_evaluation_specific.csv" # Changed filename
        all_results_df.to_csv(output_csv_path, index=False)
        print(f"\nFull evaluation results saved to: {output_csv_path}")

        # --- Show Top Results Based on Criteria (Example) ---
        if not all_results_df.empty:
            print("\n--- Top 5 Results by Balanced Accuracy ---")
            print(all_results_df.sort_values(by="Balanced Accuracy", ascending=False).head())

            print("\n--- Top 5 Results for Overall Criterion (Recall >= 0.7) ---")
            overall_candidates = all_results_df[all_results_df['Malicious Recall'] >= 0.7]
            if not overall_candidates.empty:
                print(overall_candidates.sort_values(by="F1 Weighted", ascending=False).head())
            else:
                print(f"No threshold combinations found with Malicious Recall >= 0.7")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        # Optionally add more specific error handling or re-raise
        import traceback
        traceback.print_exc()





candidate_thresholds_b = [
            4.136933033252717,  # F1
            4.131666520101135,  # F2
            -5.386647489620373, # F3, F4
            -5.433434425438463  # F5, F6, F7, F8, F9
]
candidate_thresholds_u = [
    7.643434769100277,   # 0.995
    6.824624366596982,   # 0.99
    6.22994717938127,    # 0.975
    5.4670587832588335,  # 0.95
    4.86743364053109,    # 0.935
    4.682177457706713,   # 0.925
    4.434328229246604,   # 0.9
    3.7008306163494127,  # 0.85
    1.8760037207481255,  # 0.8
    0.9190983361768303,  # 0.75
    0.4167966030363459,  # 0.7
    0.04552454883814801, # 0.65
    -0.3536344539432322, # 0.6
    -0.7702020008306131, # 0.55
    -1.1690952684002696, # 0.5
    -1.5072963169353897, # 0.45
    -1.816158812616777,  # 0.4
    -2.075113604361832,  # 0.35
    -2.29816812389181,   # 0.3
    -2.5346928942526574, # 0.25
    -2.7951741845623475, # 0.2
    -3.031223325204337,  # 0.15
    -3.3609381773523634, # 0.1
    -4.43079797317332   # 0.05
]

# ocsvm_pipeline_base = Pipeline(
#     [
#         ("imputer", SimpleImputer(strategy='mean')), 
#         ("pca", PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)), 
#         ("ocsvm", OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=True, max_iter=-1))
#     ]
# )

# # --- Define the parameter distributions to sample from --- # <-- Changed
# ocsvm_param_dist = {
#     'pca__n_components': randint(10, 15), # Sample integers from 10 up to (but not including) 16
#     'ocsvm__gamma': loguniform(6e-2, 1e0), # Sample from log-uniform distribution between 
#     'ocsvm__nu': loguniform(2e-4, 1e0), # Sample from log-uniform distribution between 
#     'ocsvm__kernel': ['rbf'] # Keep kernel fixed or add others like 'poly' if desired
# }
# # --- End Define Distributions ---

# print("\n--- Optimizing Stage 1 OCSVM using RandomizedSearchCV ---") # <-- Changed
# # Use the 100k training set size ('ae') for optimization
# # Use roc_auc scoring
# # --- Use RandomizedSearchCV --- # <-- Changed
# n_iterations = 15 # Number of parameter settings that are sampled. Adjust as needed.
# clf_ocsvm = RandomizedSearchCV(
#     estimator=ocsvm_pipeline_base,
#     param_distributions=ocsvm_param_dist, # <-- Use distributions
#     n_iter=n_iterations, # <-- Specify number of iterations
#     scoring='roc_auc',
#     cv=3,
#     verbose=2,
#     n_jobs=-1,
#     random_state=seed_value # Set random state for reproducibility of sampling
# )



# candidate_thresholds_b = [
#             7358.98843248808,    # F1
#             2412.6025796339245,  # F2
#             2270.223193305139,   # F3, F4, F5, F6
#             115.47566260323765   # F7, F8, F9
# ]
# candidate_thresholds_u = [
#     8043.460728362692,  # 0.995           F3: 94 BruF64 WA87 25
#     7845.011842171679,  # 0.99  F7: 93-25 F3: 94 BruF64 WA87 25
#     7076.462751742779,  # 0.975 F7: 92-25 F3: 92 BruF64 WA87 25
#     6089.156843347407,  # 0.95  F7: 91-70 F3: 90 BruF64 WA87 68
#     5654.048986801937,  # 0.935 F7: 89-80 F3: 88 BruF64 WA87 78
#     5563.252694668334,  # 0.925 F7: 89-80 F3: 88 BruF64 WA87 78                              F7 + 400: 87 BruF64 78
#     5427.014657281848,  # 0.9   F7: 87-80 F3: 87 BruF63 WA86 80
#     4968.257591611648,  # 0.85  F7: 81-80 F3: 82 BruF63 WA86 83
#     4283.223524173904,  # 0.8   F7: 77-80 F3: 77 BruF63 WA86 83
#     3540.779403728062,  # 0.75  F7: 72-87 F3: 72 BruF63 WA86 89
#     3084.4921398414936, # 0.7   F7: 69-87 F3: 70 BruF63 WA86 89
#     2013.1842640251689, # 0.65  F7: 65-95
#     1473.2989459851308, # 0.6   F7: 60-95                        F7 + 1000: 60 BruF63 WA87 95 F7 + 400: 65 BruF63 WA87 95
#     1138.855295944574,  # 0.55
#     599.6938502654693,  # 0.5
#     301.74265518688105, # 0.45
#     103.97320095381478, # 0.4
#     -52.44815724257981, # 0.35
#     -173.66611730097836,# 0.3
#     -287.6913082109986, # 0.25
#     -398.3050005719891, # 0.2
#     -517.8082429672334, # 0.15
#     -651.7080115487577, # 0.1
#     -828.8560532570824  # 0.05
# ]




candidate_thresholds_b = [
            4300.253439874997,   # F1
            1862.573363626765,   # F2, F3, F4
            327.9705384888148    # F5, F6, F7, F8, F9
]
candidate_thresholds_u = [
    4465.658421984566,   # 0.995  F5: 95-38
    4379.70714529618,    # 0.99
    4141.371209929084,   # 0.975
    3857.7025223646483,  # 0.95
    3656.8339468869253,  # 0.935 F5: 89-83
    3538.611707915062,   # 0.925 F5: 88-83
    3292.231254204273,   # 0.9   F5: 86-85 ok
    2749.481755640221,   # 0.85  F5: 80-85
    2376.636607853582,   # 0.8   F5: 75-89
    2109.0709320214733,  # 0.75
    1732.4371891650806,  # 0.7   F5: 67-89
    1371.203167928436,   # 0.65
    1039.7315015336858,  # 0.6
    730.7710147265658,   # 0.55
    468.3211444909516,   # 0.5   F5: 49-97
    267.8901461860602,   # 0.45
    123.95235662184157,  # 0.4
    7.393404867964631,   # 0.35
    -99.99456275060814,  # 0.3
    -217.3455552237847,  # 0.25
    -350.7122367486596,  # 0.2
    -497.07834456928924, # 0.15
    -666.3931470377094,  # 0.1
    -905.3503003994916   # 0.05
]

# 2017: ok
candidate_thresholds_b = [
    -0.03254373742560457,  # F1
    -0.13715624211525668,  # F2, F3, F4
    -0.141627077936073,    # F5, F6
    -0.14713456462666652   # F7, F8, F9
]
candidate_thresholds_u = [
    -0.00036979748732912654, # 0.995 F7: 95-40
    -0.002716603800830275,  # 0.99
    -0.011360895181434856, # 0.975
    -0.018417145530570833, # 0.95
    -0.023535124373489244, # 0.935
    -0.026038157752358973, # 0.925
    -0.03341903502330623,  # 0.9   F7: 88-53
    -0.04618108151940381,  # 0.85  F7: 83-72
    -0.05662668561211542,  # 0.815 F7: 80-78
    -0.05841619571424543,  # 0.81  F7: 79-87
    -0.059760980431092126, # 0.805 F7: 79-89
    -0.0610887238591802,   # 0.8   F7: 78-91 ok
    -0.07038237226165223,  # 0.75  F7: 73-93
    -0.08748450849032925,  # 0.7   F7: 69-95
    -0.09886488638158172,  # 0.65
    -0.11561592319373735,  # 0.6
    -0.14181118489430158,  # 0.55  F7: 55-100
    -0.161334513704265,    # 0.5
    -0.1750277011733698,   # 0.45
    -0.18560321232473787,  # 0.4
    -0.19483653797015893,  # 0.35
    -0.20271925322769016,  # 0.3
    -0.21010861510767354,  # 0.25
    -0.2180081669708738,   # 0.2
    -0.2266767390503119,   # 0.15
    -0.2351687136355554,   # 0.1
    -0.2463425618207814    # 0.05
]




