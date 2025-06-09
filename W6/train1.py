import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import collections
from tabulate import tabulate
import warnings

# Scikit-learn imports
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

# PyTorch imports
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings("ignore")

# Fully connected neural network with two hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size_2)
        self.l3 = nn.Linear(hidden_size_2, num_classes)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

# Function to plot feature importance
def plot_feature_importance(features, importances):
    """Plot feature importance from a model"""
    # Sắp xếp theo mức độ quan trọng giảm dần
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance in RandomForestClassifier")
    plt.gca().invert_yaxis()  # Đảo ngược trục để feature quan trọng nhất ở trên cùng
    plt.savefig('W6/feature_importance.png')  # Save figure instead of showing it
    print("Feature importance plot saved to 'W6/feature_importance.png'")

# NN
def train_neural_network(X_train_RFE, y_train, X_test_RFE, y_test):
    """Train and evaluate a neural network model"""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    input_size = X_train_RFE.shape[1]  # Number of features
    hidden_size = 64
    hidden_size_2 = 64
    num_classes = len(set(y_train))
    num_epochs = 40
    batch_size = 32
    learning_rate = 0.001

    print(f"Neural Network parameters: input_size={input_size}, hidden_size={hidden_size}, "
          f"hidden_size_2={hidden_size_2}, num_classes={num_classes}")

    # Initialize model
    model = NeuralNet(input_size, hidden_size, hidden_size_2, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() # Tiêu chí này kết hợp nn.LogSoftmax() và nn.NLLLoss() trong một lớp duy nhất.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Convert data to numpy arrays
    X_train_RFE_vals = X_train_RFE.values
    y_train_vals = y_train.values
    X_test_RFE_vals = X_test_RFE.values
    y_test_vals = y_test.values

    # Train the model
    n_total_steps = len(X_train_RFE_vals)
    print(f"Starting neural network training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, X_train_RFE_vals.shape[0], batch_size):
            # Get batch
            batch_end = min(i + batch_size, X_train_RFE_vals.shape[0])
            x = torch.FloatTensor(X_train_RFE_vals[i:batch_end]).to(device)
            y = torch.LongTensor(y_train_vals[i:batch_end]).to(device)
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            epoch_loss += loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print epoch stats
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/n_total_steps:.4f}')

    # Test the model
    print("Evaluating neural network...")
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i in range(0, X_test_RFE_vals.shape[0], batch_size):
            batch_end = min(i + batch_size, X_test_RFE_vals.shape[0])
            x = torch.FloatTensor(X_test_RFE_vals[i:batch_end]).to(device)
            y = torch.LongTensor(y_test_vals[i:batch_end]).to(device)
            
            outputs = model(x)
            if len(outputs.data) > 0:
                # max returns (value, index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += y.size(0)
                n_correct += (predicted == y).sum().item()
        
        acc = 100.0 * n_correct / n_samples if n_samples > 0 else 0
        print(f'Accuracy of the neural network: {acc:.2f}%')
    
    return model, acc

def main():
    print("Loading datasets...")
    try:
        train = pd.read_csv('W6/data/UNSW_NB15_training-set.csv')
        test = pd.read_csv('W6/data/UNSW_NB15_testing-set.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the data files exist in the correct location")
        return

    # Combine datasets
    combined_data = pd.concat([train, test]).drop(['id','label'], axis=1)
    print(f"Combined data shape: {combined_data.shape}")
    print("Preview of combined data:")
    print(combined_data.head(5))

    
    # Calculate normal data proportion
    tmp = train.where(train['attack_cat'] == "Normal").dropna()
    print('Train normal proportion:', round(len(tmp['attack_cat'])/len(train['attack_cat']),5))

    tmp = test.where(test['attack_cat'] == "Normal").dropna()
    print('Test normal proportion:', round(len(tmp['attack_cat'])/len(test['attack_cat']),5))

    # Label encoding
    le = LabelEncoder()
    vector = combined_data['attack_cat']
    print("Attack categories:", list(set(list(vector))))

    combined_data['attack_cat'] = le.fit_transform(vector)
    combined_data['proto'] = le.fit_transform(combined_data['proto'])
    combined_data['service'] = le.fit_transform(combined_data['service'])
    combined_data['state'] = le.fit_transform(combined_data['state'])

    print('\nDescribing attack types:')
    print("Mode:", vector.mode())
    print(f"Mode {np.sum(combined_data['attack_cat'].values==6)/vector.shape[0]:.2f}%")
    print("Looks like 6 is 'normal', but it's not that common")

    # Display counts of each attack type
    counter = collections.Counter(vector)  
    print(tabulate(counter.most_common(), headers=['Type', 'Occurrences']))

    # Create a deep copy
    combined_data = combined_data.copy(deep=True)
    
    # Feature analysis
    lowSTD = list(combined_data.std().to_frame().nsmallest(7, columns=0).index)
    lowCORR = list(combined_data.corr().abs().sort_values('attack_cat')['attack_cat'].nsmallest(7).index)

    exclude = list(set(lowCORR + lowSTD))
    if 'attack_cat' in exclude:
        exclude.remove('attack_cat')

    print('Shape before PCA:', combined_data.shape)
    print('Replace the following with their PCA(3):', exclude)

    # Apply PCA to selected features
    pca = PCA(3)
    dim_reduct = pca.fit_transform(combined_data[exclude])
    print("Explained variance ratio:", sum(pca.explained_variance_ratio_))

    # Remove original features and add PCA results
    combined_data.drop(exclude, axis=1, inplace=True)
    dim_reduction = pd.DataFrame(dim_reduct)
    combined_data = combined_data.join(dim_reduction)
    print('Shape after PCA:', combined_data.shape)

    # Feature scaling
    print('combined_data.dur is scaled up by 10,000')
    combined_data['dur'] = 10000 * combined_data['dur']
    print(combined_data.head())

    # Prepare data for modeling
    print('Before splitting:', combined_data.shape)
    data_x = combined_data.drop(['attack_cat'], axis=1)
    data_y = combined_data['attack_cat']
    print(f"X shape: {data_x.shape}, Y shape: {data_y.shape}")

    # Min-max scaling
    data_x = data_x.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.20, random_state=42
    )

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Initial model training
    print("\n--- Initial Model Training ---")
    DTC = DecisionTreeClassifier() 
    RFC = RandomForestClassifier(n_estimators=50, random_state=1)
    ETC = ExtraTreesClassifier(n_estimators=75, criterion='gini', bootstrap=False)

    # Store accuracies
    accuracies = {}

    # Train classifier models
    eclf = VotingClassifier(estimators=[('lr', DTC), ('rf', RFC), ('et', ETC)], voting='hard') 
    for clf, label in zip([DTC, RFC, ETC, eclf], 
                        ['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'Ensemble']): 
        clf.fit(X_train, y_train)
        pred = clf.score(X_test, y_test)
        accuracies[label] = pred
        print(f"Accuracy: {pred:.7f} [{label}]")

    # Feature selection with RFE
    print("\n--- Feature Selection with RFE ---")
    rfe = RFE(DecisionTreeClassifier(), n_features_to_select=10).fit(X_train, y_train)
    desiredIndices = np.where(rfe.support_==True)[0]
    selected_features = list(X_train.columns[desiredIndices])
    print("Selected features:", selected_features)

    # Convert to DataFrame if not already
    X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test

    whitelist = X_train_df.columns.values[desiredIndices]
    X_train_RFE, X_test_RFE = X_train_df[whitelist], X_test_df[whitelist]
    
    print(f"RFE selected {len(whitelist)} features")
    print(f"X_train_RFE shape: {X_train_RFE.shape}, X_test_RFE shape: {X_test_RFE.shape}")

    # Training with selected features
    print("\n--- Training with Selected Features ---")
    DTC_RFE = DecisionTreeClassifier() 
    RFC_RFE = RandomForestClassifier(n_estimators=50, random_state=1)
    ETC_RFE = ExtraTreesClassifier(n_estimators=75, criterion='gini', bootstrap=False)

    # Train classifier models with RFE
    eclf_RFE = VotingClassifier(estimators=[
        ('lr', DTC_RFE), ('rf', RFC_RFE), ('et', ETC_RFE)], voting='hard') 
    
    for clf, label in zip([DTC_RFE, RFC_RFE, ETC_RFE, eclf_RFE], 
                        ['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'Ensemble']): 
        clf.fit(X_train_RFE, y_train)
        pred = clf.score(X_test_RFE, y_test)
        accuracies[f"{label}_RFE"] = pred
        print(f"Accuracy with RFE: {pred:.7f} [{label}]")

    print("\nUnique classes in y_train:", set(y_train))
    
    # Get feature importances from Random Forest
    RFC_RFE.fit(X_train_RFE, y_train)
    importances = RFC_RFE.feature_importances_
    
    # Plot feature importance
    plot_feature_importance(selected_features, importances)
    
    # Train neural network model
    print("\n--- Neural Network Training ---")
    nn_model, nn_acc = train_neural_network(X_train_RFE, y_train, X_test_RFE, y_test)
    accuracies["Neural Network"] = nn_acc/100  # Convert percentage to decimal for consistent reporting
    
    # Print summary of all models
    print("\n--- Model Accuracy Summary ---")
    for model_name, accuracy in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:25s}: {accuracy:.7f}")
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()