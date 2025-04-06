import os
import matplotlib.pyplot as plt
import numpy as np

def read_float_from_file(filepath):
    """Reads a float value from a file."""
    try:
        with open(filepath, 'r') as file:
            return float(file.read().strip())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def collect_accuracies_by_group(base_dir):
    """Collects accuracies grouped by subdirectories."""
    grouped_accuracies = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                accuracy = read_float_from_file(filepath)
                if accuracy is not None:
                    # Group by the immediate subdirectory
                    group = os.path.relpath(root, base_dir)
                    if group not in grouped_accuracies:
                        grouped_accuracies[group] = {}
                    relative_path = os.path.relpath(filepath, base_dir)
                    grouped_accuracies[group][relative_path] = accuracy
    return grouped_accuracies
def visualize_datasets_as_grid(grouped_accuracies):
    """Visualizes accuracies for each dataset as a grid of subplots."""
    # Flatten and group by dataset names
    dataset_accuracies = {}
    
    for group, accuracies in grouped_accuracies.items():
        for file, accuracy in accuracies.items():
            dataset_name = os.path.basename(file).replace('.txt', '')  # Extract dataset name
            if dataset_name not in dataset_accuracies:
                dataset_accuracies[dataset_name] = []
            dataset_accuracies[dataset_name].append((group, accuracy))

    transposed_accuracies = {}
    for dataset_name, accuracies in dataset_accuracies.items():
        for method_name, accuracy in accuracies:
            if method_name not in transposed_accuracies:
                transposed_accuracies[method_name] = {}
            transposed_accuracies[method_name][dataset_name] = accuracy
    #if dataset can be convert to int
    #sort by int
    if all(dataset.isdigit() for dataset in dataset_accuracies.keys()):
        sorted_datasets = sorted(dataset_accuracies.keys(), key=lambda x: int(x))
    else:
        # Sort datasets alphabetically
        sorted_datasets = sorted(dataset_accuracies.keys())
    mean_dataset_accuracies = {name: np.mean([acc for _, acc in dataset_accuracies[name]]) for name in sorted_datasets}
    mean_method_accuracies = ({method: np.mean([acc for dataset, acc in accuracies.items()]) for method, accuracies in transposed_accuracies.items()})
    var_method_accuracies = {method: np.var([acc for dataset, acc in accuracies.items()]) for method, accuracies in transposed_accuracies.items()}
    # Determine grid size (e.g., 3 columns)
    num_datasets = len(sorted_datasets)+2
    num_cols = 5
    num_rows = (num_datasets + num_cols - 1) // num_cols  # Calculate rows needed

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten axes for easy indexing
    mean_dataset_axes = axes[0]  # First subplot for mean accuracies
    mean_groups, mean_values = zip(*sorted(mean_dataset_accuracies.items(), key=lambda x: x[1], reverse=True))
    mean_dataset_axes.bar(mean_groups, mean_values, color='skyblue')
    mean_dataset_axes.set_title('Mean Accuracies')
    mean_dataset_axes.set_xlabel('Dataset')
    mean_dataset_axes.set_ylabel('Mean Accuracy')
    mean_dataset_axes.set_xticks(range(len(mean_groups)))
    mean_dataset_axes.set_xticklabels(mean_groups, rotation=45, ha='right', fontsize=8)
    
    mean_method_axes = axes[1]  # Second subplot for mean accuracies
    mean_method_groups, mean_method_values = zip(*sorted(mean_method_accuracies.items(), key=lambda x: x[0], reverse=True))
    _ ,var_method_values = zip(*sorted(var_method_accuracies.items(), key=lambda x: x[0], reverse=True))
    mean_method_axes.bar(mean_method_groups, mean_method_values, color='lightgreen')
    mean_method_axes.set_title('Mean Accuracies by Method (with Variance)')
    mean_method_axes.set_xlabel('Method')
    mean_method_axes.set_ylabel('Mean Accuracy')
    mean_method_axes.set_xticks(range(len(mean_method_groups)))
    mean_method_axes.set_xticklabels(mean_method_groups, rotation=45, ha='right', fontsize=8)
    # Example: Adding scatter points for individual accuracies
    for i, method in enumerate(mean_method_groups):
        accuracies = list(transposed_accuracies[method].values())
        x_positions = [i] * len(accuracies)
        mean_method_axes.scatter(x_positions, accuracies, color='blue', alpha=0.6)
    mean_method_axes.legend()
    mean_method_axes.set_title('Mean Accuracies by Method (With Scatter Points)')
    mean_method_axes.set_xlabel('Method')
    mean_method_axes.set_ylabel('Accuracy')
    mean_method_axes.set_xticks(range(len(mean_method_groups)))
    mean_method_axes.set_xticklabels(mean_method_groups, rotation=45, ha='right', fontsize=8)
    for i, dataset_name in enumerate(sorted_datasets):
        ax = axes[i+2]
        accuracies = dataset_accuracies[dataset_name]
        groups, values = zip(*sorted(accuracies, key=lambda x: x[0], reverse=True))

        # Assign unique colors to each group
        colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))

        # Plot bar chart for the dataset
        ax.bar(groups, values, color=colors)
        ax.set_title(dataset_name)
        ax.set_xlabel('method')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(len(groups)))
        #set y range
        ax.set_ylim(0, 100)
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=8)

    # Hide unused subplots
    for j in range(num_datasets, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('accuracies_visualization_grid.png')
    plt.show()

if __name__ == "__main__":
    base_directory = "/home/lzq/workspace/guess-what-moves/log"  # Change this to your base directory
    grouped_accuracies = collect_accuracies_by_group(base_directory)
    visualize_datasets_as_grid(grouped_accuracies)