import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def load_point_cloud(file_path):
    """Load point cloud data from numpy file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.load(file_path)
    print(f"Loaded point cloud shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data range: min={data.min():.4f}, max={data.max():.4f}")
    
    return data

def visualize_point_cloud_3d(points, title="3D Point Cloud", save_path=None, show_axes=True):
    """
    Create a 3D scatter plot of point cloud data.
    
    Args:
        points: numpy array of shape (N, 3) containing x, y, z coordinates
        title: title for the plot
        save_path: path to save the plot (optional)
        show_axes: whether to show axis labels and grid
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    if points.shape[1] >= 3:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
    else:
        raise ValueError(f"Expected at least 3 dimensions, got {points.shape[1]}")
    
    # Create scatter plot with color mapping based on z-coordinate
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6, s=1)
    
    # Customize the plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if show_axes:
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='Z-coordinate')
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def visualize_multiple_views(points, base_title="3D Point Cloud", save_dir=None):
    """Create multiple views of the same point cloud."""
    views = [
        (20, 45, "Default View"),
        (0, 0, "Front View (XZ)"),
        (0, 90, "Side View (YZ)"),
        (90, 0, "Top View (XY)")
    ]
    
    fig = plt.figure(figsize=(16, 12))
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    for i, (elev, azim, view_name) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6, s=1)
        ax.set_title(f"{base_title} - {view_name}", fontsize=12)
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "multiple_views.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multiple views saved to: {save_path}")
    
    plt.show()

def visualize_multiple_batches(data, num_batches, base_title="Point Cloud Batches", save_dir=None):
    """Visualize multiple batches of point clouds in a grid layout."""
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D data (batch, points, dims), got shape {data.shape}")
    
    total_batches = data.shape[0]
    if num_batches > total_batches:
        print(f"Requested {num_batches} batches but only {total_batches} available. Using all {total_batches} batches.")
        num_batches = total_batches
    
    # Sample random batches
    import random
    batch_indices = random.sample(range(total_batches), num_batches)
    batch_indices.sort()
    
    # Calculate grid layout
    cols = min(5, num_batches)  # Max 5 columns
    rows = (num_batches + cols - 1) // cols
    
    fig = plt.figure(figsize=(4*cols, 3*rows))
    
    for i, batch_idx in enumerate(batch_indices):
        points = data[batch_idx]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.7, s=0.5)
        
        ax.set_title(f"Batch {batch_idx}", fontsize=10)
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Remove axis labels for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle(f"{base_title} - {num_batches} Samples", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f"multiple_batches_{num_batches}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multiple batches plot saved to: {save_path}")
    
    plt.show()
    
    print(f"Visualized batches: {batch_indices}")

def print_statistics(points):
    """Print basic statistics about the point cloud."""
    print("\n" + "="*50)
    print("POINT CLOUD STATISTICS")
    print("="*50)
    print(f"Number of points: {points.shape[0]:,}")
    print(f"Dimensions: {points.shape[1]}")
    
    for i in range(min(3, points.shape[1])):
        coord_name = ['X', 'Y', 'Z'][i]
        coord_data = points[:, i]
        print(f"\n{coord_name}-coordinate statistics:")
        print(f"  Min: {coord_data.min():.6f}")
        print(f"  Max: {coord_data.max():.6f}")
        print(f"  Mean: {coord_data.mean():.6f}")
        print(f"  Std: {coord_data.std():.6f}")
    
    print("\nOverall statistics:")
    print(f"  Global min: {points.min():.6f}")
    print(f"  Global max: {points.max():.6f}")
    print(f"  Global mean: {points.mean():.6f}")
    print(f"  Global std: {points.std():.6f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D point cloud from numpy file')
    parser.add_argument('file_path', help='Path to the .npy file containing point cloud data')
    parser.add_argument('--save-dir', help='Directory to save visualization plots')
    parser.add_argument('--title', default='3D Point Cloud Visualization', help='Title for the plots')
    parser.add_argument('--multiple-views', action='store_true', help='Show multiple viewing angles')
    parser.add_argument('--no-stats', action='store_true', help='Skip printing statistics')
    parser.add_argument('--sample-batches', type=int, help='Number of batches to sample and visualize (for 3D data)')
    
    args = parser.parse_args()
    
    try:
        # Load the point cloud data
        data = load_point_cloud(args.file_path)
        
        # Create save directory if specified
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
        
        # Handle sample-batches option first
        if args.sample_batches:
            if len(data.shape) != 3:
                raise ValueError(f"--sample-batches requires 3D data (batch, points, dims), got shape {data.shape}")
            
            visualize_multiple_batches(
                data,
                num_batches=args.sample_batches,
                base_title=args.title,
                save_dir=args.save_dir
            )
            return 0
        
        # Handle different data shapes for single visualization
        if len(data.shape) == 3:
            # If data is (batch, points, dims), take the first batch
            points = data[0]
        elif len(data.shape) == 2:
            # Data is already (points, dims)
            points = data
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        # Ensure we have at least 3 dimensions
        if points.shape[1] < 3:
            raise ValueError(f"Need at least 3 dimensions for 3D visualization, got {points.shape[1]}")
        
        # Print statistics
        if not args.no_stats:
            print_statistics(points)
        
        # Create visualizations
        if args.multiple_views:
            visualize_multiple_views(
                points, 
                base_title=args.title,
                save_dir=args.save_dir
            )
        else:
            save_path = None
            if args.save_dir:
                save_path = os.path.join(args.save_dir, "point_cloud_3d.png")
            
            visualize_point_cloud_3d(
                points,
                title=args.title,
                save_path=save_path
            )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
