import numpy as np
import wandb
import matplotlib.pyplot as plt
from typing import Dict, Optional
from omnimimic.data.data_utils import unnormalize_action
import os

VISUALIZATION_IMAGE_SIZE = (128, 128) 
#Define colors for convenience
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])


def visualize_train_images(
    batch_obs_images,
    save_folder,
    epoch,
    num_images_log=10
):
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        f"epoch{epoch}",
        "train_images"
    )
   
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path) 
    wandb_list = []
    batch_obs_images = np.transpose(batch_obs_images, (0, 2, 3, 1))
    save_path = os.path.join(visualize_path, f"{epoch}.png")
    plt.figure()
    fig, ax = plt.subplots(1, num_images_log)

    indices = np.random.choice(np.arange(len(batch_obs_images)), 
        size=num_images_log,
        replace=False)
    for i, axis in enumerate(ax):
        img = batch_obs_images[num_images_log[i]]
        axis.imshow(img[:, :, -3:])
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
    fig.savefig(
        save_path,
        bbox_inches="tight"
    )
    wandb.log({f"train_images": wandb.Image(
        save_path)}, commit=False)


def visualize_obs_action(obs, goal_image, action, loss_dict, num_to_visualize=5, val_key='Train',
    epoch=0):
    action_pred_key = [k for k in loss_dict.keys() if 'action_pred' in k][0]
    batch_size = len(obs)
    indices = np.random.randint(batch_size, size=num_to_visualize)

    table = wandb.Table(columns=['Image', 'Goal Image', 'Action Pred', 'Action', 'Action L2 Distance'])
    for idx in indices: 
        curr_action_pred = loss_dict[action_pred_key][idx]
        curr_action = action[idx]
        curr_action_distance = np.linalg.norm(curr_action_pred - curr_action)
        table.add_data(wandb.Image(obs[idx]), wandb.Image(goal_image[idx]), str(curr_action_pred),
            str(curr_action), curr_action_distance)

    wandb.log(data={f"{val_key} Obs Action Table" : table}, step=epoch)


def visualize_action_dist(action, loss_dict, val_key='Train', epoch=0):
    action_pred_key = [k for k in loss_dict.keys() if 'action_pred' in k][0]
    action_dim = action.shape[-1]
    for i in range(action_dim):
        data = action[:, i:i+1].tolist()
        act_table = wandb.Table(data=data, columns=["values"])

        action_pred = loss_dict[action_pred_key]
        data = action_pred[:, i:i+1].tolist()
        pred_table = wandb.Table(data=data, columns=["values"])

        
        wandb.log(data={f'{val_key} Action Histogram dim {i}': wandb.plot.histogram(act_table, 'values',
 	            title=f'Action Dim {i} Distribution')}, step=epoch)
        wandb.log(data={f'{val_key} Action Pred Histogram dim {i}': wandb.plot.histogram(pred_table, 'values',
                title=f'Action Pred Dim {i} Distribution')}, step=epoch)


def visualize_goal_reaching_plots(obs_image, goal_image, goal_state, action, loss_dict, 
    metadata, num_plots=5, val_key='Train', epoch=0, no_normalization=False,
    dataloader_config=None):
    action_pred_key = [k for k in loss_dict.keys() if 'action_pred' in k][0]
    discrete = dataloader_config['discrete']
    num_bins = dataloader_config['num_bins']

    for i in range(num_plots):
        obs_img = obs_image[i]
        goal_img = goal_image[i]
        goal_pos = goal_state[i][:2]
        pred_waypoints = loss_dict[action_pred_key][i]
        gt_waypoints = action[i]
        if not no_normalization or discrete:
            pred_waypoints = unnormalize_action(pred_waypoints, 
                    metadata,
                    discrete=discrete, num_bins=num_bins)
            gt_waypoints = unnormalize_action(gt_waypoints, 
                    metadata,
                    discrete=discrete, num_bins=num_bins)
        pred_waypoints = pred_waypoints[:, 2:0:-1]
        gt_waypoints = gt_waypoints[:, 2:0:-1]
        pred_waypoints[:, 0] *= -1
        gt_waypoints = np.copy(gt_waypoints)
        gt_waypoints[:, 0] *= -1
        if dataloader_config['gnm_delta_actions']:
            pred_waypoints = np.cumsum(pred_waypoints, axis=0)
            gt_waypoints = np.cumsum(gt_waypoints, axis=0)
        pred_waypoints = np.concatenate(([[0, 0]], pred_waypoints), axis=0)
        gt_waypoints = np.concatenate(([[0, 0]], gt_waypoints), axis=0)
        print("pred_waypoints", pred_waypoints)
        print("gt_waypoints", gt_waypoints)
        trajs = [pred_waypoints, gt_waypoints]
        fig, ax = plt.subplots(1, 3)
        start_pos = np.array([0, 0])  
        plot_trajs_and_points(
            ax[0],
            trajs,
            [start_pos, goal_pos],
            traj_colors=[CYAN, MAGENTA],
            point_colors=[GREEN, RED],
        )
        ax[1].imshow(obs_img[:, :, -3:])
        ax[2].imshow(goal_img)

        fig.set_size_inches(18.5, 10.5)
        ax[0].set_title("Action Prediction")
        ax[1].set_title("Obs")
        ax[2].set_title("Goal")
        wandb.log(data={f"{val_key} Goal Reaching Plot {i}": fig}, step=epoch) 
      


def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
    quiver_freq: int = 1,
    default_coloring: bool = True,
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
        traj_labels: list of labels for trajectories
        point_labels: list of labels for points
        traj_alphas: list of alphas for trajectories
        point_alphas: list of alphas for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """
    assert (
        len(list_trajs) <= len(traj_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
        traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
            bearings = gen_bearings_from_waypoints(traj)
            ax.quiver(
                traj[::quiver_freq, 0],
                traj[::quiver_freq, 1],
                bearings[::quiver_freq, 0],
                bearings[::quiver_freq, 1],
                color=traj_colors[i] * 0.5,
                scale=1.0,
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0], 
                pt[1], 
                color=point_colors[i], 
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )

    
    # put the legend below the plot
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")


def gen_bearings_from_waypoints(
    waypoints: np.ndarray,
    mag=0.2,
) -> np.ndarray:
    """Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    bearing = []
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # label is sin/cos repr
            v = waypoints[i, 2:]
            # normalize v
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # label is radians repr
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing
         
